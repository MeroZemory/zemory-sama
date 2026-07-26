"""Central orchestrator.

Wires the five tasks of the pipeline:

1. mic pump (PCM → TurnDetector, gated by profile/barge-in policy)
2. LLM event consumer (deltas → TTSTaskManager, turn events → Phase/interrupt)
3. TurnDetector event consumer (local profile; speech_end → STT → LLM inject)
4. generation-tagged Realtime audio relay
5. SpeakerStream.feed
6. startup ready-beep task

There is no explicit TTS worker in this group; ``TTSTaskManager`` owns its
bounded task set.

End-to-end latency measurement emits a ``turn.complete`` structlog entry
carrying ``speech_end_ts``, ``first_llm_delta_ts``, ``first_tts_byte_ts``,
``speaker_first_play_ts``, ``total_ms``, ``profile``, ``interrupted``.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections import deque
from collections.abc import Awaitable
from pathlib import Path

from openai import AsyncOpenAI

from zemory.audio import MicrophoneStream, SpeakerStream, generate_beep_pcm
from zemory.config import (
    ELEVENLABS_API_KEY,
    OPENAI_API_KEY,
    ProfileName,
    canonical_profile,
    settings,
    validate_runtime_credentials,
)
from zemory.observability import configure_logging, get_logger, metrics
from zemory.pipeline.chunker import SentenceChunker
from zemory.pipeline.context import (
    AsyncContextScheduler,
    SQLiteMemoryStore,
)
from zemory.pipeline.interrupt_bus import InterruptBus
from zemory.pipeline.realtime_events import handle_speech_started
from zemory.pipeline.runtime_cleanup import RuntimeCleanup
from zemory.pipeline.transcript_corrector import TranscriptCorrector
from zemory.pipeline.tts_manager import TTSTaskManager
from zemory.providers.base import build_pipeline
from zemory.state import Phase, StateMachine

_log = get_logger("orchestrator")

_INPUT_TRANSCRIPT_TIMEOUT_S = 5.0
_MIC_HEALTH_POLL_S = 0.05
_REALTIME_AUDIO_QUEUE_MAX_CHUNKS = 128
_PROVIDER_CONTROL_TIMEOUT_S = 1.0
_RESPONSE_TERMINAL_TIMEOUT_S = 90.0
_CLEANUP_TIMEOUT_S = 2.0
_LOCAL_ITEM_ID_TRACKING_LIMIT = 256
_PENDING_PROVIDER_CONTROL_LIMIT = 64


def build_context_scheduler(*, profile: ProfileName) -> AsyncContextScheduler:
    """Build context only for profiles that actually consume it at runtime."""
    memory = None
    if settings.memory_enabled and profile == "local_cascade":
        memory_path = Path(settings.memory_path).expanduser()
        if not memory_path.is_absolute():
            memory_path = Path.cwd() / memory_path
        memory = SQLiteMemoryStore(memory_path)
    return AsyncContextScheduler(
        memory=memory,
        memory_deadline_ms=settings.memory_recall_deadline_ms,
        tool_deadline_ms=settings.context_tool_deadline_ms,
        memory_limit=settings.memory_recall_limit,
    )


def build_transcript_corrector() -> TranscriptCorrector | None:
    """Build the optional corrector with one profile-independent deadline."""
    if not settings.transcript_correction_enabled:
        return None
    corrector = TranscriptCorrector(
        client=AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=settings.openai_base_url,
            timeout=settings.transcript_correction_timeout_s,
            max_retries=0,
        ),
        model=settings.transcript_correction_model,
        history_turns=settings.transcript_correction_history_turns,
        owns_client=True,
        timeout_s=settings.transcript_correction_timeout_s,
    )
    _log.info(
        "correction.enabled",
        model=settings.transcript_correction_model,
        history_turns=settings.transcript_correction_history_turns,
        timeout_s=settings.transcript_correction_timeout_s,
    )
    return corrector


class TurnTimer:
    """Per-turn stopwatch capturing the five latency milestones."""

    __slots__ = (
        "turn_id",
        "speech_end_ts",
        "first_llm_delta_ts",
        "first_tts_byte_ts",
        "speaker_first_play_ts",
        "speaker_buffer_ms",
        "correction_ms",
        "interrupted",
    )

    def __init__(self, turn_id: int) -> None:
        self.turn_id = turn_id
        self.speech_end_ts: float | None = None
        self.first_llm_delta_ts: float | None = None
        self.first_tts_byte_ts: float | None = None
        self.speaker_first_play_ts: float | None = None
        self.speaker_buffer_ms: float | None = None
        self.correction_ms: float | None = None
        self.interrupted = False

    def total_ms(self) -> float | None:
        if self.speech_end_ts is None or self.speaker_first_play_ts is None:
            return None
        return (self.speaker_first_play_ts - self.speech_end_ts) * 1000


async def run() -> None:
    validate_runtime_credentials()
    configure_logging()
    loop = asyncio.get_running_loop()

    pipeline = build_pipeline(
        settings.profile,
        openai_api_key=OPENAI_API_KEY,
        elevenlabs_api_key=ELEVENLABS_API_KEY,
    )
    profile = canonical_profile(settings.profile)
    manual_realtime_turns = (
        profile in {"realtime_audio", "realtime_text_external_tts"}
        and settings.realtime.turn_detection == "none"
    )

    mic = MicrophoneStream(loop)
    speaker = SpeakerStream(loop)
    state = StateMachine()

    turn_seq = 0
    timer = TurnTimer(turn_seq)
    item_ids: list[str] = []
    item_id_set: set[str] = set()
    context_scheduler = build_context_scheduler(profile=profile)

    generation_id = 0
    current_response_id: str | None = None
    expected_input_item_id: str | None = None
    current_output_item_id: str | None = None
    current_output_content_index = 0
    cancelled_response_ids: set[str] = set()
    cancelled_response_order: deque[str] = deque()
    handled_response_create_error_ids: set[str] = set()
    handled_response_create_error_order: deque[str] = deque()
    response_cancel_barriers: dict[str, asyncio.Future[bool]] = {}
    cancel_barrier_failure_pending = False
    unscoped_cancel_barrier: asyncio.Future[bool] | None = None
    input_clear_barrier: asyncio.Future[bool] | None = None
    input_clear_generation: int | None = None
    manual_input_boundary_generation: int | None = None
    item_delete_barriers: dict[str, asyncio.Future[bool]] = {}
    item_truncate_barriers: dict[str, asyncio.Future[bool]] = {}
    item_mutation_failure_pending = False
    scoped_cancel_tasks: dict[str, asyncio.Task] = {}
    stale_item_delete_tasks: dict[str, asyncio.Task] = {}
    stale_item_delete_ids: set[str] = set()
    stale_item_delete_order: deque[str] = deque()
    provider_control_lock = asyncio.Lock()
    correction_task: asyncio.Task[None] | None = None
    speculative_correction_input: tuple[int, str] | None = None
    corrector_user_history_generation: int | None = None
    finalize_task: asyncio.Task[None] | None = None
    transcript_guard_task: asyncio.Task[None] | None = None
    response_request_task: asyncio.Task[None] | None = None
    active_response_conflict_task: asyncio.Task[None] | None = None
    response_watchdog_task: asyncio.Task[None] | None = None
    response_resolution_generation: int | None = None
    response_resolution_owner: str | None = None
    background_tasks: set[asyncio.Task] = set()
    runtime_failures: asyncio.Queue[RuntimeError] = asyncio.Queue(maxsize=1)
    realtime_audio_queue: asyncio.Queue[tuple[int, str | None, bytes]] = (
        asyncio.Queue(maxsize=_REALTIME_AUDIO_QUEUE_MAX_CHUNKS)
    )

    def _drain_realtime_audio_queue() -> None:
        while True:
            try:
                realtime_audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            else:
                realtime_audio_queue.task_done()

    async def on_partial_abort(partial: str) -> None:
        """Report discarded partial text without retaining its contents."""
        _log.info("interrupt.partial_discarded", generated_chars=len(partial))

    def _remember_cancelled_response(response_id: str) -> bool:
        if response_id in cancelled_response_ids:
            return False
        if len(cancelled_response_order) >= 64:
            cancelled_response_ids.discard(cancelled_response_order.popleft())
        cancelled_response_order.append(response_id)
        cancelled_response_ids.add(response_id)
        return True

    def _remember_response_create_error(client_event_id: str) -> bool:
        if client_event_id in handled_response_create_error_ids:
            return False
        if len(handled_response_create_error_order) >= 64:
            handled_response_create_error_ids.discard(
                handled_response_create_error_order.popleft()
            )
        handled_response_create_error_order.append(client_event_id)
        handled_response_create_error_ids.add(client_event_id)
        return True

    def _prune_completed_cancel_barriers() -> None:
        """Keep completed ACK outcomes without retaining one Future per ID."""
        nonlocal cancel_barrier_failure_pending
        for response_id, barrier in tuple(response_cancel_barriers.items()):
            if not barrier.done():
                continue
            response_cancel_barriers.pop(response_id, None)
            try:
                succeeded = not barrier.cancelled() and bool(barrier.result())
            except Exception:
                succeeded = False
            cancel_barrier_failure_pending = (
                cancel_barrier_failure_pending or not succeeded
            )

    def _register_cancel_barrier(response_id: str) -> asyncio.Future[bool]:
        barrier = response_cancel_barriers.get(response_id)
        if barrier is None:
            _prune_completed_cancel_barriers()
            if len(response_cancel_barriers) >= _PENDING_PROVIDER_CONTROL_LIMIT:
                raise RuntimeError(
                    "Pending response cancellation ACK capacity exceeded"
                )
            barrier = asyncio.get_running_loop().create_future()
            response_cancel_barriers[response_id] = barrier
        return barrier

    def _resolve_cancel_barrier(response_id: str, *, succeeded: bool) -> None:
        barrier = response_cancel_barriers.get(response_id)
        if barrier is not None and not barrier.done():
            barrier.set_result(succeeded)

    def _register_unscoped_cancel_barrier() -> asyncio.Future[bool]:
        nonlocal unscoped_cancel_barrier
        if unscoped_cancel_barrier is None or unscoped_cancel_barrier.done():
            unscoped_cancel_barrier = asyncio.get_running_loop().create_future()
        return unscoped_cancel_barrier

    def _resolve_unscoped_cancel_barrier(*, succeeded: bool) -> None:
        if (
            unscoped_cancel_barrier is not None
            and not unscoped_cancel_barrier.done()
        ):
            unscoped_cancel_barrier.set_result(succeeded)

    def _register_input_clear_barrier(
        captured_generation: int,
    ) -> asyncio.Future[bool]:
        nonlocal input_clear_barrier, input_clear_generation
        if input_clear_barrier is not None and not input_clear_barrier.done():
            if input_clear_generation != captured_generation:
                raise RuntimeError(
                    "A manual input clear is already pending for another generation"
                )
            return input_clear_barrier
        input_clear_barrier = asyncio.get_running_loop().create_future()
        input_clear_generation = captured_generation
        return input_clear_barrier

    def _resolve_input_clear_barrier(
        captured_generation: int,
        *,
        succeeded: bool,
    ) -> bool:
        if (
            input_clear_barrier is None
            or input_clear_barrier.done()
            or input_clear_generation != captured_generation
        ):
            return False
        input_clear_barrier.set_result(succeeded)
        return True

    def _prune_completed_item_mutation_barriers() -> None:
        nonlocal item_mutation_failure_pending
        for barriers in (item_delete_barriers, item_truncate_barriers):
            for item_id, barrier in tuple(barriers.items()):
                if not barrier.done():
                    continue
                barriers.pop(item_id, None)
                try:
                    succeeded = not barrier.cancelled() and bool(barrier.result())
                except Exception:
                    succeeded = False
                item_mutation_failure_pending = (
                    item_mutation_failure_pending or not succeeded
                )

    def _register_item_delete_barrier(item_id: str) -> asyncio.Future[bool]:
        _prune_completed_item_mutation_barriers()
        barrier = item_delete_barriers.get(item_id)
        if barrier is None:
            if (
                len(item_delete_barriers) + len(item_truncate_barriers)
                >= _PENDING_PROVIDER_CONTROL_LIMIT
            ):
                raise RuntimeError(
                    "Pending item mutation ACK capacity exceeded"
                )
            barrier = asyncio.get_running_loop().create_future()
            item_delete_barriers[item_id] = barrier
        return barrier

    def _resolve_item_delete_barrier(item_id: str, *, succeeded: bool) -> None:
        barrier = item_delete_barriers.get(item_id)
        if barrier is not None and not barrier.done():
            barrier.set_result(succeeded)

    def _register_item_truncate_barrier(item_id: str) -> asyncio.Future[bool]:
        _prune_completed_item_mutation_barriers()
        barrier = item_truncate_barriers.get(item_id)
        if barrier is None:
            if (
                len(item_delete_barriers) + len(item_truncate_barriers)
                >= _PENDING_PROVIDER_CONTROL_LIMIT
            ):
                raise RuntimeError(
                    "Pending item mutation ACK capacity exceeded"
                )
            barrier = asyncio.get_running_loop().create_future()
            item_truncate_barriers[item_id] = barrier
        return barrier

    def _resolve_item_truncate_barrier(item_id: str, *, succeeded: bool) -> None:
        barrier = item_truncate_barriers.get(item_id)
        if barrier is not None and not barrier.done():
            barrier.set_result(succeeded)

    def _signal_runtime_failure(message: str) -> None:
        """Fail the owning TaskGroup without retaining provider payloads."""
        if runtime_failures.empty():
            runtime_failures.put_nowait(RuntimeError(message))

    async def _runtime_failure_monitor() -> None:
        raise await runtime_failures.get()

    def _forget_item(item_id: str) -> None:
        item_id_set.discard(item_id)
        try:
            item_ids.remove(item_id)
        except ValueError:
            pass

    def on_output_interrupted(*, register_cancel: bool = True) -> Awaitable[None]:
        """Snapshot the output cursor, then return remote history synchronization."""
        nonlocal current_output_item_id
        timer.interrupted = True
        response_id = current_response_id
        item_id = current_output_item_id
        content_index = current_output_content_index
        played_audio_ms = speaker.played_audio_ms
        # Audio still waiting in the relay was never heard and must not cross
        # into the next response generation.
        _drain_realtime_audio_queue()
        if response_id and register_cancel:
            _remember_cancelled_response(response_id)
            _register_cancel_barrier(response_id)
        mutation_barrier: asyncio.Future[bool] | None = None
        if item_id:
            mutation_barrier = (
                _register_item_truncate_barrier(item_id)
                if profile == "realtime_audio"
                else _register_item_delete_barrier(item_id)
            )
        current_output_item_id = None

        async def synchronize() -> None:
            if not item_id:
                return
            assert mutation_barrier is not None
            try:
                if profile == "realtime_audio":
                    truncate = getattr(pipeline.llm, "truncate_item", None)
                    if not callable(truncate):
                        _resolve_item_truncate_barrier(item_id, succeeded=False)
                    else:
                        await truncate(
                            item_id,
                            content_index=content_index,
                            audio_end_ms=played_audio_ms,
                        )
                else:
                    # Text responses are synthesized outside Realtime and
                    # cannot be audio-truncated. Remove the full item.
                    delete = getattr(pipeline.llm, "delete_item", None)
                    if not callable(delete):
                        _resolve_item_delete_barrier(item_id, succeeded=False)
                    else:
                        deleted = await delete(item_id)
                        if deleted is False:
                            _resolve_item_delete_barrier(item_id, succeeded=False)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if profile == "realtime_audio":
                    _resolve_item_truncate_barrier(item_id, succeeded=False)
                else:
                    _resolve_item_delete_barrier(item_id, succeeded=False)
                _log.warning(
                    "interrupt.item_mutation_send_failed",
                    error_type=type(e).__name__,
                )
                _signal_runtime_failure(
                    "Interrupted response history mutation failed"
                )
                return

            # Shield preserves the authoritative ACK barrier if InterruptBus's
            # short latency budget expires; the next response still waits for
            # it rather than reusing partially-mutated provider history.
            mutation_succeeded = bool(await asyncio.shield(mutation_barrier))
            if not mutation_succeeded:
                _signal_runtime_failure(
                    "Interrupted response history mutation was not acknowledged"
                )
                return
            if profile != "realtime_audio":
                _forget_item(item_id)
            record = getattr(pipeline.llm, "record_system_note", None)
            if callable(record):
                await record("(The previous assistant response was interrupted.)")

        return synchronize()

    interrupt_bus = InterruptBus(
        state,
        speaker,
        on_partial=on_partial_abort,
        on_output_interrupted=on_output_interrupted,
        get_response_id=lambda: current_response_id,
        on_remote_failure=lambda action: _signal_runtime_failure(
            f"Interrupt {action} synchronization failed"
        ),
    )

    # Optional transcript corrector (context-aware ASR fix-up).
    corrector = build_transcript_corrector()

    def _record_corrector_user_once(text: str, captured_generation: int) -> None:
        """Commit exactly one user entry before this generation's assistant."""
        nonlocal corrector_user_history_generation
        if (
            corrector is None
            or corrector_user_history_generation == captured_generation
        ):
            return
        corrector.record_user(text)
        corrector_user_history_generation = captured_generation

    def _on_first_tts(seq: int, ttfb_ms: float) -> None:
        # Record the first TTS chunk OF THE CURRENT TURN.
        # ``reset_for_new_turn`` wipes counters, so seq 0 here means the
        # first sentence of this turn (not the session).
        if seq == 0 and timer.first_tts_byte_ts is None:
            timer.first_tts_byte_ts = time.monotonic()

    tts_manager = TTSTaskManager(
        tts=pipeline.tts,
        speaker=speaker,
        max_concurrent=settings.tts.max_concurrent,
        on_first_chunk=_on_first_tts,
    )
    interrupt_bus.bind(tts_manager, pipeline.llm)

    def _track_background(coro, *, name: str) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)
        background_tasks.add(task)

        def _done(completed: asyncio.Task) -> None:
            background_tasks.discard(completed)
            if completed.cancelled():
                return
            try:
                error = completed.exception()
            except Exception as e:  # pragma: no cover - defensive task cleanup
                _log.error(
                    "background_task.failed",
                    task=name,
                    error_type=type(e).__name__,
                )
                return
            if error is not None:
                _log.error(
                    "background_task.failed",
                    task=name,
                    error_type=type(error).__name__,
                )

        task.add_done_callback(_done)
        return task

    async def _cancel_response_by_id(
        response_id: str,
        *,
        failure_message: str = "Scoped response cancellation failed",
    ) -> None:
        """Cancel a stale response and hold a barrier until its server ACK."""
        barrier = _register_cancel_barrier(response_id)
        try:
            async with asyncio.timeout(_PROVIDER_CONTROL_TIMEOUT_S):
                async with provider_control_lock:
                    await pipeline.llm.cancel_current(response_id=response_id)
        except TimeoutError:
            _resolve_cancel_barrier(response_id, succeeded=False)
            _log.warning(
                "response.scoped_cancel_timeout",
                timeout_ms=int(_PROVIDER_CONTROL_TIMEOUT_S * 1000),
            )
            _signal_runtime_failure(failure_message)
            return
        except Exception as e:
            _resolve_cancel_barrier(response_id, succeeded=False)
            _log.warning(
                "response.scoped_cancel_failed",
                error_type=type(e).__name__,
            )
            _signal_runtime_failure(failure_message)
            return

        done, _ = await asyncio.wait(
            {barrier},
            timeout=_PROVIDER_CONTROL_TIMEOUT_S,
        )
        if not done:
            _resolve_cancel_barrier(response_id, succeeded=False)
            _log.warning(
                "response.scoped_cancel_ack_timeout",
                response_id=response_id,
                timeout_ms=int(_PROVIDER_CONTROL_TIMEOUT_S * 1000),
            )
            _signal_runtime_failure(failure_message)
        elif barrier.cancelled() or not bool(barrier.result()):
            _signal_runtime_failure(failure_message)

    def _schedule_scoped_cancel(response_id: str) -> None:
        # Install the barrier synchronously with scheduling so a transcript
        # event already queued behind response.created cannot race ahead of it.
        _register_cancel_barrier(response_id)
        existing = scoped_cancel_tasks.get(response_id)
        if existing is not None and not existing.done():
            return
        for stale_response_id, completed in tuple(scoped_cancel_tasks.items()):
            if completed.done():
                scoped_cancel_tasks.pop(stale_response_id, None)
        if len(scoped_cancel_tasks) >= _PENDING_PROVIDER_CONTROL_LIMIT:
            raise RuntimeError(
                "Pending stale response cancellation capacity exceeded"
            )
        task = _track_background(
            _cancel_response_by_id(response_id),
            name=f"cancel_stale_response_{response_id}",
        )
        scoped_cancel_tasks[response_id] = task

        def discard(completed: asyncio.Task) -> None:
            if scoped_cancel_tasks.get(response_id) is completed:
                scoped_cancel_tasks.pop(response_id, None)

        task.add_done_callback(discard)

    async def _wait_for_cancel_barriers() -> bool:
        """Wait until every earlier response is confirmed inactive."""
        nonlocal cancel_barrier_failure_pending, unscoped_cancel_barrier
        _prune_completed_cancel_barriers()
        prior_failure = cancel_barrier_failure_pending
        cancel_barrier_failure_pending = False
        scoped_snapshot = tuple(response_cancel_barriers.items())
        unscoped_snapshot = unscoped_cancel_barrier
        if not scoped_snapshot and unscoped_snapshot is None:
            return not prior_failure
        futures = {barrier for _, barrier in scoped_snapshot}
        if unscoped_snapshot is not None:
            futures.add(unscoped_snapshot)
        done, pending = await asyncio.wait(
            futures,
            timeout=_PROVIDER_CONTROL_TIMEOUT_S,
        )
        for barrier in pending:
            if not barrier.done():
                barrier.set_result(False)
        succeeded = not prior_failure
        for response_id, barrier in scoped_snapshot:
            if response_cancel_barriers.get(response_id) is barrier:
                response_cancel_barriers.pop(response_id, None)
            if barrier not in done or barrier.cancelled():
                succeeded = False
                continue
            try:
                succeeded = bool(barrier.result()) and succeeded
            except Exception:
                succeeded = False
        if unscoped_snapshot is not None:
            if unscoped_cancel_barrier is unscoped_snapshot:
                unscoped_cancel_barrier = None
            if unscoped_snapshot not in done or unscoped_snapshot.cancelled():
                succeeded = False
            else:
                try:
                    succeeded = bool(unscoped_snapshot.result()) and succeeded
                except Exception:
                    succeeded = False
        if not succeeded:
            _log.warning(
                "response.cancel_barrier_failed",
                barrier_count=len(futures),
            )
        return succeeded

    async def _wait_for_item_mutation_barriers() -> bool:
        """Block response reuse until remote history mutations are authoritative."""
        nonlocal item_mutation_failure_pending
        _prune_completed_item_mutation_barriers()
        prior_failure = item_mutation_failure_pending
        item_mutation_failure_pending = False
        snapshots = (
            tuple(("delete", item_id, barrier) for item_id, barrier in item_delete_barriers.items())
            + tuple(
                ("truncate", item_id, barrier)
                for item_id, barrier in item_truncate_barriers.items()
            )
        )
        if not snapshots:
            return not prior_failure
        futures = {barrier for _, _, barrier in snapshots}
        done, pending = await asyncio.wait(
            futures,
            timeout=_PROVIDER_CONTROL_TIMEOUT_S,
        )
        for barrier in pending:
            if not barrier.done():
                barrier.set_result(False)
        succeeded = not prior_failure
        for operation, item_id, barrier in snapshots:
            registry = (
                item_delete_barriers
                if operation == "delete"
                else item_truncate_barriers
            )
            if registry.get(item_id) is barrier:
                registry.pop(item_id, None)
            if barrier not in done or barrier.cancelled():
                succeeded = False
                continue
            try:
                succeeded = bool(barrier.result()) and succeeded
            except Exception:
                succeeded = False
        if not succeeded:
            _log.warning(
                "response.item_mutation_barrier_failed",
                barrier_count=len(futures),
            )
        return succeeded

    async def _delete_stale_item(item_id: str) -> None:
        if not await _delete_item_with_ack(item_id):
            _signal_runtime_failure(
                "Stale provider item deletion was not acknowledged"
            )

    async def _delete_item_with_ack(item_id: str) -> bool:
        """Delete one context item only after the server confirms the mutation."""
        barrier = _register_item_delete_barrier(item_id)
        try:
            async with asyncio.timeout(_PROVIDER_CONTROL_TIMEOUT_S):
                async with provider_control_lock:
                    accepted = await pipeline.llm.delete_item(item_id)
                if accepted is False:
                    _resolve_item_delete_barrier(item_id, succeeded=False)
                succeeded = bool(await barrier)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            succeeded = False
            _log.warning(
                "response.item_delete_ack_timeout",
                timeout_ms=int(_PROVIDER_CONTROL_TIMEOUT_S * 1000),
            )
        except Exception as e:
            succeeded = False
            _log.warning(
                "response.item_delete_failed",
                error_type=type(e).__name__,
            )
        finally:
            if item_delete_barriers.get(item_id) is barrier:
                item_delete_barriers.pop(item_id, None)
        if succeeded:
            _forget_item(item_id)
        return succeeded

    def _schedule_stale_item_delete(item_id: str) -> None:
        if item_id in stale_item_delete_ids:
            return
        for stale_item_id, completed in tuple(stale_item_delete_tasks.items()):
            if completed.done():
                stale_item_delete_tasks.pop(stale_item_id, None)
        if len(stale_item_delete_tasks) >= _PENDING_PROVIDER_CONTROL_LIMIT:
            raise RuntimeError(
                "Pending stale item deletion capacity exceeded"
            )
        if len(stale_item_delete_order) >= _LOCAL_ITEM_ID_TRACKING_LIMIT:
            stale_item_delete_ids.discard(stale_item_delete_order.popleft())
        stale_item_delete_order.append(item_id)
        stale_item_delete_ids.add(item_id)
        task = _track_background(
            _delete_stale_item(item_id),
            name=f"delete_stale_item_{item_id}",
        )
        stale_item_delete_tasks[item_id] = task

        def discard(completed: asyncio.Task) -> None:
            if stale_item_delete_tasks.get(item_id) is completed:
                stale_item_delete_tasks.pop(item_id, None)

        task.add_done_callback(discard)

    async def _synchronize_terminal_partial(
        item_id: str,
        *,
        content_index: int,
        audio_end_ms: int,
    ) -> bool:
        """Keep failed/incomplete output out of future server context."""
        barrier = (
            _register_item_truncate_barrier(item_id)
            if profile == "realtime_audio"
            else _register_item_delete_barrier(item_id)
        )
        try:
            async with asyncio.timeout(_PROVIDER_CONTROL_TIMEOUT_S):
                if profile == "realtime_audio":
                    await pipeline.llm.truncate_item(
                        item_id,
                        content_index=content_index,
                        audio_end_ms=audio_end_ms,
                    )
                else:
                    accepted = await pipeline.llm.delete_item(item_id)
                    if accepted is False:
                        _resolve_item_delete_barrier(item_id, succeeded=False)
                mutation_succeeded = bool(await asyncio.shield(barrier))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning(
                "response.terminal_partial_sync_failed",
                error_type=type(e).__name__,
            )
            if profile == "realtime_audio":
                _resolve_item_truncate_barrier(item_id, succeeded=False)
            else:
                _resolve_item_delete_barrier(item_id, succeeded=False)
            mutation_succeeded = False
        if not mutation_succeeded:
            _signal_runtime_failure(
                "Terminal response history mutation was not acknowledged"
            )
            return False
        elif profile != "realtime_audio":
            _forget_item(item_id)
        return True

    async def _finish_terminal_partial(
        *,
        captured_generation: int,
        reason: str,
        item_id: str | None,
        content_index: int,
        audio_end_ms: int,
    ) -> None:
        """Synchronize partial provider history before reopening the mic."""
        if item_id and not await _synchronize_terminal_partial(
            item_id,
            content_index=content_index,
            audio_end_ms=audio_end_ms,
        ):
            return
        if (
            captured_generation != generation_id
            or state.phase != Phase.RESPONDING
        ):
            return
        await _play_ready_beep()
        if (
            captured_generation != generation_id
            or state.phase != Phase.RESPONDING
        ):
            return
        _invalidate_generation(reason=reason)
        if state.phase == Phase.RESPONDING:
            await state.transition(Phase.LISTENING)

    async def _recover_async_commit_error(captured_generation: int) -> None:
        nonlocal manual_input_boundary_generation
        if (
            captured_generation != generation_id
            or state.phase != Phase.RESPONDING
            or not gen_response_active[0]
        ):
            _log.info(
                "input.stale_commit_error_ignored",
                captured_generation=captured_generation,
                current_generation=generation_id,
                phase=state.phase.name,
            )
            return
        if not await _clear_failed_manual_input(captured_generation):
            return
        await _abandon_unusable_input(
            reason="input_commit_server_error",
            captured_generation=captured_generation,
        )
        if state.phase != Phase.LISTENING:
            return
        manual_input_boundary_generation = None
        reset = getattr(pipeline.turn, "reset", None)
        if callable(reset):
            reset()
        mic_output_enabled.set()

    # Pre-compute the "ready to speak" beep PCM once at startup.
    ready_beep_pcm = (
        generate_beep_pcm(
            frequency_hz=settings.ready_beep_frequency_hz,
            duration_ms=settings.ready_beep_duration_ms,
            sample_rate=settings.sample_rate,
            volume=settings.ready_beep_volume,
        )
        if settings.enable_ready_beep
        else b""
    )
    mic_output_enabled = asyncio.Event()
    if not ready_beep_pcm:
        mic_output_enabled.set()

    async def _play_ready_beep() -> None:
        """Play the mic-ready beep while the mic is still muted.

        Call this BEFORE transitioning to ``Phase.LISTENING`` so the beep
        can never leak back into the transcript via speaker→mic echo.
        Waits for the speaker buffer to drain + a short post-gap before
        returning so the mic doesn't open on the beep's tail.
        """
        if not ready_beep_pcm:
            mic_output_enabled.set()
            return
        mic_output_enabled.clear()
        try:
            await speaker.queue.put(ready_beep_pcm)
            await speaker.wait_until_done()
            if settings.ready_beep_post_gap_s > 0:
                await asyncio.sleep(settings.ready_beep_post_gap_s)
        finally:
            # The capture callback kept running while muted. Drop every frame
            # that could contain beep/tail echo before reopening outbound mic.
            mic.clear()
            mic_output_enabled.set()

    # --- Per-response generation state (shared between llm_event_consumer and
    # speculative correction task so the latter can abort+replace cleanly). ---
    gen_chunker: list[SentenceChunker] = [SentenceChunker()]
    gen_assistant_text: list[str] = [""]
    # True from ``speech_stopped`` until ``response.done``. Distinct from
    # ``Phase.RESPONDING`` which stays True through the post-response
    # TTS/speaker drain + safety_delay window.
    gen_response_active: list[bool] = [False]
    # With Realtime VAD, response creation is app-owned and happens only after
    # a non-empty input transcript is received. This prevents echo/noise turns
    # from recursively eliciting more assistant audio.
    gen_response_requested: list[bool] = [False]
    gen_response_create_attempts: list[int] = [0]

    def _reset_generation_state() -> None:
        nonlocal response_resolution_generation, response_resolution_owner
        gen_chunker[0] = SentenceChunker()
        gen_assistant_text[0] = ""
        gen_response_requested[0] = False
        gen_response_create_attempts[0] = 0
        response_resolution_generation = generation_id
        response_resolution_owner = None

    def _claim_response_resolution(
        captured_generation: int,
        *,
        owner: str,
    ) -> bool:
        """Give one task exclusive ownership of response recovery/replacement."""
        nonlocal response_resolution_generation, response_resolution_owner
        if captured_generation != generation_id:
            return False
        if response_resolution_generation != captured_generation:
            response_resolution_generation = captured_generation
            response_resolution_owner = owner
            return True
        if response_resolution_owner is None:
            response_resolution_owner = owner
            return True
        return response_resolution_owner == owner

    def _cancel_response_terminal_watchdog() -> None:
        nonlocal response_watchdog_task
        current = asyncio.current_task()
        if (
            response_watchdog_task is not None
            and response_watchdog_task is not current
            and not response_watchdog_task.done()
        ):
            response_watchdog_task.cancel()
        response_watchdog_task = None

    async def _guard_response_terminal(captured_generation: int) -> None:
        """Recover when a response request never receives a terminal event."""
        await asyncio.sleep(_RESPONSE_TERMINAL_TIMEOUT_S)
        if (
            captured_generation != generation_id
            or state.phase != Phase.RESPONDING
            or not gen_response_active[0]
        ):
            return

        _log.error(
            "response.terminal_timeout",
            generation=captured_generation,
            timeout_ms=int(_RESPONSE_TERMINAL_TIMEOUT_S * 1000),
        )
        timed_out_item_id = current_output_item_id
        timed_out_content_index = current_output_content_index
        timed_out_audio_end_ms = speaker.played_audio_ms
        timed_out_partial_text = gen_assistant_text[0]
        speaker.clear()
        _drain_realtime_audio_queue()
        try:
            async with asyncio.timeout(_PROVIDER_CONTROL_TIMEOUT_S):
                await tts_manager.abort()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning(
                "response.terminal_timeout_tts_abort_failed",
                error_type=type(e).__name__,
            )

        # A barge-in or terminal event may have won while TTS was aborting.
        if (
            captured_generation != generation_id
            or state.phase != Phase.RESPONDING
            or not gen_response_active[0]
        ):
            return

        timed_out_response_id = current_response_id
        try:
            if timed_out_response_id is not None:
                _remember_cancelled_response(timed_out_response_id)
                await _cancel_response_by_id(
                    timed_out_response_id,
                    failure_message=(
                        "Timed-out response cancellation was not acknowledged"
                    ),
                )
                if not await _wait_for_cancel_barriers():
                    _signal_runtime_failure(
                        "Timed-out response cancellation was not acknowledged"
                    )
                    return
            else:
                # response.create returned but response.created itself may
                # have been lost. In that ambiguous state only an unscoped
                # cancellation can make the server session safe to reuse.
                _register_unscoped_cancel_barrier()
                try:
                    async with asyncio.timeout(_PROVIDER_CONTROL_TIMEOUT_S):
                        async with provider_control_lock:
                            await pipeline.llm.cancel_current()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    _resolve_unscoped_cancel_barrier(succeeded=False)
                    raise
                cancel_acknowledged = await _wait_for_cancel_barriers()
                if not cancel_acknowledged:
                    _signal_runtime_failure(
                        "Timed-out response cancellation was not acknowledged"
                    )
                    return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning(
                "response.terminal_timeout_cancel_failed",
                error_type=type(e).__name__,
            )
            _signal_runtime_failure(
                "Timed-out response cancellation failed"
            )
            return

        if captured_generation != generation_id:
            return
        if timed_out_partial_text:
            await on_partial_abort(timed_out_partial_text)
        await _finish_terminal_partial(
            captured_generation=captured_generation,
            reason="response_terminal_timeout",
            item_id=timed_out_item_id,
            content_index=timed_out_content_index,
            audio_end_ms=timed_out_audio_end_ms,
        )

    def _arm_response_terminal_watchdog(captured_generation: int) -> None:
        nonlocal response_watchdog_task
        if (
            captured_generation != generation_id
            or state.phase != Phase.RESPONDING
            or not gen_response_active[0]
        ):
            return
        _cancel_response_terminal_watchdog()
        response_watchdog_task = _track_background(
            _guard_response_terminal(captured_generation),
            name=f"response_terminal_watchdog_{captured_generation}",
        )

    def _cancel_turn_background() -> None:
        nonlocal correction_task, finalize_task, transcript_guard_task
        nonlocal response_request_task, active_response_conflict_task
        nonlocal response_watchdog_task
        current = asyncio.current_task()
        for task in (
            correction_task,
            finalize_task,
            transcript_guard_task,
            response_request_task,
            active_response_conflict_task,
            response_watchdog_task,
        ):
            if task is not None and task is not current and not task.done():
                task.cancel()
        correction_task = None
        finalize_task = None
        transcript_guard_task = None
        response_request_task = None
        active_response_conflict_task = None
        response_watchdog_task = None

    def _invalidate_generation(*, reason: str) -> None:
        nonlocal generation_id, current_response_id, expected_input_item_id
        nonlocal current_output_item_id, current_output_content_index
        generation_id += 1
        gen_response_active[0] = False
        _cancel_turn_background()
        _drain_realtime_audio_queue()
        current_response_id = None
        expected_input_item_id = None
        current_output_item_id = None
        current_output_content_index = 0
        _log.info("response.generation_invalidated", generation=generation_id, reason=reason)

    async def _abandon_unusable_input(
        *,
        reason: str,
        captured_generation: int,
        item_id: str | None = None,
    ) -> None:
        """Return to listening without ever creating a response for bad input."""
        if captured_generation != generation_id:
            return
        if state.phase != Phase.RESPONDING:
            return
        speaker.clear()
        await tts_manager.abort()
        if item_id and not await _delete_item_with_ack(item_id):
            _signal_runtime_failure(
                "Unusable input deletion was not acknowledged"
            )
            return
        _invalidate_generation(reason=reason)
        await state.transition(Phase.LISTENING)

    async def _guard_input_transcript(
        captured_generation: int,
        captured_item_id: str | None,
    ) -> None:
        await asyncio.sleep(_INPUT_TRANSCRIPT_TIMEOUT_S)
        if (
            captured_generation == generation_id
            and state.phase == Phase.RESPONDING
            and gen_response_active[0]
            and not gen_response_requested[0]
        ):
            _log.warning(
                "input.transcript_timeout",
                timeout_ms=int(_INPUT_TRANSCRIPT_TIMEOUT_S * 1000),
            )
            if not isinstance(captured_item_id, str) or not captured_item_id:
                _signal_runtime_failure(
                    "Realtime input item identity was unavailable"
                )
                return
            await _abandon_unusable_input(
                reason="input_transcript_timeout",
                captured_generation=captured_generation,
                item_id=captured_item_id,
            )

    async def _request_response_for_transcript(
        raw_text: str,
        raw_item_id: str | None,
        captured_generation: int,
        turn_timer: TurnTimer,
    ) -> None:
        nonlocal correction_task, speculative_correction_input
        if not await _wait_for_cancel_barriers():
            _signal_runtime_failure(
                "Response cancellation was not acknowledged"
            )
            return
        if not await _wait_for_item_mutation_barriers():
            _signal_runtime_failure(
                "Provider history mutation was not acknowledged"
            )
            return
        if (
            captured_generation != generation_id
            or state.phase != Phase.RESPONDING
            or not gen_response_active[0]
        ):
            return
        try:
            async with asyncio.timeout(_PROVIDER_CONTROL_TIMEOUT_S):
                async with provider_control_lock:
                    gen_response_create_attempts[0] += 1
                    await pipeline.llm.trigger_response(
                        generation_id=captured_generation,
                    )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            _log.warning(
                "response.create_timeout",
                timeout_ms=int(_PROVIDER_CONTROL_TIMEOUT_S * 1000),
            )
            _signal_runtime_failure(
                "Response creation timed out in an ambiguous provider state"
            )
            return
        except Exception as e:
            _log.warning(
                "response.create_failed",
                error_type=type(e).__name__,
            )
            _signal_runtime_failure(
                "Response creation failed in an ambiguous provider state"
            )
            return

        if (
            captured_generation != generation_id
            or state.phase != Phase.RESPONDING
            or not gen_response_active[0]
        ):
            return
        _arm_response_terminal_watchdog(captured_generation)
        if corrector is not None:
            speculative_correction_input = (captured_generation, raw_text)
            correction_task = _track_background(
                _speculative_correction(
                    raw_text,
                    raw_item_id,
                    captured_generation,
                    turn_timer,
                ),
                name=f"transcript_correction_{captured_generation}",
            )

    async def _recover_active_response_conflict(
        captured_generation: int,
    ) -> None:
        """Resolve a server-active stale response, then retry create once."""
        if (
            captured_generation != generation_id
            or state.phase != Phase.RESPONDING
            or not gen_response_active[0]
        ):
            return

        had_scoped_barrier = bool(response_cancel_barriers)
        if had_scoped_barrier:
            if not await _wait_for_cancel_barriers():
                _signal_runtime_failure(
                    "Active response cancellation was not acknowledged"
                )
                return
        else:
            # The server explicitly reported an active response, but its ID
            # has not reached the client. This is the one safe case for an
            # unscoped cancel. Only a cancel-not-active error carrying the
            # original client event ID is safely correlated; an unscoped
            # response.done cannot authorize session reuse.
            _register_unscoped_cancel_barrier()
            try:
                async with asyncio.timeout(_PROVIDER_CONTROL_TIMEOUT_S):
                    async with provider_control_lock:
                        await pipeline.llm.cancel_current()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                _resolve_unscoped_cancel_barrier(succeeded=False)
                _log.warning(
                    "response.active_conflict_cancel_failed",
                    error_type=type(e).__name__,
                )
                _signal_runtime_failure(
                    "Active response cancellation failed"
                )
                return
            if not await _wait_for_cancel_barriers():
                _signal_runtime_failure(
                    "Active response cancellation was not acknowledged"
                )
                return

        if (
            captured_generation != generation_id
            or state.phase != Phase.RESPONDING
            or not gen_response_active[0]
        ):
            return
        try:
            async with asyncio.timeout(_PROVIDER_CONTROL_TIMEOUT_S):
                async with provider_control_lock:
                    gen_response_create_attempts[0] += 1
                    await pipeline.llm.trigger_response(
                        generation_id=captured_generation,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning(
                "response.active_conflict_retry_failed",
                error_type=type(e).__name__,
            )
            _signal_runtime_failure(
                "Active response retry failed in an ambiguous provider state"
            )
            return

        if (
            captured_generation == generation_id
            and state.phase == Phase.RESPONDING
            and gen_response_active[0]
        ):
            # The retry owns a fresh terminal deadline. Keeping the original
            # request's timer can cancel the replacement immediately when the
            # conflict arrived near the end of that window.
            _arm_response_terminal_watchdog(captured_generation)

    async def _clear_failed_manual_input(captured_generation: int) -> bool:
        nonlocal input_clear_barrier, input_clear_generation
        clear = getattr(pipeline.llm, "clear_input_buffer", None)
        if not callable(clear):
            _signal_runtime_failure("Manual input buffer cannot be cleared")
            return False
        barrier = _register_input_clear_barrier(captured_generation)
        try:
            async with asyncio.timeout(_PROVIDER_CONTROL_TIMEOUT_S):
                async with provider_control_lock:
                    await clear(generation_id=captured_generation)
                succeeded = bool(await asyncio.shield(barrier))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _resolve_input_clear_barrier(
                captured_generation,
                succeeded=False,
            )
            _log.warning(
                "input.clear_after_commit_failed",
                error_type=type(e).__name__,
            )
            succeeded = False
        finally:
            if input_clear_barrier is barrier:
                input_clear_barrier = None
                input_clear_generation = None
        if not succeeded:
            _signal_runtime_failure(
                "Manual input buffer clear was not acknowledged"
            )
        return succeeded

    async def _commit_manual_input(captured_generation: int) -> None:
        nonlocal transcript_guard_task
        try:
            async with asyncio.timeout(_PROVIDER_CONTROL_TIMEOUT_S):
                async with provider_control_lock:
                    await pipeline.llm.commit_input_audio_buffer(
                        generation_id=captured_generation,
                    )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            _log.warning(
                "input.commit_timeout",
                timeout_ms=int(_PROVIDER_CONTROL_TIMEOUT_S * 1000),
            )
            _signal_runtime_failure(
                "Manual input commit timed out in an ambiguous provider state"
            )
            return
        except Exception as e:
            _log.warning(
                "input.commit_failed",
                error_type=type(e).__name__,
            )
            _signal_runtime_failure(
                "Manual input commit failed in an ambiguous provider state"
            )
            return

        if (
            captured_generation == generation_id
            and state.phase == Phase.RESPONDING
            and not gen_response_requested[0]
            and expected_input_item_id is None
            and (
                transcript_guard_task is None
                or transcript_guard_task.done()
            )
        ):
            transcript_guard_task = _track_background(
                _guard_input_transcript(captured_generation, None),
                name=f"input_transcript_guard_{captured_generation}",
            )

    async def _speculative_correction(
        raw: str,
        raw_item_id: str | None,
        captured_generation: int,
        turn_timer: TurnTimer,
    ) -> None:
        """Run correction in parallel with the speculative (raw) response.

        If correction returns the same text → no-op; the already-streaming
        response runs to completion with zero added latency.

        If correction differs AND the response is still being generated
        (``Phase == RESPONDING``), we abort the speculative response, delete
        the server-transcribed user item, inject the corrected text, and
        trigger a new response. The user will hear the start of the raw
        response cut off and the corrected response begin (tradeoff vs. pure
        sequential correction's fixed latency cost).

        If the speculative response already finished before correction
        completes, we just record the user text in history and skip.
        """
        assert corrector is not None
        nonlocal generation_id, current_response_id
        nonlocal current_output_item_id, current_output_content_index

        corrected, correction_ms = await corrector.correct(raw)
        if captured_generation != generation_id:
            _log.info(
                "correction.stale_generation",
                captured_generation=captured_generation,
                current_generation=generation_id,
            )
            return

        turn_timer.correction_ms = correction_ms
        _record_corrector_user_once(corrected, captured_generation)

        if corrected == raw:
            return

        if not gen_response_active[0] or state.phase != Phase.RESPONDING:
            # The speculative response already completed before correction
            # arrived — don't kick off a duplicate response. The user's
            # corrected text is still recorded in history for the next turn.
            _log.info(
                "correction.too_late_to_replace",
                raw_len=len(raw),
                corrected_len=len(corrected),
            )
            return

        if not _claim_response_resolution(
            captured_generation,
            owner="correction",
        ):
            _log.info(
                "correction.replacement_owned_by_recovery",
                generation=captured_generation,
                owner=response_resolution_owner,
            )
            return

        _log.info(
            "correction.speculative_replace",
            raw_len=len(raw),
            corrected_len=len(corrected),
            ms=round(correction_ms, 1),
        )
        print(f"\n[corrected]: {corrected}")

        # The old response no longer owns this turn. Every provider control
        # action below is independently bounded; the replacement gets a new
        # watchdog only after its response.create has been sent.
        _cancel_response_terminal_watchdog()

        # Abort the speculative stream (same shape as barge-in abort chain,
        # but we stay in RESPONDING because the user isn't talking).
        response_id_to_cancel = current_response_id
        speaker.clear()
        await tts_manager.abort()

        if response_id_to_cancel is None:
            _register_unscoped_cancel_barrier()
        else:
            _remember_cancelled_response(response_id_to_cancel)
            _register_cancel_barrier(response_id_to_cancel)
        try:
            async with asyncio.timeout(_PROVIDER_CONTROL_TIMEOUT_S):
                async with provider_control_lock:
                    await pipeline.llm.cancel_current(
                        response_id=response_id_to_cancel,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning("correction.cancel_failed", error_type=type(e).__name__)
            if response_id_to_cancel is not None:
                _resolve_cancel_barrier(
                    response_id_to_cancel,
                    succeeded=False,
                )
            else:
                _resolve_unscoped_cancel_barrier(succeeded=False)
            _signal_runtime_failure(
                "Correction response cancellation failed"
            )
            return

        cancel_acknowledged = await _wait_for_cancel_barriers()
        if not cancel_acknowledged:
            _signal_runtime_failure(
                "Correction response cancellation was not acknowledged"
            )
            return

        output_sync = on_output_interrupted(register_cancel=False)
        try:
            async with asyncio.timeout(_PROVIDER_CONTROL_TIMEOUT_S):
                await output_sync
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning(
                "correction.output_sync_failed",
                error_type=type(e).__name__,
            )
            _signal_runtime_failure(
                "Correction output history synchronization failed"
            )
            return
        if not await _wait_for_item_mutation_barriers():
            _signal_runtime_failure(
                "Correction history mutation was not acknowledged"
            )
            return

        if captured_generation != generation_id or state.phase != Phase.RESPONDING:
            _log.info("correction.replace_abandoned_after_cancel")
            return

        if raw_item_id and not await _delete_item_with_ack(raw_item_id):
            _signal_runtime_failure(
                "Corrected input item deletion was not acknowledged"
            )
            return

        if captured_generation != generation_id or state.phase != Phase.RESPONDING:
            _log.info("correction.replace_abandoned_after_delete")
            return

        # A replacement is a new response generation even though it belongs
        # to the same user turn and latency timer.
        generation_id += 1
        current_response_id = None
        current_output_item_id = None
        current_output_content_index = 0
        gen_response_active[0] = True

        # Reset per-response generation state so the replacement stream is
        # clean and the latency milestones are re-captured.
        _reset_generation_state()
        gen_response_requested[0] = True
        gen_response_create_attempts[0] = 1
        tts_manager.reset_for_new_turn()
        speaker.arm()
        interrupt_bus.reset_partial()
        turn_timer.first_llm_delta_ts = None
        turn_timer.first_tts_byte_ts = None

        replacement_generation = generation_id
        try:
            async with asyncio.timeout(_PROVIDER_CONTROL_TIMEOUT_S):
                async with provider_control_lock:
                    await pipeline.llm.send_user_text(
                        corrected,
                        injections=[],
                        generation_id=replacement_generation,
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _log.warning(
                "correction.replace_send_failed",
                error_type=type(e).__name__,
            )
            _signal_runtime_failure(
                "Corrected response request failed in an ambiguous provider state"
            )
            return
        _arm_response_terminal_watchdog(replacement_generation)

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------
    async def mic_pump() -> None:
        while True:
            health = mic.capture_health()
            if health.failure_reason is not None:
                reason = health.failure_reason
                _log.error(
                    "microphone.capture_failed",
                    reason=reason,
                    restart_required=True,
                )
                raise RuntimeError(
                    f"Microphone capture failed ({reason}); restart Zemory"
                )
            try:
                async with asyncio.timeout(_MIC_HEALTH_POLL_S):
                    pcm = await mic.queue.get()
            except TimeoutError:
                continue
            if not mic_output_enabled.is_set():
                continue
            if state.phase == Phase.RESPONDING:
                # While AI is responding:
                #   - barge-in ON:  keep pushing mic so server VAD can detect
                #     interruption (but echo will cause false triggers on
                #     devices without hardware AEC)
                #   - barge-in OFF: suppress mic entirely → speaker audio
                #     leaking into mic cannot be mis-transcribed as user input
                if profile in {"realtime_audio", "realtime_text_external_tts"} and settings.enable_barge_in:
                    await pipeline.turn.feed(pcm)
                # else: drop frame
            else:
                await pipeline.turn.feed(pcm)

    async def realtime_audio_output_consumer() -> None:
        """Relay response audio without blocking the sole control-event reader.

        A slow speaker must not prevent a following speech_started event from
        executing the interruption chain. Each chunk is generation-tagged;
        stale chunks are discarded before and after a potentially blocking
        speaker enqueue.
        """
        while True:
            captured_generation, response_id, audio = (
                await realtime_audio_queue.get()
            )
            try:
                if (
                    captured_generation != generation_id
                    or state.phase != Phase.RESPONDING
                    or (response_id and response_id in cancelled_response_ids)
                    or (
                        response_id
                        and current_response_id
                        and response_id != current_response_id
                    )
                ):
                    continue
                await speaker.queue.put(audio)
                if (
                    captured_generation != generation_id
                    or state.phase != Phase.RESPONDING
                    or (response_id and response_id in cancelled_response_ids)
                ):
                    # A clear may have freed a blocked put after interruption.
                    # Clear again so that just-unblocked stale chunk cannot play.
                    speaker.clear()
            finally:
                realtime_audio_queue.task_done()

    async def turn_event_consumer() -> None:
        """Handle local or manual-Realtime turn detector events."""
        nonlocal turn_seq, timer, generation_id
        nonlocal current_response_id, current_output_item_id, expected_input_item_id
        nonlocal current_output_content_index, transcript_guard_task
        nonlocal response_request_task
        nonlocal manual_input_boundary_generation
        if profile != "local_cascade" and not manual_realtime_turns:
            return
        while True:
            event = await pipeline.turn.events.get()
            if event == "speech_start":
                if (
                    manual_realtime_turns
                    and manual_input_boundary_generation is not None
                ):
                    _log.error(
                        "input.manual_boundary_interleaved",
                        boundary_generation=manual_input_boundary_generation,
                        current_generation=generation_id,
                    )
                    _signal_runtime_failure(
                        "New speech began before the manual input boundary was acknowledged"
                    )
                    continue
                # Cancel speculative correction/finalization immediately;
                # remote response synchronization is deliberately off the
                # latency-critical interruption path.
                if state.phase == Phase.RESPONDING:
                    _cancel_turn_background()
                interrupted = await handle_speech_started(
                    state,
                    interrupt_bus,
                    reason="local_speech_started",
                )
                if interrupted:
                    _invalidate_generation(reason="local_barge_in")
            elif event == "speech_end":
                if (
                    manual_realtime_turns
                    and manual_input_boundary_generation is not None
                ):
                    _signal_runtime_failure(
                        "A second manual input boundary arrived before acknowledgement"
                    )
                    continue
                if state.phase != Phase.ACTIVE:
                    _log.warning(
                        "turn.speech_end_out_of_phase_dropped",
                        phase=state.phase.name,
                    )
                    continue
                state.mark_speech_end()
                turn_seq += 1
                timer = TurnTimer(turn_seq)
                timer.speech_end_ts = state.speech_end_ts
                generation_id += 1
                _cancel_turn_background()
                current_response_id = None
                expected_input_item_id = None
                current_output_item_id = None
                current_output_content_index = 0
                await state.transition(Phase.RESPONDING)
                speaker.arm()
                interrupt_bus.reset_partial()
                _reset_generation_state()

                if manual_realtime_turns:
                    manual_input_boundary_generation = generation_id
                    # The local endpoint detector is paused at this exact
                    # watermark. Drop, rather than backlog, microphone frames
                    # until the server acknowledges commit (or clear).
                    mic_output_enabled.clear()
                    tts_manager.reset_for_new_turn()
                    gen_response_active[0] = True
                    response_request_task = _track_background(
                        _commit_manual_input(generation_id),
                        name=f"commit_input_{generation_id}",
                    )
                    continue

                chunks = pipeline.turn.consume_audio()
                text = await pipeline.stt.transcribe(chunks)
                if not text or text == "[transcription failed]":
                    _log.warning("stt.empty_or_failed")
                    _invalidate_generation(reason="stt_empty_or_failed")
                    await _play_ready_beep()
                    await state.transition(Phase.LISTENING)
                    continue
                raw_text = text

                if corrector is not None:
                    corrected, correction_ms = await corrector.correct(raw_text)
                    timer.correction_ms = correction_ms
                    text = corrected
                    if corrected != raw_text:
                        print(f"\n[You (raw)]:       {raw_text}")
                        print(f"[You (corrected)]: {corrected}")
                    else:
                        print(f"\n[You]: {text}")
                    corrector.record_user(text)
                else:
                    print(f"\n[You]: {text}")

                _log.info(
                    "user.text",
                    text_len=len(text),
                    corrected=raw_text != text,
                )
                if not await _wait_for_cancel_barriers():
                    _signal_runtime_failure(
                        "Response cancellation was not acknowledged"
                    )
                    continue
                if not await _wait_for_item_mutation_barriers():
                    _signal_runtime_failure(
                        "Provider history mutation was not acknowledged"
                    )
                    continue
                tts_manager.reset_for_new_turn()
                context = await context_scheduler.gather_for_turn(text)
                gen_response_active[0] = True
                gen_response_requested[0] = True
                gen_response_create_attempts[0] = 1
                local_generation = generation_id
                try:
                    async with asyncio.timeout(_PROVIDER_CONTROL_TIMEOUT_S):
                        async with provider_control_lock:
                            await pipeline.llm.send_user_text(
                                text,
                                injections=context.injections,
                                generation_id=local_generation,
                            )
                except asyncio.CancelledError:
                    raise
                except TimeoutError:
                    _log.error(
                        "response.local_send_timeout",
                        timeout_ms=int(_PROVIDER_CONTROL_TIMEOUT_S * 1000),
                    )
                    # A timed-out send may already have created a response.
                    # Closing this ambiguous session is safer than issuing a
                    # later response.create against unknown provider state.
                    raise RuntimeError(
                        "Local response request timed out"
                    ) from None
                except Exception as e:
                    _log.error(
                        "response.local_send_failed",
                        error_type=type(e).__name__,
                    )
                    raise RuntimeError(
                        "Local response request failed"
                    ) from None
                _arm_response_terminal_watchdog(local_generation)
                # Reset local VAD state for the next turn
                from zemory.providers.turn.silero import SileroTurnDetector
                if isinstance(pipeline.turn, SileroTurnDetector):
                    pipeline.turn.reset()

    def _accept_response_event(event: dict) -> bool:
        """Bind an event to the active generation or reject a stale response."""
        nonlocal current_response_id
        response_id = event.get("response_id")
        if state.phase != Phase.RESPONDING or not gen_response_active[0]:
            if response_id and _remember_cancelled_response(response_id):
                _schedule_scoped_cancel(response_id)
            item_id = event.get("item_id")
            if item_id:
                _schedule_stale_item_delete(item_id)
            _log.debug(
                "response.inactive_event_dropped",
                response_id=response_id,
                phase=state.phase.name,
            )
            return False
        if response_id and response_id in cancelled_response_ids:
            _log.debug("response.stale_event_dropped", response_id=response_id)
            return False
        if response_id and current_response_id is None:
            current_response_id = response_id
        if response_id and current_response_id and response_id != current_response_id:
            _log.warning(
                "response.foreign_event_dropped",
                response_id=response_id,
                current_response_id=current_response_id,
            )
            return False
        return True

    def _record_transcription_usage(usage: dict | None) -> None:
        if not usage:
            return
        if usage.get("type") == "duration":
            seconds = usage.get("seconds")
            if isinstance(seconds, (int, float)) and seconds >= 0:
                metrics.observe("transcription.seconds", float(seconds))
            _log.info("input.transcription_usage", seconds=seconds)
            return
        for metric_name, field_name in (
            ("tokens.transcription_input", "input_tokens"),
            ("tokens.transcription_input_text", "input_text_tokens"),
            ("tokens.transcription_input_audio", "input_audio_tokens"),
            ("tokens.transcription_output", "output_tokens"),
        ):
            value = usage.get(field_name)
            if isinstance(value, int) and value >= 0:
                metrics.observe(metric_name, value)
        _log.info(
            "input.transcription_usage",
            total_tokens=usage.get("total_tokens"),
            input_tokens=usage.get("input_tokens"),
            input_text_tokens=usage.get("input_text_tokens"),
            input_audio_tokens=usage.get("input_audio_tokens"),
            output_tokens=usage.get("output_tokens"),
        )

    def _record_response_usage(usage: dict | None, *, status: str | None) -> None:
        if not usage:
            return
        input_tokens = usage.get("input_tokens")
        cached_tokens = usage.get("cached_tokens")
        if isinstance(input_tokens, int) and input_tokens >= 0:
            metrics.observe("tokens.input", input_tokens)
        if isinstance(cached_tokens, int) and cached_tokens >= 0:
            metrics.observe("tokens.cached", cached_tokens)
        for metric_name, field_name in (
            ("tokens.input_text", "input_text_tokens"),
            ("tokens.input_audio", "input_audio_tokens"),
            ("tokens.input_image", "input_image_tokens"),
            ("tokens.cached_text", "cached_text_tokens"),
            ("tokens.cached_audio", "cached_audio_tokens"),
            ("tokens.cached_image", "cached_image_tokens"),
            ("tokens.output_text", "output_text_tokens"),
            ("tokens.output_audio", "output_audio_tokens"),
        ):
            value = usage.get(field_name)
            if isinstance(value, int) and value >= 0:
                metrics.observe(metric_name, value)
        cache_hit_pct = (
            cached_tokens / input_tokens * 100
            if isinstance(input_tokens, int)
            and input_tokens > 0
            and isinstance(cached_tokens, int)
            else None
        )
        _log.info(
            "response.usage",
            status=status,
            total_tokens=usage.get("total_tokens"),
            input_tokens=input_tokens,
            output_tokens=usage.get("output_tokens"),
            input_text_tokens=usage.get("input_text_tokens"),
            input_audio_tokens=usage.get("input_audio_tokens"),
            input_image_tokens=usage.get("input_image_tokens"),
            cached_tokens=cached_tokens,
            cached_text_tokens=usage.get("cached_text_tokens"),
            cached_audio_tokens=usage.get("cached_audio_tokens"),
            cached_image_tokens=usage.get("cached_image_tokens"),
            output_text_tokens=usage.get("output_text_tokens"),
            output_audio_tokens=usage.get("output_audio_tokens"),
            cache_hit_pct=(round(cache_hit_pct, 1) if cache_hit_pct is not None else None),
        )

    async def llm_event_consumer() -> None:
        nonlocal turn_seq, timer, generation_id
        nonlocal current_response_id, current_output_item_id, expected_input_item_id
        nonlocal current_output_content_index, correction_task, finalize_task
        nonlocal transcript_guard_task, response_request_task
        nonlocal active_response_conflict_task
        nonlocal manual_input_boundary_generation
        async for event in pipeline.llm.events():
            t = event.get("type")

            if t == "session.created":
                _log.info("session.created", id=event.get("session_id"))
            elif t == "session.updated":
                _log.info("session.configured")

            elif t == "input.speech_started":
                if state.phase == Phase.RESPONDING:
                    _cancel_turn_background()
                interrupted = await handle_speech_started(state, interrupt_bus)
                if interrupted:
                    _invalidate_generation(reason="realtime_barge_in")

            elif t == "input.speech_stopped":
                if state.phase != Phase.ACTIVE:
                    _log.warning(
                        "input.speech_stopped_out_of_phase_dropped",
                        phase=state.phase.name,
                    )
                    continue
                stopped_item_id = event.get("item_id")
                if not isinstance(stopped_item_id, str) or not stopped_item_id:
                    _log.error("input.speech_stopped_item_id_missing")
                    _signal_runtime_failure(
                        "Realtime input item identity was unavailable"
                    )
                    continue
                # Realtime: user finished — move to RESPONDING and time turn
                state.mark_speech_end()
                turn_seq += 1
                timer = TurnTimer(turn_seq)
                timer.speech_end_ts = state.speech_end_ts
                generation_id += 1
                _cancel_turn_background()
                current_response_id = None
                expected_input_item_id = stopped_item_id
                current_output_item_id = None
                current_output_content_index = 0
                await state.transition(Phase.RESPONDING)
                speaker.arm()
                # Clear abort flag + reset seq counter so this turn's TTS
                # submissions are accepted (prior interrupt would have stuck).
                tts_manager.reset_for_new_turn()
                interrupt_bus.reset_partial()
                _reset_generation_state()
                gen_response_active[0] = True
                transcript_guard_task = _track_background(
                    _guard_input_transcript(
                        generation_id,
                        expected_input_item_id,
                    ),
                    name=f"input_transcript_guard_{generation_id}",
                )

            elif t == "input.committed":
                if not manual_realtime_turns:
                    continue
                committed_generation = event.get("generation_id")
                committed_item_id = event.get("item_id")
                if (
                    not isinstance(committed_generation, int)
                    or committed_generation != generation_id
                    or manual_input_boundary_generation != committed_generation
                    or state.phase != Phase.RESPONDING
                    or not gen_response_active[0]
                ):
                    _log.error(
                        "input.stale_commit_ack",
                        committed_generation=committed_generation,
                        current_generation=generation_id,
                        boundary_generation=manual_input_boundary_generation,
                    )
                    _signal_runtime_failure(
                        "Stale manual input commit acknowledgement"
                    )
                    continue
                if not isinstance(committed_item_id, str) or not committed_item_id:
                    _signal_runtime_failure(
                        "Manual input commit acknowledgement had no item identity"
                    )
                    continue
                expected_input_item_id = committed_item_id
                manual_input_boundary_generation = None
                reset = getattr(pipeline.turn, "reset", None)
                if callable(reset):
                    reset()
                mic_output_enabled.set()
                if transcript_guard_task is not None:
                    transcript_guard_task.cancel()
                transcript_guard_task = _track_background(
                    _guard_input_transcript(
                        committed_generation,
                        committed_item_id,
                    ),
                    name=f"input_transcript_guard_{committed_generation}",
                )

            elif t == "input.cleared":
                if not manual_realtime_turns:
                    continue
                cleared_generation = event.get("generation_id")
                if (
                    not isinstance(cleared_generation, int)
                    or cleared_generation != generation_id
                    or manual_input_boundary_generation != cleared_generation
                    or not _resolve_input_clear_barrier(
                        cleared_generation,
                        succeeded=True,
                    )
                ):
                    _log.error(
                        "input.unowned_clear_ack",
                        cleared_generation=cleared_generation,
                        current_generation=generation_id,
                        boundary_generation=manual_input_boundary_generation,
                    )
                    _signal_runtime_failure(
                        "Unowned manual input clear acknowledgement"
                    )

            elif t == "input.transcript":
                raw_text = event.get("text", "")
                raw_item_id = event.get("item_id")
                _record_transcription_usage(event.get("transcription_usage"))

                if profile not in {
                    "realtime_audio",
                    "realtime_text_external_tts",
                }:
                    continue
                if state.phase != Phase.RESPONDING or not gen_response_active[0]:
                    _log.info(
                        "input.late_transcript_dropped",
                        text_len=len(raw_text),
                    )
                    continue
                if not isinstance(expected_input_item_id, str) or not expected_input_item_id:
                    _log.error("input.expected_item_id_missing")
                    _signal_runtime_failure(
                        "Realtime input item identity was unavailable"
                    )
                    continue
                if not isinstance(raw_item_id, str) or not raw_item_id:
                    _log.error("input.transcript_item_id_missing")
                    _signal_runtime_failure(
                        "Realtime transcript item identity was unavailable"
                    )
                    continue
                if (
                    raw_item_id != expected_input_item_id
                ):
                    _log.warning(
                        "input.stale_transcript_dropped",
                        expected_item_id=expected_input_item_id,
                        received_item_id=raw_item_id,
                    )
                    if raw_item_id:
                        _schedule_stale_item_delete(raw_item_id)
                    continue
                if gen_response_requested[0]:
                    _log.warning("input.duplicate_transcript_dropped")
                    continue
                if transcript_guard_task is not None:
                    transcript_guard_task.cancel()
                    transcript_guard_task = None

                if not raw_text.strip():
                    _log.warning("input.empty_transcript_ignored")
                    response_request_task = _track_background(
                        _abandon_unusable_input(
                            reason="empty_input_transcript",
                            captured_generation=generation_id,
                            item_id=raw_item_id,
                        ),
                        name=f"discard_empty_input_{generation_id}",
                    )
                    continue

                gen_response_requested[0] = True
                print(f"\n[You]: {raw_text}")

                # Provider control I/O runs generation-scoped in the
                # background. The sole event reader remains free to process a
                # following speech_started even if response.create stalls.
                response_request_task = _track_background(
                    _request_response_for_transcript(
                            raw_text,
                            raw_item_id,
                            generation_id,
                            timer,
                    ),
                    name=f"create_response_{generation_id}",
                )

                _log.info("user.text", text_len=len(raw_text))

            elif t == "input.transcript.failed":
                failed_item_id = event.get("item_id")
                if not isinstance(expected_input_item_id, str) or not expected_input_item_id:
                    _log.error("input.expected_item_id_missing")
                    _signal_runtime_failure(
                        "Realtime input item identity was unavailable"
                    )
                    continue
                if not isinstance(failed_item_id, str) or not failed_item_id:
                    _log.error("input.transcript_failure_item_id_missing")
                    _signal_runtime_failure(
                        "Realtime transcript item identity was unavailable"
                    )
                    continue
                if (
                    failed_item_id != expected_input_item_id
                ):
                    _log.warning(
                        "input.stale_transcript_failure_dropped",
                        expected_item_id=expected_input_item_id,
                        received_item_id=failed_item_id,
                    )
                    if failed_item_id:
                        _schedule_stale_item_delete(failed_item_id)
                    continue
                _log.warning("input.transcript_failed")
                response_request_task = _track_background(
                    _abandon_unusable_input(
                        reason="input_transcript_failed",
                        captured_generation=generation_id,
                        item_id=failed_item_id,
                    ),
                    name=f"recover_transcript_failure_{generation_id}",
                )

            elif t == "conversation.item.created":
                item_id = event.get("item_id")
                if item_id and item_id not in item_id_set:
                    item_ids.append(item_id)
                    item_id_set.add(item_id)
                    if len(item_ids) > _LOCAL_ITEM_ID_TRACKING_LIMIT:
                        item_id_set.discard(item_ids.pop(0))
                if event.get("role") == "assistant" and item_id:
                    if state.phase != Phase.RESPONDING or not gen_response_active[0]:
                        _schedule_stale_item_delete(item_id)
                    # Generic conversation.item events carry no response_id.
                    # Never bind them to the current generation: a delayed old
                    # item could otherwise be truncated with the new turn's
                    # playback cursor. response.output_item/audio events own
                    # the authoritative response-scoped binding.

            elif t == "conversation.item.deleted":
                deleted_item_id = event.get("item_id")
                if deleted_item_id:
                    _resolve_item_delete_barrier(
                        deleted_item_id,
                        succeeded=True,
                    )
                    _forget_item(deleted_item_id)

            elif t == "conversation.item.truncated":
                truncated_item_id = event.get("item_id")
                if truncated_item_id:
                    _resolve_item_truncate_barrier(
                        truncated_item_id,
                        succeeded=True,
                    )

            elif t == "response.created":
                response_id = event.get("response_id")
                if response_id:
                    response_generation = event.get("generation_id")
                    # Every runtime-owned response.create carries metadata.
                    # Missing metadata is an automatic/foreign response and
                    # must never bind to the active local generation.
                    stale_generation = response_generation != generation_id
                    if (
                        stale_generation
                        or state.phase != Phase.RESPONDING
                        or not gen_response_active[0]
                    ):
                        # A barge-in can happen before the server assigns an
                        # ID.  Once that late ID arrives, cancel that exact
                        # response on the server; merely quarantining local
                        # deltas leaves the session with an active response
                        # and makes the next response.create fail.
                        if _remember_cancelled_response(response_id):
                            _schedule_scoped_cancel(response_id)
                        _log.info(
                            "response.inactive_created_dropped",
                            response_id=response_id,
                            response_generation=response_generation,
                            current_generation=generation_id,
                            phase=state.phase.name,
                        )
                    elif response_id in cancelled_response_ids:
                        _schedule_scoped_cancel(response_id)
                    elif current_response_id is None:
                        current_response_id = response_id
                    elif current_response_id != response_id:
                        # Bind once. A second response.created cannot silently
                        # steal the active generation; quarantine it remotely.
                        if _remember_cancelled_response(response_id):
                            _schedule_scoped_cancel(response_id)
                        _log.warning(
                            "response.foreign_created_dropped",
                            response_id=response_id,
                            current_response_id=current_response_id,
                        )

            elif t == "response.output_item":
                if not _accept_response_event(event):
                    continue
                item_id = event.get("item_id")
                if item_id:
                    current_output_item_id = item_id

            elif t == "text.delta":
                if not _accept_response_event(event):
                    continue
                if timer.first_llm_delta_ts is None:
                    timer.first_llm_delta_ts = time.monotonic()
                delta = event.get("delta", "")
                print(delta, end="", flush=True)
                interrupt_bus.record_partial(delta)
                gen_assistant_text[0] += delta
                if profile != "realtime_audio":
                    for sentence in gen_chunker[0].add(delta):
                        tts_manager.submit(sentence)

            elif t == "text.done":
                if not _accept_response_event(event):
                    continue
                print()
                if profile != "realtime_audio":
                    tail = gen_chunker[0].flush()
                    if tail:
                        tts_manager.submit(tail)
                # History remains provisional until playback has fully
                # drained. Barge-in may still truncate this response.

            elif t == "audio.delta":
                if not _accept_response_event(event):
                    continue
                item_id = event.get("item_id")
                if item_id:
                    current_output_item_id = item_id
                content_index = event.get("content_index")
                if content_index is not None:
                    current_output_content_index = int(content_index)
                if timer.first_tts_byte_ts is None:
                    timer.first_tts_byte_ts = time.monotonic()
                audio = event.get("audio", b"")
                if audio:
                    try:
                        realtime_audio_queue.put_nowait(
                            (generation_id, event.get("response_id"), audio)
                        )
                    except asyncio.QueueFull as e:
                        # Continuing after a dropped middle chunk would make
                        # local playback non-contiguous while server truncation
                        # assumes a contiguous heard prefix. Fail closed.
                        raise RuntimeError(
                            "Realtime audio relay exceeded its bounded capacity"
                        ) from e

            elif t == "audio.transcript.delta":
                if not _accept_response_event(event):
                    continue
                delta = event.get("delta", "")
                print(delta, end="", flush=True)
                interrupt_bus.record_partial(delta)
                gen_assistant_text[0] += delta

            elif t == "audio.transcript.done":
                if not _accept_response_event(event):
                    continue
                print()
                # Commit to local history only after response finalization and
                # audible playback drain; until then this text is provisional.

            elif t == "response.done":
                response_id = event.get("response_id")
                status = event.get("status")
                _record_response_usage(event.get("usage"), status=status)
                had_scoped_cancel = bool(
                    response_id
                    and response_id in response_cancel_barriers
                    and not response_cancel_barriers[response_id].done()
                )
                if response_id:
                    _resolve_cancel_barrier(response_id, succeeded=True)
                locally_cancelled = bool(
                    response_id and response_id in cancelled_response_ids
                )
                if had_scoped_cancel or locally_cancelled:
                    _log.info(
                        "response.cancel_owned_terminal_skipped_finalize",
                        turn_id=timer.turn_id,
                        response_id=response_id,
                        status=status,
                    )
                    continue

                # A terminal is not allowed to bind an otherwise-unidentified
                # response to the current turn. Runtime-owned responses carry
                # generation metadata; an untagged/foreign delayed terminal
                # can otherwise finalize or cancel the wrong generation while
                # an unscoped cancellation barrier is still pending.
                response_generation = event.get("generation_id")
                if (
                    current_response_id is None
                    and response_generation != generation_id
                ):
                    _log.info(
                        "response.unowned_terminal_dropped",
                        response_id=response_id,
                        response_generation=response_generation,
                        current_generation=generation_id,
                        status=status,
                    )
                    continue

                if status == "cancelled":
                    exact_current_terminal = (
                        state.phase == Phase.RESPONDING
                        and gen_response_active[0]
                        and isinstance(response_id, str)
                        and response_id == current_response_id
                        and response_generation == generation_id
                    )
                    if not exact_current_terminal:
                        _log.info(
                            "response.foreign_cancelled_terminal_dropped",
                            turn_id=timer.turn_id,
                            response_id=response_id,
                            response_generation=response_generation,
                            current_response_id=current_response_id,
                            current_generation=generation_id,
                        )
                        continue

                    # Cancellation not owned by interruption/correction must
                    # not leave the runtime permanently stuck in RESPONDING.
                    _cancel_response_terminal_watchdog()
                    cancelled_generation = generation_id
                    cancelled_item_id = current_output_item_id
                    cancelled_content_index = current_output_content_index
                    cancelled_audio_end_ms = speaker.played_audio_ms
                    cancelled_partial_text = gen_assistant_text[0]
                    if response_id:
                        # The response is already terminal. Quarantine any
                        # delayed deltas without sending a redundant cancel.
                        _remember_cancelled_response(response_id)
                    gen_response_active[0] = False
                    speaker.clear()
                    await tts_manager.abort()
                    if cancelled_partial_text:
                        await on_partial_abort(cancelled_partial_text)
                    if finalize_task is not None and not finalize_task.done():
                        finalize_task.cancel()
                    finalize_task = _track_background(
                        _finish_terminal_partial(
                            captured_generation=cancelled_generation,
                            reason="unowned_response_cancel",
                            item_id=cancelled_item_id,
                            content_index=cancelled_content_index,
                            audio_end_ms=cancelled_audio_end_ms,
                        ),
                        name=(
                            "sync_unowned_cancel_"
                            f"{cancelled_generation}"
                        ),
                    )
                    continue
                if not _accept_response_event(event):
                    continue
                if status != "completed":
                    _cancel_response_terminal_watchdog()
                    failed_generation = generation_id
                    partial_text = gen_assistant_text[0]
                    failed_item_id = current_output_item_id
                    failed_content_index = current_output_content_index
                    failed_audio_end_ms = speaker.played_audio_ms
                    if response_id:
                        # The provider has already terminated this response;
                        # quarantine delayed deltas without another cancel.
                        _remember_cancelled_response(response_id)
                    gen_response_active[0] = False
                    speaker.clear()
                    await tts_manager.abort()
                    if partial_text:
                        await on_partial_abort(partial_text)
                    if failed_item_id:
                        if profile == "realtime_audio":
                            _register_item_truncate_barrier(failed_item_id)
                        else:
                            _register_item_delete_barrier(failed_item_id)
                    if finalize_task is not None and not finalize_task.done():
                        finalize_task.cancel()
                    finalize_task = _track_background(
                        _finish_terminal_partial(
                            captured_generation=failed_generation,
                            reason=f"response_{status or 'unknown'}",
                            item_id=failed_item_id,
                            content_index=failed_content_index,
                            audio_end_ms=failed_audio_end_ms,
                        ),
                        name=(
                            f"sync_terminal_{status}_"
                            f"{failed_item_id or failed_generation}"
                        ),
                    )
                    continue
                _cancel_response_terminal_watchdog()
                gen_response_active[0] = False
                completed_generation = generation_id
                completed_timer = timer
                completed_assistant_text = gen_assistant_text[0]
                if finalize_task is not None and not finalize_task.done():
                    finalize_task.cancel()
                finalize_task = _track_background(
                    _finalize_response(
                        completed_timer,
                        completed_generation,
                        completed_assistant_text,
                    ),
                    name=f"finalize_response_{completed_generation}",
                )

            elif t == "error":
                err = event.get("error")
                operation = event.get("operation")
                operation_response_id = event.get("response_id")
                operation_generation = event.get("generation_id")
                item_create_purpose = event.get("item_create_purpose")
                client_event_id = event.get("client_event_id")
                # Expected race: speculative abort sometimes arrives after
                # the server has already finished the response. Log at info.
                code = event.get("error_code") or (
                    err.get("code")
                    if isinstance(err, dict)
                    else getattr(err, "code", None) if err is not None else None
                )
                if code == "response_cancel_not_active":
                    if isinstance(operation_response_id, str):
                        _resolve_cancel_barrier(
                            operation_response_id,
                            succeeded=True,
                        )
                    elif operation == "response.cancel":
                        _resolve_unscoped_cancel_barrier(succeeded=True)
                    _log.info(
                        "llm.cancel_race_ignored",
                        operation=operation,
                    )
                elif (
                    code == "conversation_already_has_active_response"
                    and isinstance(client_event_id, str)
                    and not _remember_response_create_error(client_event_id)
                ):
                    _log.info(
                        "response.duplicate_active_conflict_ignored",
                        client_event_id=client_event_id,
                    )
                elif (
                    code == "conversation_already_has_active_response"
                    and operation in {None, "response.create"}
                    and (
                        not isinstance(operation_generation, int)
                        or operation_generation == generation_id
                    )
                    and state.phase == Phase.RESPONDING
                    and gen_response_active[0]
                    and gen_response_requested[0]
                ):
                    if (
                        active_response_conflict_task is not None
                        and not active_response_conflict_task.done()
                    ):
                        _log.info(
                            "response.active_conflict_recovery_already_running",
                            generation=generation_id,
                        )
                        continue
                    if gen_response_create_attempts[0] < 2:
                        if not _claim_response_resolution(
                            generation_id,
                            owner="active_conflict",
                        ):
                            _log.info(
                                "response.active_conflict_owned_by_correction",
                                generation=generation_id,
                                owner=response_resolution_owner,
                            )
                            continue
                        _log.warning(
                            "response.active_conflict_retrying",
                            generation=generation_id,
                        )
                        active_response_conflict_task = _track_background(
                            _recover_active_response_conflict(generation_id),
                            name=f"recover_active_response_{generation_id}",
                        )
                        response_request_task = active_response_conflict_task
                    else:
                        _log.error(
                            "response.active_conflict_exhausted",
                            generation=generation_id,
                        )
                        _signal_runtime_failure(
                            "Active response conflict recovery was exhausted"
                        )
                elif operation == "response.cancel":
                    if isinstance(operation_response_id, str):
                        _resolve_cancel_barrier(
                            operation_response_id,
                            succeeded=False,
                        )
                    else:
                        _resolve_unscoped_cancel_barrier(succeeded=False)
                    _signal_runtime_failure(
                        "Response cancellation was rejected"
                    )
                    _log.warning(
                        "response.cancel_server_error",
                        code=code,
                    )
                elif operation in {
                    "conversation.item.delete",
                    "conversation.item.truncate",
                    "input_audio_buffer.clear",
                }:
                    if operation == "conversation.item.delete":
                        operation_item_id = event.get("item_id")
                        if isinstance(operation_item_id, str):
                            _resolve_item_delete_barrier(
                                operation_item_id,
                                succeeded=False,
                            )
                    elif operation == "conversation.item.truncate":
                        operation_item_id = event.get("item_id")
                        if isinstance(operation_item_id, str):
                            _resolve_item_truncate_barrier(
                                operation_item_id,
                                succeeded=False,
                            )
                    elif operation == "input_audio_buffer.clear":
                        if not isinstance(operation_generation, int):
                            _signal_runtime_failure(
                                "Manual input clear rejection had no generation"
                            )
                        else:
                            _resolve_input_clear_barrier(
                                operation_generation,
                                succeeded=False,
                            )
                        _signal_runtime_failure(
                            "Manual input buffer clear was rejected"
                        )
                    if operation in {
                        "conversation.item.delete",
                        "conversation.item.truncate",
                    }:
                        _signal_runtime_failure(
                            "Provider history mutation was rejected"
                        )
                    _log.warning(
                        "llm.control_action_error",
                        operation=operation,
                        code=code,
                    )
                elif operation == "input_audio_buffer.commit":
                    if not isinstance(operation_generation, int):
                        _signal_runtime_failure(
                            "Manual input commit rejection had no generation"
                        )
                        continue
                    captured = operation_generation
                    response_request_task = _track_background(
                        _recover_async_commit_error(captured),
                        name=f"recover_commit_error_{captured}",
                    )
                elif (
                    operation == "conversation.item.create"
                    and item_create_purpose == "system_note"
                ):
                    _log.warning(
                        "llm.system_note_create_error",
                        code=code,
                    )
                elif (
                    code == "conversation_already_has_active_response"
                    and operation == "response.create"
                    and isinstance(operation_generation, int)
                    and operation_generation != generation_id
                ):
                    _log.info(
                        "llm.stale_user_turn_transaction_error_ignored",
                        operation=operation,
                        operation_generation=operation_generation,
                        current_generation=generation_id,
                    )
                elif operation in {
                    "response.create",
                    "conversation.item.create",
                }:
                    # These operations are one ordered user-turn transaction.
                    # An asynchronous rejection can arrive after earlier item
                    # creates succeeded, and Realtime does not expose an
                    # atomic rollback identity for the whole transaction.
                    # Reusing the session could therefore retain a ghost user
                    # turn or an already-started response.
                    _log.error(
                        "llm.user_turn_transaction_error",
                        operation=operation,
                        generation=operation_generation,
                        code=code,
                    )
                    _signal_runtime_failure(
                        "Provider user-turn transaction failed"
                    )
                else:
                    _log.error(
                        "llm.error",
                        error_type=(
                            event.get("error_type")
                            or (type(err).__name__ if err is not None else None)
                        ),
                        code=code,
                    )
                    # Unknown Realtime errors may be session-fatal. Exiting
                    # explicitly is safer than leaving the state machine stuck
                    # in RESPONDING while silently dropping microphone frames.
                    failure = RuntimeError(
                        f"Realtime provider error ({code or 'unknown'})"
                    )
                    raise failure from None

        raise ConnectionError("Realtime event stream ended")

    async def _finalize_response(
        t: TurnTimer,
        captured_generation: int,
        completed_assistant_text: str,
    ) -> None:
        nonlocal current_response_id, current_output_item_id
        nonlocal current_output_content_index, speculative_correction_input
        # Wait for all response audio/TTS chunks to reach speaker, then for
        # speaker playback to drain. A stopped output callback returns False
        # after clearing pending bytes instead of leaving this background
        # finalizer (and Phase.RESPONDING) stuck forever.
        if profile == "realtime_audio":
            await realtime_audio_queue.join()
        tts_succeeded = await tts_manager.wait_until_empty()
        playback_result = await speaker.wait_until_done()
        playback_succeeded = playback_result is not False

        if captured_generation != generation_id or state.phase != Phase.RESPONDING:
            _log.info(
                "response.finalize_stale",
                captured_generation=captured_generation,
                current_generation=generation_id,
                phase=state.phase.name,
            )
            return

        # A provider-side response.done only proves generation completed. For
        # external TTS, every accepted sentence must also have synthesized and
        # reached the speaker; for all profiles at least one non-silent frame
        # must have reached the device callback. Otherwise the user did not
        # hear the response and persisting its full text would corrupt history.
        output_was_audible = speaker.first_play_at is not None
        delivery_succeeded = output_was_audible and playback_succeeded and (
            profile == "realtime_audio" or tts_succeeded
        )
        if completed_assistant_text and delivery_succeeded:
            if corrector is not None:
                correction_input = speculative_correction_input
                if (
                    correction_input is not None
                    and correction_input[0] == captured_generation
                ):
                    if corrector_user_history_generation != captured_generation:
                        pending_correction = correction_task
                        if (
                            pending_correction is not None
                            and not pending_correction.done()
                        ):
                            pending_correction.cancel()
                            await asyncio.gather(
                                pending_correction,
                                return_exceptions=True,
                            )
                        _record_corrector_user_once(
                            correction_input[1],
                            captured_generation,
                        )
                    speculative_correction_input = None
                corrector.record_assistant(completed_assistant_text)
        elif completed_assistant_text:
            _log.warning(
                "response.unheard_history_discarded",
                generated_chars=len(completed_assistant_text),
                output_was_audible=output_was_audible,
                playback_succeeded=playback_succeeded,
                tts_succeeded=(
                    tts_succeeded if profile != "realtime_audio" else None
                ),
                tts_failure_reasons=(
                    tts_manager.generation_failure_reasons
                    if profile != "realtime_audio"
                    else ()
                ),
            )
        # Delivery, not transcript availability, decides whether the provider's
        # assistant item is safe to retain. A completed item can arrive without
        # any transcript delta; if no output was heard it must not become a
        # ghost assistant turn in subsequent server context.
        if not delivery_succeeded and current_output_item_id:
            if not await _delete_item_with_ack(current_output_item_id):
                _signal_runtime_failure(
                    "Unheard response item deletion was not acknowledged"
                )
                return

        # Capture speaker timing BEFORE beep plays so the next turn's
        # ttfb measurement isn't overwritten by the beep's buffer write.
        if speaker.first_play_at is not None:
            t.speaker_first_play_ts = speaker.first_play_at
        elif speaker.first_write_at is not None:
            t.speaker_first_play_ts = speaker.first_write_at
        if speaker.first_write_at is not None and speaker.first_play_at is not None:
            t.speaker_buffer_ms = (
                speaker.first_play_at - speaker.first_write_at
            ) * 1000

        # Immediately play the "mic live" beep. Its built-in post-gap
        # (ready_beep_post_gap_s) doubles as the echo-decay window that
        # ``safety_delay_s`` used to provide. Fall back to safety_delay
        # whenever no beep waveform was generated (disabled or zero-duration).
        if ready_beep_pcm:
            await _play_ready_beep()
        elif settings.safety_delay_s > 0:
            await asyncio.sleep(settings.safety_delay_s)

        if captured_generation != generation_id or state.phase != Phase.RESPONDING:
            return

        # Emit the turn.complete summary — but only if this response
        # actually produced audio. A speculative response that was
        # replaced by correction will call _finalize_response after being
        # aborted; skipping the log avoids a noisy ``total_ms=None`` entry.
        total = t.total_ms()
        if total is not None:
            metrics.observe("turn.total_ms", total)
            _log.info(
                "turn.complete",
                turn_id=t.turn_id,
                total_ms=round(total, 1),
                first_llm_delta_ms=(
                    round((t.first_llm_delta_ts - t.speech_end_ts) * 1000, 1)
                    if t.first_llm_delta_ts and t.speech_end_ts else None
                ),
                first_tts_byte_ms=(
                    round((t.first_tts_byte_ts - t.speech_end_ts) * 1000, 1)
                    if t.first_tts_byte_ts and t.speech_end_ts else None
                ),
                speaker_buffer_ms=(
                    round(t.speaker_buffer_ms, 1)
                    if t.speaker_buffer_ms is not None else None
                ),
                correction_ms=(
                    round(t.correction_ms, 1) if t.correction_ms is not None else None
                ),
                interrupted=t.interrupted,
                profile=settings.profile,
            )
        else:
            _log.debug(
                "turn.aborted_before_audio",
                turn_id=t.turn_id,
                correction_ms=(
                    round(t.correction_ms, 1) if t.correction_ms is not None else None
                ),
            )

        if captured_generation != generation_id or state.phase != Phase.RESPONDING:
            return
        current_response_id = None
        current_output_item_id = None
        current_output_content_index = 0
        await state.transition(Phase.LISTENING)

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------
    try:
        await pipeline.llm.open_session()
        tts_manager.start()
        mic.start()
        speaker.start()
        _log.info("orchestrator.started", profile=settings.profile)
        print(f"Zemory is listening… profile={settings.profile} (Ctrl+C to quit)\n")

        async with asyncio.TaskGroup() as tg:
            tg.create_task(mic_pump(), name="mic_pump")
            tg.create_task(turn_event_consumer(), name="turn_events")
            tg.create_task(llm_event_consumer(), name="llm_events")
            tg.create_task(
                realtime_audio_output_consumer(),
                name="realtime_audio_output",
            )
            tg.create_task(speaker.feed(), name="speaker_feed")
            tg.create_task(
                _runtime_failure_monitor(),
                name="runtime_failure_monitor",
            )
            # Fire-and-forget startup beep so the user knows the mic is live.
            # Runs inside the TaskGroup so speaker.feed is already pumping.
            tg.create_task(_play_ready_beep(), name="startup_beep")
    finally:
        active_error = sys.exception()
        cleanup = RuntimeCleanup(timeout_s=_CLEANUP_TIMEOUT_S, logger=_log)

        pending_background = tuple(background_tasks)
        for task in pending_background:
            task.cancel()
        if pending_background:
            _, still_pending = await asyncio.wait(
                pending_background,
                timeout=_CLEANUP_TIMEOUT_S,
            )
            if still_pending:
                cleanup.errors.append(
                    TimeoutError(
                        "background task shutdown exceeded "
                        f"{_CLEANUP_TIMEOUT_S:.3f}s"
                    )
                )
                _log.error(
                    "orchestrator.background_shutdown_timeout",
                    pending_count=len(still_pending),
                    timeout_ms=int(_CLEANUP_TIMEOUT_S * 1000),
                )

        await cleanup.run("microphone", mic.stop)
        await cleanup.run("tts_manager", tts_manager.stop)

        close_turn = getattr(pipeline.turn, "close", None)
        if callable(close_turn):
            await cleanup.run("turn_detector", close_turn)

        await cleanup.run("context_scheduler", context_scheduler.aclose)
        await cleanup.run("interrupt_bus", interrupt_bus.aclose)

        await cleanup.run("llm", pipeline.llm.close)

        for provider in (pipeline.stt, pipeline.tts, corrector):
            if provider is None:
                continue
            close_provider = getattr(provider, "aclose", None)
            if not callable(close_provider):
                close_provider = getattr(provider, "close", None)
            if callable(close_provider):
                await cleanup.run(type(provider).__name__, close_provider)

        await cleanup.run("speaker", speaker.stop)

        # Do not mask the primary runtime failure. If shutdown itself is the
        # only failure, make it explicit after every resource had a chance to
        # close.
        if cleanup.errors and active_error is None:
            raise ExceptionGroup("orchestrator cleanup failures", cleanup.errors)
