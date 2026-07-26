"""Barge-in abort chain.

When the user starts speaking during ``Phase.RESPONDING`` the orchestrator
fires :meth:`InterruptBus.trigger`. This runs the following sequence
(target: ≤ 150 ms p95):

1. ``SpeakerStream.clear()`` — drop in-flight audio
2. ``TTSTaskManager.abort()`` — cancel synthesis tasks, drop queued bytes
3. transition state to ACTIVE immediately
4. ``LLMProvider.cancel_current()`` — bounded remote cancellation
5. truncate Realtime audio to the heard cursor, or delete an external-TTS item

The Realtime input buffer is deliberately *not* cleared here: a server-VAD
``speech_started`` event means it already contains the new user utterance.

Duplicate triggers are suppressed by the phase transition under ``_lock``. A
time-based debounce is deliberately avoided because it can overlap the next
response generation and suppress a legitimate second interruption.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from zemory.observability import get_logger, metrics
from zemory.state import Phase

if TYPE_CHECKING:
    from zemory.audio import SpeakerStream
    from zemory.pipeline.tts_manager import TTSTaskManager
    from zemory.providers.base import LLMProvider
    from zemory.state import StateMachine

_log = get_logger(__name__)

_REMOTE_SYNC_TIMEOUT_S = 0.1
_REMOTE_ACTION_TIMEOUT_S = 0.5
_CANCEL_ACTION_TIMEOUT_S = 0.1


class InterruptBus:
    def __init__(
        self,
        state: StateMachine,
        speaker: SpeakerStream,
        on_partial: Callable[[str], Awaitable[None]] | None = None,
        on_output_interrupted: Callable[[], Awaitable[None]] | None = None,
        get_response_id: Callable[[], str | None] | None = None,
        on_remote_failure: Callable[[str], None] | None = None,
    ) -> None:
        self._state = state
        self._speaker = speaker
        self._tts_manager: TTSTaskManager | None = None
        self._llm: LLMProvider | None = None
        self._on_partial = on_partial
        self._on_output_interrupted = on_output_interrupted
        self._get_response_id = get_response_id
        self._on_remote_failure = on_remote_failure
        self._partial_text = ""
        self._lock = asyncio.Lock()
        self._remote_tasks: set[asyncio.Task[None]] = set()

    def bind(
        self,
        tts_manager: TTSTaskManager | None,
        llm: LLMProvider,
    ) -> None:
        self._tts_manager = tts_manager
        self._llm = llm

    def record_partial(self, delta: str) -> None:
        """Accumulate provisional deltas for abort accounting and callbacks."""
        self._partial_text += delta

    def reset_partial(self) -> None:
        self._partial_text = ""

    async def aclose(self) -> None:
        """Cancel and retrieve remote synchronization tasks at shutdown."""
        tasks = tuple(self._remote_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._remote_tasks.clear()

    def _track_remote(self, coro: Awaitable[None]) -> asyncio.Task[None]:
        task = asyncio.create_task(coro, name="interrupt_remote_sync")
        self._remote_tasks.add(task)

        def done(completed: asyncio.Task[None]) -> None:
            self._remote_tasks.discard(completed)
            if completed.cancelled():
                return
            error = completed.exception()
            if error is not None:  # pragma: no cover - individual actions isolate errors
                _log.warning(
                    "interrupt.remote_task_failed",
                    error_type=type(error).__name__,
                )

        task.add_done_callback(done)
        return task

    async def trigger(self, reason: str) -> bool:
        """Run the abort chain. Return False if this response was already handled.

        Only effective when ``phase == RESPONDING`` — harmless no-op otherwise.
        """
        now = time.monotonic()
        if self._state.phase != Phase.RESPONDING:
            return False

        async with self._lock:
            # re-check under lock
            if self._state.phase != Phase.RESPONDING:
                return False
            _log.warning("interrupt.trigger", reason=reason,
                         partial_len=len(self._partial_text))

            # [1] Drop audio in the speaker immediately.
            self._speaker.clear()

            # [2] Cancel all TTS synthesis tasks.
            if self._tts_manager is not None:
                await self._tts_manager.abort()

            partial_text = self._partial_text
            self._partial_text = ""
            response_id = (
                self._get_response_id() if self._get_response_id is not None else None
            )
            allow_unscoped_cancel = self._get_response_id is None

            # Calling the hook here lets it snapshot response/item/playback
            # identity before the orchestrator invalidates the generation.
            output_sync: Awaitable[None] | None = None
            if self._on_output_interrupted is not None:
                try:
                    output_sync = self._on_output_interrupted()
                except Exception as e:
                    _log.warning(
                        "interrupt.output_prepare_failed",
                        error_type=type(e).__name__,
                    )
                    if self._on_remote_failure is not None:
                        self._on_remote_failure("output_prepare")

            # [3] Open the local capture path before any network await. This
            # keeps interruption latency independent of provider stalls.
            await self._state.transition(Phase.ACTIVE)

        # Remote conversation synchronization is important, but must never
        # hold the state lock or violate the 150 ms local interruption budget.
        remote_task = self._track_remote(
            self._sync_remote_interruption(
                partial_text,
                output_sync,
                response_id,
                allow_unscoped_cancel=allow_unscoped_cancel,
            )
        )
        try:
            async with asyncio.timeout(_REMOTE_SYNC_TIMEOUT_S):
                await asyncio.shield(remote_task)
        except TimeoutError:
            # The local interruption is complete. Keep the bounded remote task
            # alive so a slow cancel cannot permanently skip conversation
            # truncation and corrupt the next turn's server context.
            _log.info(
                "interrupt.remote_sync_deferred",
                wait_budget_ms=int(_REMOTE_SYNC_TIMEOUT_S * 1000),
            )

        elapsed_ms = (time.monotonic() - now) * 1000
        metrics.observe("interrupt.chain_total_ms", elapsed_ms)
        _log.info("interrupt.done", elapsed_ms=round(elapsed_ms, 1))
        return True

    async def _sync_remote_interruption(
        self,
        partial_text: str,
        output_sync: Awaitable[None] | None,
        response_id: str | None,
        *,
        allow_unscoped_cancel: bool,
    ) -> None:
        actions: list[tuple[str, Callable[[], Awaitable[None]]]] = []
        output_started = False

        async def run_output_sync() -> None:
            nonlocal output_started
            assert output_sync is not None
            output_started = True
            await output_sync

        if self._llm is not None and (
            response_id is not None or allow_unscoped_cancel
        ):
            async def cancel_response() -> None:
                assert self._llm is not None
                try:
                    await self._llm.cancel_current(response_id=response_id)
                except TypeError:
                    # Compatibility for third-party providers implementing the
                    # pre-response-id protocol.
                    await self._llm.cancel_current()

            actions.append(("cancel", cancel_response))
        if output_sync is not None:
            actions.append(("truncate", run_output_sync))
        if self._on_partial is not None and partial_text:
            actions.append(("partial", lambda: self._on_partial(partial_text)))

        try:
            for action, run in actions:
                try:
                    timeout_s = (
                        _CANCEL_ACTION_TIMEOUT_S
                        if action == "cancel"
                        else _REMOTE_ACTION_TIMEOUT_S
                    )
                    async with asyncio.timeout(timeout_s):
                        await run()
                except TimeoutError:
                    _log.warning(
                        "interrupt.remote_action_timeout",
                        action=action,
                        timeout_ms=int(timeout_s * 1000),
                    )
                    if (
                        action in {"cancel", "truncate"}
                        and self._on_remote_failure is not None
                    ):
                        self._on_remote_failure(action)
                except Exception as e:  # pragma: no cover - best-effort remote sync
                    _log.warning(
                        "interrupt.remote_sync_failed",
                        action=action,
                        error_type=type(e).__name__,
                    )
                    if (
                        action in {"cancel", "truncate"}
                        and self._on_remote_failure is not None
                    ):
                        self._on_remote_failure(action)
        finally:
            if (
                output_sync is not None
                and not output_started
                and inspect.iscoroutine(output_sync)
            ):
                output_sync.close()
