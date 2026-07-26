"""Parallel TTS synthesis with strict sequence-ordered playback.

Inspired by Open-LLM-VTuber's ``TTSTaskManager`` (docs/ref/open-llm-vtuber.md:163-198).
Each ``submit(text)`` assigns a monotonically increasing seq number and
spawns a task that synthesizes + buffers all audio chunks. A single
dispatcher forwards buffered payloads to the speaker in seq order, so
sentence N+1 can generate in parallel but still plays after sentence N.

A :class:`TTSTaskManager` instance is bound to one response. The
orchestrator creates a fresh instance per turn (or calls :meth:`reset`
between turns).

On :meth:`abort` — called by the InterruptBus during barge-in — all
in-flight synthesis tasks are cancelled and queued audio is dropped
before it can reach the speaker.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from zemory.observability import get_logger, metrics

if TYPE_CHECKING:
    from zemory.audio import SpeakerStream
    from zemory.providers.base import TTSProvider

_log = get_logger(__name__)
_MAX_PENDING_SENTENCES = 32
_MAX_BUFFERED_AUDIO_BYTES = 240_000  # 5 seconds of 24 kHz mono PCM16
_BUFFER_CHUNK_BYTES = 4096
_TASK_SHUTDOWN_TIMEOUT_S = 1.0


@dataclass
class _GenerationOutcome:
    """Conservative delivery accounting for one response generation."""

    generation: int
    accepted_sequences: set[int] = field(default_factory=set)
    completed_sequences: set[int] = field(default_factory=set)
    produced_bytes: dict[int, int] = field(default_factory=dict)
    handed_off_bytes: dict[int, int] = field(default_factory=dict)
    failure_reasons: set[str] = field(default_factory=set)
    drain_impossible: bool = False

    @property
    def completed_successfully(self) -> bool:
        if not self.accepted_sequences or self.failure_reasons:
            return False
        if self.completed_sequences != self.accepted_sequences:
            return False
        return all(
            self.produced_bytes.get(seq, 0) > 0
            and self.handed_off_bytes.get(seq, 0) == self.produced_bytes[seq]
            for seq in self.accepted_sequences
        )


class TTSTaskManager:
    def __init__(
        self,
        tts: TTSProvider,
        speaker: SpeakerStream,
        max_concurrent: int,
        on_first_chunk: Callable[[int, float], None] | None = None,
    ) -> None:
        self._tts = tts
        self._speaker = speaker
        self._sem = asyncio.Semaphore(max_concurrent)
        self._on_first_chunk = on_first_chunk

        # seq → buffered payload (chunks may arrive faster than playback)
        self._buffers: dict[int, deque[bytes]] = {}
        # seq → event set when synthesis for seq has finished producing chunks
        self._done: dict[int, asyncio.Event] = {}

        self._next_seq = 0
        self._next_play_seq = 0
        self._generation = 0
        self._outcome = _GenerationOutcome(generation=self._generation)
        # Keep every live task until its done callback runs.  Resetting a turn
        # must not lose the task reference: an old provider may take time to
        # unwind after cancellation and ``stop()`` still owns that cleanup.
        self._tasks: dict[asyncio.Task, tuple[int, int]] = {}
        self._aborted = False
        self._dispatcher_task: asyncio.Task | None = None
        self._dispatch_wake = asyncio.Event()
        self._buffer_space = asyncio.Event()
        self._buffer_space.set()
        self._buffered_bytes = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._reset_state()
        self._dispatcher_task = asyncio.create_task(self._dispatcher())

    def reset_for_new_turn(self) -> None:
        """Clear abort flag + seq counters so the next turn can submit again.

        Keeps the dispatcher task alive. Must be called by the orchestrator
        whenever a new response begins (e.g. on ``speech_stopped``) because
        an earlier barge-in will have set ``_aborted=True``, which would
        otherwise reject every future :meth:`submit`.
        """
        self._reset_state()
        self._dispatch_wake.set()

    def _reset_state(self) -> None:
        if self._outcome.accepted_sequences and not self._outcome.completed_successfully:
            self._outcome.failure_reasons.add("generation_replaced")
        self._generation += 1
        self._outcome = _GenerationOutcome(generation=self._generation)
        self._aborted = False
        self._next_seq = 0
        self._next_play_seq = 0
        self._buffers.clear()
        self._done.clear()
        self._buffered_bytes = 0
        self._buffer_space.set()

    async def stop(self) -> None:
        """Abort provider work and shut down without an unbounded network wait."""
        await self.abort()
        if self._dispatcher_task is not None:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except (asyncio.CancelledError, Exception):
                pass
            self._dispatcher_task = None
        tasks = tuple(self._tasks)
        if tasks:
            done, pending = await asyncio.wait(
                tasks,
                timeout=_TASK_SHUTDOWN_TIMEOUT_S,
            )
            for task in done:
                if not task.cancelled():
                    task.exception()
            if pending:
                _log.warning(
                    "tts.shutdown_timeout",
                    pending_count=len(pending),
                    timeout_ms=int(_TASK_SHUTDOWN_TIMEOUT_S * 1000),
                )

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------
    def submit(self, text: str) -> int:
        """Queue ``text`` for synthesis. Returns assigned sequence number."""
        if self._aborted:
            self._outcome.failure_reasons.add("submission_rejected")
            return -1
        if self._next_seq - self._next_play_seq >= _MAX_PENDING_SENTENCES:
            _log.warning("tts.pending_limit", limit=_MAX_PENDING_SENTENCES)
            self._outcome.failure_reasons.add("submission_rejected")
            return -1
        seq = self._next_seq
        self._next_seq += 1
        generation = self._generation
        buffer: deque[bytes] = deque()
        done = asyncio.Event()
        self._buffers[seq] = buffer
        self._done[seq] = done
        outcome = self._outcome
        outcome.accepted_sequences.add(seq)
        outcome.produced_bytes[seq] = 0
        outcome.handed_off_bytes[seq] = 0
        task = asyncio.create_task(self._synthesize(outcome, generation, seq, text, buffer, done))
        self._tasks[task] = (generation, seq)
        task.add_done_callback(self._on_synthesis_done)
        return seq

    def _owns_sequence(
        self,
        generation: int,
        seq: int,
        buffer: deque[bytes],
        done: asyncio.Event,
    ) -> bool:
        """Return whether a task still owns the current turn's sequence."""
        return (
            generation == self._generation
            and self._buffers.get(seq) is buffer
            and self._done.get(seq) is done
        )

    def _on_synthesis_done(self, task: asyncio.Task) -> None:
        identity = self._tasks.pop(task, None)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            generation, seq = identity if identity is not None else (-1, -1)
            _log.error(
                "tts.synthesis_failed",
                generation=generation,
                seq=seq,
                error_type=type(error).__name__,
            )

    async def _append_audio(
        self,
        outcome: _GenerationOutcome,
        generation: int,
        seq: int,
        payload: bytes,
        buffer: deque[bytes],
        done: asyncio.Event,
    ) -> bool:
        for offset in range(0, len(payload), _BUFFER_CHUNK_BYTES):
            chunk = payload[offset : offset + _BUFFER_CHUNK_BYTES]
            while self._buffered_bytes + len(chunk) > _MAX_BUFFERED_AUDIO_BYTES and not (
                seq == self._next_play_seq
                and sum(len(queued) for queued in buffer) < _BUFFER_CHUNK_BYTES
            ):
                self._buffer_space.clear()
                await self._buffer_space.wait()
                if self._aborted or not self._owns_sequence(generation, seq, buffer, done):
                    return False
            if self._aborted or not self._owns_sequence(generation, seq, buffer, done):
                return False
            buffer.append(chunk)
            self._buffered_bytes += len(chunk)
            outcome.produced_bytes[seq] += len(chunk)
            self._dispatch_wake.set()
        return True

    async def _synthesize(
        self,
        outcome: _GenerationOutcome,
        generation: int,
        seq: int,
        text: str,
        buffer: deque[bytes],
        done: asyncio.Event,
    ) -> None:
        quick = seq == 0  # RVC-style Quick for first sentence only
        async with self._sem:
            if self._aborted or not self._owns_sequence(generation, seq, buffer, done):
                if self._owns_sequence(generation, seq, buffer, done):
                    done.set()
                return
            started = time.monotonic()
            first_chunk_logged = False
            try:
                async for chunk in self._tts.synthesize(text, quick=quick):
                    if not chunk:
                        continue
                    # A reset reuses sequence numbers.  Check both generation
                    # and object identity so an old task can never append to a
                    # replacement turn's seq entry.
                    if self._aborted or not self._owns_sequence(generation, seq, buffer, done):
                        break
                    if not first_chunk_logged:
                        ttfb_ms = (time.monotonic() - started) * 1000
                        metrics.observe("ttfb.tts", ttfb_ms)
                        if self._on_first_chunk is not None:
                            self._on_first_chunk(seq, ttfb_ms)
                        _log.info("tts.first_chunk", seq=seq, ttfb_ms=round(ttfb_ms))
                        first_chunk_logged = True
                    if not await self._append_audio(outcome, generation, seq, chunk, buffer, done):
                        break
            except asyncio.CancelledError:
                outcome.failure_reasons.add("synthesis_cancelled")
                raise
            except Exception:
                outcome.failure_reasons.add("synthesis_failed")
                raise
            else:
                if self._owns_sequence(generation, seq, buffer, done):
                    outcome.completed_sequences.add(seq)
                    if outcome.produced_bytes[seq] == 0:
                        outcome.failure_reasons.add("empty_audio")
            finally:
                # In particular, do not complete a reused seq after the old
                # generation finishes cancellation cleanup.
                if self._owns_sequence(generation, seq, buffer, done):
                    done.set()
                self._dispatch_wake.set()

    # ------------------------------------------------------------------
    # Dispatch (seq-ordered playback)
    # ------------------------------------------------------------------
    async def _dispatcher(self) -> None:
        try:
            while True:
                # Drain buffered audio for the current target seq, in order.
                while self._next_play_seq in self._buffers:
                    generation = self._generation
                    outcome = self._outcome
                    seq = self._next_play_seq
                    buf = self._buffers[seq]
                    done = self._done[seq]
                    if buf:
                        chunk = buf.popleft()
                        self._buffered_bytes -= len(chunk)
                        self._buffer_space.set()
                        if not self._aborted:
                            try:
                                await self._speaker.queue.put(chunk)
                            except asyncio.CancelledError:
                                raise
                            except Exception as exc:
                                outcome.failure_reasons.add("speaker_handoff_failed")
                                outcome.drain_impossible = True
                                _log.error(
                                    "tts.dispatch_failed",
                                    generation=generation,
                                    seq=seq,
                                    error_type=type(exc).__name__,
                                )
                                if self._owns_sequence(generation, seq, buf, done):
                                    self._fail_current_dispatch()
                                break
                            if self._owns_sequence(generation, seq, buf, done):
                                outcome.handed_off_bytes[seq] += len(chunk)
                        continue
                    # buffer empty — either synthesis still running or finished
                    if self._done[self._next_play_seq].is_set():
                        # finished + drained → advance
                        del self._buffers[self._next_play_seq]
                        del self._done[self._next_play_seq]
                        self._next_play_seq += 1
                        continue
                    # wait for more chunks or completion
                    break

                self._dispatch_wake.clear()
                try:
                    await asyncio.wait_for(self._dispatch_wake.wait(), timeout=0.5)
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise

    def _fail_current_dispatch(self) -> None:
        """Drop a generation that can no longer reach the speaker."""
        self._aborted = True
        for task, (generation, _seq) in tuple(self._tasks.items()):
            if generation == self._generation:
                task.cancel()
        self._buffers.clear()
        for event in self._done.values():
            event.set()
        self._done.clear()
        self._buffered_bytes = 0
        self._next_play_seq = self._next_seq
        self._buffer_space.set()
        self._dispatch_wake.set()

    # ------------------------------------------------------------------
    # Barge-in
    # ------------------------------------------------------------------
    async def abort(self) -> None:
        """Cancel all in-flight synthesis + drop any queued payloads.

        Fire-and-forget: we cancel tasks but do NOT await their cleanup,
        since the latency-critical path must return within the interrupt
        budget. Tasks complete asynchronously; abort state plus generation
        ownership gates any chunks they might still try to enqueue. If a
        caller needs to wait for full teardown, they call :meth:`stop`.
        """
        self._aborted = True
        self._outcome.failure_reasons.add("aborted")
        self._outcome.drain_impossible = True
        for task in list(self._tasks):
            task.cancel()
        self._buffers.clear()
        self._buffered_bytes = 0
        self._buffer_space.set()
        for ev in self._done.values():
            ev.set()
        self._done.clear()
        self._dispatch_wake.set()

    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def generation_completed_successfully(self) -> bool:
        """Whether the current generation was fully synthesized and handed off.

        This deliberately reports ``False`` for an empty generation, partial
        synthesis, zero-byte provider output, rejected submission, abort, or
        speaker/dispatcher failure.
        """
        current_tasks = any(
            generation == self._generation for generation, _seq in self._tasks.values()
        )
        return (
            not self._aborted
            and not current_tasks
            and not self._buffers
            and self._next_play_seq == self._next_seq
            and self._outcome.completed_successfully
        )

    @property
    def generation_failure_reasons(self) -> tuple[str, ...]:
        """Stable, payload-free diagnostics for a failed generation."""
        return tuple(sorted(self._outcome.failure_reasons))

    async def wait_until_empty(self) -> bool:
        """Wait for speaker handoff and return the conservative outcome."""
        while True:
            if self._outcome.drain_impossible:
                return False
            current_tasks = any(
                generation == self._generation for generation, _seq in self._tasks.values()
            )
            if not self._buffers and not current_tasks:
                return self.generation_completed_successfully
            if self._dispatcher_task is None or self._dispatcher_task.done():
                self._outcome.failure_reasons.add("dispatcher_unavailable")
                self._outcome.drain_impossible = True
                self._fail_current_dispatch()
                return False
            await asyncio.sleep(0.05)
