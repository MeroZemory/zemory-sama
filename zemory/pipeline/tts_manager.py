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
from collections.abc import Callable
from typing import TYPE_CHECKING

from zemory.observability import get_logger, metrics

if TYPE_CHECKING:
    from zemory.audio import SpeakerStream
    from zemory.providers.base import TTSProvider

_log = get_logger(__name__)


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

        # seq → list[bytes] payload (chunks may arrive faster than playback)
        self._buffers: dict[int, list[bytes]] = {}
        # seq → event set when synthesis for seq has finished producing chunks
        self._done: dict[int, asyncio.Event] = {}

        self._next_seq = 0
        self._next_play_seq = 0
        self._tasks: set[asyncio.Task] = set()
        self._aborted = False
        self._dispatcher_task: asyncio.Task | None = None
        self._dispatch_wake = asyncio.Event()

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
        self._aborted = False
        self._next_seq = 0
        self._next_play_seq = 0
        self._buffers.clear()
        self._done.clear()
        self._tasks.clear()

    async def stop(self) -> None:
        """Flush pending playback and shut down the dispatcher cleanly."""
        if self._dispatcher_task is not None:
            # Let remaining buffered audio drain; dispatcher exits when no more
            # seqs are expected and _closed is set.
            self._dispatch_wake.set()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._dispatch_wake.set()
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except (asyncio.CancelledError, Exception):
                pass
            self._dispatcher_task = None
        self._tasks.clear()

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------
    def submit(self, text: str) -> int:
        """Queue ``text`` for synthesis. Returns assigned sequence number."""
        if self._aborted:
            return -1
        seq = self._next_seq
        self._next_seq += 1
        self._buffers[seq] = []
        self._done[seq] = asyncio.Event()
        task = asyncio.create_task(self._synthesize(seq, text))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return seq

    async def _synthesize(self, seq: int, text: str) -> None:
        quick = seq == 0  # RVC-style Quick for first sentence only
        async with self._sem:
            if self._aborted:
                self._done[seq].set()
                return
            started = time.monotonic()
            first_chunk_logged = False
            try:
                async for chunk in self._tts.synthesize(text, quick=quick):
                    if self._aborted:
                        break
                    self._buffers[seq].append(chunk)
                    if not first_chunk_logged:
                        ttfb_ms = (time.monotonic() - started) * 1000
                        metrics.observe("ttfb.tts", ttfb_ms)
                        if self._on_first_chunk is not None:
                            self._on_first_chunk(seq, ttfb_ms)
                        _log.info("tts.first_chunk", seq=seq, ttfb_ms=round(ttfb_ms))
                        first_chunk_logged = True
                    self._dispatch_wake.set()
            finally:
                self._done[seq].set()
                self._dispatch_wake.set()

    # ------------------------------------------------------------------
    # Dispatch (seq-ordered playback)
    # ------------------------------------------------------------------
    async def _dispatcher(self) -> None:
        try:
            while True:
                # Drain buffered audio for the current target seq, in order.
                while self._next_play_seq in self._buffers:
                    buf = self._buffers[self._next_play_seq]
                    if buf:
                        chunk = buf.pop(0)
                        if not self._aborted:
                            await self._speaker.queue.put(chunk)
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

    # ------------------------------------------------------------------
    # Barge-in
    # ------------------------------------------------------------------
    async def abort(self) -> None:
        """Cancel all in-flight synthesis + drop any queued payloads.

        Fire-and-forget: we cancel tasks but do NOT await their cleanup,
        since the latency-critical path must return within the interrupt
        budget. Tasks complete asynchronously; ``self._aborted`` gates any
        chunks they might still try to enqueue. If a caller needs to wait
        for full teardown, they call :meth:`stop`.
        """
        self._aborted = True
        for task in list(self._tasks):
            task.cancel()
        self._buffers.clear()
        for ev in self._done.values():
            ev.set()
        self._dispatch_wake.set()

    @property
    def aborted(self) -> bool:
        return self._aborted

    async def wait_until_empty(self) -> None:
        """Wait until all submitted sentences have been handed to the speaker."""
        while True:
            if not self._buffers and not self._tasks:
                return
            await asyncio.sleep(0.05)
