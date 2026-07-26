"""Realtime audio turn detector with local endpoint commits.

This experimental path keeps audio-native Realtime output, but disables
server-side turn detection. Microphone PCM is still streamed to Realtime while
a local endpoint detector emits ``speech_start`` / ``speech_end`` events. The
orchestrator commits the Realtime input buffer on local ``speech_end``.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING, Any

from zemory.config import settings
from zemory.observability import get_logger

if TYPE_CHECKING:
    from zemory.providers.llm.openai_realtime import OpenAIRealtimeLLM

_EVENT_QUEUE_MAXSIZE = 4
_AUDIO_QUEUE_MAXSIZE = 64
_SIGNAL_QUEUE_MAXSIZE = 8

_log = get_logger(__name__)


class RealtimeAudioBackpressureError(RuntimeError):
    """The bounded Realtime audio sender queue could not accept a frame."""


class RealtimeAudioSenderError(RuntimeError):
    """The background Realtime audio sender terminated unexpectedly."""


class RealtimeEndpointStateMachine:
    """Endpoint-only VAD state machine for manual Realtime commits."""

    def __init__(
        self,
        *,
        prob_threshold: float,
        db_threshold: float,
        required_hits: int,
        required_misses: int,
        smoothing_window: int,
    ) -> None:
        self.prob_threshold = prob_threshold
        self.db_threshold = db_threshold
        self.required_hits = required_hits
        self.required_misses = required_misses
        self._prob_win: deque[float] = deque(maxlen=smoothing_window)
        self._db_win: deque[float] = deque(maxlen=smoothing_window)
        self._speaking = False
        self._hit = 0
        self._miss = 0

    def reset(self) -> None:
        self._prob_win.clear()
        self._db_win.clear()
        self._speaking = False
        self._hit = 0
        self._miss = 0

    def process(self, prob: float, db: float) -> str | None:
        self._prob_win.append(prob)
        self._db_win.append(db)
        sp = sum(self._prob_win) / len(self._prob_win)
        sd = sum(self._db_win) / len(self._db_win)
        is_speech = sp >= self.prob_threshold and sd >= self.db_threshold

        if not self._speaking:
            if is_speech:
                self._hit += 1
                if self._hit >= self.required_hits:
                    self._speaking = True
                    self._hit = 0
                    self._miss = 0
                    return "speech_start"
            else:
                self._hit = 0
            return None

        if is_speech:
            self._miss = 0
            return None

        self._miss += 1
        if self._miss >= self.required_misses:
            self._speaking = False
            self._miss = 0
            self._hit = 0
            return "speech_end"
        return None


def _build_endpoint_state_machine() -> RealtimeEndpointStateMachine:
    return RealtimeEndpointStateMachine(
        prob_threshold=settings.vad.prob_threshold,
        db_threshold=settings.vad.db_threshold,
        required_hits=settings.vad.required_hits,
        required_misses=settings.realtime.local_endpoint_required_misses,
        smoothing_window=settings.vad.smoothing_window,
    )


class RealtimeManualTurnDetector:
    """Run local endpoint analysis independently from ordered network sends.

    ``speech_end`` is relayed only after its frame watermark has been appended
    to the Realtime input buffer. The detector then pauses at that exact turn
    boundary until ``reset()`` acknowledges that the orchestrator committed
    the buffer; response creation may happen later after transcript validation.
    Shutdown cancels queued/in-flight sends instead of risking an unbounded
    wait on a stalled network call.
    """

    def __init__(
        self,
        *,
        llm: OpenAIRealtimeLLM,
        endpoint_detector: Any | None = None,
        audio_queue_maxsize: int = _AUDIO_QUEUE_MAXSIZE,
    ) -> None:
        if audio_queue_maxsize <= 0:
            raise ValueError("audio_queue_maxsize must be positive")

        self._llm = llm
        self._endpoint_detector = endpoint_detector
        self.events: asyncio.Queue[str] = asyncio.Queue(maxsize=_EVENT_QUEUE_MAXSIZE)

        # ``feed`` only performs local endpoint analysis. Network I/O runs in a
        # single ordered sender so a slow append cannot delay speech_start/end
        # detection. Queue overflow fails explicitly: dropping PCM would make
        # the local endpoint and the server input buffer disagree.
        self._audio_queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue(
            maxsize=audio_queue_maxsize
        )
        self._signal_queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue(
            maxsize=_SIGNAL_QUEUE_MAXSIZE
        )
        self._feed_lock = asyncio.Lock()
        self._flush_condition = asyncio.Condition()
        self._accepting_audio = asyncio.Event()
        self._accepting_audio.set()

        self._sender_task: asyncio.Task[None] | None = None
        self._endpoint_event_task: asyncio.Task[None] | None = None
        self._signal_task: asyncio.Task[None] | None = None
        self._next_frame_id = 0
        self._analyzing_frame_id: int | None = None
        self._last_analyzed_frame_id = 0
        self._flushed_frame_id = 0
        self._sender_failure_type: str | None = None
        self._background_failure_type: str | None = None
        self._closed = False

    def _endpoint(self) -> Any:
        if self._endpoint_detector is None:
            from zemory.providers.turn.silero import SileroTurnDetector

            self._endpoint_detector = SileroTurnDetector(
                state_machine=_build_endpoint_state_machine(),
                capture_audio=False,
            )
        return self._endpoint_detector

    def _ensure_background_tasks(self) -> None:
        if self._sender_task is not None:
            return
        self._endpoint()
        self._sender_task = asyncio.create_task(
            self._audio_sender(), name="realtime-manual-audio-sender"
        )
        self._endpoint_event_task = asyncio.create_task(
            self._endpoint_event_relay(), name="realtime-manual-endpoint-events"
        )
        self._signal_task = asyncio.create_task(
            self._signal_relay(), name="realtime-manual-signal-relay"
        )
        self._sender_task.add_done_callback(self._background_task_done)
        self._endpoint_event_task.add_done_callback(self._background_task_done)
        self._signal_task.add_done_callback(self._background_task_done)

    def _background_task_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        self._background_failure_type = type(exc).__name__
        self._accepting_audio.set()
        _log.error(
            "realtime_manual.background_failed",
            task=task.get_name(),
            error_type=type(exc).__name__,
        )

    def _raise_if_unavailable(self) -> None:
        if self._closed:
            raise RuntimeError("Realtime manual turn detector is closed")
        if self._sender_failure_type is not None:
            raise RealtimeAudioSenderError("Realtime audio sender failed") from None
        if self._background_failure_type is not None:
            raise RealtimeAudioSenderError("Realtime manual turn background task failed") from None

    async def _audio_sender(self) -> None:
        while True:
            frame_id, pcm24k = await self._audio_queue.get()
            try:
                await self._llm.push_audio(pcm24k)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._sender_failure_type = type(exc).__name__
                self._accepting_audio.set()
                _log.error(
                    "realtime_manual.audio_sender_failed",
                    error_type=type(exc).__name__,
                )
                async with self._flush_condition:
                    self._flush_condition.notify_all()
                return
            finally:
                self._audio_queue.task_done()

            async with self._flush_condition:
                self._flushed_frame_id = frame_id
                self._flush_condition.notify_all()

    async def _wait_until_flushed(self, frame_id: int) -> bool:
        async with self._flush_condition:
            await self._flush_condition.wait_for(
                lambda: (
                    self._flushed_frame_id >= frame_id
                    or self._sender_failure_type is not None
                    or self._closed
                )
            )
        return self._flushed_frame_id >= frame_id

    async def _signal_relay(self) -> None:
        while True:
            signal, frame_id = await self._signal_queue.get()
            try:
                if signal == "speech_end" and not await self._wait_until_flushed(frame_id):
                    return
                await self.events.put(signal)
            finally:
                self._signal_queue.task_done()

    async def _endpoint_event_relay(self) -> None:
        endpoint_events = self._endpoint().events
        while True:
            signal = await endpoint_events.get()

            try:
                frame_id = self._analyzing_frame_id or self._last_analyzed_frame_id
                if signal not in {"speech_start", "speech_end"}:
                    _log.warning(
                        "realtime_manual.unknown_endpoint_signal",
                        signal_type=type(signal).__name__,
                    )
                    continue
                if signal == "speech_end":
                    # No frame from the next turn may reach the server before
                    # the orchestrator commits this exact watermark. reset()
                    # is the acknowledgement immediately after that commit.
                    self._accepting_audio.clear()
                try:
                    self._signal_queue.put_nowait((signal, frame_id))
                except asyncio.QueueFull as exc:
                    raise RealtimeAudioBackpressureError(
                        "Realtime endpoint signal queue is full"
                    ) from exc
            finally:
                endpoint_events.task_done()

    async def feed(self, pcm24k: bytes) -> None:
        self._raise_if_unavailable()
        self._ensure_background_tasks()
        while True:
            await self._accepting_audio.wait()
            self._raise_if_unavailable()

            async with self._feed_lock:
                # Another feed may have passed wait() just before the
                # preceding frame detected speech_end. Recheck under the
                # ordering lock so it cannot cross the commit boundary.
                if not self._accepting_audio.is_set():
                    continue
                self._raise_if_unavailable()
                self._next_frame_id += 1
                frame_id = self._next_frame_id
                try:
                    self._audio_queue.put_nowait((frame_id, pcm24k))
                except asyncio.QueueFull as exc:
                    raise RealtimeAudioBackpressureError(
                        "Realtime audio sender queue is full"
                    ) from exc

                self._analyzing_frame_id = frame_id
                try:
                    await self._endpoint().feed(pcm24k)
                    self._last_analyzed_frame_id = frame_id
                    # Endpoint signals are local queue operations. Waiting only
                    # for their watermark assignment keeps feed independent from
                    # network I/O while ensuring the commit barrier is installed
                    # before another microphone frame can enter.
                    await self._endpoint().events.join()
                finally:
                    self._analyzing_frame_id = None
                break

        # Give the independent sender/relay a scheduling opportunity without
        # ever waiting for the network append itself.
        await asyncio.sleep(0)
        self._raise_if_unavailable()

    def consume_audio(self) -> list[bytes]:
        return []

    def reset(self) -> None:
        reset = getattr(self._endpoint_detector, "reset", None)
        if callable(reset):
            reset()
        if not self._closed:
            self._accepting_audio.set()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._accepting_audio.set()
        async with self._flush_condition:
            self._flush_condition.notify_all()

        tasks = [
            task
            for task in (
                self._sender_task,
                self._endpoint_event_task,
                self._signal_task,
            )
            if task is not None
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        close = getattr(self._endpoint_detector, "close", None)
        if callable(close):
            await close()
