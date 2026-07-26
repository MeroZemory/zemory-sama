"""Realtime API server_vad turn detector.

In the realtime profile, VAD runs on OpenAI's server. This adapter's
:meth:`feed` forwards mic audio to the LLM (which drives VAD), and
speech_start / speech_end events are populated by the orchestrator when
it receives the corresponding Realtime events.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from zemory.observability import get_logger

if TYPE_CHECKING:
    from zemory.providers.llm.openai_realtime import OpenAIRealtimeLLM

# With the default 20 ms mic frame, 32 waiting frames represent 640 ms of
# already-stale audio (plus one in flight). A larger buffer only postpones a
# terminal WebSocket-backpressure signal beyond the low-latency turn budget.
_AUDIO_QUEUE_MAXSIZE = 32
_CLOSE_TIMEOUT_S = 0.5

_log = get_logger(__name__)


class ServerVADAudioBackpressureError(RuntimeError):
    """The ordered server-VAD audio queue can no longer accept input."""


class ServerVADAudioSenderError(RuntimeError):
    """The ordered server-VAD audio sender is no longer healthy."""


class ServerVADTurnDetector:
    """Forward mic frames through one ordered, bounded background sender.

    A Realtime WebSocket append must not stall the microphone consumer: audio
    drivers commonly use a separate bounded callback queue and would otherwise
    discard newer frames without surfacing the network fault. ``feed`` therefore
    enqueues synchronously and yields once only to expose an immediate sender
    failure. The queue accepts at most ``audio_queue_maxsize`` waiting frames,
    in addition to the single in-flight frame owned by the sender.

    Overflow is terminal because accepting later frames would create a silent
    hole in the server's input stream. Provider exception details and PCM are
    never retained in the propagated error.
    """

    def __init__(
        self,
        llm: OpenAIRealtimeLLM,
        *,
        audio_queue_maxsize: int = _AUDIO_QUEUE_MAXSIZE,
        close_timeout_s: float = _CLOSE_TIMEOUT_S,
    ) -> None:
        if audio_queue_maxsize <= 0:
            raise ValueError("audio_queue_maxsize must be positive")
        if close_timeout_s <= 0:
            raise ValueError("close_timeout_s must be positive")

        self._llm = llm
        self.events: asyncio.Queue[str] = asyncio.Queue()
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=audio_queue_maxsize
        )
        self._close_timeout_s = close_timeout_s
        self._sender_task: asyncio.Task[None] | None = None
        self._failure_kind: str | None = None
        self._closed = False

    def _discard_queued_frames(self) -> int:
        discarded = 0
        while True:
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                return discarded
            self._audio_queue.task_done()
            discarded += 1

    def _mark_sender_failure(self, failure_type: str) -> None:
        if self._failure_kind is not None:
            return
        self._failure_kind = "sender"
        discarded = self._discard_queued_frames()
        _log.error(
            "server_vad.audio_sender_failed",
            error_type=failure_type,
            discarded_frames=discarded,
        )

    def _observe_sender_completion(self, task: asyncio.Task[None]) -> None:
        if (
            self._failure_kind is not None
            or self._closed
            or not task.done()
        ):
            return
        if task.cancelled():
            self._mark_sender_failure("CancelledError")
            return
        try:
            error = task.exception()
        except asyncio.CancelledError:
            self._mark_sender_failure("CancelledError")
            return
        self._mark_sender_failure(
            type(error).__name__ if error is not None else "UnexpectedExit"
        )

    def _sender_done(self, task: asyncio.Task[None]) -> None:
        self._observe_sender_completion(task)

    def _ensure_sender(self) -> None:
        if self._sender_task is not None:
            return
        self._sender_task = asyncio.create_task(
            self._audio_sender(),
            name="server-vad-audio-sender",
        )
        self._sender_task.add_done_callback(self._sender_done)

    def check_health(self) -> None:
        """Raise a payload-free terminal error at an explicit health boundary."""

        if self._sender_task is not None:
            self._observe_sender_completion(self._sender_task)
        if self._failure_kind == "backpressure":
            raise ServerVADAudioBackpressureError(
                "Realtime server VAD sender is unavailable after queue overflow"
            ) from None
        if self._failure_kind == "sender":
            raise ServerVADAudioSenderError(
                "Realtime server VAD sender failed"
            ) from None
        if self._closed:
            raise RuntimeError("Realtime server VAD turn detector is closed")

    async def _audio_sender(self) -> None:
        while True:
            pcm24k = await self._audio_queue.get()
            try:
                await self._llm.push_audio(pcm24k)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self._mark_sender_failure(type(error).__name__)
                return
            finally:
                self._audio_queue.task_done()

    async def feed(self, pcm24k: bytes) -> None:
        self.check_health()
        self._ensure_sender()
        try:
            self._audio_queue.put_nowait(pcm24k)
        except asyncio.QueueFull as error:
            # Once a frame is rejected, later audio must not be accepted into
            # the now-discontinuous Realtime stream.
            self._failure_kind = "backpressure"
            discarded = self._discard_queued_frames()
            _log.error(
                "server_vad.audio_queue_overflow",
                queue_capacity=self._audio_queue.maxsize,
                discarded_frames=discarded,
            )
            raise ServerVADAudioBackpressureError(
                "Realtime server VAD audio sender queue is full"
            ) from error

        # This is a scheduler/health boundary, not a flush. A stuck WebSocket
        # append remains owned by the background task.
        await asyncio.sleep(0)
        self.check_health()

    def consume_audio(self) -> list[bytes]:
        # Realtime server sees audio; no local replay buffer.
        return []

    async def notify(self, event: str) -> None:
        await self.events.put(event)

    async def close(self) -> None:
        if self._closed:
            if self._sender_task is not None and not self._sender_task.done():
                self._sender_task.cancel()
                raise ServerVADAudioSenderError(
                    "Realtime server VAD sender did not stop before shutdown deadline"
                ) from None
            if self._failure_kind == "sender":
                raise ServerVADAudioSenderError(
                    "Realtime server VAD sender failed"
                ) from None
            return

        sender = self._sender_task
        # A task may have terminated immediately before close(), with its
        # call_soon done callback not run yet. Observe that completion here so
        # close remains a reliable failure boundary.
        if sender is not None:
            self._observe_sender_completion(sender)

        self._closed = True
        if sender is not None and not sender.done():
            sender.cancel()
        discarded = self._discard_queued_frames()
        if discarded:
            _log.info(
                "server_vad.audio_queue_discarded_on_close",
                discarded_frames=discarded,
            )

        sender_timed_out = False
        if sender is not None and not sender.done():
            try:
                _, pending = await asyncio.wait(
                    {sender},
                    timeout=self._close_timeout_s,
                )
            except asyncio.CancelledError:
                sender.cancel()
                raise
            if pending:
                # A task that repeatedly suppresses CancelledError cannot be
                # killed by asyncio. Re-signal cancellation, return within the
                # detector deadline, and let the CLI-owned bounded loop abandon
                # it if the provider remains hostile.
                sender.cancel()
                sender_timed_out = True
                _log.error(
                    "server_vad.audio_sender_shutdown_timeout",
                    timeout_ms=int(self._close_timeout_s * 1000),
                )

        if sender_timed_out:
            raise ServerVADAudioSenderError(
                "Realtime server VAD sender did not stop before shutdown deadline"
            ) from None
        if self._failure_kind == "sender":
            raise ServerVADAudioSenderError(
                "Realtime server VAD sender failed"
            ) from None
