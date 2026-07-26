from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from zemory.config import CHUNK_DURATION_MS, SAMPLE_RATE

_MIC_QUEUE_MAX_FRAMES = max(2, 1000 // CHUNK_DURATION_MS)
_MIC_STARTUP_GRACE_S = 2.0
_MIC_CALLBACK_STALL_TIMEOUT_S = 2.0
_SPEAKER_QUEUE_MAX_CHUNKS = 32
_SPEAKER_QUEUE_CHUNK_BYTES = 4096
_SPEAKER_BUFFER_MAX_BYTES = SAMPLE_RATE * 2 // 2  # 500 ms of mono PCM16
_PLAYBACK_HEALTH_POLL_S = 0.05
_PLAYBACK_STALL_TIMEOUT_S = 2.0


@dataclass(frozen=True, slots=True)
class MicrophoneHealth:
    """Payload-free snapshot of the input callback's device health."""

    started: bool
    stopping: bool
    finished: bool
    active: bool | None
    last_callback_at: float | None
    failure_reason: str | None


class _EpochAudioQueue:
    """Bounded byte queue whose entries can be invalidated by ``clear``.

    Producers capture the current epoch before waiting for capacity. This
    prevents an old response that was blocked on backpressure from reappearing
    after an interruption has flushed playback.
    """

    def __init__(self, epoch: Callable[[], int]) -> None:
        self._epoch = epoch
        self._queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue(
            maxsize=_SPEAKER_QUEUE_MAX_CHUNKS
        )

    @property
    def maxsize(self) -> int:
        return self._queue.maxsize

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def full(self) -> bool:
        return self._queue.full()

    async def put(self, data: bytes) -> None:
        epoch = self._epoch()
        for offset in range(0, len(data), _SPEAKER_QUEUE_CHUNK_BYTES):
            await self._queue.put(
                (epoch, data[offset : offset + _SPEAKER_QUEUE_CHUNK_BYTES])
            )

    def put_nowait(self, data: bytes) -> None:
        chunk_count = (
            len(data) + _SPEAKER_QUEUE_CHUNK_BYTES - 1
        ) // _SPEAKER_QUEUE_CHUNK_BYTES
        if self.qsize() + chunk_count > self.maxsize:
            raise asyncio.QueueFull
        epoch = self._epoch()
        for offset in range(0, len(data), _SPEAKER_QUEUE_CHUNK_BYTES):
            self._queue.put_nowait(
                (epoch, data[offset : offset + _SPEAKER_QUEUE_CHUNK_BYTES])
            )

    async def get(self) -> bytes:
        _epoch, data = await self._queue.get()
        return data

    def get_nowait(self) -> bytes:
        _epoch, data = self._queue.get_nowait()
        return data

    async def get_tagged(self) -> tuple[int, bytes]:
        return await self._queue.get()


def generate_beep_pcm(
    frequency_hz: float,
    duration_ms: int,
    sample_rate: int = SAMPLE_RATE,
    volume: float = 0.15,
) -> bytes:
    """Generate a short sine-wave beep as PCM16 bytes at ``sample_rate``.

    Includes a 5 ms fade-in/out envelope to avoid audible clicks at the
    tone boundaries. Output is mono, little-endian int16 (matching the
    SpeakerStream expected format).
    """
    n = int(sample_rate * duration_ms / 1000)
    if n <= 0:
        return b""
    t = np.arange(n) / sample_rate
    tone = np.sin(2 * np.pi * frequency_hz * t) * max(0.0, min(volume, 1.0))
    fade = min(n // 4, int(sample_rate * 0.005))
    if fade > 0:
        tone[:fade] *= np.linspace(0.0, 1.0, fade)
        tone[-fade:] *= np.linspace(1.0, 0.0, fade)
    pcm16 = (tone * 32767).astype(np.int16)
    return pcm16.tobytes()


def output_block_size(sample_rate: int = SAMPLE_RATE, block_ms: int = 10) -> int:
    """Return output callback frames for a low-latency PCM block."""
    return int(sample_rate * block_ms / 1000)


class MicrophoneStream:
    """Captures PCM16 audio from the default microphone."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=_MIC_QUEUE_MAX_FRAMES
        )
        self.chunk_size = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 480 samples
        self._loop = loop
        self._stream: sd.InputStream | None = None
        self._health_lock = threading.Lock()
        self._started = False
        self._stopping = False
        self._started_at: float | None = None
        self._last_callback_at: float | None = None
        self._stream_finished = threading.Event()
        self._stream_finished.set()
        self.dropped_frames = 0

    def _enqueue_captured(self, pcm: bytes) -> None:
        if self.queue.full():
            try:
                self.queue.get_nowait()
                self.dropped_frames += 1
            except asyncio.QueueEmpty:  # pragma: no cover - same-loop race guard
                pass
        self.queue.put_nowait(pcm)

    def _callback(
        self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags
    ) -> None:
        with self._health_lock:
            if self._stopping:
                return
            self._last_callback_at = time.monotonic()
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        self._loop.call_soon_threadsafe(self._enqueue_captured, indata.tobytes())

    def start(self) -> None:
        self._stream_finished.clear()
        with self._health_lock:
            self._started = False
            self._stopping = False
            self._started_at = None
            self._last_callback_at = None
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=self.chunk_size,
            latency="low",
            callback=self._callback,
            finished_callback=self._stream_finished.set,
        )
        with self._health_lock:
            self._stream = stream
        try:
            stream.start()
        except BaseException:
            self._stream_finished.set()
            raise
        with self._health_lock:
            self._started = True
            self._started_at = time.monotonic()

    def capture_health(self) -> MicrophoneHealth:
        """Return a coherent, thread-safe capture callback health snapshot.

        Silent input remains healthy because PortAudio continues invoking the
        callback with zero-valued frames. A missing first callback gets a
        startup grace period; a previously healthy callback gets its own stall
        deadline. Intentional shutdown always masks terminal device state.
        """
        with self._health_lock:
            started = self._started
            stopping = self._stopping
            started_at = self._started_at
            last_callback_at = self._last_callback_at
            stream = self._stream

            active: bool | None = None
            status_unavailable = False
            if stream is not None:
                try:
                    active = bool(stream.active)
                except Exception:
                    status_unavailable = True

            finished = self._stream_finished.is_set()
            failure_reason: str | None = None
            if started and not stopping:
                now = time.monotonic()
                within_startup_grace = (
                    last_callback_at is None
                    and started_at is not None
                    and now - started_at < _MIC_STARTUP_GRACE_S
                )
                if finished or stream is None:
                    failure_reason = "stream_finished"
                elif status_unavailable and not within_startup_grace:
                    failure_reason = "stream_status_unavailable"
                elif not active and not within_startup_grace:
                    failure_reason = "stream_inactive"
                elif active:
                    if last_callback_at is None:
                        if (
                            started_at is not None
                            and now - started_at >= _MIC_STARTUP_GRACE_S
                        ):
                            failure_reason = "callback_stalled"
                    elif (
                        now - last_callback_at
                        >= _MIC_CALLBACK_STALL_TIMEOUT_S
                    ):
                        failure_reason = "callback_stalled"

            return MicrophoneHealth(
                started=started,
                stopping=stopping,
                finished=finished,
                active=active,
                last_callback_at=last_callback_at,
                failure_reason=failure_reason,
            )

    def stop(self) -> None:
        with self._health_lock:
            self._stopping = True
            stream = self._stream
        if stream:
            try:
                try:
                    stream.stop()
                finally:
                    stream.close()
            finally:
                with self._health_lock:
                    self._started = False
                    self._stream = None
                self._stream_finished.set()
        else:
            with self._health_lock:
                self._started = False
                self._stream = None
            self._stream_finished.set()

    def clear(self) -> None:
        """Drop captured frames that belong to an intentionally muted window."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class SpeakerStream:
    """Plays PCM16 audio to the default speaker.

    Exposes ``first_write_at`` — the ``time.monotonic()`` timestamp of the
    first byte written to the playback buffer after the most recent
    :meth:`arm` call. Used by the orchestrator to measure end-to-end turn
    latency (speech_end → first audio out).
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._clear_epoch = 0
        self.queue = _EpochAudioQueue(lambda: self._clear_epoch)
        self._loop = loop
        self._stream: sd.OutputStream | None = None
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self.first_write_at: float | None = None
        self.first_play_at: float | None = None
        self._played_samples = 0
        self._armed = False
        self._feed_pending = False
        self._ever_started = False
        self._stopping = False
        self._stream_finished = threading.Event()
        self._stream_finished.set()
        self._playback_failed = threading.Event()
        self._terminal_failure_reason: str | None = None
        self._last_playback_progress_at = time.monotonic()

    def arm(self) -> None:
        """Mark the start of a new response. Next buffer write records its timestamp."""
        self.first_write_at = None
        self.first_play_at = None
        self._played_samples = 0
        self._armed = True

    @property
    def played_audio_ms(self) -> int:
        """Source audio submitted to output callbacks since the last arm."""
        return int(self._played_samples * 1000 / SAMPLE_RATE)

    def _callback(
        self, outdata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags
    ) -> None:
        bytes_needed = frames * 2  # 2 bytes per int16 sample
        with self._lock:
            source_bytes = min(len(self._buffer), bytes_needed)
            if len(self._buffer) >= bytes_needed:
                data = bytes(self._buffer[:bytes_needed])
                del self._buffer[:bytes_needed]
            else:
                data = bytes(self._buffer) + b"\x00" * (bytes_needed - len(self._buffer))
                self._buffer.clear()
            if self.first_play_at is None and data.strip(b"\x00"):
                self.first_play_at = time.monotonic()
            self._played_samples += source_bytes // 2
            if source_bytes:
                self._last_playback_progress_at = time.monotonic()
        outdata[:] = np.frombuffer(data, dtype="int16").reshape(-1, 1)

    def _playback_failure_reason(self) -> str | None:
        """Return the payload-free terminal output-device failure reason."""
        with self._lock:
            buffered = len(self._buffer)
            last_progress_at = self._last_playback_progress_at
            stopping = self._stopping
            terminal_reason = self._terminal_failure_reason
            stream = self._stream
            ever_started = self._ever_started
        if terminal_reason is not None:
            return terminal_reason
        if stopping or not ever_started:
            return None
        if self._stream_finished.is_set() or stream is None:
            return "stream_finished"
        try:
            active = stream.active
        except Exception:
            return "stream_status_unavailable"
        if not active:
            return "stream_inactive"
        if (
            buffered > 0
            and time.monotonic() - last_progress_at >= _PLAYBACK_STALL_TIMEOUT_S
        ):
            return "callback_stalled"
        return None

    def _fail_pending_playback(self, reason: str) -> str | None:
        """Persist a device failure unless intentional shutdown won the race."""
        with self._lock:
            if self._stopping:
                return None
            if self._terminal_failure_reason is None:
                self._terminal_failure_reason = reason
            terminal_reason = self._terminal_failure_reason
        self._playback_failed.set()
        self.clear()
        return terminal_reason

    @staticmethod
    def _terminal_failure(reason: str) -> RuntimeError:
        return RuntimeError(f"Speaker output failed ({reason}); restart Zemory")

    def _raise_if_terminal(self) -> None:
        reason = self._playback_failure_reason()
        if reason is None:
            return
        terminal_reason = self._fail_pending_playback(reason)
        if terminal_reason is not None:
            raise self._terminal_failure(terminal_reason)

    async def feed(self) -> None:
        """Pump audio and terminate the owning TaskGroup on device failure."""
        while True:
            self._raise_if_terminal()
            try:
                async with asyncio.timeout(_PLAYBACK_HEALTH_POLL_S):
                    epoch, chunk = await self.queue.get_tagged()
            except TimeoutError:
                self._raise_if_terminal()
                continue
            self._feed_pending = True
            try:
                offset = 0
                while offset < len(chunk):
                    appended = 0
                    with self._lock:
                        if epoch != self._clear_epoch:
                            break
                        capacity = _SPEAKER_BUFFER_MAX_BYTES - len(self._buffer)
                        if capacity > 0:
                            appended = min(capacity, len(chunk) - offset)
                            if self._armed and not self._buffer:
                                # Record TTFB when current-generation audio first
                                # reaches the device-facing buffer.
                                self.first_write_at = time.monotonic()
                                self._armed = False
                            if not self._buffer:
                                self._last_playback_progress_at = time.monotonic()
                            self._buffer.extend(chunk[offset : offset + appended])
                    offset += appended
                    if offset < len(chunk) and epoch == self._clear_epoch:
                        self._raise_if_terminal()
                        await asyncio.sleep(0.005)
            finally:
                self._feed_pending = False

    def clear(self) -> None:
        """Flush all buffered audio (called on interruption)."""
        self._clear_epoch += 1
        with self._lock:
            self._buffer.clear()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def wait_until_done(self) -> bool:
        """Wait for playback, or fail closed if the device callback stops.

        Returns ``False`` after dropping pending bytes when the output stream is
        inactive or makes no playback progress for a bounded interval. Response
        finalization can then discard unheard full text while :meth:`feed`
        propagates the same terminal failure to the runtime TaskGroup.
        """
        while True:
            if self._playback_failed.is_set():
                self.clear()
                return False
            failure_reason = self._playback_failure_reason()
            if failure_reason is not None:
                terminal_reason = self._fail_pending_playback(failure_reason)
                return terminal_reason is None
            with self._lock:
                buf_remaining = len(self._buffer)
            if buf_remaining == 0 and self.queue.empty() and not self._feed_pending:
                return True
            await asyncio.sleep(_PLAYBACK_HEALTH_POLL_S)

    def start(self) -> None:
        with self._lock:
            terminal_reason = self._terminal_failure_reason
            self._stopping = False
        if terminal_reason is not None:
            raise self._terminal_failure(terminal_reason)
        self._stream_finished.clear()
        self._playback_failed.clear()
        stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=output_block_size(),
            latency="low",
            callback=self._callback,
            finished_callback=self._stream_finished.set,
        )
        with self._lock:
            self._stream = stream
        try:
            stream.start()
        except BaseException:
            self._stream_finished.set()
            raise
        with self._lock:
            self._ever_started = True

    def stop(self) -> None:
        with self._lock:
            self._stopping = True
            stream = self._stream
        if stream:
            try:
                try:
                    stream.stop()
                finally:
                    stream.close()
            finally:
                with self._lock:
                    self._stream = None
                self._stream_finished.set()
        else:
            self._stream_finished.set()
