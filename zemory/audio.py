from __future__ import annotations

import asyncio
import sys
import threading
import time

import numpy as np
import sounddevice as sd

from zemory.config import CHUNK_DURATION_MS, SAMPLE_RATE


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


def output_block_size(sample_rate: int = SAMPLE_RATE, block_ms: int = 20) -> int:
    """Return output callback frames for a low-latency PCM block."""
    return int(sample_rate * block_ms / 1000)


class MicrophoneStream:
    """Captures PCM16 audio from the default microphone."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.chunk_size = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 480 samples
        self._loop = loop
        self._stream: sd.InputStream | None = None

    def _callback(
        self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags
    ) -> None:
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        self._loop.call_soon_threadsafe(self.queue.put_nowait, indata.tobytes())

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=self.chunk_size,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class SpeakerStream:
    """Plays PCM16 audio to the default speaker.

    Exposes ``first_write_at`` — the ``time.monotonic()`` timestamp of the
    first byte written to the playback buffer after the most recent
    :meth:`arm` call. Used by the orchestrator to measure end-to-end turn
    latency (speech_end → first audio out).
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._loop = loop
        self._stream: sd.OutputStream | None = None
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self.first_write_at: float | None = None
        self._armed = False

    def arm(self) -> None:
        """Mark the start of a new response. Next buffer write records its timestamp."""
        self.first_write_at = None
        self._armed = True

    def _callback(
        self, outdata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags
    ) -> None:
        bytes_needed = frames * 2  # 2 bytes per int16 sample
        with self._lock:
            if len(self._buffer) >= bytes_needed:
                data = bytes(self._buffer[:bytes_needed])
                del self._buffer[:bytes_needed]
            else:
                data = bytes(self._buffer) + b"\x00" * (bytes_needed - len(self._buffer))
                self._buffer.clear()
        outdata[:] = np.frombuffer(data, dtype="int16").reshape(-1, 1)

    async def feed(self) -> None:
        """Async task: pull decoded audio from queue into the playback buffer."""
        while True:
            chunk = await self.queue.get()
            with self._lock:
                if self._armed and not self._buffer:
                    # record TTFB the moment non-empty audio first reaches the buffer
                    self.first_write_at = time.monotonic()
                    self._armed = False
                self._buffer.extend(chunk)

    def clear(self) -> None:
        """Flush all buffered audio (called on interruption)."""
        with self._lock:
            self._buffer.clear()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def wait_until_done(self) -> None:
        """Wait until all queued and buffered audio has finished playing."""
        while True:
            with self._lock:
                buf_remaining = len(self._buffer)
            if buf_remaining == 0 and self.queue.empty():
                break
            await asyncio.sleep(0.05)

    def start(self) -> None:
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=output_block_size(),
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
