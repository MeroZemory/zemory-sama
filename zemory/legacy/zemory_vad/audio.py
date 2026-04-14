from __future__ import annotations

import asyncio
import sys
import threading

import numpy as np
import sounddevice as sd
from zemory_vad.config import CHUNK_DURATION_MS, SAMPLE_RATE


class MicrophoneStream:
    """Captures PCM16 audio from the default microphone."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.chunk_size = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
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
    """Plays PCM16 audio to the default speaker."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._loop = loop
        self._stream: sd.OutputStream | None = None
        self._buffer = bytearray()
        self._lock = threading.Lock()

    def _callback(
        self, outdata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags
    ) -> None:
        bytes_needed = frames * 2
        with self._lock:
            if len(self._buffer) >= bytes_needed:
                data = bytes(self._buffer[:bytes_needed])
                del self._buffer[:bytes_needed]
            else:
                data = bytes(self._buffer) + b"\x00" * (bytes_needed - len(self._buffer))
                self._buffer.clear()
        outdata[:] = np.frombuffer(data, dtype="int16").reshape(-1, 1)

    async def feed(self) -> None:
        while True:
            chunk = await self.queue.get()
            with self._lock:
                self._buffer.extend(chunk)

    def clear(self) -> None:
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
                remaining = len(self._buffer)
            if remaining == 0 and self.queue.empty():
                break
            await asyncio.sleep(0.05)

    def start(self) -> None:
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=960,  # 40ms
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
