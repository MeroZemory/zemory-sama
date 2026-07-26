"""Shared test fixtures and fakes.

We can't stand up the real Realtime/ElevenLabs/sounddevice stack in a
unit test, so the fakes below mirror the public surfaces of the real
providers just enough to drive the pipeline code paths.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator

# --- Required before importing zemory.config (which validates keys on import) ---
os.environ.setdefault("OPENAI_API_KEY", "test-openai")
os.environ.setdefault("ELEVENLABS_API_KEY", "test-elevenlabs")

# --- Stub heavy optional deps so importing zemory.* doesn't require them ---
# sounddevice pulls in PortAudio — skip when tests run headless.
if "sounddevice" not in sys.modules:
    try:
        import sounddevice  # noqa: F401
    except Exception:
        import types
        sd = types.ModuleType("sounddevice")

        class _DummyStream:
            def start(self) -> None: ...
            def stop(self) -> None: ...
            def close(self) -> None: ...

        sd.InputStream = lambda **kw: _DummyStream()
        sd.OutputStream = lambda **kw: _DummyStream()
        sd.CallbackFlags = object
        sys.modules["sounddevice"] = sd


class FakeSpeaker:
    """Mimics :class:`zemory.audio.SpeakerStream` for ordering/timing tests."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._buf = bytearray()
        self.first_write_at: float | None = None
        self._armed = False
        self.cleared = 0

    def arm(self) -> None:
        self.first_write_at = None
        self._armed = True

    def clear(self) -> None:
        self.cleared += 1
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._buf.clear()

    async def wait_until_done(self) -> None:
        while not self.queue.empty():
            await asyncio.sleep(0.01)

    async def drain_for_test(self) -> list[bytes]:
        """Collect everything currently in the queue. Test-only helper."""
        out: list[bytes] = []
        while not self.queue.empty():
            try:
                out.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return out


class FakeTTS:
    """Returns deterministic "chunk" bytes; optional per-sentence delay."""

    def __init__(self, delay_ms: dict[str, float] | None = None) -> None:
        self._delay_ms = delay_ms or {}

    async def synthesize(self, text: str, quick: bool = False) -> AsyncIterator[bytes]:
        # Allow test to inject latency per text.
        if text in self._delay_ms:
            await asyncio.sleep(self._delay_ms[text] / 1000)
        # Emit 3 chunks: the text, then a marker indicating quick/final.
        yield f"<start:{text}>".encode()
        await asyncio.sleep(0)
        yield f"<{text}>".encode()
        await asyncio.sleep(0)
        yield f"<end:{text}:q={quick}>".encode()


class FakeLLM:
    """Records cancel() calls and partial-text callbacks."""

    def __init__(self) -> None:
        self.cancel_called = 0
        self.clear_called = 0

    async def cancel_current(self, response_id: str | None = None) -> None:
        self.cancel_called += 1

    async def clear_input_buffer(
        self,
        *,
        generation_id: int | None = None,
    ) -> None:
        self.clear_called += 1

    async def send_user_text(
        self,
        text: str,
        injections=None,
        *,
        generation_id: int | None = None,
    ) -> None:
        return None
