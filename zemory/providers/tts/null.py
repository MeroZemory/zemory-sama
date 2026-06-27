"""No-op TTS provider for audio-native Realtime sessions."""

from __future__ import annotations

from collections.abc import AsyncIterator


class NullTTS:
    """Satisfies the TTSProvider protocol without requiring an external TTS key."""

    async def synthesize(self, text: str, quick: bool = False) -> AsyncIterator[bytes]:
        if False:
            yield b""
