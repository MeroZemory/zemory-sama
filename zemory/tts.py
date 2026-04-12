from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from zemory.config import ELEVENLABS_API_KEY, ELEVENLABS_MODEL_ID, ELEVENLABS_VOICE_ID

_SENTENCE_BOUNDARIES = frozenset(".?!。？！\n")

_ELEVENLABS_URL = (
    f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
)
_ELEVENLABS_HEADERS = {
    "xi-api-key": ELEVENLABS_API_KEY,
    "Content-Type": "application/json",
}


class SentenceChunker:
    """Accumulates streaming text and yields complete sentences."""

    def __init__(self) -> None:
        self._buf = ""

    def add(self, text: str) -> list[str]:
        self._buf += text
        sentences: list[str] = []
        while True:
            idx = next(
                (i for i, ch in enumerate(self._buf) if ch in _SENTENCE_BOUNDARIES),
                -1,
            )
            if idx == -1:
                break
            sentence = self._buf[: idx + 1].strip()
            self._buf = self._buf[idx + 1 :]
            if sentence:
                sentences.append(sentence)
        return sentences

    def flush(self) -> str | None:
        remaining = self._buf.strip()
        self._buf = ""
        return remaining or None


async def elevenlabs_tts(
    http: httpx.AsyncClient, text: str
) -> AsyncIterator[bytes]:
    """Stream PCM 24kHz audio from ElevenLabs for the given text."""
    async with http.stream(
        "POST",
        _ELEVENLABS_URL,
        headers=_ELEVENLABS_HEADERS,
        json={"text": text, "model_id": ELEVENLABS_MODEL_ID},
        params={"output_format": "pcm_24000"},
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes(4096):
            yield chunk
