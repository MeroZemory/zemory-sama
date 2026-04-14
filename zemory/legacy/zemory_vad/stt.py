"""Modular speech-to-text.

Current implementation: OpenAI Whisper API (gpt-4o-transcribe).
Swappable for other backends by implementing the same interface.
"""

from __future__ import annotations

import io
import wave

from openai import AsyncOpenAI
from zemory_vad.config import SAMPLE_RATE, STT_MODEL


class WhisperSTT:
    """Transcribe PCM audio via OpenAI Whisper API."""

    def __init__(self, client: AsyncOpenAI, model: str = STT_MODEL) -> None:
        self._client = client
        self._model = model

    async def transcribe(self, pcm_chunks: list[bytes]) -> str:
        """Convert PCM 24 kHz int16 chunks → text."""
        pcm_data = b"".join(pcm_chunks)
        if not pcm_data:
            return ""

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_data)
        wav_buf.seek(0)
        wav_buf.name = "audio.wav"

        resp = await self._client.audio.transcriptions.create(
            model=self._model,
            file=wav_buf,
        )
        return resp.text.strip()
