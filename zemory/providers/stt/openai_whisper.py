"""Whisper STT via OpenAI ``/audio/transcriptions`` endpoint.

Wraps PCM24k chunks in a WAV header and submits to ``gpt-4o-transcribe``.
Retries once on transient failure; on final failure returns
``"[transcription failed]"`` so the LLM gets a clear signal.
"""

from __future__ import annotations

import io
import wave

from openai import AsyncOpenAI

from zemory.config import settings
from zemory.observability import get_logger

_log = get_logger(__name__)


class WhisperSTT:
    def __init__(self, api_key: str, model: str | None = None) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model or settings.stt_model

    async def transcribe(self, pcm_chunks: list[bytes]) -> str:
        pcm_data = b"".join(pcm_chunks)
        if not pcm_data:
            return ""

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(settings.sample_rate)
            wf.writeframes(pcm_data)
        wav_buf.seek(0)
        wav_buf.name = "audio.wav"

        retries = 2
        while True:
            try:
                resp = await self._client.audio.transcriptions.create(
                    model=self._model, file=wav_buf
                )
                return resp.text.strip()
            except Exception as e:  # pragma: no cover — network dependent
                if retries > 0:
                    retries -= 1
                    _log.warning("stt.retry", error=str(e))
                    wav_buf.seek(0)
                    continue
                _log.error("stt.failed", error=str(e))
                return "[transcription failed]"
