"""Whisper STT via OpenAI ``/audio/transcriptions`` endpoint.

Wraps PCM24k chunks in a WAV header and submits to ``gpt-4o-transcribe``.
Retries once only when the SDK exposes an underlying connection-establishment
failure; otherwise returns ``"[transcription failed]"`` so the LLM gets a
clear signal without risking a duplicate billable transcription.
"""

from __future__ import annotations

import inspect
import io
import wave

import httpx
from openai import APIConnectionError, AsyncOpenAI

from zemory.config import settings
from zemory.observability import get_logger

_log = get_logger(__name__)


class WhisperSTT:
    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        *,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=settings.openai_base_url,
            timeout=15.0,
            # Keep retry ownership in this latency-sensitive adapter. The SDK
            # otherwise retries each attempt internally as well.
            max_retries=0,
        )
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

        retries = 1
        while True:
            try:
                resp = await self._client.audio.transcriptions.create(
                    model=self._model, file=wav_buf
                )
                return resp.text.strip()
            except Exception as e:  # pragma: no cover — network dependent
                retryable = _is_preconnect_failure(e)
                if retryable and retries > 0:
                    retries -= 1
                    _log.warning(
                        "stt.retry",
                        error_type=type(e).__name__,
                        status=getattr(e, "status_code", None),
                    )
                    wav_buf.seek(0)
                    continue
                _log.error(
                    "stt.failed",
                    error_type=type(e).__name__,
                    status=getattr(e, "status_code", None),
                    retryable=retryable,
                )
                return "[transcription failed]"

    async def aclose(self) -> None:
        if not self._owns_client:
            return
        close = getattr(self._client, "close", None)
        if not callable(close):
            return
        result = close()
        if inspect.isawaitable(result):
            await result


def _is_preconnect_failure(error: Exception) -> bool:
    # APIConnectionError also wraps ambiguous read/write/protocol failures.
    # Only its direct HTTPX ConnectError cause proves no request reached the
    # service. Timeouts and status responses therefore fail closed.
    return isinstance(error, APIConnectionError) and isinstance(
        error.__cause__, httpx.ConnectError
    )
