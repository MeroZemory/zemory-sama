"""ElevenLabs Flash v2.5 TTS provider.

Implements RVC Quick/Final by varying ``optimize_streaming_latency``:
``quick=True`` uses the fastest setting (level 4) for the first sentence,
``quick=False`` uses level 2 (higher quality) for subsequent sentences.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from zemory.config import settings
from zemory.observability import get_logger

_log = get_logger(__name__)


class ElevenLabsTTS:
    def __init__(self, api_key: str, http: httpx.AsyncClient | None = None) -> None:
        self._api_key = api_key
        self._http = http
        self._owns_http = http is None

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=30.0)
        return self._http

    async def synthesize(self, text: str, quick: bool = False) -> AsyncIterator[bytes]:
        voice = settings.tts.voice_id
        latency = (
            settings.tts.quick_latency_level
            if quick
            else settings.tts.final_latency_level
        )
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream"
        headers = {"xi-api-key": self._api_key, "Content-Type": "application/json"}
        body = {"text": text, "model_id": settings.tts.model_id}
        params = {"output_format": "pcm_24000", "optimize_streaming_latency": latency}

        http = await self._client()
        # A second POST is safe only when HTTPX proves that no connection was
        # established. Once connected, this endpoint has no idempotency
        # guarantee, so availability must yield to duplicate-audio/cost safety.
        retries_left = 1
        emitted_audio = False
        while True:
            try:
                async with http.stream(
                    "POST", url, headers=headers, json=body, params=params
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes(4096):
                        emitted_audio = True
                        yield chunk
                return
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                _log.error("tts.http_error", status=status)
                raise
            except httpx.RequestError as e:
                preconnect_failure = isinstance(
                    e, (httpx.ConnectError, httpx.ConnectTimeout)
                )
                if preconnect_failure and not emitted_audio and retries_left > 0:
                    retries_left -= 1
                    _log.warning(
                        "tts.connect_retry",
                        error_type=type(e).__name__,
                        text_len=len(text),
                    )
                    continue
                _log.error(
                    "tts.request_error",
                    error_type=type(e).__name__,
                    partial_audio=emitted_audio,
                    preconnect_failure=preconnect_failure,
                )
                raise

    async def aclose(self) -> None:
        if self._http and self._owns_http:
            await self._http.aclose()
            self._http = None
