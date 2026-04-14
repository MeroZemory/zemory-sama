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
        retries_left = 2
        while True:
            try:
                async with http.stream(
                    "POST", url, headers=headers, json=body, params=params
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes(4096):
                        yield chunk
                return
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status in (429, 500, 502, 503, 504) and retries_left > 0:
                    retries_left -= 1
                    _log.warning("tts.retry", status=status, text_len=len(text))
                    continue
                _log.error("tts.http_error", status=status)
                return
            except httpx.RequestError as e:
                if retries_left > 0:
                    retries_left -= 1
                    _log.warning("tts.request_error_retry", error=str(e))
                    continue
                _log.error("tts.request_error", error=str(e))
                return

    async def warmup(self) -> None:
        """Establish the HTTPS connection pool + DNS + TLS ahead of first use.

        ElevenLabs' first stream response incurs ~2-3 s of cold-start
        overhead (handshake + model load). Submitting a trivial
        single-character request at startup primes the httpx connection
        pool so the first real user turn sees a ~300 ms TTFB instead.
        Failures are logged and swallowed — warmup is best-effort.
        """
        try:
            async for _ in self.synthesize(".", quick=True):
                # Consume a few bytes and stop — we don't need the audio.
                break
            _log.info("tts.warmup.done")
        except Exception as e:  # pragma: no cover
            _log.warning("tts.warmup.failed", error=str(e))

    async def aclose(self) -> None:
        if self._http and self._owns_http:
            await self._http.aclose()
