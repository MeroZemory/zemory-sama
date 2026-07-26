from __future__ import annotations

import httpx
import pytest

from zemory.providers.tts.elevenlabs import ElevenLabsTTS


class _FailAfterFirstChunk(httpx.AsyncByteStream):
    def __init__(self, request: httpx.Request) -> None:
        self._request = request

    async def __aiter__(self):
        yield b"x" * 4096
        raise httpx.ReadError("stream broke", request=self._request)

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_partial_stream_failure_is_never_retried_or_duplicated() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            request=request,
            stream=_FailAfterFirstChunk(request),
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tts = ElevenLabsTTS("secret", http=http)
    chunks: list[bytes] = []

    with pytest.raises(httpx.ReadError):
        async for chunk in tts.synthesize("hello"):
            chunks.append(chunk)

    await http.aclose()
    assert calls == 1
    assert chunks == [b"x" * 4096]


@pytest.mark.asyncio
async def test_connection_failure_before_audio_is_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("connect failed", request=request)
        return httpx.Response(200, request=request, content=b"pcm")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tts = ElevenLabsTTS("secret", http=http)

    chunks = [chunk async for chunk in tts.synthesize("hello")]

    await http.aclose()
    assert calls == 2
    assert chunks == [b"pcm"]


@pytest.mark.asyncio
async def test_connect_timeout_before_audio_is_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectTimeout("connect timed out", request=request)
        return httpx.Response(200, request=request, content=b"pcm")

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tts = ElevenLabsTTS("secret", http=http)

    chunks = [chunk async for chunk in tts.synthesize("hello")]

    await http.aclose()
    assert calls == 2
    assert chunks == [b"pcm"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [httpx.ReadTimeout, httpx.WriteError])
async def test_ambiguous_transport_failure_before_audio_is_not_retried(
    error_type: type[httpx.RequestError],
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise error_type("ambiguous request state", request=request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tts = ElevenLabsTTS("secret", http=http)

    with pytest.raises(error_type):
        _ = [chunk async for chunk in tts.synthesize("hello")]

    await http.aclose()
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_server_response_is_not_retried_without_idempotency_guarantee(
    status: int,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, request=request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tts = ElevenLabsTTS("secret", http=http)

    with pytest.raises(httpx.HTTPStatusError):
        _ = [chunk async for chunk in tts.synthesize("hello")]

    await http.aclose()
    assert calls == 1
