from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError
from structlog.testing import capture_logs

from zemory.providers.stt import openai_whisper
from zemory.providers.stt.openai_whisper import WhisperSTT


class _FakeTranscriptions:
    def __init__(self, results: list[object]) -> None:
        self._results = iter(results)
        self.calls = 0
        self.file_positions: list[int] = []

    async def create(self, *, model: str, file):
        self.calls += 1
        self.file_positions.append(file.tell())
        result = next(self._results)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(text=result)


class _FakeClient:
    def __init__(self, results: list[object]) -> None:
        self.audio = SimpleNamespace(transcriptions=_FakeTranscriptions(results))
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _connection_error(
    request: httpx.Request, cause: Exception | None = None
) -> APIConnectionError:
    error = APIConnectionError(request=request)
    error.__cause__ = cause
    return error


@pytest.mark.asyncio
async def test_non_transient_failure_is_not_retried_or_logged_with_payload() -> None:
    secret = "private-payload-must-not-appear"
    client = _FakeClient([ValueError(secret)])
    stt = WhisperSTT("unused", client=client)

    with capture_logs() as logs:
        result = await stt.transcribe([b"\x01\x00" * 32])

    assert result == "[transcription failed]"
    assert client.audio.transcriptions.calls == 1
    assert secret not in repr(logs)


@pytest.mark.asyncio
async def test_connection_failure_retries_once_with_rewound_file() -> None:
    request = httpx.Request("POST", "https://example.invalid/transcribe")
    client = _FakeClient(
        [
            _connection_error(
                request, httpx.ConnectError("connect failed", request=request)
            ),
            "  recovered transcript  ",
        ]
    )
    stt = WhisperSTT("unused", client=client)

    result = await stt.transcribe([b"\x01\x00" * 32])

    assert result == "recovered transcript"
    assert client.audio.transcriptions.calls == 2
    assert client.audio.transcriptions.file_positions == [0, 0]


@pytest.mark.asyncio
@pytest.mark.parametrize("cause_type", [httpx.ReadError, httpx.WriteError])
async def test_ambiguous_connection_failure_is_not_retried(
    cause_type: type[httpx.RequestError],
) -> None:
    request = httpx.Request("POST", "https://example.invalid/transcribe")
    client = _FakeClient(
        [
            _connection_error(
                request, cause_type("ambiguous request state", request=request)
            ),
            "must not be requested",
        ]
    )
    stt = WhisperSTT("unused", client=client)

    result = await stt.transcribe([b"\x01\x00" * 32])

    assert result == "[transcription failed]"
    assert client.audio.transcriptions.calls == 1


@pytest.mark.asyncio
async def test_connection_error_without_transport_cause_is_not_retried() -> None:
    request = httpx.Request("POST", "https://example.invalid/transcribe")
    client = _FakeClient(
        [_connection_error(request), "must not be requested"]
    )
    stt = WhisperSTT("unused", client=client)

    result = await stt.transcribe([b"\x01\x00" * 32])

    assert result == "[transcription failed]"
    assert client.audio.transcriptions.calls == 1


@pytest.mark.asyncio
async def test_timeout_is_not_retried_even_when_connect_phase_is_the_cause() -> None:
    request = httpx.Request("POST", "https://example.invalid/transcribe")
    error = APITimeoutError(request=request)
    error.__cause__ = httpx.ConnectTimeout("connect timed out", request=request)
    client = _FakeClient([error, "must not be requested"])
    stt = WhisperSTT("unused", client=client)

    result = await stt.transcribe([b"\x01\x00" * 32])

    assert result == "[transcription failed]"
    assert client.audio.transcriptions.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_server_response_is_not_retried_without_idempotency_guarantee(
    status: int,
) -> None:
    request = httpx.Request("POST", "https://example.invalid/transcribe")
    response = httpx.Response(status, request=request)
    error = APIStatusError("server failure", response=response, body=None)
    client = _FakeClient([error, "must not be requested"])
    stt = WhisperSTT("unused", client=client)

    result = await stt.transcribe([b"\x01\x00" * 32])

    assert result == "[transcription failed]"
    assert client.audio.transcriptions.calls == 1


@pytest.mark.asyncio
async def test_injected_client_is_not_closed() -> None:
    client = _FakeClient(["ok"])
    stt = WhisperSTT("unused", client=client)

    await stt.aclose()

    assert client.closed is False


@pytest.mark.asyncio
async def test_owned_client_disables_sdk_retries_and_is_closed(monkeypatch) -> None:
    client = _FakeClient(["ok"])
    received: dict[str, object] = {}

    def build_client(**kwargs):
        received.update(kwargs)
        return client

    monkeypatch.setattr(openai_whisper, "AsyncOpenAI", build_client)
    stt = WhisperSTT("test-key")

    await stt.aclose()

    assert received == {
        "api_key": "test-key",
        "base_url": openai_whisper.settings.openai_base_url,
        "timeout": 15.0,
        "max_retries": 0,
    }
    assert client.closed is True
