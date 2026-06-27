"""OpenAI Realtime adapter contract tests with a fake SDK client."""

from __future__ import annotations

import base64

import pytest

from zemory.providers.base import Injection
from zemory.providers.llm.openai_realtime import OpenAIRealtimeLLM


class FakeSession:
    def __init__(self) -> None:
        self.updated_with: dict | None = None

    async def update(self, *, session: dict) -> None:
        self.updated_with = session


class FakeInputAudioBuffer:
    def __init__(self) -> None:
        self.appended: list[str] = []
        self.cleared = 0
        self.committed = 0

    async def append(self, *, audio: str) -> None:
        self.appended.append(audio)

    async def clear(self) -> None:
        self.cleared += 1

    async def commit(self) -> None:
        self.committed += 1


class FakeConversationItem:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.deleted: list[str] = []

    async def create(self, *, item: dict) -> None:
        self.created.append(item)

    async def delete(self, *, item_id: str) -> None:
        self.deleted.append(item_id)


class FakeConversation:
    def __init__(self) -> None:
        self.item = FakeConversationItem()


class FakeResponse:
    def __init__(self) -> None:
        self.created = 0
        self.cancelled = 0

    async def create(self) -> None:
        self.created += 1

    async def cancel(self) -> None:
        self.cancelled += 1


class FakeConnection:
    def __init__(self) -> None:
        self.session = FakeSession()
        self.input_audio_buffer = FakeInputAudioBuffer()
        self.conversation = FakeConversation()
        self.response = FakeResponse()

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class FakeConnectionManager:
    def __init__(self, conn: FakeConnection) -> None:
        self.conn = conn
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> FakeConnection:
        self.entered = True
        return self.conn

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.exited = True


class FakeRealtimeResource:
    def __init__(self, manager: FakeConnectionManager) -> None:
        self.manager = manager
        self.connect_calls: list[dict] = []

    def connect(self, **kwargs) -> FakeConnectionManager:
        self.connect_calls.append(kwargs)
        return self.manager


class FakeBetaRealtimeResource:
    def connect(self, **kwargs):  # pragma: no cover - should never be called
        raise AssertionError("beta realtime connect must not be used")


class FakeClient:
    def __init__(self, manager: FakeConnectionManager) -> None:
        self.realtime = FakeRealtimeResource(manager)
        self.beta = type("Beta", (), {"realtime": FakeBetaRealtimeResource()})()


@pytest.mark.asyncio
async def test_open_session_uses_ga_realtime_connect_and_updates_session() -> None:
    conn = FakeConnection()
    manager = FakeConnectionManager(conn)
    client = FakeClient(manager)
    llm = OpenAIRealtimeLLM(api_key="test", client=client)

    await llm.open_session()

    assert client.realtime.connect_calls == [{"model": "gpt-realtime-2"}]
    assert manager.entered is True
    assert conn.session.updated_with is not None
    assert conn.session.updated_with["type"] == "realtime"
    assert conn.session.updated_with["output_modalities"] == ["audio"]

    await llm.close()
    assert manager.exited is True


@pytest.mark.asyncio
async def test_adapter_pushes_audio_and_text_items_through_connection() -> None:
    conn = FakeConnection()
    client = FakeClient(FakeConnectionManager(conn))
    llm = OpenAIRealtimeLLM(api_key="test", client=client)
    await llm.open_session()

    await llm.push_audio(b"abc")
    await llm.send_user_text(
        "hello",
        injections=[
            Injection(source="late", priority=200, text="late context"),
            Injection(source="early", priority=10, text="early context"),
        ],
    )
    await llm.record_system_note("interrupted note")

    assert conn.input_audio_buffer.appended == [
        base64.b64encode(b"abc").decode("ascii")
    ]
    roles = [item["role"] for item in conn.conversation.item.created]
    assert roles == ["system", "system", "user", "system"]
    assert conn.conversation.item.created[0]["content"][0]["text"] == "early context"
    assert conn.conversation.item.created[-1]["content"][0]["text"] == "interrupted note"
    assert conn.response.created == 1

    await llm.close()


@pytest.mark.asyncio
async def test_adapter_commits_input_audio_buffer() -> None:
    conn = FakeConnection()
    client = FakeClient(FakeConnectionManager(conn))
    llm = OpenAIRealtimeLLM(api_key="test", client=client)
    await llm.open_session()

    await llm.commit_input_audio_buffer()

    assert conn.input_audio_buffer.committed == 1

    await llm.close()
