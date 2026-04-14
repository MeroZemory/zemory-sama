"""TranscriptCorrector unit tests.

Uses a stub OpenAI chat-completions client so we can verify:
1. History is built in chronological order and sent to the model
2. Whitespace is stripped from the corrected output
3. Failures fall back to the raw transcript
4. Correction latency is reported
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from zemory.pipeline.transcript_corrector import TranscriptCorrector


@dataclass
class _FakeMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]


class _FakeCompletions:
    def __init__(self, response: str | Exception) -> None:
        self._response = response
        self.received_messages: list[list[dict]] = []
        self.received_model: list[str] = []

    async def create(self, *, model: str, messages: list[dict], temperature: float):
        self.received_messages.append(messages)
        self.received_model.append(model)
        if isinstance(self._response, Exception):
            raise self._response
        return _FakeResponse(choices=[_FakeChoice(_FakeMessage(self._response))])


class _FakeChat:
    def __init__(self, response: str | Exception) -> None:
        self.completions = _FakeCompletions(response)


class _FakeClient:
    def __init__(self, response: str | Exception) -> None:
        self.chat = _FakeChat(response)


@pytest.mark.asyncio
async def test_correct_returns_stripped_text_and_positive_latency():
    client = _FakeClient("  zemory, hello  ")
    corrector = TranscriptCorrector(client=client, model="test-model", history_turns=3)

    corrected, elapsed_ms = await corrector.correct("Jemori, hello")

    assert corrected == "zemory, hello"
    assert elapsed_ms >= 0
    # Model name forwarded
    assert client.chat.completions.received_model == ["test-model"]


@pytest.mark.asyncio
async def test_correct_includes_history_in_messages():
    client = _FakeClient("corrected")
    corrector = TranscriptCorrector(client=client, model="m", history_turns=3)
    corrector.record_user("first user msg")
    corrector.record_assistant("first assistant reply")

    await corrector.correct("new raw")

    sent = client.chat.completions.received_messages[0]
    # system prompt (1) + conversation context (2) + user raw (3)
    assert len(sent) == 3
    assert sent[0]["role"] == "system"
    assert "transcript post-processor" in sent[0]["content"]
    assert "first user msg" in sent[1]["content"]
    assert "first assistant reply" in sent[1]["content"]
    assert "new raw" in sent[2]["content"]


@pytest.mark.asyncio
async def test_correct_empty_input_returns_unchanged_and_skips_api():
    client = _FakeClient("should not be called")
    corrector = TranscriptCorrector(client=client, model="m", history_turns=3)

    corrected, elapsed_ms = await corrector.correct("")
    assert corrected == ""
    assert elapsed_ms == 0.0
    assert client.chat.completions.received_messages == []


@pytest.mark.asyncio
async def test_correct_falls_back_to_raw_on_api_exception():
    client = _FakeClient(RuntimeError("rate limited"))
    corrector = TranscriptCorrector(client=client, model="m", history_turns=3)

    corrected, elapsed_ms = await corrector.correct("raw utterance")
    assert corrected == "raw utterance"
    assert elapsed_ms >= 0


@pytest.mark.asyncio
async def test_correct_falls_back_to_raw_on_empty_result():
    client = _FakeClient("")
    corrector = TranscriptCorrector(client=client, model="m", history_turns=3)

    corrected, _ = await corrector.correct("raw")
    assert corrected == "raw"


@pytest.mark.asyncio
async def test_history_capped_at_history_turns_times_two():
    client = _FakeClient("x")
    corrector = TranscriptCorrector(client=client, model="m", history_turns=2)

    # 2 turns * 2 roles = 4 entries max
    for i in range(5):
        corrector.record_user(f"u{i}")
        corrector.record_assistant(f"a{i}")

    await corrector.correct("new")
    convo = client.chat.completions.received_messages[0][1]["content"]
    # Earliest entries should have been evicted
    assert "u0" not in convo
    assert "a0" not in convo
    # Latest should be present
    assert "u4" in convo
    assert "a4" in convo
