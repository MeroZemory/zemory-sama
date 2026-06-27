"""Async memory/context scheduler tests."""

from __future__ import annotations

import asyncio

import pytest

from zemory.pipeline.context import (
    AsyncContextScheduler,
    MemoryHit,
    TranscriptLedger,
)


class FakeMemoryStore:
    def __init__(self, *, delay_ms: int = 0) -> None:
        self.delay_ms = delay_ms
        self.queries: list[str] = []

    async def recall(self, query: str, *, limit: int, deadline_ms: int) -> list[MemoryHit]:
        self.queries.append(query)
        if self.delay_ms:
            await asyncio.sleep(self.delay_ms / 1000)
        return [
            MemoryHit(
                text="사용자는 짧은 한국어 답변을 선호한다.",
                score=0.91,
                metadata={"kind": "preference"},
            )
        ][:limit]


def test_transcript_ledger_records_interrupted_assistant_turn() -> None:
    ledger = TranscriptLedger(max_turns=3)

    ledger.record_user("안녕")
    ledger.record_assistant("안녕하세요.", interrupted=True)

    assert [turn.role for turn in ledger.window()] == ["user", "assistant"]
    assert ledger.window()[-1].interrupted is True
    assert "interrupted" in ledger.as_context_text()


@pytest.mark.asyncio
async def test_scheduler_returns_memory_injections_before_deadline() -> None:
    scheduler = AsyncContextScheduler(memory=FakeMemoryStore(), memory_deadline_ms=50)

    bundle = await scheduler.gather_for_turn("내 말투 기억해?")

    assert [inj.source for inj in bundle.injections] == ["memory"]
    assert bundle.injections[0].priority == 60
    assert "짧은 한국어" in bundle.injections[0].text
    assert bundle.late is False


@pytest.mark.asyncio
async def test_scheduler_does_not_block_first_audio_for_late_memory() -> None:
    scheduler = AsyncContextScheduler(
        memory=FakeMemoryStore(delay_ms=80),
        memory_deadline_ms=10,
    )

    started = asyncio.get_running_loop().time()
    bundle = await scheduler.gather_for_turn("늦은 메모리 테스트")
    elapsed_ms = (asyncio.get_running_loop().time() - started) * 1000

    assert elapsed_ms < 50
    assert bundle.injections == []
    assert bundle.late is True

    await asyncio.sleep(0.1)
    assert len(scheduler.late_results) == 1
    assert scheduler.late_results[0][0].text.startswith("사용자는")
