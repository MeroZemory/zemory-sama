"""SQLite memory store contract tests."""

from __future__ import annotations

import pytest

from zemory.pipeline.context import (
    AsyncContextScheduler,
    MemoryItem,
    SQLiteMemoryStore,
)


@pytest.mark.asyncio
async def test_sqlite_memory_store_persists_and_recalls_ranked_hits(tmp_path) -> None:
    path = tmp_path / "memory.sqlite3"
    store = SQLiteMemoryStore(path)

    await store.write_reflection(
        [
            MemoryItem(
                text="사용자는 한국어로 짧고 자연스러운 답변을 좋아한다.",
                importance=8,
                metadata={"source": "reflection"},
            ),
            MemoryItem(
                text="사용자는 장황한 영어 설명을 싫어한다.",
                importance=3,
                metadata={"source": "reflection"},
            ),
        ]
    )

    reopened = SQLiteMemoryStore(path)
    hits = await reopened.recall("한국어 짧은 답변", limit=2, deadline_ms=50)

    assert [hit.text for hit in hits] == [
        "사용자는 한국어로 짧고 자연스러운 답변을 좋아한다.",
        "사용자는 장황한 영어 설명을 싫어한다.",
    ]
    assert hits[0].score > hits[1].score
    assert hits[0].metadata["source"] == "reflection"
    assert hits[0].metadata["importance"] == 8


@pytest.mark.asyncio
async def test_sqlite_memory_store_empty_query_returns_no_hits(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    await store.write_reflection([MemoryItem(text="무관한 기억", importance=5)])

    assert await store.recall("", limit=5, deadline_ms=50) == []


@pytest.mark.asyncio
async def test_scheduler_uses_sqlite_memory_store_without_blocking(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    await store.write_reflection(
        [MemoryItem(text="사용자는 반말보다 존댓말을 선호한다.", importance=7)]
    )
    scheduler = AsyncContextScheduler(memory=store, memory_deadline_ms=50)

    bundle = await scheduler.gather_for_turn("존댓말로 말해줘")

    assert bundle.late is False
    assert bundle.injections
    assert bundle.injections[0].source == "memory"
    assert "존댓말" in bundle.injections[0].text


@pytest.mark.asyncio
async def test_orchestrator_builds_configured_sqlite_memory_scheduler(
    tmp_path,
    monkeypatch,
) -> None:
    from zemory import orchestrator
    from zemory.config import settings

    memory_path = tmp_path / "runtime-memory.sqlite3"
    monkeypatch.setattr(settings, "memory_enabled", True, raising=False)
    monkeypatch.setattr(settings, "memory_path", str(memory_path), raising=False)
    monkeypatch.setattr(settings, "memory_recall_deadline_ms", 50, raising=False)
    monkeypatch.setattr(settings, "memory_recall_limit", 2, raising=False)
    monkeypatch.setattr(settings, "context_tool_deadline_ms", 50, raising=False)

    scheduler = orchestrator.build_context_scheduler()
    await scheduler._memory.write_reflection(
        [MemoryItem(text="사용자는 차분하고 짧은 답변을 선호한다.", importance=9)]
    )

    bundle = await scheduler.gather_for_turn("차분한 답변으로 말해줘")

    assert memory_path.exists()
    assert bundle.late is False
    assert [injection.source for injection in bundle.injections] == ["memory"]
    assert "차분" in bundle.injections[0].text
