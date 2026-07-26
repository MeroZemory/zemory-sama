"""SQLite memory store contract tests."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import stat
import threading
import time

import pytest

from zemory.pipeline import context as context_module
from zemory.pipeline.context import (
    AsyncContextScheduler,
    MemoryItem,
    NullMemoryStore,
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
async def test_sqlite_memory_store_uses_insertion_order_for_tied_scores(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    await store.write_reflection(
        [
            MemoryItem(text="alpha first", importance=5),
            MemoryItem(text="alpha second", importance=5),
        ]
    )

    hits = await store.recall("alpha", limit=2, deadline_ms=50)

    assert [hit.text for hit in hits] == ["alpha first", "alpha second"]


@pytest.mark.asyncio
async def test_sqlite_memory_store_bounds_python_scoring_candidates(
    tmp_path,
    monkeypatch,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    await store.write_reflection(
        [
            MemoryItem(text=f"candidate-{index}", importance=index % 10)
            for index in range(context_module._SQLITE_RECALL_MAX_CANDIDATES * 3)
        ]
    )
    original_tokenize = context_module._tokenize
    tokenized_texts: list[str] = []

    def counted_tokenize(text: str) -> set[str]:
        tokenized_texts.append(text)
        return original_tokenize(text)

    monkeypatch.setattr(context_module, "_tokenize", counted_tokenize)

    hits = await store.recall("candidate", limit=5, deadline_ms=500)

    assert len(hits) == 5
    # One query tokenization plus at most the SQL candidate LIMIT reaches
    # Python; the remaining database rows are never scored in-process.
    expected_candidates = min(
        context_module._SQLITE_RECALL_MAX_CANDIDATES,
        5 * context_module._SQLITE_RECALL_CANDIDATE_MULTIPLIER,
    )
    assert len(tokenized_texts) <= 1 + expected_candidates


@pytest.mark.asyncio
async def test_sqlite_memory_store_enforces_deadline_while_database_is_locked(
    tmp_path,
) -> None:
    path = tmp_path / "memory.sqlite3"
    store = SQLiteMemoryStore(path)
    await store.write_reflection([MemoryItem(text="locked memory")])
    blocker = sqlite3.connect(path, isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="deadline"):
            await store.recall("locked", limit=1, deadline_ms=30)
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()

    assert time.monotonic() - started < 0.25


@pytest.mark.asyncio
async def test_sqlite_recall_cancellation_waits_for_real_worker_shutdown(
    tmp_path,
    monkeypatch,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    worker_started = threading.Event()
    worker_finished = threading.Event()

    def slow_recall(query: str, limit: int, deadline_ms: int):
        worker_started.set()
        time.sleep(0.05)
        worker_finished.set()
        return []

    monkeypatch.setattr(store, "_recall_sync", slow_recall)
    recall_task = asyncio.create_task(
        store.recall("query", limit=1, deadline_ms=1_000)
    )
    await asyncio.wait_for(asyncio.to_thread(worker_started.wait), timeout=0.2)

    recall_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await recall_task

    assert worker_finished.is_set()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are not portable")
def test_sqlite_memory_store_creates_new_database_with_private_permissions(
    tmp_path,
) -> None:
    path = tmp_path / "memory.sqlite3"
    previous_umask = os.umask(0)
    try:
        SQLiteMemoryStore(path)
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are not portable")
def test_sqlite_memory_store_preserves_existing_database_permissions(tmp_path) -> None:
    path = tmp_path / "memory.sqlite3"
    path.touch(mode=0o644)
    path.chmod(0o644)

    SQLiteMemoryStore(path)

    assert stat.S_IMODE(path.stat().st_mode) == 0o644


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

    scheduler = orchestrator.build_context_scheduler(profile="local_cascade")
    await scheduler._memory.write_reflection(
        [MemoryItem(text="사용자는 차분하고 짧은 답변을 선호한다.", importance=9)]
    )

    bundle = await scheduler.gather_for_turn("차분한 답변으로 말해줘")

    assert memory_path.exists()
    assert bundle.late is False
    assert [injection.source for injection in bundle.injections] == ["memory"]
    assert "차분" in bundle.injections[0].text


@pytest.mark.parametrize(
    "profile",
    ["realtime_audio", "realtime_text_external_tts"],
)
@pytest.mark.asyncio
async def test_realtime_profiles_do_not_initialize_unused_sqlite_memory(
    profile,
    tmp_path,
    monkeypatch,
) -> None:
    from zemory import orchestrator
    from zemory.config import settings

    memory_path = tmp_path / "must-not-exist.sqlite3"
    sqlite_initializations: list[object] = []
    monkeypatch.setattr(settings, "memory_enabled", True, raising=False)
    monkeypatch.setattr(settings, "memory_path", str(memory_path), raising=False)
    monkeypatch.setattr(
        orchestrator,
        "SQLiteMemoryStore",
        lambda path: sqlite_initializations.append(path) or NullMemoryStore(),
    )

    scheduler = orchestrator.build_context_scheduler(profile=profile)
    bundle = await scheduler.gather_for_turn("unused realtime memory")

    assert isinstance(scheduler._memory, NullMemoryStore)
    assert bundle.injections == []
    assert sqlite_initializations == []
    assert memory_path.exists() is False


def test_context_scheduler_builder_requires_explicit_profile() -> None:
    from zemory import orchestrator

    with pytest.raises(TypeError, match="profile"):
        orchestrator.build_context_scheduler()
