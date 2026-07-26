"""Async memory/context scheduler tests."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
import threading

import pytest
from structlog.testing import capture_logs

from zemory.pipeline.context import (
    AsyncContextScheduler,
    ContextTool,
    MemoryHit,
    MemoryItem,
    SQLiteMemoryStore,
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


@pytest.mark.asyncio
async def test_sqlite_recall_cancellation_drains_daemon_worker(
    tmp_path,
    monkeypatch,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    started = threading.Event()
    release = threading.Event()

    def controlled_recall(
        query: str,
        limit: int,
        deadline_ms: int,
    ) -> list[MemoryHit]:
        started.set()
        release.wait()
        return [MemoryHit(text=query, score=float(limit + deadline_ms))]

    monkeypatch.setattr(store, "_recall_sync", controlled_recall)
    recall_task = asyncio.create_task(
        store.recall("cancelled", limit=1, deadline_ms=10_000)
    )
    while not started.is_set():
        await asyncio.sleep(0)

    recall_task.cancel()
    await asyncio.sleep(0)
    assert recall_task.done() is False

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(recall_task, timeout=0.2)


@pytest.mark.asyncio
async def test_sqlite_write_cancellation_drains_daemon_worker(
    tmp_path,
    monkeypatch,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    started = threading.Event()
    release = threading.Event()

    def controlled_write(items: list[MemoryItem]) -> None:
        assert items
        started.set()
        release.wait()

    monkeypatch.setattr(store, "_write_sync", controlled_write)
    write_task = asyncio.create_task(
        store.write_reflection([MemoryItem(text="cancelled")])
    )
    while not started.is_set():
        await asyncio.sleep(0)

    write_task.cancel()
    await asyncio.sleep(0)
    assert write_task.done() is False

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(write_task, timeout=0.2)


@pytest.mark.asyncio
async def test_sqlite_concurrent_writes_start_at_most_one_worker(
    tmp_path,
    monkeypatch,
) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    started = threading.Event()
    release = threading.Event()
    counter_lock = threading.Lock()
    calls = 0
    active = 0
    max_active = 0

    def controlled_write(items: list[MemoryItem]) -> None:
        nonlocal calls, active, max_active
        assert items
        with counter_lock:
            calls += 1
            active += 1
            max_active = max(max_active, active)
        started.set()
        release.wait()
        with counter_lock:
            active -= 1

    monkeypatch.setattr(store, "_write_sync", controlled_write)
    tasks = [
        asyncio.create_task(
            store.write_reflection([MemoryItem(text=f"reflection-{index}")])
        )
        for index in range(3)
    ]
    while not started.is_set():
        await asyncio.sleep(0)
    await asyncio.sleep(0.01)

    with counter_lock:
        assert calls == 1
        assert max_active == 1

    release.set()
    await asyncio.gather(*tasks)
    with counter_lock:
        assert calls == 3
        assert max_active == 1


def test_hung_sqlite_workers_cannot_block_interpreter_exit() -> None:
    script = textwrap.dedent(
        """
        import asyncio
        import tempfile
        import threading
        from pathlib import Path

        from zemory.__main__ import _run_with_bounded_shutdown
        from zemory.pipeline.context import MemoryItem, SQLiteMemoryStore

        async def exercise(path: Path) -> None:
            store = SQLiteMemoryStore(path)
            recall_started = threading.Event()
            write_started = threading.Event()

            def hang_recall(query: str, limit: int, deadline_ms: int):
                recall_started.set()
                threading.Event().wait()

            def hang_write(items: list[MemoryItem]) -> None:
                write_started.set()
                threading.Event().wait()

            store._recall_sync = hang_recall
            store._write_sync = hang_write
            asyncio.create_task(
                store.recall("private-query", limit=1, deadline_ms=10_000)
            )
            asyncio.create_task(
                store.write_reflection([MemoryItem(text="private-note")])
            )
            while not (recall_started.is_set() and write_started.is_set()):
                await asyncio.sleep(0)
            raise RuntimeError("stop")

        with tempfile.TemporaryDirectory() as directory:
            try:
                _run_with_bounded_shutdown(
                    exercise(Path(directory) / "memory.sqlite3"),
                    shutdown_timeout_s=0.05,
                )
            except RuntimeError:
                pass
        print("PROCESS_EXIT_COMPLETED", flush=True)
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        pytest.fail(
            "hung SQLite worker blocked interpreter exit; "
            f"stdout={stdout!r}, stderr={stderr!r}"
        )

    assert process.returncode == 0, stderr
    assert "PROCESS_EXIT_COMPLETED" in stdout


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


@pytest.mark.asyncio
async def test_late_memory_is_injected_once_on_the_next_gather() -> None:
    release_first = asyncio.Event()

    class ControlledMemoryStore(FakeMemoryStore):
        async def recall(
            self,
            query: str,
            *,
            limit: int,
            deadline_ms: int,
        ) -> list[MemoryHit]:
            if query == "first":
                await release_first.wait()
                return [
                    MemoryHit(
                        text="late memory",
                        score=0.75,
                        metadata={"kind": "preference"},
                    )
                ]
            return []

    scheduler = AsyncContextScheduler(
        memory=ControlledMemoryStore(),
        memory_deadline_ms=1,
    )

    first = await scheduler.gather_for_turn("first")
    assert first.injections == []
    assert first.late is True

    release_first.set()
    async with asyncio.timeout(0.2):
        while not scheduler.late_results:
            await asyncio.sleep(0)

    second = await scheduler.gather_for_turn("second")
    assert [injection.text for injection in second.injections] == ["late memory"]
    assert second.injections[0].source == "memory"
    assert second.injections[0].priority == 60
    assert second.injections[0].metadata == {
        "score": 0.75,
        "kind": "preference",
    }
    assert scheduler.late_results == []

    third = await scheduler.gather_for_turn("third")
    assert third.injections == []


@pytest.mark.asyncio
async def test_scheduler_runs_memory_and_tools_concurrently() -> None:
    delay_s = 0.06

    class SlowMemoryStore(FakeMemoryStore):
        async def recall(
            self,
            query: str,
            *,
            limit: int,
            deadline_ms: int,
        ) -> list[MemoryHit]:
            await asyncio.sleep(delay_s)
            return await super().recall(query, limit=limit, deadline_ms=deadline_ms)

    async def slow_tool(query: str) -> str:
        await asyncio.sleep(delay_s)
        return f"tool result for {query}"

    scheduler = AsyncContextScheduler(
        memory=SlowMemoryStore(),
        tools=[ContextTool(name="lookup", run=slow_tool)],
        memory_deadline_ms=200,
        tool_deadline_ms=200,
    )

    started = asyncio.get_running_loop().time()
    bundle = await scheduler.gather_for_turn("병렬 조회")
    elapsed_ms = (asyncio.get_running_loop().time() - started) * 1000

    assert elapsed_ms < 95
    assert [injection.source for injection in bundle.injections] == [
        "memory",
        "tool:lookup",
    ]


@pytest.mark.asyncio
async def test_memory_failure_isolated_without_logging_provider_message() -> None:
    secret = "private-query-010-1234-5678"

    class FailingMemoryStore(FakeMemoryStore):
        async def recall(
            self,
            query: str,
            *,
            limit: int,
            deadline_ms: int,
        ) -> list[MemoryHit]:
            raise RuntimeError(f"provider echoed {secret}")

    async def healthy_tool(query: str) -> str:
        return "healthy result"

    scheduler = AsyncContextScheduler(
        memory=FailingMemoryStore(),
        tools=[ContextTool(name="healthy", run=healthy_tool)],
    )

    with capture_logs() as logs:
        bundle = await scheduler.gather_for_turn(secret)

    assert [injection.source for injection in bundle.injections] == ["tool:healthy"]
    assert any(entry["event"] == "context.memory_failed" for entry in logs)
    assert secret not in repr(logs)


@pytest.mark.asyncio
async def test_scheduler_bounds_provider_results_and_late_result_history() -> None:
    class OversizedMemoryStore(FakeMemoryStore):
        async def recall(
            self,
            query: str,
            *,
            limit: int,
            deadline_ms: int,
        ) -> list[MemoryHit]:
            await asyncio.sleep(0.01)
            return [MemoryHit(text=f"{query}-{index}-oversized", score=1.0) for index in range(10)]

    scheduler = AsyncContextScheduler(
        memory=OversizedMemoryStore(),
        memory_deadline_ms=0,
        memory_limit=2,
        max_injection_chars=8,
        max_late_results=2,
    )

    for query in ("first", "second", "third"):
        bundle = await scheduler.gather_for_turn(query)
        assert bundle.late is True

    await asyncio.sleep(0.05)

    assert len(scheduler.late_results) == 2
    assert all(len(batch) <= 2 for batch in scheduler.late_results)
    assert all(len(hit.text) <= 8 for batch in scheduler.late_results for hit in batch)
    assert [batch[0].text for batch in scheduler.late_results] == [
        "second-0",
        "third-0-",
    ]


@pytest.mark.asyncio
async def test_scheduler_applies_aggregate_budget_after_merging_late_results() -> None:
    scheduler = AsyncContextScheduler(
        memory_limit=0,
        late_result_ttl_turns=10,
        max_injections_per_turn=3,
        max_total_injection_chars=10,
    )
    scheduler._turn_sequence = 10
    scheduler._late_memory_entries = [
        (turn_id, [MemoryHit(text="abcd", score=1.0)]) for turn_id in range(10)
    ]
    scheduler.late_results = [hits for _, hits in scheduler._late_memory_entries]

    bundle = await scheduler.gather_for_turn("next")

    assert len(bundle.injections) == 3
    # Newest results win selection, then the surviving entries are restored
    # to their original old-to-new prompt order.
    assert [injection.text for injection in bundle.injections] == [
        "ab",
        "abcd",
        "abcd",
    ]
    assert sum(len(injection.text) for injection in bundle.injections) == 10
    # Eligible late entries are consumed once even when the aggregate budget
    # drops their overflow; they cannot flood a later turn instead.
    assert scheduler.late_results == []


@pytest.mark.asyncio
async def test_fresh_high_priority_tool_survives_stale_aggregate_flood() -> None:
    async def current_tool(_query: str) -> str:
        return "FRESH_TOOL"

    scheduler = AsyncContextScheduler(
        memory_limit=0,
        tools=[ContextTool(name="current", run=current_tool, priority=200)],
        late_result_ttl_turns=10,
        max_injections_per_turn=32,
        max_total_injection_chars=16_000,
    )
    scheduler._turn_sequence = 10
    scheduler._late_memory_entries = [
        (turn_id, [MemoryHit(text=str(turn_id) * 4_000, score=1.0)]) for turn_id in range(6, 10)
    ]
    scheduler.late_results = [hits for _, hits in scheduler._late_memory_entries]

    bundle = await scheduler.gather_for_turn("current turn")

    assert bundle.injections[-1].source == "tool:current"
    assert bundle.injections[-1].text == "FRESH_TOOL"
    assert sum(len(injection.text) for injection in bundle.injections) == 16_000


@pytest.mark.asyncio
async def test_current_turn_freshness_precedes_older_higher_priority() -> None:
    async def current_tool(_query: str) -> str:
        return "fresh"

    scheduler = AsyncContextScheduler(
        memory_limit=0,
        tools=[ContextTool(name="current", run=current_tool, priority=10)],
        max_injections_per_turn=1,
    )
    scheduler._turn_sequence = 1
    scheduler._late_tool_entries = [(0, 0, "stale", 200, "stale")]
    scheduler.late_tool_results = [("stale", "stale")]

    bundle = await scheduler.gather_for_turn("current turn")

    assert [injection.source for injection in bundle.injections] == ["tool:current"]


@pytest.mark.asyncio
async def test_same_turn_priority_selection_preserves_original_output_order() -> None:
    async def low_priority_tool(_query: str) -> str:
        return "low"

    async def high_priority_tool(_query: str) -> str:
        return "high"

    scheduler = AsyncContextScheduler(
        memory=FakeMemoryStore(),
        tools=[
            ContextTool(name="low", run=low_priority_tool, priority=80),
            ContextTool(name="high", run=high_priority_tool, priority=200),
        ],
        max_injections_per_turn=2,
    )

    bundle = await scheduler.gather_for_turn("priority")

    assert [injection.source for injection in bundle.injections] == [
        "tool:low",
        "tool:high",
    ]


@pytest.mark.asyncio
async def test_late_ttl_drops_expired_results_and_keeps_newest_order() -> None:
    scheduler = AsyncContextScheduler(
        memory_limit=0,
        late_result_ttl_turns=2,
        max_injections_per_turn=2,
    )
    scheduler._turn_sequence = 5
    scheduler._late_memory_entries = [
        (2, [MemoryHit(text="expired", score=1.0)]),
        (3, [MemoryHit(text="newer", score=1.0)]),
        (4, [MemoryHit(text="newest", score=1.0)]),
    ]
    scheduler.late_results = [hits for _, hits in scheduler._late_memory_entries]

    bundle = await scheduler.gather_for_turn("next")

    assert [injection.text for injection in bundle.injections] == [
        "newer",
        "newest",
    ]
    assert scheduler.late_results == []


@pytest.mark.asyncio
async def test_scheduler_cancels_late_task_after_bounded_grace() -> None:
    cancelled = asyncio.Event()

    class HangingMemoryStore(FakeMemoryStore):
        async def recall(
            self,
            query: str,
            *,
            limit: int,
            deadline_ms: int,
        ) -> list[MemoryHit]:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return []

    scheduler = AsyncContextScheduler(
        memory=HangingMemoryStore(),
        memory_deadline_ms=1,
        late_result_grace_ms=10,
    )

    bundle = await scheduler.gather_for_turn("never returns")
    assert bundle.late is True

    await asyncio.wait_for(cancelled.wait(), timeout=0.2)
    await asyncio.sleep(0)
    assert scheduler.pending_task_count == 0


@pytest.mark.asyncio
async def test_scheduler_aclose_reclaims_pending_provider_tasks() -> None:
    cancelled = asyncio.Event()

    class HangingMemoryStore(FakeMemoryStore):
        async def recall(
            self,
            query: str,
            *,
            limit: int,
            deadline_ms: int,
        ) -> list[MemoryHit]:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return []

    scheduler = AsyncContextScheduler(
        memory=HangingMemoryStore(),
        memory_deadline_ms=1,
    )
    await scheduler.gather_for_turn("shutdown")

    await scheduler.aclose()

    assert cancelled.is_set()
    assert scheduler.pending_task_count == 0


@pytest.mark.asyncio
async def test_cancelled_gather_reclaims_provider_tasks() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class HangingMemoryStore(FakeMemoryStore):
        async def recall(
            self,
            query: str,
            *,
            limit: int,
            deadline_ms: int,
        ) -> list[MemoryHit]:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return []

    scheduler = AsyncContextScheduler(
        memory=HangingMemoryStore(),
        memory_deadline_ms=10_000,
    )
    gather_task = asyncio.create_task(scheduler.gather_for_turn("cancelled turn"))
    await started.wait()

    gather_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await gather_task

    assert cancelled.is_set()
    assert scheduler.pending_task_count == 0
