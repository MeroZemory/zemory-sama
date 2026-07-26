"""Async tool/RAG scheduling tests."""

from __future__ import annotations

import asyncio

import pytest

from zemory.pipeline.context import AsyncContextScheduler, ContextTool, MemoryHit


@pytest.mark.asyncio
async def test_scheduler_returns_tool_injections_before_deadline() -> None:
    async def lookup(query: str) -> str:
        return f"tool result for {query}"

    scheduler = AsyncContextScheduler(
        tools=[ContextTool(name="lookup", run=lookup, priority=80)],
        tool_deadline_ms=50,
    )

    bundle = await scheduler.gather_for_turn("날씨")

    tool_injections = [inj for inj in bundle.injections if inj.source == "tool:lookup"]
    assert len(tool_injections) == 1
    assert tool_injections[0].priority == 80
    assert tool_injections[0].text == "tool result for 날씨"
    assert bundle.late is False


@pytest.mark.asyncio
async def test_scheduler_does_not_block_on_late_tool_result() -> None:
    async def slow_lookup(query: str) -> str:
        await asyncio.sleep(0.08)
        return f"late result for {query}"

    scheduler = AsyncContextScheduler(
        tools=[ContextTool(name="slow_lookup", run=slow_lookup, priority=80)],
        tool_deadline_ms=10,
    )

    started = asyncio.get_running_loop().time()
    bundle = await scheduler.gather_for_turn("검색")
    elapsed_ms = (asyncio.get_running_loop().time() - started) * 1000

    assert elapsed_ms < 50
    assert [inj for inj in bundle.injections if inj.source.startswith("tool:")] == []
    assert bundle.late is True

    await asyncio.sleep(0.1)
    assert scheduler.late_tool_results == [("slow_lookup", "late result for 검색")]


@pytest.mark.asyncio
async def test_late_tool_is_injected_once_without_consuming_same_turn_result() -> None:
    release_first = asyncio.Event()

    class SlowSecondMemory:
        async def recall(
            self,
            query: str,
            *,
            limit: int,
            deadline_ms: int,
        ) -> list[MemoryHit]:
            if query == "second":
                await asyncio.sleep(0.03)
            return []

        async def write_reflection(self, items: list) -> None:
            return None

    async def lookup(query: str) -> str:
        if query == "first":
            await release_first.wait()
            return "late first"
        if query == "second":
            await asyncio.sleep(0.01)
            return "late second"
        return ""

    scheduler = AsyncContextScheduler(
        memory=SlowSecondMemory(),
        tools=[ContextTool(name="lookup", run=lookup, priority=93)],
        memory_deadline_ms=50,
        tool_deadline_ms=1,
    )

    first = await scheduler.gather_for_turn("first")
    assert first.injections == []
    release_first.set()
    async with asyncio.timeout(0.2):
        while not scheduler.late_tool_results:
            await asyncio.sleep(0)

    second = await scheduler.gather_for_turn("second")
    assert [(item.source, item.priority, item.text) for item in second.injections] == [
        ("tool:lookup", 93, "late first")
    ]
    assert scheduler.late_tool_results == [("lookup", "late second")]

    third = await scheduler.gather_for_turn("third")
    assert [(item.source, item.priority, item.text) for item in third.injections] == [
        ("tool:lookup", 93, "late second")
    ]
    assert scheduler.late_tool_results == []

    fourth = await scheduler.gather_for_turn("fourth")
    assert fourth.injections == []


@pytest.mark.asyncio
async def test_tool_failures_are_isolated_and_results_use_configured_order() -> None:
    async def slow_first(query: str) -> str:
        await asyncio.sleep(0.02)
        return "first"

    async def failing(query: str) -> str:
        raise RuntimeError("provider failure")

    async def fast_last(query: str) -> str:
        return "last"

    scheduler = AsyncContextScheduler(
        tools=[
            ContextTool(name="first", run=slow_first),
            ContextTool(name="failing", run=failing),
            ContextTool(name="last", run=fast_last),
        ],
        tool_deadline_ms=100,
    )

    bundle = await scheduler.gather_for_turn("query")

    assert [(injection.source, injection.text) for injection in bundle.injections] == [
        ("tool:first", "first"),
        ("tool:last", "last"),
    ]


@pytest.mark.asyncio
async def test_late_tool_results_use_configured_order_not_completion_order() -> None:
    async def first(query: str) -> str:
        await asyncio.sleep(0.03)
        return "first result"

    async def second(query: str) -> str:
        await asyncio.sleep(0.01)
        return "second result"

    scheduler = AsyncContextScheduler(
        tools=[
            ContextTool(name="first", run=first),
            ContextTool(name="second", run=second),
        ],
        tool_deadline_ms=0,
    )

    bundle = await scheduler.gather_for_turn("query")
    assert bundle.late is True
    await asyncio.sleep(0.05)

    assert scheduler.late_tool_results == [
        ("first", "first result"),
        ("second", "second result"),
    ]


@pytest.mark.asyncio
async def test_scheduler_bounds_tool_count_and_result_size() -> None:
    calls: list[int] = []

    def make_tool(index: int) -> ContextTool:
        async def run(query: str) -> str:
            calls.append(index)
            return f"result-{index}-oversized"

        return ContextTool(name=f"tool-{index}", run=run)

    scheduler = AsyncContextScheduler(
        tools=[make_tool(index) for index in range(5)],
        max_tools_per_turn=2,
        max_injection_chars=8,
    )

    bundle = await scheduler.gather_for_turn("query")

    assert calls == [0, 1]
    assert [injection.text for injection in bundle.injections] == [
        "result-0",
        "result-1",
    ]


@pytest.mark.asyncio
async def test_scheduler_bounds_outstanding_provider_tasks() -> None:
    started: list[int] = []

    def make_hanging_tool(index: int) -> ContextTool:
        async def run(query: str) -> str:
            started.append(index)
            await asyncio.Event().wait()
            return "unreachable"

        return ContextTool(name=f"tool-{index}", run=run)

    scheduler = AsyncContextScheduler(
        tools=[make_hanging_tool(index) for index in range(3)],
        memory_limit=0,
        tool_deadline_ms=1,
        max_pending_tasks=1,
    )

    bundle = await scheduler.gather_for_turn("query")

    assert bundle.late is True
    assert started == [0]
    assert scheduler.pending_task_count == 1
    await scheduler.aclose()
    assert scheduler.pending_task_count == 0
