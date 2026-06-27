"""Async tool/RAG scheduling tests."""

from __future__ import annotations

import asyncio

import pytest

from zemory.pipeline.context import AsyncContextScheduler, ContextTool


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
