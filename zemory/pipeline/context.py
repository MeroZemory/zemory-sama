"""Async transcript, memory, and context scheduling primitives.

The scheduler keeps retrieval off the first-audio critical path: memory recall
gets a small deadline, returns injections only when ready in time, and stores
late results for a later turn instead of blocking response start.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from zemory.observability import get_logger, metrics
from zemory.providers.base import Injection

_log = get_logger(__name__)


@dataclass(frozen=True)
class TranscriptTurn:
    role: str
    text: str
    timestamp: float = field(default_factory=time.time)
    interrupted: bool = False


@dataclass(frozen=True)
class MemoryHit:
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryItem:
    text: str
    importance: int = 5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextBundle:
    injections: list[Injection]
    late: bool = False


@dataclass(frozen=True)
class ContextTool:
    name: str
    run: Callable[[str], Awaitable[str]]
    priority: int = 80


class MemoryStore(Protocol):
    async def recall(self, query: str, *, limit: int, deadline_ms: int) -> list[MemoryHit]:
        ...

    async def write_reflection(self, items: list[MemoryItem]) -> None:
        ...


class NullMemoryStore:
    async def recall(self, query: str, *, limit: int, deadline_ms: int) -> list[MemoryHit]:
        return []

    async def write_reflection(self, items: list[MemoryItem]) -> None:
        return None


class SQLiteMemoryStore:
    """Small local memory store with deterministic lexical recall.

    This is the default no-service implementation. A vector store can replace
    it behind the same protocol later, but SQLite gives us durable memory and
    fully offline tests now.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._init_db()

    async def recall(self, query: str, *, limit: int, deadline_ms: int) -> list[MemoryHit]:
        if not query.strip() or limit <= 0:
            return []
        return await asyncio.to_thread(self._recall_sync, query, limit)

    async def write_reflection(self, items: list[MemoryItem]) -> None:
        if not items:
            return
        await asyncio.to_thread(self._write_sync, items)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    importance INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_created_at "
                "ON memories(created_at)"
            )

    def _write_sync(self, items: list[MemoryItem]) -> None:
        rows = [
            (
                item.text,
                max(1, min(10, int(item.importance))),
                json.dumps(item.metadata, ensure_ascii=False, sort_keys=True),
                time.time(),
            )
            for item in items
            if item.text.strip()
        ]
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO memories (text, importance, metadata_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )

    def _recall_sync(self, query: str, limit: int) -> list[MemoryHit]:
        query_terms = _tokenize(query)
        if not query_terms:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT text, importance, metadata_json FROM memories"
            ).fetchall()

        scored: list[MemoryHit] = []
        fallback: list[MemoryHit] = []
        for row in rows:
            text = str(row["text"])
            overlap = len(query_terms & _tokenize(text))
            importance = int(row["importance"])
            metadata = json.loads(row["metadata_json"])
            metadata["importance"] = importance
            hit = MemoryHit(
                text=text,
                score=overlap + (importance / 100),
                metadata=metadata,
            )
            if overlap == 0:
                fallback.append(hit)
            else:
                scored.append(hit)

        scored.sort(key=lambda hit: hit.score, reverse=True)
        fallback.sort(key=lambda hit: hit.score, reverse=True)
        if len(scored) < limit:
            scored.extend(fallback[: limit - len(scored)])
        return scored[:limit]


def _tokenize(text: str) -> set[str]:
    tokens = {
        token
        for token in re.findall(r"[\w가-힣]+", text.casefold())
        if token
    }
    expanded = set(tokens)
    for token in tokens:
        if re.search(r"[가-힣]", token):
            for size in range(2, min(5, len(token) + 1)):
                for idx in range(0, len(token) - size + 1):
                    expanded.add(token[idx : idx + size])
    return expanded


class TranscriptLedger:
    def __init__(self, max_turns: int) -> None:
        self._max_turns = max_turns
        self._turns: list[TranscriptTurn] = []

    def record_user(self, text: str) -> None:
        self._append(TranscriptTurn(role="user", text=text))

    def record_assistant(self, text: str, *, interrupted: bool = False) -> None:
        self._append(
            TranscriptTurn(role="assistant", text=text, interrupted=interrupted)
        )

    def record_system(self, text: str) -> None:
        self._append(TranscriptTurn(role="system", text=text))

    def window(self) -> list[TranscriptTurn]:
        return list(self._turns)

    def as_context_text(self) -> str:
        lines: list[str] = []
        for turn in self._turns:
            suffix = " interrupted=true" if turn.interrupted else ""
            lines.append(f"{turn.role}{suffix}: {turn.text}")
        return "\n".join(lines)

    def _append(self, turn: TranscriptTurn) -> None:
        self._turns.append(turn)
        overflow = len(self._turns) - self._max_turns
        if overflow > 0:
            del self._turns[:overflow]


class AsyncContextScheduler:
    def __init__(
        self,
        *,
        memory: MemoryStore | None = None,
        tools: list[ContextTool] | None = None,
        memory_deadline_ms: int = 80,
        tool_deadline_ms: int = 200,
        memory_limit: int = 5,
    ) -> None:
        self._memory = memory or NullMemoryStore()
        self._tools = tools or []
        self._memory_deadline_ms = memory_deadline_ms
        self._tool_deadline_ms = tool_deadline_ms
        self._memory_limit = memory_limit
        self.late_results: list[list[MemoryHit]] = []
        self.late_tool_results: list[tuple[str, str]] = []

    async def gather_for_turn(self, user_text: str) -> ContextBundle:
        """Return context injections that are ready before the deadline."""
        memory_bundle = await self._gather_memory(user_text)
        tool_bundle = await self._gather_tools(user_text)
        return ContextBundle(
            injections=memory_bundle.injections + tool_bundle.injections,
            late=memory_bundle.late or tool_bundle.late,
        )

    async def _gather_memory(self, user_text: str) -> ContextBundle:
        started = time.monotonic()
        recall_task = asyncio.create_task(
            self._memory.recall(
                user_text,
                limit=self._memory_limit,
                deadline_ms=self._memory_deadline_ms,
            )
        )

        done, pending = await asyncio.wait(
            {recall_task},
            timeout=self._memory_deadline_ms / 1000,
        )
        elapsed_ms = (time.monotonic() - started) * 1000
        metrics.observe("context.memory_wait_ms", elapsed_ms)

        if pending:
            recall_task.add_done_callback(self._capture_late_result)
            _log.info(
                "context.memory_late",
                deadline_ms=self._memory_deadline_ms,
                elapsed_ms=round(elapsed_ms, 1),
            )
            return ContextBundle(injections=[], late=True)

        hits = recall_task.result()
        injections = [
            Injection(
                source="memory",
                priority=60,
                text=hit.text,
                metadata={"score": hit.score, **hit.metadata},
            )
            for hit in hits
        ]
        return ContextBundle(injections=injections, late=False)

    async def _gather_tools(self, user_text: str) -> ContextBundle:
        if not self._tools:
            return ContextBundle(injections=[], late=False)

        started = time.monotonic()
        tasks = {
            asyncio.create_task(tool.run(user_text)): tool
            for tool in self._tools
        }
        done, pending = await asyncio.wait(
            set(tasks),
            timeout=self._tool_deadline_ms / 1000,
        )
        elapsed_ms = (time.monotonic() - started) * 1000
        metrics.observe("context.tool_wait_ms", elapsed_ms)

        late = False
        injections: list[Injection] = []
        for task in done:
            tool = tasks[task]
            result = task.result()
            if result:
                injections.append(
                    Injection(
                        source=f"tool:{tool.name}",
                        priority=tool.priority,
                        text=result,
                    )
                )

        for task in pending:
            late = True
            tool = tasks[task]
            task.add_done_callback(
                lambda done_task, tool_name=tool.name: self._capture_late_tool_result(
                    tool_name,
                    done_task,
                )
            )

        if late:
            _log.info(
                "context.tool_late",
                deadline_ms=self._tool_deadline_ms,
                elapsed_ms=round(elapsed_ms, 1),
            )
        return ContextBundle(injections=injections, late=late)

    def _capture_late_result(self, task: asyncio.Task[list[MemoryHit]]) -> None:
        try:
            hits = task.result()
        except Exception as e:  # pragma: no cover - defensive background path
            _log.warning("context.memory_late_failed", error=str(e))
            return
        if hits:
            self.late_results.append(hits)

    def _capture_late_tool_result(self, tool_name: str, task: asyncio.Task[str]) -> None:
        try:
            result = task.result()
        except Exception as e:  # pragma: no cover - defensive background path
            _log.warning("context.tool_late_failed", tool=tool_name, error=str(e))
            return
        if result:
            self.late_tool_results.append((tool_name, result))
