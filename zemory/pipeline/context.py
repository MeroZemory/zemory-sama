"""Async transcript, memory, and context scheduling primitives.

The scheduler deadline-bounds retrieval before response creation, returns
injections only when ready in time, and stores late results for a later turn.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import threading
import time
from collections.abc import Awaitable, Callable
from contextlib import closing
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, TypeVar

from zemory.observability import get_logger, metrics
from zemory.providers.base import Injection

_log = get_logger(__name__)

_SQLITE_RECALL_CANDIDATE_MULTIPLIER = 16
_SQLITE_RECALL_MAX_CANDIDATES = 512
_SQLITE_RECALL_QUERY_MAX_CHARS = 4_000
_SQLITE_RECALL_TEXT_MAX_CHARS = 4_000
_SQLITE_RECALL_METADATA_MAX_CHARS = 16_000
_SQLITE_PROGRESS_CHECK_OPS = 100

_WorkerResult = TypeVar("_WorkerResult")


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
    async def recall(self, query: str, *, limit: int, deadline_ms: int) -> list[MemoryHit]: ...

    async def write_reflection(self, items: list[MemoryItem]) -> None: ...


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
        # Serialize submissions so a busy/slow database cannot start an
        # unbounded number of recall workers across concurrent turns.
        self._recall_lock = asyncio.Lock()
        # Preserve the default executor's bounded-write behavior when using
        # dedicated daemon workers: one SQLite write may run per store.
        self._write_lock = asyncio.Lock()
        self._init_db()

    async def recall(self, query: str, *, limit: int, deadline_ms: int) -> list[MemoryHit]:
        if not query.strip() or limit <= 0:
            return []
        deadline_s = max(0, deadline_ms) / 1000
        deadline_at = time.monotonic() + deadline_s
        try:
            async with asyncio.timeout(deadline_s):
                async with self._recall_lock:
                    remaining_ms = max(
                        1,
                        int((deadline_at - time.monotonic()) * 1000),
                    )
                    worker = self._start_worker(
                        lambda: self._recall_sync(query, limit, remaining_ms),
                        name="zemory-sqlite-recall",
                    )
                    return await self._await_worker(worker)
        except TimeoutError:
            raise TimeoutError("SQLite memory recall exceeded its deadline") from None

    async def write_reflection(self, items: list[MemoryItem]) -> None:
        if not items:
            return
        async with self._write_lock:
            worker = self._start_worker(
                lambda: self._write_sync(items),
                name="zemory-sqlite-write",
            )
            await self._await_worker(worker)

    @staticmethod
    def _start_worker(
        operation: Callable[[], _WorkerResult],
        *,
        name: str,
    ) -> asyncio.Future[_WorkerResult]:
        """Run SQLite work off-loop without creating an exit-blocking worker."""
        loop = asyncio.get_running_loop()
        completion: asyncio.Future[_WorkerResult] = loop.create_future()

        def finish_success(result: _WorkerResult) -> None:
            if not completion.done():
                completion.set_result(result)

        def finish_failure(error: BaseException) -> None:
            if not completion.done():
                completion.set_exception(error)

        def run_operation() -> None:
            try:
                result = operation()
            except BaseException as error:
                try:
                    loop.call_soon_threadsafe(finish_failure, error)
                except RuntimeError:
                    pass
            else:
                try:
                    loop.call_soon_threadsafe(finish_success, result)
                except RuntimeError:
                    pass

        threading.Thread(
            target=run_operation,
            name=name,
            daemon=True,
        ).start()
        return completion

    @staticmethod
    async def _await_worker(
        worker: asyncio.Future[_WorkerResult],
    ) -> _WorkerResult:
        """Do not detach SQLite work when the awaiting task is cancelled.

        The drain keeps cancellation accounting honest while real SQLite work
        finishes. The dedicated daemon thread also lets the interpreter exit
        if an OS-level database operation never returns.
        """
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if worker.done() and not worker.cancelled():
                try:
                    worker.result()
                except Exception:
                    pass
            raise

    def _connect(self, *, timeout_s: float = 5.0) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=max(0.0, timeout_s))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._create_private_db_file_if_missing()
        with closing(self._connect()) as conn:
            with conn:
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
                # Candidate retrieval uses this index in importance/id order,
                # allowing SQLite to stop after the SQL LIMIT instead of
                # sorting or returning the whole table to Python.
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_importance_id "
                    "ON memories(importance DESC, id ASC)"
                )

    def _create_private_db_file_if_missing(self) -> None:
        """Create a new database no broader than 0600; preserve existing modes."""
        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
        try:
            descriptor = os.open(self._path, flags, 0o600)
        except FileExistsError:
            return
        try:
            os.close(descriptor)
        except OSError:  # pragma: no cover - close failure is platform-specific
            pass

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
        with closing(self._connect()) as conn:
            with conn:
                conn.executemany(
                    """
                    INSERT INTO memories (text, importance, metadata_json, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    rows,
                )

    def _recall_sync(
        self,
        query: str,
        limit: int,
        deadline_ms: int,
    ) -> list[MemoryHit]:
        deadline_at = time.monotonic() + max(0, deadline_ms) / 1000

        def expired() -> bool:
            return time.monotonic() >= deadline_at

        def require_time() -> None:
            if expired():
                raise TimeoutError("SQLite memory recall exceeded its deadline")

        result_limit = min(max(0, limit), _SQLITE_RECALL_MAX_CANDIDATES)
        candidate_limit = min(
            _SQLITE_RECALL_MAX_CANDIDATES,
            max(result_limit, result_limit * _SQLITE_RECALL_CANDIDATE_MULTIPLIER),
        )
        query_terms = _tokenize(query[:_SQLITE_RECALL_QUERY_MAX_CHARS])
        if not query_terms:
            return []
        require_time()

        timeout_s = max(0, deadline_ms) / 1000
        with closing(self._connect(timeout_s=timeout_s)) as conn:
            conn.set_progress_handler(
                lambda: int(expired()),
                _SQLITE_PROGRESS_CHECK_OPS,
            )
            try:
                rows = conn.execute(
                    """
                    SELECT
                        substr(text, 1, ?) AS text,
                        importance,
                        CASE
                            WHEN length(metadata_json) <= ? THEN metadata_json
                            ELSE '{}'
                        END AS metadata_json
                    FROM memories INDEXED BY idx_memories_importance_id
                    ORDER BY importance DESC, id ASC
                    LIMIT ?
                    """,
                    (
                        _SQLITE_RECALL_TEXT_MAX_CHARS,
                        _SQLITE_RECALL_METADATA_MAX_CHARS,
                        candidate_limit,
                    ),
                ).fetchall()
            except sqlite3.OperationalError as error:
                timeout_codes = {
                    sqlite3.SQLITE_BUSY,
                    sqlite3.SQLITE_INTERRUPT,
                    sqlite3.SQLITE_LOCKED,
                }
                if expired() or getattr(error, "sqlite_errorcode", None) in timeout_codes:
                    raise TimeoutError(
                        "SQLite memory recall exceeded its deadline"
                    ) from None
                raise
            finally:
                conn.set_progress_handler(None, 0)
        require_time()

        scored: list[MemoryHit] = []
        fallback: list[MemoryHit] = []
        for row in rows:
            require_time()
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
        if len(scored) < result_limit:
            scored.extend(fallback[: result_limit - len(scored)])
        require_time()
        return scored[:result_limit]


def _tokenize(text: str) -> set[str]:
    tokens = {token for token in re.findall(r"[\w가-힣]+", text.casefold()) if token}
    expanded = set(tokens)
    for token in tokens:
        if re.search(r"[가-힣]", token):
            for size in range(2, min(5, len(token) + 1)):
                for idx in range(0, len(token) - size + 1):
                    expanded.add(token[idx : idx + size])
    return expanded


class AsyncContextScheduler:
    def __init__(
        self,
        *,
        memory: MemoryStore | None = None,
        tools: list[ContextTool] | None = None,
        memory_deadline_ms: int = 80,
        tool_deadline_ms: int = 200,
        memory_limit: int = 5,
        max_injection_chars: int = 4_000,
        max_tools_per_turn: int = 16,
        max_late_results: int = 32,
        max_pending_tasks: int = 64,
        late_result_grace_ms: int = 5_000,
        late_result_ttl_turns: int = 1,
        max_injections_per_turn: int = 32,
        max_total_injection_chars: int = 16_000,
    ) -> None:
        self._memory = memory or NullMemoryStore()
        self._memory_deadline_ms = max(0, memory_deadline_ms)
        self._tool_deadline_ms = max(0, tool_deadline_ms)
        self._memory_limit = max(0, memory_limit)
        self._max_injection_chars = max(0, max_injection_chars)
        self._max_late_results = max(0, max_late_results)
        self._max_pending_tasks = max(0, max_pending_tasks)
        self._late_result_grace_ms = max(0, late_result_grace_ms)
        self._late_result_ttl_turns = max(0, late_result_ttl_turns)
        self._max_injections_per_turn = max(0, max_injections_per_turn)
        self._max_total_injection_chars = max(0, max_total_injection_chars)

        configured_tools = list(tools or [])
        tool_limit = max(0, max_tools_per_turn)
        self._tools = configured_tools[:tool_limit]
        if len(configured_tools) > len(self._tools):
            _log.warning(
                "context.tools_truncated",
                configured_count=len(configured_tools),
                scheduled_count=len(self._tools),
            )

        self.late_results: list[list[MemoryHit]] = []
        self.late_tool_results: list[tuple[str, str]] = []
        self._late_memory_entries: list[tuple[int, list[MemoryHit]]] = []
        self._late_tool_entries: list[tuple[int, int, str, int, str]] = []
        self._pending_tasks: set[asyncio.Task[Any]] = set()
        self._late_cancel_handles: dict[asyncio.Task[Any], asyncio.TimerHandle] = {}
        self._turn_sequence = 0
        self._closed = False

    @property
    def pending_task_count(self) -> int:
        return len(self._pending_tasks)

    async def gather_for_turn(self, user_text: str) -> ContextBundle:
        """Return context injections that are ready before the deadline."""
        if self._closed:
            raise RuntimeError("context scheduler is closed")

        turn_id = self._turn_sequence
        self._turn_sequence += 1
        memory_bundle, tool_bundle = await asyncio.gather(
            self._gather_memory(user_text, turn_id),
            self._gather_tools(user_text, turn_id),
        )
        previous_late_injections = self._consume_late_injections(before_turn_id=turn_id)
        current_injections = memory_bundle.injections + tool_bundle.injections
        gathered_injections = previous_late_injections + [
            (turn_id, injection) for injection in current_injections
        ]
        bounded_injections = self._bound_turn_injections(gathered_injections)
        if len(bounded_injections) < len(gathered_injections):
            _log.warning(
                "context.turn_injections_truncated",
                gathered_count=len(gathered_injections),
                injected_count=len(bounded_injections),
            )
        return ContextBundle(
            injections=bounded_injections,
            late=memory_bundle.late or tool_bundle.late,
        )

    async def aclose(self) -> None:
        """Cancel and retrieve provider tasks that outlived their deadline."""
        self._closed = True
        tasks = tuple(self._pending_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._pending_tasks.clear()
        for handle in self._late_cancel_handles.values():
            handle.cancel()
        self._late_cancel_handles.clear()

    async def _gather_memory(
        self,
        user_text: str,
        turn_id: int,
    ) -> ContextBundle:
        if self._memory_limit == 0:
            return ContextBundle(injections=[], late=False)

        started = time.monotonic()
        recall_task = self._start_task(
            lambda: self._memory.recall(
                user_text,
                limit=self._memory_limit,
                deadline_ms=self._memory_deadline_ms,
            )
        )
        if recall_task is None:
            _log.warning("context.provider_capacity_reached", provider="memory")
            return ContextBundle(injections=[], late=True)

        try:
            _, pending = await asyncio.wait(
                {recall_task},
                timeout=self._memory_deadline_ms / 1000,
            )
        except asyncio.CancelledError:
            await self._cancel_and_retrieve({recall_task})
            raise
        elapsed_ms = (time.monotonic() - started) * 1000
        metrics.observe("context.memory_wait_ms", elapsed_ms)

        if pending:
            recall_task.add_done_callback(lambda task: self._capture_late_result(turn_id, task))
            self._schedule_late_cancellation(recall_task)
            _log.info(
                "context.memory_late",
                deadline_ms=self._memory_deadline_ms,
                elapsed_ms=round(elapsed_ms, 1),
            )
            return ContextBundle(injections=[], late=True)

        if recall_task.cancelled():
            return ContextBundle(injections=[], late=False)
        try:
            hits = self._bounded_hits(recall_task.result())
        except Exception as error:
            _log.warning(
                "context.memory_failed",
                error_type=type(error).__name__,
            )
            return ContextBundle(injections=[], late=False)
        injections = [self._memory_injection(hit) for hit in hits if hit.text]
        return ContextBundle(injections=injections, late=False)

    async def _gather_tools(
        self,
        user_text: str,
        turn_id: int,
    ) -> ContextBundle:
        if not self._tools:
            return ContextBundle(injections=[], late=False)

        started = time.monotonic()
        tasks: dict[asyncio.Task[Any], tuple[int, ContextTool]] = {}
        capacity_reached = False
        for index, tool in enumerate(self._tools):
            task = self._start_task(lambda tool=tool: tool.run(user_text))
            if task is None:
                capacity_reached = True
                continue
            tasks[task] = (index, tool)

        if not tasks:
            if capacity_reached:
                _log.warning("context.provider_capacity_reached", provider="tool")
            return ContextBundle(injections=[], late=capacity_reached)

        try:
            done, pending = await asyncio.wait(
                set(tasks),
                timeout=self._tool_deadline_ms / 1000,
            )
        except asyncio.CancelledError:
            await self._cancel_and_retrieve(set(tasks))
            raise
        elapsed_ms = (time.monotonic() - started) * 1000
        metrics.observe("context.tool_wait_ms", elapsed_ms)

        late = capacity_reached
        injections: list[Injection] = []
        for task in sorted(done, key=lambda done_task: tasks[done_task][0]):
            _, tool = tasks[task]
            if task.cancelled():
                continue
            try:
                result = task.result()
            except Exception as error:
                _log.warning(
                    "context.tool_failed",
                    tool=tool.name,
                    error_type=type(error).__name__,
                )
                continue
            if not isinstance(result, str):
                _log.warning(
                    "context.tool_invalid_result",
                    tool=tool.name,
                    result_type=type(result).__name__,
                )
                continue
            result = self._bounded_text(result)
            if result:
                injections.append(
                    Injection(
                        source=f"tool:{tool.name}",
                        priority=tool.priority,
                        text=result,
                    )
                )

        for task in sorted(pending, key=lambda pending_task: tasks[pending_task][0]):
            late = True
            tool_index, tool = tasks[task]
            task.add_done_callback(
                lambda done_task, tool_name=tool.name, index=tool_index, priority=tool.priority: (
                    self._capture_late_tool_result(
                        turn_id,
                        index,
                        tool_name,
                        priority,
                        done_task,
                    )
                )
            )
            self._schedule_late_cancellation(task)

        if late:
            _log.info(
                "context.tool_late",
                deadline_ms=self._tool_deadline_ms,
                elapsed_ms=round(elapsed_ms, 1),
            )
        return ContextBundle(injections=injections, late=late)

    def _start_task(
        self,
        factory: Callable[[], Awaitable[Any]],
    ) -> asyncio.Task[Any] | None:
        if len(self._pending_tasks) >= self._max_pending_tasks:
            return None
        task = asyncio.create_task(factory())
        self._pending_tasks.add(task)
        task.add_done_callback(self._task_finished)
        return task

    def _task_finished(self, task: asyncio.Task[Any]) -> None:
        self._pending_tasks.discard(task)
        handle = self._late_cancel_handles.pop(task, None)
        if handle is not None:
            handle.cancel()

    def _schedule_late_cancellation(self, task: asyncio.Task[Any]) -> None:
        if task.done():
            return
        handle = asyncio.get_running_loop().call_later(
            self._late_result_grace_ms / 1000,
            task.cancel,
        )
        self._late_cancel_handles[task] = handle

    async def _cancel_and_retrieve(self, tasks: set[asyncio.Task[Any]]) -> None:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _bounded_text(self, text: str) -> str:
        return text[: self._max_injection_chars]

    def _bound_turn_injections(
        self,
        injections: list[tuple[int, Injection]],
    ) -> list[Injection]:
        """Select by freshness/priority, then restore chronological ordering."""
        ranked = sorted(
            enumerate(injections),
            key=lambda candidate: (
                -candidate[1][0],
                -candidate[1][1].priority,
                candidate[0],
            ),
        )
        selected: dict[int, Injection] = {}
        remaining_chars = self._max_total_injection_chars
        for original_index, (_, injection) in ranked:
            if len(selected) >= self._max_injections_per_turn or remaining_chars <= 0:
                break
            text = injection.text[:remaining_chars]
            if not text:
                continue
            selected[original_index] = replace(injection, text=text)
            remaining_chars -= len(text)
        return [selected[index] for index in sorted(selected)]

    def _bounded_hits(self, hits: list[MemoryHit]) -> list[MemoryHit]:
        return [
            MemoryHit(
                text=self._bounded_text(hit.text),
                score=hit.score,
                metadata=dict(hit.metadata),
            )
            for hit in hits[: self._memory_limit]
        ]

    @staticmethod
    def _memory_injection(hit: MemoryHit) -> Injection:
        return Injection(
            source="memory",
            priority=60,
            text=hit.text,
            metadata={"score": hit.score, **hit.metadata},
        )

    def _consume_late_injections(
        self,
        *,
        before_turn_id: int,
    ) -> list[tuple[int, Injection]]:
        """Drain non-expired older results while retaining their turn freshness."""
        oldest_turn_id = max(
            0,
            before_turn_id - self._late_result_ttl_turns,
        )
        eligible_memory = [
            entry
            for entry in self._late_memory_entries
            if oldest_turn_id <= entry[0] < before_turn_id
        ]
        self._late_memory_entries[:] = [
            entry for entry in self._late_memory_entries if entry[0] >= before_turn_id
        ]
        self.late_results[:] = [hits for _, hits in self._late_memory_entries]

        eligible_tools = [
            entry
            for entry in self._late_tool_entries
            if oldest_turn_id <= entry[0] < before_turn_id
        ]
        self._late_tool_entries[:] = [
            entry for entry in self._late_tool_entries if entry[0] >= before_turn_id
        ]
        self.late_tool_results[:] = [
            (tool_name, result) for _, _, tool_name, _, result in self._late_tool_entries
        ]

        ordered: list[tuple[int, int, int, Injection]] = []
        for turn_id, hits in eligible_memory:
            ordered.extend(
                (turn_id, 0, hit_index, self._memory_injection(hit))
                for hit_index, hit in enumerate(hits)
                if hit.text
            )
        ordered.extend(
            (
                turn_id,
                1,
                tool_index,
                Injection(
                    source=f"tool:{tool_name}",
                    priority=priority,
                    text=result,
                ),
            )
            for turn_id, tool_index, tool_name, priority, result in eligible_tools
        )
        ordered.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
        return [(turn_id, injection) for turn_id, _, _, injection in ordered]

    def _capture_late_result(
        self,
        turn_id: int,
        task: asyncio.Task[list[MemoryHit]],
    ) -> None:
        if task.cancelled():
            return
        try:
            hits = self._bounded_hits(task.result())
        except Exception as error:
            _log.warning(
                "context.memory_late_failed",
                error_type=type(error).__name__,
            )
            return
        if not hits or self._max_late_results == 0:
            return

        self._late_memory_entries.append((turn_id, hits))
        self._late_memory_entries.sort(key=lambda entry: entry[0])
        del self._late_memory_entries[: -self._max_late_results]
        self.late_results[:] = [stored_hits for _, stored_hits in self._late_memory_entries]

    def _capture_late_tool_result(
        self,
        turn_id: int,
        tool_index: int,
        tool_name: str,
        priority: int,
        task: asyncio.Task[str],
    ) -> None:
        if task.cancelled():
            return
        try:
            result = task.result()
        except Exception as error:
            _log.warning(
                "context.tool_late_failed",
                tool=tool_name,
                error_type=type(error).__name__,
            )
            return
        if not isinstance(result, str):
            _log.warning(
                "context.tool_late_invalid_result",
                tool=tool_name,
                result_type=type(result).__name__,
            )
            return
        result = self._bounded_text(result)
        if not result or self._max_late_results == 0:
            return

        self._late_tool_entries.append((turn_id, tool_index, tool_name, priority, result))
        self._late_tool_entries.sort(key=lambda entry: (entry[0], entry[1]))
        del self._late_tool_entries[: -self._max_late_results]
        self.late_tool_results[:] = [
            (stored_name, stored_result)
            for _, _, stored_name, _, stored_result in self._late_tool_entries
        ]
