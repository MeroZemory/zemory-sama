"""Bounded resource cleanup independent of turn/response orchestration."""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Callable
from typing import Any


class RuntimeCleanup:
    """Run each resource cleanup with a real wall-clock deadline."""

    def __init__(self, *, timeout_s: float, logger: Any) -> None:
        self.timeout_s = max(0.0, timeout_s)
        self.logger = logger
        self.errors: list[Exception] = []
        self._orphaned_tasks: set[asyncio.Task[Any]] = set()

    def _retrieve_orphan(self, task: asyncio.Task[Any]) -> None:
        self._orphaned_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.exception()
        except Exception:
            pass

    async def _invoke(self, callback: Callable[[], Any]) -> None:
        if inspect.iscoroutinefunction(callback):
            await callback()
            return

        # The default executor owns non-daemon workers. A dedicated daemon
        # makes this resource deadline real even if a device driver never
        # returns from its synchronous close method.
        loop = asyncio.get_running_loop()
        completion: asyncio.Future[object] = loop.create_future()

        def finish_success(result: object) -> None:
            if not completion.done():
                completion.set_result(result)

        def finish_failure(error: BaseException) -> None:
            if not completion.done():
                completion.set_exception(
                    RuntimeError(
                        f"synchronous cleanup raised {type(error).__name__}"
                    )
                )

        def run_synchronous_cleanup() -> None:
            try:
                result = callback()
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
            target=run_synchronous_cleanup,
            name="zemory-cleanup",
            daemon=True,
        ).start()
        result = await completion
        if inspect.isawaitable(result):
            await result

    async def run(self, name: str, callback: Callable[[], Any]) -> None:
        task = asyncio.create_task(
            self._invoke(callback),
            name=f"cleanup_{name}",
        )
        try:
            done, pending = await asyncio.wait({task}, timeout=self.timeout_s)
            if pending:
                task.cancel()
                self._orphaned_tasks.add(task)
                task.add_done_callback(self._retrieve_orphan)
                error = TimeoutError(
                    f"{name} cleanup exceeded {self.timeout_s:.3f}s"
                )
                self.errors.append(error)
                self.logger.error(
                    "orchestrator.cleanup_timeout",
                    resource=name,
                    timeout_ms=int(self.timeout_s * 1000),
                )
                return
            if task.cancelled():
                raise RuntimeError(f"{name} cleanup cancelled unexpectedly")
            task.result()
        except asyncio.CancelledError as error:
            failure = RuntimeError(f"{name} cleanup cancelled unexpectedly")
            self.errors.append(failure)
            self.logger.error(
                "orchestrator.cleanup_failed",
                resource=name,
                error_type=type(error).__name__,
            )
        except Exception as error:
            # Provider/device cleanup messages may contain request data. Keep
            # only the exception type at this terminal boundary.
            failure = RuntimeError(f"{name} cleanup raised {type(error).__name__}")
            self.errors.append(failure)
            self.logger.error(
                "orchestrator.cleanup_failed",
                resource=name,
                error_type=type(error).__name__,
            )
