import asyncio
import sys
import time
from collections.abc import Awaitable

from zemory.config import RuntimeCredentialError
from zemory.orchestrator import run

_ASYNC_SHUTDOWN_TIMEOUT_S = 2.0
_MAIN_TASK_CLEANUP_TIMEOUT_S = 30.0


def _remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _retrieve_finished(tasks: set[asyncio.Task[object]]) -> None:
    for task in tasks:
        if task.cancelled():
            continue
        try:
            task.exception()
        except Exception:
            # Runtime failures have already been surfaced by the owning task.
            # Shutdown only retrieves them so asyncio does not report a second,
            # payload-bearing "exception was never retrieved" message.
            pass


def _abandon_pending(
    tasks: set[asyncio.Task[object]],
    *,
    label: str,
) -> None:
    if not tasks:
        return
    for task in tasks:
        task.cancel()
        # A coroutine that deliberately suppresses cancellation cannot be
        # stopped safely in-process. The CLI owns this loop and closes it below,
        # so suppress only asyncio's duplicate destruction warning; emit one
        # bounded, payload-free diagnostic here instead.
        task._log_destroy_pending = False  # type: ignore[attr-defined]
    print(
        f"[shutdown] abandoned {len(tasks)} cancellation-resistant {label}",
        file=sys.stderr,
    )


def _run_shutdown_step(
    loop: asyncio.AbstractEventLoop,
    awaitable: Awaitable[object],
    *,
    deadline: float,
    label: str,
) -> None:
    remaining = _remaining(deadline)
    if remaining <= 0:
        if asyncio.iscoroutine(awaitable):
            awaitable.close()
        return
    task = loop.create_task(awaitable)
    done, pending = loop.run_until_complete(
        asyncio.wait({task}, timeout=remaining)
    )
    _retrieve_finished(done)
    _abandon_pending(pending, label=label)


def _run_with_bounded_shutdown[T](
    awaitable: Awaitable[T],
    *,
    shutdown_timeout_s: float = _ASYNC_SHUTDOWN_TIMEOUT_S,
    main_cleanup_timeout_s: float = _MAIN_TASK_CLEANUP_TIMEOUT_S,
) -> T:
    """Run the CLI coroutine without asyncio.run's unbounded final gather.

    ``asyncio.run`` waits forever for a task that catches every
    ``CancelledError``. Provider cleanup is still given a bounded grace period,
    then this CLI-owned loop is closed so a cancellation-resistant async task
    cannot hold the CLI open. Python cannot forcibly terminate arbitrary
    non-daemon executor threads; production SQLite and cleanup work therefore
    use explicit daemon-thread bridges instead of the default executor.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def await_root() -> T:
        return await awaitable

    root_task = loop.create_task(await_root(), name="zemory-main")
    try:
        return loop.run_until_complete(root_task)
    finally:
        # SIGINT can stop run_until_complete before run() reaches its finally
        # block. Give the owning task enough bounded time to execute every
        # per-resource cleanup deadline before abandoning unrelated orphans.
        if not root_task.done():
            root_task.cancel()
            done, pending_root = loop.run_until_complete(
                asyncio.wait(
                    {root_task},
                    timeout=max(0.0, main_cleanup_timeout_s),
                )
            )
            _retrieve_finished(done)
            _abandon_pending(pending_root, label="main cleanup task(s)")

        deadline = time.monotonic() + max(0.0, shutdown_timeout_s)
        pending = {
            task
            for task in asyncio.all_tasks(loop)
            if task is not root_task and not task.done()
        }
        for task in pending:
            task.cancel()
        if pending and _remaining(deadline) > 0:
            done, pending = loop.run_until_complete(
                asyncio.wait(pending, timeout=_remaining(deadline))
            )
            _retrieve_finished(done)
        _abandon_pending(pending, label="async task(s)")

        _run_shutdown_step(
            loop,
            loop.shutdown_asyncgens(),
            deadline=deadline,
            label="async-generator shutdown task(s)",
        )
        if _remaining(deadline) > 0:
            _run_shutdown_step(
                loop,
                loop.shutdown_default_executor(timeout=_remaining(deadline)),
                deadline=deadline,
                label="default-executor shutdown task(s)",
            )
        asyncio.set_event_loop(None)
        loop.close()


def main() -> None:
    try:
        _run_with_bounded_shutdown(run())
    except RuntimeCredentialError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
