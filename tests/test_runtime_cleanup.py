"""Direct contracts for the bounded runtime cleanup boundary."""

from __future__ import annotations

import traceback

import pytest

from zemory.pipeline.runtime_cleanup import RuntimeCleanup


class RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def error(self, event: str, **fields) -> None:
        self.events.append((event, fields))


@pytest.mark.asyncio
async def test_async_cleanup_failure_retains_type_not_provider_payload() -> None:
    logger = RecordingLogger()
    cleanup = RuntimeCleanup(timeout_s=0.1, logger=logger)

    async def fail() -> None:
        raise RuntimeError("private provider cleanup payload")

    await cleanup.run("provider", fail)

    assert len(cleanup.errors) == 1
    assert str(cleanup.errors[0]) == "provider cleanup raised RuntimeError"
    formatted = "".join(traceback.format_exception(cleanup.errors[0]))
    assert "private provider cleanup payload" not in formatted
    assert logger.events == [
        (
            "orchestrator.cleanup_failed",
            {"resource": "provider", "error_type": "RuntimeError"},
        )
    ]


@pytest.mark.asyncio
async def test_sync_cleanup_failure_retains_type_not_provider_payload() -> None:
    cleanup = RuntimeCleanup(timeout_s=0.1, logger=RecordingLogger())

    def fail() -> None:
        raise ValueError("private device cleanup payload")

    await cleanup.run("device", fail)

    assert len(cleanup.errors) == 1
    assert str(cleanup.errors[0]) == "device cleanup raised RuntimeError"
    formatted = "".join(traceback.format_exception(cleanup.errors[0]))
    assert "private device cleanup payload" not in formatted
