"""TTSTaskManager: sequence-order preservation under parallel synthesis.

The critical property: even when later sentences finish synthesis before
earlier ones (because the Semaphore allows parallelism), the speaker
receives bytes strictly in submission order.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest
from structlog.testing import capture_logs

from tests.conftest import FakeSpeaker, FakeTTS
from zemory.pipeline.tts_manager import TTSTaskManager


@pytest.mark.asyncio
async def test_single_sentence_reaches_speaker():
    tts = FakeTTS()
    sp = FakeSpeaker()
    mgr = TTSTaskManager(tts=tts, speaker=sp, max_concurrent=3)
    mgr.start()
    mgr.submit("hello")
    # Wait for synthesis to complete + dispatcher to forward everything
    outcome = await mgr.wait_until_empty()
    # Drain the scheduler once more so dispatcher flushes any final chunks.
    for _ in range(10):
        if not sp.queue.empty():
            break
        await asyncio.sleep(0.02)
    chunks = await sp.drain_for_test()
    assert outcome is True
    assert mgr.generation_completed_successfully is True
    await mgr.stop()

    joined = b"".join(chunks).decode()
    assert "<start:hello>" in joined
    assert "<hello>" in joined
    assert "<end:hello" in joined


@pytest.mark.asyncio
async def test_sequence_order_preserved_under_parallel_synthesis():
    """Submit seq-0 slow, seq-1 fast. Playback order must still be 0 then 1."""
    tts = FakeTTS(delay_ms={"slow": 80, "fast": 5})
    sp = FakeSpeaker()
    mgr = TTSTaskManager(tts=tts, speaker=sp, max_concurrent=3)
    mgr.start()
    mgr.submit("slow")
    mgr.submit("fast")

    await mgr.wait_until_empty()
    for _ in range(15):
        if not sp.queue.empty():
            await asyncio.sleep(0.02)
        else:
            break
    # Collect and wait a moment for late chunks
    await asyncio.sleep(0.05)
    chunks = await sp.drain_for_test()
    await mgr.stop()

    joined = b"".join(chunks).decode()
    slow_start = joined.index("<start:slow>")
    fast_start = joined.index("<start:fast>")
    slow_end = joined.index("<end:slow")
    fast_end = joined.index("<end:fast")

    # All "slow" bytes must come before any "fast" byte
    assert slow_end < fast_start
    assert slow_start < fast_start
    assert slow_end < fast_end


@pytest.mark.asyncio
async def test_abort_prevents_future_chunks_from_reaching_speaker():
    tts = FakeTTS(delay_ms={"long": 200, "tail": 200})
    sp = FakeSpeaker()
    mgr = TTSTaskManager(tts=tts, speaker=sp, max_concurrent=3)
    mgr.start()
    mgr.submit("long")
    mgr.submit("tail")
    await asyncio.sleep(0.02)  # let first chunk maybe dispatch
    await mgr.abort()
    await mgr.stop()

    # After abort, the manager must be in aborted state
    assert mgr.aborted is True
    # Further submit should return -1 (rejected)
    rejected = mgr.submit("post_abort")
    assert rejected == -1


@pytest.mark.asyncio
async def test_reset_for_new_turn_allows_submission_after_abort():
    """Regression: after one barge-in, the manager was permanently aborted.

    The orchestrator now calls ``reset_for_new_turn()`` at each new
    ``speech_stopped`` / ``speech_end`` — this test protects that path.
    """
    tts = FakeTTS()
    sp = FakeSpeaker()
    mgr = TTSTaskManager(tts=tts, speaker=sp, max_concurrent=3)
    mgr.start()

    # Turn 1: submit, then abort (simulates barge-in mid-response).
    seq_a = mgr.submit("first")
    assert seq_a == 0
    await mgr.abort()
    assert mgr.aborted is True
    assert mgr.submit("rejected") == -1  # confirms permanent-abort behavior pre-reset

    # Turn 2: new response begins → orchestrator resets the manager.
    mgr.reset_for_new_turn()
    assert mgr.aborted is False

    seq_b = mgr.submit("second")
    assert seq_b == 0, "seq counter should reset to 0 for the new turn"
    seq_c = mgr.submit("third")
    assert seq_c == 1

    await mgr.wait_until_empty()
    await asyncio.sleep(0.05)
    chunks = await sp.drain_for_test()
    await mgr.stop()

    joined = b"".join(chunks).decode()
    assert "<second>" in joined
    assert "<third>" in joined


@pytest.mark.asyncio
async def test_first_chunk_callback_fires_for_seq_zero():
    tts = FakeTTS()
    sp = FakeSpeaker()
    ttfb_observed: list[tuple[int, float]] = []
    mgr = TTSTaskManager(
        tts=tts,
        speaker=sp,
        max_concurrent=3,
        on_first_chunk=lambda seq, ttfb: ttfb_observed.append((seq, ttfb)),
    )
    mgr.start()
    mgr.submit("one")
    mgr.submit("two")
    await mgr.wait_until_empty()
    await mgr.stop()

    # Each sentence reports a first_chunk ttfb
    seqs = [s for s, _ in ttfb_observed]
    assert 0 in seqs and 1 in seqs


class _DelayedCancellationTTS:
    """Keep an aborted generation alive until a replacement turn exists."""

    def __init__(self) -> None:
        self.old_started = asyncio.Event()
        self.old_cancelled = asyncio.Event()
        self.release_old_cleanup = asyncio.Event()
        self.new_started = asyncio.Event()
        self.release_new_audio = asyncio.Event()

    async def synthesize(self, text: str, quick: bool = False) -> AsyncIterator[bytes]:
        if text == "old":
            self.old_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.old_cancelled.set()
                await self.release_old_cleanup.wait()
                raise
            yield b"unreachable"
            return

        self.new_started.set()
        await self.release_new_audio.wait()
        yield b"<new-turn>"


@pytest.mark.asyncio
async def test_aborted_generation_cannot_complete_reused_sequence() -> None:
    """An old seq-0 cleanup must not mark a new turn's seq-0 as done."""
    tts = _DelayedCancellationTTS()
    sp = FakeSpeaker()
    mgr = TTSTaskManager(tts=tts, speaker=sp, max_concurrent=2)
    mgr.start()

    mgr.submit("old")
    await asyncio.wait_for(tts.old_started.wait(), timeout=1)
    await mgr.abort()
    await asyncio.wait_for(tts.old_cancelled.wait(), timeout=1)

    mgr.reset_for_new_turn()
    assert mgr.submit("new") == 0
    await asyncio.wait_for(tts.new_started.wait(), timeout=1)

    # Finish the old task while the replacement seq-0 is still synthesizing.
    tts.release_old_cleanup.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    tts.release_new_audio.set()

    await asyncio.wait_for(mgr.wait_until_empty(), timeout=1)
    await asyncio.sleep(0.05)
    chunks = await sp.drain_for_test()
    await mgr.stop()

    assert b"<new-turn>" in chunks


class _FailingTTS:
    async def synthesize(self, text: str, quick: bool = False) -> AsyncIterator[bytes]:
        await asyncio.sleep(0)
        raise RuntimeError("synthesis exploded")
        yield b"unreachable"


@pytest.mark.asyncio
async def test_synthesis_failure_is_observable() -> None:
    """A background provider failure must be retrieved and logged."""
    mgr = TTSTaskManager(tts=_FailingTTS(), speaker=FakeSpeaker(), max_concurrent=1)
    mgr.start()

    with capture_logs() as logs:
        mgr.submit("broken")
        await asyncio.wait_for(mgr.wait_until_empty(), timeout=1)

    await mgr.stop()

    failures = [entry for entry in logs if entry["event"] == "tts.synthesis_failed"]
    assert len(failures) == 1
    assert failures[0]["seq"] == 0
    assert failures[0]["error_type"] == "RuntimeError"
    assert "synthesis exploded" not in repr(logs)


class _EmptyAudioTTS:
    async def synthesize(self, text: str, quick: bool = False) -> AsyncIterator[bytes]:
        if False:
            yield b""


@pytest.mark.asyncio
async def test_zero_byte_generation_is_not_successful() -> None:
    mgr = TTSTaskManager(tts=_EmptyAudioTTS(), speaker=FakeSpeaker(), max_concurrent=1)
    mgr.start()
    assert mgr.generation_completed_successfully is False

    assert mgr.submit("silent") == 0
    assert await asyncio.wait_for(mgr.wait_until_empty(), timeout=1) is False
    assert mgr.generation_completed_successfully is False
    assert mgr.generation_failure_reasons == ("empty_audio",)
    await mgr.stop()


class _PartialThenFailingTTS:
    async def synthesize(self, text: str, quick: bool = False) -> AsyncIterator[bytes]:
        yield b"partial-audio"
        await asyncio.sleep(0)
        raise RuntimeError("provider included sensitive input")


@pytest.mark.asyncio
async def test_partial_audio_followed_by_provider_failure_is_not_successful() -> None:
    speaker = FakeSpeaker()
    mgr = TTSTaskManager(tts=_PartialThenFailingTTS(), speaker=speaker, max_concurrent=1)
    mgr.start()

    mgr.submit("private text")
    assert await asyncio.wait_for(mgr.wait_until_empty(), timeout=1) is False
    assert mgr.generation_completed_successfully is False
    assert mgr.generation_failure_reasons == ("synthesis_failed",)
    assert b"partial-audio" in await speaker.drain_for_test()
    await mgr.stop()


@pytest.mark.asyncio
async def test_reset_isolates_failed_generation_outcome() -> None:
    class SelectiveTTS:
        async def synthesize(self, text: str, quick: bool = False) -> AsyncIterator[bytes]:
            if text == "works":
                yield b"new-generation-audio"

    mgr = TTSTaskManager(tts=SelectiveTTS(), speaker=FakeSpeaker(), max_concurrent=1)
    mgr.start()
    mgr.submit("silent")
    assert await asyncio.wait_for(mgr.wait_until_empty(), timeout=1) is False

    mgr.reset_for_new_turn()
    assert mgr.generation_failure_reasons == ()
    assert mgr.submit("works") == 0
    assert await asyncio.wait_for(mgr.wait_until_empty(), timeout=1) is True
    assert mgr.generation_completed_successfully is True
    await mgr.stop()


@pytest.mark.asyncio
async def test_wait_requires_every_byte_to_be_handed_to_speaker() -> None:
    class GatedQueue:
        def __init__(self) -> None:
            self.put_started = asyncio.Event()
            self.release = asyncio.Event()

        async def put(self, payload: bytes) -> None:
            self.put_started.set()
            await self.release.wait()

    class GatedSpeaker:
        def __init__(self) -> None:
            self.queue = GatedQueue()

    speaker = GatedSpeaker()
    mgr = TTSTaskManager(tts=FakeTTS(), speaker=speaker, max_concurrent=1)
    mgr.start()
    mgr.submit("held")
    waiter = asyncio.create_task(mgr.wait_until_empty())

    await asyncio.wait_for(speaker.queue.put_started.wait(), timeout=1)
    assert waiter.done() is False
    assert mgr.generation_completed_successfully is False
    speaker.queue.release.set()
    assert await asyncio.wait_for(waiter, timeout=1) is True
    await mgr.stop()


@pytest.mark.asyncio
async def test_speaker_handoff_failure_returns_false_without_hanging() -> None:
    class FailingQueue:
        async def put(self, payload: bytes) -> None:
            raise RuntimeError("speaker error included private payload")

    class FailingSpeaker:
        def __init__(self) -> None:
            self.queue = FailingQueue()

    mgr = TTSTaskManager(tts=FakeTTS(), speaker=FailingSpeaker(), max_concurrent=1)
    mgr.start()

    with capture_logs() as logs:
        mgr.submit("do not log this")
        assert await asyncio.wait_for(mgr.wait_until_empty(), timeout=1) is False

    assert mgr.generation_completed_successfully is False
    assert "speaker_handoff_failed" in mgr.generation_failure_reasons
    dispatch_failures = [entry for entry in logs if entry["event"] == "tts.dispatch_failed"]
    assert len(dispatch_failures) == 1
    assert dispatch_failures[0]["error_type"] == "RuntimeError"
    assert "private payload" not in repr(logs)
    assert "do not log this" not in repr(logs)
    await mgr.stop()


class _UnboundedFutureAudioTTS:
    def __init__(self) -> None:
        self.first_started = asyncio.Event()

    async def synthesize(self, text: str, quick: bool = False) -> AsyncIterator[bytes]:
        if text == "blocking-first":
            self.first_started.set()
            await asyncio.Event().wait()
        while True:
            yield b"x" * 4096
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_future_sentence_audio_buffer_has_a_hard_limit() -> None:
    tts = _UnboundedFutureAudioTTS()
    mgr = TTSTaskManager(tts=tts, speaker=FakeSpeaker(), max_concurrent=2)
    mgr.start()
    mgr.submit("blocking-first")
    mgr.submit("runaway-second")

    await asyncio.wait_for(tts.first_started.wait(), timeout=1)
    for _ in range(200):
        if mgr._buffered_bytes >= 240_000:
            break
        await asyncio.sleep(0)

    assert 240_000 - 4096 < mgr._buffered_bytes <= 240_000
    assert any(not task.done() for task in mgr._tasks)
    await mgr.abort()
    await mgr.stop()


@pytest.mark.asyncio
async def test_future_sentence_cannot_starve_head_sequence_buffer() -> None:
    class ControlledTTS:
        def __init__(self) -> None:
            self.release_head = asyncio.Event()
            self.future_filled = asyncio.Event()

        async def synthesize(self, text: str, quick: bool = False) -> AsyncIterator[bytes]:
            if text == "head":
                await self.release_head.wait()
                yield b"head-audio"
                return
            while True:
                yield b"x" * 4096
                if mgr._buffered_bytes >= 240_000 - 4096:
                    self.future_filled.set()
                await asyncio.sleep(0)

    tts = ControlledTTS()
    speaker = FakeSpeaker()
    mgr = TTSTaskManager(tts=tts, speaker=speaker, max_concurrent=2)
    mgr.start()
    mgr.submit("head")
    mgr.submit("future")

    await asyncio.wait_for(tts.future_filled.wait(), timeout=1)
    tts.release_head.set()
    for _ in range(100):
        if not speaker.queue.empty():
            break
        await asyncio.sleep(0)

    chunks = await speaker.drain_for_test()
    assert b"head-audio" in chunks
    assert mgr._buffered_bytes <= 240_000 + 4096
    await mgr.abort()
    await mgr.stop()


@pytest.mark.asyncio
async def test_pending_sentence_count_is_bounded() -> None:
    tts = FakeTTS(delay_ms={f"sentence-{index}": 100 for index in range(40)})
    mgr = TTSTaskManager(tts=tts, speaker=FakeSpeaker(), max_concurrent=1)
    mgr.start()

    accepted = [mgr.submit(f"sentence-{index}") for index in range(40)]

    assert accepted[:32] == list(range(32))
    assert accepted[32:] == [-1] * 8
    await mgr.abort()
    await mgr.stop()


@pytest.mark.asyncio
async def test_stop_is_bounded_when_provider_suppresses_cancellation(monkeypatch) -> None:
    from zemory.pipeline import tts_manager as manager_module

    monkeypatch.setattr(manager_module, "_TASK_SHUTDOWN_TIMEOUT_S", 0.01)

    class StubbornTTS:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def synthesize(self, text: str, quick: bool = False) -> AsyncIterator[bytes]:
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await self.release.wait()
            if False:
                yield b""

    tts = StubbornTTS()
    mgr = TTSTaskManager(tts=tts, speaker=FakeSpeaker(), max_concurrent=1)
    mgr.start()
    mgr.submit("stubborn")
    await asyncio.wait_for(tts.started.wait(), timeout=1)

    started = time.monotonic()
    await mgr.stop()
    assert (time.monotonic() - started) < 0.05
    assert mgr._tasks

    tts.release.set()
    await asyncio.wait_for(
        asyncio.gather(*tuple(mgr._tasks), return_exceptions=True),
        timeout=1,
    )
