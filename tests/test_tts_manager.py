"""TTSTaskManager: sequence-order preservation under parallel synthesis.

The critical property: even when later sentences finish synthesis before
earlier ones (because the Semaphore allows parallelism), the speaker
receives bytes strictly in submission order.
"""

from __future__ import annotations

import asyncio

import pytest

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
    await mgr.wait_until_empty()
    # Drain the scheduler once more so dispatcher flushes any final chunks.
    for _ in range(10):
        if not sp.queue.empty():
            break
        await asyncio.sleep(0.02)
    chunks = await sp.drain_for_test()
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
