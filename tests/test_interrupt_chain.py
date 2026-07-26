"""InterruptBus: abort ordering, duplicate suppression, and timing budget."""

from __future__ import annotations

import asyncio
import time

import pytest

from tests.conftest import FakeLLM, FakeSpeaker, FakeTTS
from zemory.pipeline.interrupt_bus import InterruptBus
from zemory.pipeline.tts_manager import TTSTaskManager
from zemory.state import Phase, StateMachine


@pytest.mark.asyncio
async def test_trigger_is_noop_unless_phase_is_responding():
    sm = StateMachine()  # LISTENING
    sp = FakeSpeaker()
    bus = InterruptBus(sm, sp)
    fired = await bus.trigger("unsolicited")
    assert fired is False
    assert sm.phase == Phase.LISTENING


@pytest.mark.asyncio
async def test_abort_chain_runs_in_order_and_under_budget():
    # Short delay — represents a sentence where TTS has started streaming
    # chunks. Abort must stop the flow quickly.
    tts = FakeTTS(delay_ms={"one": 30})
    sp = FakeSpeaker()
    llm = FakeLLM()

    sm = StateMachine()
    await sm.transition(Phase.ACTIVE)
    await sm.transition(Phase.RESPONDING)

    mgr = TTSTaskManager(tts=tts, speaker=sp, max_concurrent=3)
    mgr.start()
    mgr.submit("one")
    # Let a chunk or two dispatch to the speaker so clear() has work to do.
    await asyncio.sleep(0.05)

    bus = InterruptBus(sm, sp)
    bus.bind(mgr, llm)
    bus.record_partial("Hello, I was sa")

    started = time.monotonic()
    fired = await bus.trigger("user_barge_in")
    elapsed_ms = (time.monotonic() - started) * 1000

    assert fired is True
    assert sp.cleared >= 1            # [1] speaker cleared
    assert mgr.aborted is True        # [2] tts aborted
    assert llm.cancel_called == 1     # [3] llm cancel invoked
    assert sm.phase == Phase.ACTIVE   # [5] state transitioned back to ACTIVE
    # speech_started means the new utterance is already in the server input
    # buffer. Clearing here would destroy its prefix.
    assert llm.clear_called == 0
    # Budget: design §4 targets p95 ≤ 150 ms. Under pytest-asyncio scheduling
    # overhead we allow 200 ms as the CI floor.
    assert elapsed_ms < 200, f"abort chain took {elapsed_ms:.1f} ms"

    await mgr.stop()


@pytest.mark.asyncio
async def test_interrupt_truncates_output_without_blocking_local_state() -> None:
    class SlowLLM(FakeLLM):
        async def cancel_current(self) -> None:
            await asyncio.sleep(0.02)
            await super().cancel_current()

    sm = StateMachine()
    await sm.transition(Phase.RESPONDING)
    llm = SlowLLM()
    truncated: list[Phase] = []

    async def truncate_played_output() -> None:
        truncated.append(sm.phase)

    bus = InterruptBus(
        sm,
        FakeSpeaker(),
        on_output_interrupted=truncate_played_output,
    )
    bus.bind(None, llm)

    assert await bus.trigger("user_barge_in") is True
    assert sm.phase == Phase.ACTIVE
    assert truncated == [Phase.ACTIVE]
    assert llm.cancel_called == 1
    assert llm.clear_called == 0
    await bus.aclose()


@pytest.mark.asyncio
async def test_slow_cancel_cannot_skip_captured_output_truncation(monkeypatch) -> None:
    from zemory.pipeline import interrupt_bus as interrupt_module

    monkeypatch.setattr(interrupt_module, "_REMOTE_SYNC_TIMEOUT_S", 0.005)
    monkeypatch.setattr(interrupt_module, "_REMOTE_ACTION_TIMEOUT_S", 0.01)
    monkeypatch.setattr(interrupt_module, "_CANCEL_ACTION_TIMEOUT_S", 0.01)

    class HangingCancelLLM(FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self.cancelled_by_timeout = asyncio.Event()

        async def cancel_current(self, response_id: str | None = None) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled_by_timeout.set()
                raise

    sm = StateMachine()
    await sm.transition(Phase.RESPONDING)
    prepared_in: list[Phase] = []
    synchronized_in: list[Phase] = []

    def prepare_truncation():
        prepared_in.append(sm.phase)

        async def synchronize() -> None:
            synchronized_in.append(sm.phase)

        return synchronize()

    bus = InterruptBus(
        sm,
        FakeSpeaker(),
        on_output_interrupted=prepare_truncation,
    )
    llm = HangingCancelLLM()
    bus.bind(None, llm)

    started = time.monotonic()
    assert await bus.trigger("slow_remote") is True
    assert (time.monotonic() - started) * 1000 < 50
    assert prepared_in == [Phase.RESPONDING]
    assert sm.phase == Phase.ACTIVE

    for _ in range(30):
        if synchronized_in:
            break
        await asyncio.sleep(0.002)
    assert synchronized_in == [Phase.ACTIVE]
    assert llm.cancelled_by_timeout.is_set()
    await bus.aclose()


@pytest.mark.asyncio
async def test_phase_transition_rejects_concurrent_duplicate_trigger():
    sm = StateMachine()
    await sm.transition(Phase.ACTIVE)
    await sm.transition(Phase.RESPONDING)

    mgr = TTSTaskManager(tts=FakeTTS(), speaker=FakeSpeaker(), max_concurrent=3)
    mgr.start()
    bus = InterruptBus(sm, FakeSpeaker())
    bus.bind(mgr, FakeLLM())

    fired = await asyncio.gather(
        bus.trigger("duplicate-a"),
        bus.trigger("duplicate-b"),
    )

    assert sorted(fired) == [False, True]
    assert sm.phase == Phase.ACTIVE

    await mgr.stop()


@pytest.mark.asyncio
async def test_rapid_interrupt_in_new_response_generation_is_not_suppressed():
    sm = StateMachine()
    await sm.transition(Phase.RESPONDING)
    speaker = FakeSpeaker()
    llm = FakeLLM()
    bus = InterruptBus(sm, speaker)
    bus.bind(None, llm)

    assert await bus.trigger("generation-one") is True
    # A short user turn can legitimately put the next generation into
    # RESPONDING within the old 250 ms debounce window.
    await sm.transition(Phase.RESPONDING)
    assert await bus.trigger("generation-two") is True

    assert sm.phase == Phase.ACTIVE
    assert speaker.cleared == 2
    assert llm.cancel_called == 2
    await bus.aclose()


@pytest.mark.asyncio
async def test_partial_text_accumulation_and_reset():
    sp = FakeSpeaker()
    sm = StateMachine()
    bus = InterruptBus(sm, sp)
    bus.record_partial("foo")
    bus.record_partial(" bar")
    assert bus._partial_text == "foo bar"
    bus.reset_partial()
    assert bus._partial_text == ""
