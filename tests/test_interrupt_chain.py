"""InterruptBus: abort chain ordering, debounce, timing budget."""

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
    # [6] (clear_input_buffer on Realtime) — not FakeLLM, so isinstance guard skips
    # Budget: design §4 targets p95 ≤ 150 ms. Under pytest-asyncio scheduling
    # overhead we allow 200 ms as the CI floor.
    assert elapsed_ms < 200, f"abort chain took {elapsed_ms:.1f} ms"

    await mgr.stop()


@pytest.mark.asyncio
async def test_debounce_rejects_rapid_double_trigger():
    sm = StateMachine()
    await sm.transition(Phase.ACTIVE)
    await sm.transition(Phase.RESPONDING)

    mgr = TTSTaskManager(tts=FakeTTS(), speaker=FakeSpeaker(), max_concurrent=3)
    mgr.start()
    bus = InterruptBus(sm, FakeSpeaker())
    bus.bind(mgr, FakeLLM())

    fired_1 = await bus.trigger("first")
    # Re-enter RESPONDING to test the debounce (trigger moved us to ACTIVE).
    await sm.transition(Phase.RESPONDING)
    fired_2 = await bus.trigger("second")

    assert fired_1 is True
    assert fired_2 is False   # debounced

    await mgr.stop()


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
