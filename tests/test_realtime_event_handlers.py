"""Small Realtime event handler tests for orchestration edge cases."""

from __future__ import annotations

import pytest

from tests.conftest import FakeLLM, FakeSpeaker, FakeTTS
from zemory.pipeline.interrupt_bus import InterruptBus
from zemory.pipeline.realtime_events import handle_speech_started
from zemory.pipeline.tts_manager import TTSTaskManager
from zemory.state import Phase, StateMachine


@pytest.mark.asyncio
async def test_realtime_speech_started_preserves_partial_before_interrupt() -> None:
    captured: list[str] = []

    async def on_partial(text: str) -> None:
        captured.append(text)

    state = StateMachine()
    await state.transition(Phase.ACTIVE)
    await state.transition(Phase.RESPONDING)

    speaker = FakeSpeaker()
    llm = FakeLLM()
    manager = TTSTaskManager(tts=FakeTTS(), speaker=speaker, max_concurrent=1)
    manager.start()

    bus = InterruptBus(state, speaker, on_partial=on_partial)
    bus.bind(manager, llm)
    bus.record_partial("말하던 문장")

    fired = await handle_speech_started(state, bus)

    assert fired is True
    assert captured == ["말하던 문장"]
    assert state.phase == Phase.ACTIVE
    assert llm.cancel_called == 1

    await manager.stop()


@pytest.mark.asyncio
async def test_realtime_speech_started_enters_active_when_listening() -> None:
    state = StateMachine()
    bus = InterruptBus(state, FakeSpeaker())

    fired = await handle_speech_started(state, bus)

    assert fired is False
    assert state.phase == Phase.ACTIVE
