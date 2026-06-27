"""Small Realtime event handlers shared by orchestrator and tests."""

from __future__ import annotations

from zemory.pipeline.interrupt_bus import InterruptBus
from zemory.state import Phase, StateMachine


async def handle_speech_started(
    state: StateMachine,
    interrupt_bus: InterruptBus,
    *,
    reason: str = "realtime_speech_started",
) -> bool:
    """Handle Realtime speech-start without dropping assistant partial text."""
    if state.phase == Phase.RESPONDING:
        return await interrupt_bus.trigger(reason)
    await state.transition(Phase.ACTIVE)
    return False
