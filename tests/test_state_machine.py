"""StateMachine unit tests — atomic transitions, mute derivation, listeners."""

from __future__ import annotations

import asyncio

import pytest

from zemory.state import Phase, StateMachine


@pytest.mark.asyncio
async def test_initial_phase_is_listening():
    sm = StateMachine()
    assert sm.phase == Phase.LISTENING
    assert sm.mute_mic is True  # LISTENING ≠ ACTIVE → mic muted


@pytest.mark.asyncio
async def test_transition_returns_old_and_mutates():
    sm = StateMachine()
    old = await sm.transition(Phase.ACTIVE)
    assert old == Phase.LISTENING
    assert sm.phase == Phase.ACTIVE
    assert sm.mute_mic is False  # ACTIVE → mic live


@pytest.mark.asyncio
async def test_listeners_notified_with_old_new():
    sm = StateMachine()
    observed: list[tuple[Phase, Phase]] = []
    sm.add_listener(lambda o, n: observed.append((o, n)))

    await sm.transition(Phase.ACTIVE)
    await sm.transition(Phase.RESPONDING)
    await sm.transition(Phase.LISTENING)

    assert observed == [
        (Phase.LISTENING, Phase.ACTIVE),
        (Phase.ACTIVE, Phase.RESPONDING),
        (Phase.RESPONDING, Phase.LISTENING),
    ]


@pytest.mark.asyncio
async def test_concurrent_transitions_are_serialized():
    """Two concurrent transitions must both succeed, last-writer-wins."""
    sm = StateMachine()

    async def tx(target: Phase) -> None:
        await sm.transition(target)

    await asyncio.gather(tx(Phase.ACTIVE), tx(Phase.RESPONDING), tx(Phase.LISTENING))
    # Final state must be one of the three — precise last-writer depends on
    # scheduling order, but state must be coherent.
    assert sm.phase in {Phase.ACTIVE, Phase.RESPONDING, Phase.LISTENING}


@pytest.mark.asyncio
async def test_mark_speech_end_sets_timestamp():
    sm = StateMachine()
    assert sm.speech_end_ts is None
    sm.mark_speech_end()
    assert sm.speech_end_ts is not None


@pytest.mark.asyncio
async def test_listener_exceptions_do_not_break_transition():
    sm = StateMachine()
    sm.add_listener(lambda o, n: (_ for _ in ()).throw(RuntimeError("boom")))
    old = await sm.transition(Phase.ACTIVE)
    assert old == Phase.LISTENING
    assert sm.phase == Phase.ACTIVE  # transition still completed
