"""Realtime manual turn detector tests."""

from __future__ import annotations

import asyncio

import pytest

from zemory.providers.turn.realtime_manual import (
    RealtimeEndpointStateMachine,
    RealtimeManualTurnDetector,
)


class FakeEndpointDetector:
    def __init__(self) -> None:
        self.events: asyncio.Queue[str] = asyncio.Queue()
        self.fed: list[bytes] = []
        self.closed = 0
        self.reset_called = 0

    async def feed(self, pcm24k: bytes) -> None:
        self.fed.append(pcm24k)

    def consume_audio(self) -> list[bytes]:
        return [b"local-only"]

    def reset(self) -> None:
        self.reset_called += 1

    async def close(self) -> None:
        self.closed += 1


class FakeRealtimeLLM:
    def __init__(self) -> None:
        self.pushed: list[bytes] = []

    async def push_audio(self, pcm_bytes: bytes) -> None:
        self.pushed.append(pcm_bytes)


def test_realtime_endpoint_state_machine_ends_after_single_miss_window() -> None:
    sm = RealtimeEndpointStateMachine(
        prob_threshold=0.5,
        db_threshold=10.0,
        required_hits=1,
        required_misses=2,
        smoothing_window=1,
    )

    assert sm.process(0.9, 30.0) == "speech_start"
    assert sm.process(0.0, 0.0) is None
    assert sm.process(0.0, 0.0) == "speech_end"


@pytest.mark.asyncio
async def test_realtime_manual_detector_uses_fast_endpoint_state_machine(
    monkeypatch,
) -> None:
    from zemory import config as cfg
    from zemory.providers.turn import silero

    captured: dict[str, object] = {}

    class FakeSileroTurnDetector:
        def __init__(self, *, state_machine) -> None:
            self.events: asyncio.Queue[str] = asyncio.Queue()
            self.fed: list[bytes] = []
            captured["state_machine"] = state_machine

        async def feed(self, pcm24k: bytes) -> None:
            self.fed.append(pcm24k)

    monkeypatch.setattr(silero, "SileroTurnDetector", FakeSileroTurnDetector)

    detector = RealtimeManualTurnDetector(llm=FakeRealtimeLLM())

    await detector.feed(b"pcm")

    state_machine = captured["state_machine"]
    assert cfg.settings.realtime.local_endpoint_required_misses == 14
    assert isinstance(state_machine, RealtimeEndpointStateMachine)
    assert state_machine.required_misses == 14


@pytest.mark.asyncio
async def test_realtime_manual_detector_streams_audio_and_relays_endpoint_events() -> None:
    endpoint = FakeEndpointDetector()
    llm = FakeRealtimeLLM()
    detector = RealtimeManualTurnDetector(llm=llm, endpoint_detector=endpoint)

    await detector.feed(b"pcm")
    await endpoint.events.put("speech_end")

    assert llm.pushed == [b"pcm"]
    assert endpoint.fed == [b"pcm"]
    assert await asyncio.wait_for(detector.events.get(), timeout=0.1) == "speech_end"
    assert detector.consume_audio() == []

    detector.reset()
    await detector.close()

    assert endpoint.reset_called == 1
    assert endpoint.closed == 1
