"""Realtime manual turn detector tests."""

from __future__ import annotations

import asyncio

import pytest

from zemory.providers.turn.realtime_manual import RealtimeManualTurnDetector


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
