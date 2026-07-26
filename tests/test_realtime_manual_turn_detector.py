"""Realtime manual turn detector tests."""

from __future__ import annotations

import asyncio

import pytest

from zemory.providers.turn.realtime_manual import (
    RealtimeAudioBackpressureError,
    RealtimeAudioSenderError,
    RealtimeEndpointStateMachine,
    RealtimeManualTurnDetector,
)


class FakeEndpointDetector:
    def __init__(self) -> None:
        self.events: asyncio.Queue[str] = asyncio.Queue()
        self.fed: list[bytes] = []
        self.signals: asyncio.Queue[str] = asyncio.Queue()
        self.closed = 0
        self.reset_called = 0

    async def feed(self, pcm24k: bytes) -> None:
        self.fed.append(pcm24k)
        while not self.signals.empty():
            await self.events.put(self.signals.get_nowait())

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
        def __init__(self, *, state_machine, capture_audio: bool) -> None:
            self.events: asyncio.Queue[str] = asyncio.Queue()
            self.fed: list[bytes] = []
            captured["state_machine"] = state_machine
            captured["capture_audio"] = capture_audio

        async def feed(self, pcm24k: bytes) -> None:
            self.fed.append(pcm24k)

    monkeypatch.setattr(silero, "SileroTurnDetector", FakeSileroTurnDetector)

    detector = RealtimeManualTurnDetector(llm=FakeRealtimeLLM())

    await detector.feed(b"pcm")

    state_machine = captured["state_machine"]
    assert cfg.settings.realtime.local_endpoint_required_misses == 14
    assert isinstance(state_machine, RealtimeEndpointStateMachine)
    assert state_machine.required_misses == 14
    assert captured["capture_audio"] is False
    assert detector.events.maxsize > 0

    await detector.close()


@pytest.mark.asyncio
async def test_realtime_manual_detector_streams_audio_and_relays_endpoint_events() -> None:
    endpoint = FakeEndpointDetector()
    llm = FakeRealtimeLLM()
    detector = RealtimeManualTurnDetector(llm=llm, endpoint_detector=endpoint)

    await endpoint.signals.put("speech_end")
    await detector.feed(b"pcm")

    assert llm.pushed == [b"pcm"]
    assert endpoint.fed == [b"pcm"]
    assert await asyncio.wait_for(detector.events.get(), timeout=0.1) == "speech_end"
    assert detector.consume_audio() == []

    detector.reset()
    await detector.close()

    assert endpoint.reset_called == 1
    assert endpoint.closed == 1


class BlockingRealtimeLLM:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.push_started = asyncio.Event()
        self.pushed: list[bytes] = []

    async def push_audio(self, pcm_bytes: bytes) -> None:
        self.push_started.set()
        await self.release.wait()
        self.pushed.append(pcm_bytes)


@pytest.mark.asyncio
async def test_network_backpressure_does_not_block_local_vad_and_end_waits_for_flush() -> None:
    endpoint = FakeEndpointDetector()
    llm = BlockingRealtimeLLM()
    detector = RealtimeManualTurnDetector(llm=llm, endpoint_detector=endpoint)

    await endpoint.signals.put("speech_start")
    await asyncio.wait_for(detector.feed(b"first"), timeout=0.1)
    await asyncio.wait_for(llm.push_started.wait(), timeout=0.1)
    assert await asyncio.wait_for(detector.events.get(), timeout=0.1) == "speech_start"

    await endpoint.signals.put("speech_end")
    await asyncio.wait_for(detector.feed(b"last"), timeout=0.1)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(detector.events.get(), timeout=0.02)

    llm.release.set()
    assert await asyncio.wait_for(detector.events.get(), timeout=0.1) == "speech_end"
    assert llm.pushed == [b"first", b"last"]

    await detector.close()


@pytest.mark.asyncio
async def test_frames_after_speech_end_wait_for_commit_reset_boundary() -> None:
    endpoint = FakeEndpointDetector()
    llm = FakeRealtimeLLM()
    detector = RealtimeManualTurnDetector(llm=llm, endpoint_detector=endpoint)

    await endpoint.signals.put("speech_end")
    await detector.feed(b"end")
    assert await asyncio.wait_for(detector.events.get(), timeout=0.1) == "speech_end"

    next_feed = asyncio.create_task(detector.feed(b"next"))
    await asyncio.sleep(0)
    assert not next_feed.done()
    assert llm.pushed == [b"end"]

    detector.reset()
    await asyncio.wait_for(next_feed, timeout=0.1)
    assert llm.pushed == [b"end", b"next"]

    await detector.close()


@pytest.mark.asyncio
async def test_concurrent_feed_cannot_race_across_speech_end_boundary() -> None:
    class EndpointThatBlocksFirstFeed(FakeEndpointDetector):
        def __init__(self) -> None:
            super().__init__()
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()

        async def feed(self, pcm24k: bytes) -> None:
            self.fed.append(pcm24k)
            if len(self.fed) == 1:
                self.first_started.set()
                await self.release_first.wait()
                await self.events.put("speech_end")

    endpoint = EndpointThatBlocksFirstFeed()
    llm = FakeRealtimeLLM()
    detector = RealtimeManualTurnDetector(llm=llm, endpoint_detector=endpoint)

    first_feed = asyncio.create_task(detector.feed(b"end"))
    await asyncio.wait_for(endpoint.first_started.wait(), timeout=0.1)
    next_feed = asyncio.create_task(detector.feed(b"next"))
    endpoint.release_first.set()

    await asyncio.wait_for(first_feed, timeout=0.1)
    assert await asyncio.wait_for(detector.events.get(), timeout=0.1) == "speech_end"
    await asyncio.sleep(0)
    assert not next_feed.done()
    assert endpoint.fed == [b"end"]

    detector.reset()
    await asyncio.wait_for(next_feed, timeout=0.1)
    assert endpoint.fed == [b"end", b"next"]

    await detector.close()


@pytest.mark.asyncio
async def test_audio_sender_queue_overflow_fails_without_silent_pcm_loss() -> None:
    endpoint = FakeEndpointDetector()
    llm = BlockingRealtimeLLM()
    detector = RealtimeManualTurnDetector(
        llm=llm,
        endpoint_detector=endpoint,
        audio_queue_maxsize=1,
    )

    await detector.feed(b"sending")
    await asyncio.wait_for(llm.push_started.wait(), timeout=0.1)
    await detector.feed(b"queued")

    with pytest.raises(RealtimeAudioBackpressureError, match="queue is full"):
        await detector.feed(b"rejected")
    assert endpoint.fed == [b"sending", b"queued"]

    await asyncio.wait_for(detector.close(), timeout=0.1)
    assert endpoint.closed == 1


@pytest.mark.asyncio
async def test_endpoint_events_emitted_outside_feed_are_still_relayed() -> None:
    endpoint = FakeEndpointDetector()
    detector = RealtimeManualTurnDetector(llm=FakeRealtimeLLM(), endpoint_detector=endpoint)

    await detector.feed(b"pcm")
    await endpoint.events.put("speech_start")

    assert await asyncio.wait_for(detector.events.get(), timeout=0.1) == "speech_start"
    await detector.close()


@pytest.mark.asyncio
async def test_audio_sender_failure_surfaces_without_provider_error_text() -> None:
    class FailingRealtimeLLM:
        async def push_audio(self, pcm_bytes: bytes) -> None:
            del pcm_bytes
            raise RuntimeError("sensitive provider payload")

    detector = RealtimeManualTurnDetector(
        llm=FailingRealtimeLLM(),  # type: ignore[arg-type]
        endpoint_detector=FakeEndpointDetector(),
    )

    with pytest.raises(RealtimeAudioSenderError) as exc_info:
        await detector.feed(b"pcm")

    assert "sensitive" not in str(exc_info.value)
    await detector.close()
