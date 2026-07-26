"""Failure-injection tests for the server-VAD Realtime audio sender."""

from __future__ import annotations

import asyncio

import pytest

from zemory.providers.turn.server_vad import (
    ServerVADAudioBackpressureError,
    ServerVADAudioSenderError,
    ServerVADTurnDetector,
)


class RecordingRealtimeLLM:
    def __init__(self, expected_frames: int = 1) -> None:
        self.pushed: list[bytes] = []
        self.complete = asyncio.Event()
        self.expected_frames = expected_frames

    async def push_audio(self, pcm_bytes: bytes) -> None:
        self.pushed.append(pcm_bytes)
        if len(self.pushed) >= self.expected_frames:
            self.complete.set()


class BlockingRealtimeLLM:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.pushed: list[bytes] = []

    async def push_audio(self, pcm_bytes: bytes) -> None:
        self.pushed.append(pcm_bytes)
        self.started.set()
        await self.release.wait()


@pytest.mark.asyncio
async def test_server_vad_sender_preserves_frame_order() -> None:
    llm = RecordingRealtimeLLM(expected_frames=4)
    detector = ServerVADTurnDetector(llm=llm)
    assert detector._audio_queue.maxsize == 32

    for frame in (b"one", b"two", b"three", b"four"):
        await detector.feed(frame)

    await asyncio.wait_for(llm.complete.wait(), timeout=0.1)
    assert llm.pushed == [b"one", b"two", b"three", b"four"]

    detector.check_health()
    await detector.notify("speech_start")
    assert detector.events.get_nowait() == "speech_start"
    assert detector.consume_audio() == []
    await detector.close()


@pytest.mark.asyncio
async def test_server_vad_feed_does_not_wait_for_network_append() -> None:
    llm = BlockingRealtimeLLM()
    detector = ServerVADTurnDetector(llm=llm)

    await asyncio.wait_for(detector.feed(b"pcm"), timeout=0.05)
    await asyncio.wait_for(llm.started.wait(), timeout=0.05)
    assert llm.pushed == [b"pcm"]

    llm.release.set()
    await asyncio.sleep(0)
    await detector.close()


@pytest.mark.asyncio
async def test_server_vad_queue_accepts_exact_capacity_then_fails_closed() -> None:
    llm = BlockingRealtimeLLM()
    detector = ServerVADTurnDetector(llm=llm, audio_queue_maxsize=2)

    await detector.feed(b"in-flight")
    await asyncio.wait_for(llm.started.wait(), timeout=0.05)
    await detector.feed(b"queued-one")
    await detector.feed(b"queued-two")

    with pytest.raises(ServerVADAudioBackpressureError, match="queue is full"):
        await detector.feed(b"overflow")
    with pytest.raises(ServerVADAudioBackpressureError, match="unavailable"):
        await detector.feed(b"must-not-resume")

    await detector.close()
    assert llm.pushed == [b"in-flight"]
    await asyncio.wait_for(detector._audio_queue.join(), timeout=0.05)


@pytest.mark.asyncio
async def test_server_vad_sender_failure_surfaces_without_provider_payload() -> None:
    class DeferredFailingRealtimeLLM:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.fail = asyncio.Event()

        async def push_audio(self, pcm_bytes: bytes) -> None:
            del pcm_bytes
            self.started.set()
            await self.fail.wait()
            raise RuntimeError("sensitive provider payload and audio")

    llm = DeferredFailingRealtimeLLM()
    detector = ServerVADTurnDetector(llm=llm)
    await detector.feed(b"pcm")
    await asyncio.wait_for(llm.started.wait(), timeout=0.05)

    llm.fail.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    with pytest.raises(ServerVADAudioSenderError) as feed_error:
        await detector.feed(b"next")
    assert "sensitive" not in str(feed_error.value)
    assert "audio" not in str(feed_error.value).lower()

    with pytest.raises(ServerVADAudioSenderError) as close_error:
        await detector.close()
    assert "sensitive" not in str(close_error.value)
    await asyncio.wait_for(detector._audio_queue.join(), timeout=0.05)


@pytest.mark.asyncio
async def test_server_vad_explicit_health_boundary_surfaces_sender_failure() -> None:
    class FailingRealtimeLLM:
        async def push_audio(self, pcm_bytes: bytes) -> None:
            del pcm_bytes
            raise LookupError("provider detail")

    detector = ServerVADTurnDetector(llm=FailingRealtimeLLM())

    with pytest.raises(ServerVADAudioSenderError):
        await detector.feed(b"pcm")
    with pytest.raises(ServerVADAudioSenderError, match="sender failed"):
        detector.check_health()
    with pytest.raises(ServerVADAudioSenderError):
        await detector.close()


@pytest.mark.asyncio
async def test_server_vad_health_observes_sender_cancel_before_done_callback() -> None:
    llm = RecordingRealtimeLLM()
    detector = ServerVADTurnDetector(llm=llm)
    await detector.feed(b"pcm")
    await asyncio.wait_for(llm.complete.wait(), timeout=0.05)
    await asyncio.sleep(0)

    assert detector._sender_task is not None
    assert detector._sender_task.remove_done_callback(detector._sender_done) == 1
    detector._sender_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await detector._sender_task

    with pytest.raises(ServerVADAudioSenderError, match="sender failed"):
        detector.check_health()
    with pytest.raises(ServerVADAudioSenderError):
        await detector.close()


@pytest.mark.asyncio
async def test_server_vad_close_is_bounded_if_sender_swallows_cancellation() -> None:
    class CancellationResistantRealtimeLLM:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.allow_cancel = False
            self.cancel_count = 0

        async def push_audio(self, pcm_bytes: bytes) -> None:
            del pcm_bytes
            self.started.set()
            while True:
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    self.cancel_count += 1
                    if self.allow_cancel:
                        raise

    llm = CancellationResistantRealtimeLLM()
    detector = ServerVADTurnDetector(
        llm=llm,
        close_timeout_s=0.01,
    )
    await detector.feed(b"pcm")
    await asyncio.wait_for(llm.started.wait(), timeout=0.05)

    try:
        with pytest.raises(ServerVADAudioSenderError, match="did not stop"):
            await asyncio.wait_for(detector.close(), timeout=0.05)
        assert llm.cancel_count >= 1
        with pytest.raises(ServerVADAudioSenderError, match="did not stop"):
            await detector.close()
    finally:
        # A Python task cannot be forcibly killed. Always release this
        # deliberately hostile fake, including when an assertion regresses,
        # so the pytest event loop itself never inherits a leaked task.
        llm.allow_cancel = True
        assert detector._sender_task is not None
        detector._sender_task.cancel()
        try:
            await asyncio.wait_for(detector._sender_task, timeout=0.05)
        except asyncio.CancelledError:
            pass
    assert detector._sender_task.done()
    await asyncio.wait_for(detector._audio_queue.join(), timeout=0.05)


@pytest.mark.asyncio
async def test_server_vad_close_remains_cancellable_during_hung_send() -> None:
    class CancellationResistantRealtimeLLM:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.allow_cancel = False

        async def push_audio(self, pcm_bytes: bytes) -> None:
            del pcm_bytes
            self.started.set()
            while True:
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    if self.allow_cancel:
                        raise

    llm = CancellationResistantRealtimeLLM()
    detector = ServerVADTurnDetector(
        llm=llm,
        close_timeout_s=30.0,
    )
    await detector.feed(b"pcm")
    await asyncio.wait_for(llm.started.wait(), timeout=0.05)

    close_task = asyncio.create_task(detector.close())
    try:
        await asyncio.sleep(0)
        close_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(close_task, timeout=0.05)
        with pytest.raises(ServerVADAudioSenderError, match="did not stop"):
            await detector.close()
    finally:
        if not close_task.done():
            close_task.cancel()
        llm.allow_cancel = True
        assert detector._sender_task is not None
        detector._sender_task.cancel()
        try:
            await asyncio.wait_for(detector._sender_task, timeout=0.05)
        except asyncio.CancelledError:
            pass
    assert close_task.done()
    assert detector._sender_task.done()
    await asyncio.wait_for(detector._audio_queue.join(), timeout=0.05)
