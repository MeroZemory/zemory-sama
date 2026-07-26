"""Tests for the ready-to-speak beep PCM generator."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from zemory.audio import MicrophoneStream, SpeakerStream, generate_beep_pcm, output_block_size


def test_beep_has_expected_length():
    sample_rate = 24000
    pcm = generate_beep_pcm(
        frequency_hz=880,
        duration_ms=80,
        sample_rate=sample_rate,
        volume=0.15,
    )
    # 80 ms at 24 kHz × 2 bytes (int16) = 3840 bytes
    expected = int(sample_rate * 0.080) * 2
    assert len(pcm) == expected


def test_beep_has_fade_edges_to_avoid_clicks():
    pcm = generate_beep_pcm(frequency_hz=880, duration_ms=80, sample_rate=24000, volume=0.5)
    samples = np.frombuffer(pcm, dtype=np.int16)
    # Fade means first and last samples are near zero.
    assert abs(samples[0]) < 1000
    assert abs(samples[-1]) < 1000
    # But the middle should be much louder (sine wave near peak).
    mid = len(samples) // 2
    middle_window = samples[max(0, mid - 10): mid + 10]
    assert np.max(np.abs(middle_window)) > 1000


def test_beep_volume_respected():
    quiet = generate_beep_pcm(880, 80, 24000, volume=0.05)
    loud = generate_beep_pcm(880, 80, 24000, volume=0.5)
    q_samples = np.frombuffer(quiet, dtype=np.int16)
    loud_samples = np.frombuffer(loud, dtype=np.int16)
    assert np.max(np.abs(loud_samples)) > np.max(np.abs(q_samples))


def test_zero_duration_returns_empty():
    assert generate_beep_pcm(880, 0, 24000, volume=0.15) == b""


def test_volume_clamped_to_one():
    # Volume > 1 should be clamped; samples must stay in int16 range.
    pcm = generate_beep_pcm(880, 80, 24000, volume=10.0)
    samples = np.frombuffer(pcm, dtype=np.int16)
    assert samples.max() <= 32767
    assert samples.min() >= -32768


def test_output_block_size_uses_10ms_chunks_by_default():
    assert output_block_size(sample_rate=24_000) == 240


def test_microphone_stream_requests_low_latency_input(monkeypatch):
    captured: dict = {}

    class FakeStream:
        def start(self) -> None:
            return None

    def fake_input_stream(**kwargs):
        captured.update(kwargs)
        return FakeStream()

    monkeypatch.setattr("zemory.audio.sd.InputStream", fake_input_stream)

    stream = MicrophoneStream(asyncio.new_event_loop())
    stream.start()

    assert captured["latency"] == "low"
    assert captured["blocksize"] == stream.chunk_size


def test_microphone_queue_is_bounded_and_keeps_freshest_frames():
    stream = MicrophoneStream(asyncio.new_event_loop())
    total = stream.queue.maxsize + 7

    for index in range(total):
        stream._enqueue_captured(index.to_bytes(2, "little"))

    retained = [
        int.from_bytes(stream.queue.get_nowait(), "little")
        for _ in range(stream.queue.qsize())
    ]
    assert stream.dropped_frames == 7
    assert retained == list(range(7, total))


def test_microphone_stop_attempts_close_after_driver_stop_failure():
    class FailingStopStream:
        def __init__(self) -> None:
            self.closed = False

        def stop(self) -> None:
            raise RuntimeError("driver stop failed")

        def close(self) -> None:
            self.closed = True

    microphone = MicrophoneStream(asyncio.new_event_loop())
    stream = FailingStopStream()
    microphone._stream = stream

    with np.testing.assert_raises(RuntimeError):
        microphone.stop()

    assert stream.closed is True
    assert microphone._stream is None


def test_speaker_stream_requests_low_latency_output(monkeypatch):
    captured: dict = {}

    class FakeStream:
        def start(self) -> None:
            return None

    def fake_output_stream(**kwargs):
        captured.update(kwargs)
        return FakeStream()

    monkeypatch.setattr("zemory.audio.sd.OutputStream", fake_output_stream)

    stream = SpeakerStream(asyncio.new_event_loop())
    stream.start()

    assert captured["latency"] == "low"
    assert captured["blocksize"] == output_block_size()


async def test_speaker_records_first_callback_playback_time():
    speaker = SpeakerStream(asyncio.get_running_loop())
    speaker.arm()
    assert speaker.first_write_at is None
    assert speaker.first_play_at is None

    feed_task = asyncio.create_task(speaker.feed())
    try:
        await speaker.queue.put(np.ones(240, dtype=np.int16).tobytes())
        for _ in range(10):
            if speaker.first_write_at is not None:
                break
            await asyncio.sleep(0)

        outdata = np.empty((240, 1), dtype=np.int16)
        speaker._callback(outdata, 240, None, None)

        assert speaker.first_write_at is not None
        assert speaker.first_play_at is not None
        assert speaker.first_play_at >= speaker.first_write_at
        assert np.max(np.abs(outdata)) > 0
        assert speaker.played_audio_ms == 10
    finally:
        feed_task.cancel()
        try:
            await feed_task
        except asyncio.CancelledError:
            pass


async def test_speaker_played_cursor_counts_source_audio_not_padding():
    speaker = SpeakerStream(asyncio.get_running_loop())
    speaker.arm()
    feed_task = asyncio.create_task(speaker.feed())
    try:
        # Only 5 ms of source audio, rendered into a 10 ms callback block.
        await speaker.queue.put(np.ones(120, dtype=np.int16).tobytes())
        for _ in range(10):
            if speaker.first_write_at is not None:
                break
            await asyncio.sleep(0)
        speaker._callback(np.empty((240, 1), dtype=np.int16), 240, None, None)
        assert speaker.played_audio_ms == 5

        speaker.arm()
        assert speaker.played_audio_ms == 0
    finally:
        feed_task.cancel()
        try:
            await feed_task
        except asyncio.CancelledError:
            pass


async def test_speaker_clear_invalidates_producer_blocked_on_backpressure():
    speaker = SpeakerStream(asyncio.get_running_loop())
    for _ in range(speaker.queue.maxsize):
        speaker.queue.put_nowait(b"x")

    stale_put = asyncio.create_task(speaker.queue.put(b"stale"))
    await asyncio.sleep(0)
    assert stale_put.done() is False

    speaker.clear()
    await asyncio.wait_for(stale_put, timeout=0.2)
    feed_task = asyncio.create_task(speaker.feed())
    try:
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        with speaker._lock:
            assert speaker._buffer == b""

        await speaker.queue.put(b"fresh")
        for _ in range(20):
            with speaker._lock:
                if speaker._buffer:
                    break
            await asyncio.sleep(0)
        with speaker._lock:
            assert speaker._buffer == b"fresh"
    finally:
        feed_task.cancel()
        await asyncio.gather(feed_task, return_exceptions=True)


async def test_speaker_device_buffer_is_bounded_under_fast_producer():
    speaker = SpeakerStream(asyncio.get_running_loop())
    feed_task = asyncio.create_task(speaker.feed())
    puts = [
        asyncio.create_task(speaker.queue.put(b"x" * 4096))
        for _ in range(speaker.queue.maxsize + 8)
    ]
    try:
        for _ in range(100):
            with speaker._lock:
                buffered = len(speaker._buffer)
            if buffered >= 24_000:
                break
            await asyncio.sleep(0.001)

        with speaker._lock:
            assert len(speaker._buffer) <= 24_000
        assert any(not task.done() for task in puts)
    finally:
        speaker.clear()
        for task in puts:
            task.cancel()
        feed_task.cancel()
        await asyncio.gather(*puts, feed_task, return_exceptions=True)


async def test_speaker_inactive_device_fails_closed_without_waiting_forever():
    speaker = SpeakerStream(asyncio.get_running_loop())
    speaker._ever_started = True
    speaker._stream_finished.clear()
    speaker._stream = SimpleNamespace(active=False)
    with speaker._lock:
        speaker._buffer.extend(b"unplayed-pcm")

    assert await asyncio.wait_for(speaker.wait_until_done(), timeout=0.1) is False
    with speaker._lock:
        assert speaker._buffer == b""


async def test_speaker_terminal_failure_survives_response_arm():
    speaker = SpeakerStream(asyncio.get_running_loop())
    speaker._ever_started = True
    speaker._stream_finished.clear()
    speaker._stream = SimpleNamespace(active=False)
    with speaker._lock:
        speaker._buffer.extend(b"unplayed-pcm")

    assert await asyncio.wait_for(speaker.wait_until_done(), timeout=0.1) is False

    speaker.arm()

    assert await asyncio.wait_for(speaker.wait_until_done(), timeout=0.1) is False
    with pytest.raises(
        RuntimeError,
        match=r"^Speaker output failed \(stream_inactive\); restart Zemory$",
    ):
        speaker.start()


async def test_speaker_feed_propagates_inactive_device_as_terminal_failure():
    speaker = SpeakerStream(asyncio.get_running_loop())
    speaker._ever_started = True
    speaker._stream_finished.clear()
    speaker._stream = SimpleNamespace(active=False)

    with pytest.raises(
        RuntimeError,
        match=r"^Speaker output failed \(stream_inactive\); restart Zemory$",
    ):
        await asyncio.wait_for(speaker.feed(), timeout=0.2)


async def test_speaker_feed_propagates_finished_device_without_pending_audio():
    speaker = SpeakerStream(asyncio.get_running_loop())
    speaker._ever_started = True
    speaker._stream = SimpleNamespace(active=True)
    speaker._stream_finished.set()

    with pytest.raises(
        RuntimeError,
        match=r"^Speaker output failed \(stream_finished\); restart Zemory$",
    ):
        await asyncio.wait_for(speaker.feed(), timeout=0.2)


async def test_speaker_active_but_stalled_callback_has_bounded_wait(monkeypatch):
    from zemory import audio as audio_module

    monkeypatch.setattr(audio_module, "_PLAYBACK_STALL_TIMEOUT_S", 0.01)
    speaker = SpeakerStream(asyncio.get_running_loop())
    speaker._ever_started = True
    speaker._stream_finished.clear()
    speaker._stream = SimpleNamespace(active=True)
    with speaker._lock:
        speaker._buffer.extend(b"unplayed-pcm")
        speaker._last_playback_progress_at = 0.0

    assert await asyncio.wait_for(speaker.wait_until_done(), timeout=0.1) is False
    with speaker._lock:
        assert speaker._buffer == b""


async def test_speaker_feed_propagates_callback_stall(monkeypatch):
    from zemory import audio as audio_module

    monkeypatch.setattr(audio_module, "_PLAYBACK_STALL_TIMEOUT_S", 0.01)
    speaker = SpeakerStream(asyncio.get_running_loop())
    speaker._ever_started = True
    speaker._stream_finished.clear()
    speaker._stream = SimpleNamespace(active=True)
    with speaker._lock:
        speaker._buffer.extend(b"unplayed-pcm")
        speaker._last_playback_progress_at = 0.0

    with pytest.raises(
        RuntimeError,
        match=r"^Speaker output failed \(callback_stalled\); restart Zemory$",
    ):
        await asyncio.wait_for(speaker.feed(), timeout=0.2)


def test_speaker_stop_attempts_close_after_driver_stop_failure():
    class FailingStopStream:
        def __init__(self) -> None:
            self.closed = False

        def stop(self) -> None:
            raise RuntimeError("driver stop failed")

        def close(self) -> None:
            self.closed = True

    speaker = SpeakerStream(asyncio.new_event_loop())
    stream = FailingStopStream()
    speaker._stream = stream

    with np.testing.assert_raises(RuntimeError):
        speaker.stop()

    assert stream.closed is True
    assert speaker._stream is None
    assert speaker._stream_finished.is_set()


def test_speaker_intentional_stop_is_not_a_terminal_device_failure():
    class StoppableStream:
        active = True

        def stop(self) -> None:
            self.active = False

        def close(self) -> None:
            return None

    speaker = SpeakerStream(asyncio.new_event_loop())
    speaker._ever_started = True
    speaker._stream_finished.clear()
    speaker._stream = StoppableStream()
    with speaker._lock:
        speaker._buffer.extend(b"pending-during-shutdown")

    speaker.stop()

    assert speaker._playback_failure_reason() is None
