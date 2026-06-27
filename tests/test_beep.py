"""Tests for the ready-to-speak beep PCM generator."""

from __future__ import annotations

import asyncio

import numpy as np

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
    finally:
        feed_task.cancel()
        try:
            await feed_task
        except asyncio.CancelledError:
            pass
