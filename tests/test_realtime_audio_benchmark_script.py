"""Realtime audio benchmark script helper tests."""

from __future__ import annotations

import pytest

from scripts.bench_realtime_audio_fixture import (
    _chunk_pcm,
    _event_from_timings,
)


def test_chunk_pcm_splits_pcm_without_losing_tail() -> None:
    assert list(_chunk_pcm(b"abcdefg", chunk_size=3)) == [b"abc", b"def", b"g"]


def test_event_from_timings_records_vad_and_model_audio_segments() -> None:
    event = _event_from_timings(
        fixture="ko_short",
        voice="Yuna",
        eagerness="high",
        turn_detection="semantic_vad",
        mode="semantic_vad",
        audio_end_at=100.0,
        speech_stopped_at=100.25,
        first_audio_at=100.75,
    )

    assert event["fixture"] == "ko_short"
    assert event["voice"] == "Yuna"
    assert event["eagerness"] == "high"
    assert event["turn_detection"] == "semantic_vad"
    assert event["sample_source"] == "macos_say_semantic_vad"
    assert event["total_ms"] == pytest.approx(750.0)
    assert event["vad_wait_ms"] == pytest.approx(250.0)
    assert event["first_audio_after_speech_stopped_ms"] == pytest.approx(500.0)


def test_event_from_timings_excludes_early_cutoff_from_latency_samples() -> None:
    event = _event_from_timings(
        fixture="en_short",
        voice="Samantha",
        eagerness="high",
        turn_detection="server_vad",
        mode="semantic_vad",
        audio_end_at=100.0,
        speech_stopped_at=99.5,
        first_audio_at=99.9,
    )

    assert event["early_cutoff"] is True
    assert event["total_ms"] is None
    assert event["first_tts_byte_ms"] is None
    assert event["first_audio_after_speech_stopped_ms"] == pytest.approx(400.0)


def test_event_from_timings_excludes_speech_stop_before_audio_end() -> None:
    event = _event_from_timings(
        fixture="en_short",
        voice="Samantha",
        eagerness="high",
        turn_detection="server_vad",
        mode="semantic_vad",
        audio_end_at=100.0,
        speech_stopped_at=99.7,
        first_audio_at=100.2,
    )

    assert event["early_cutoff"] is True
    assert event["total_ms"] is None
