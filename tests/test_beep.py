"""Tests for the ready-to-speak beep PCM generator."""

from __future__ import annotations

import numpy as np

from zemory.audio import generate_beep_pcm


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
