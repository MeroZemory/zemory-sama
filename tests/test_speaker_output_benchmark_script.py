"""Speaker output benchmark artifact tests."""

from __future__ import annotations

import json
from xml.etree import ElementTree as ET

import pytest

from scripts.bench_speaker_output import (
    _make_summary,
    _write_speaker_artifacts,
)


def test_make_summary_separates_callback_outlier() -> None:
    events = [
        {
            "event": "speaker.playback",
            "total_ms": 4.0,
            "queue_to_buffer_ms": 1.0,
            "speaker_buffer_ms": 3.0,
        },
        {
            "event": "speaker.playback",
            "total_ms": 4.2,
            "queue_to_buffer_ms": 1.1,
            "speaker_buffer_ms": 3.1,
        },
        {
            "event": "speaker.playback",
            "total_ms": 4.4,
            "queue_to_buffer_ms": 1.2,
            "speaker_buffer_ms": 3.2,
        },
        {
            "event": "speaker.playback",
            "total_ms": 4.6,
            "queue_to_buffer_ms": 1.3,
            "speaker_buffer_ms": 3.3,
        },
        {
            "event": "speaker.playback",
            "total_ms": 20.0,
            "queue_to_buffer_ms": 1.0,
            "speaker_buffer_ms": 19.0,
        },
    ]

    summary = _make_summary(
        events,
        title="speaker bench",
        source_note="numeric-only",
        output_block_ms=10,
    )

    assert summary["sample_count"] == 5
    assert summary["queue_to_play_p50_ms"] == pytest.approx(4.4)
    assert summary["queue_to_play_representative_max_ms"] == pytest.approx(4.6)
    assert summary["queue_to_play_max_ms"] == pytest.approx(20.0)
    assert summary["queue_to_play_extreme_outlier_count"] == 1
    assert summary["speaker_buffer_p50_ms"] == pytest.approx(3.2)


def test_write_speaker_artifacts_without_transcript_text(tmp_path) -> None:
    events = [
        {
            "event": "speaker.playback",
            "trial": 1,
            "total_ms": 4.0,
            "queue_to_buffer_ms": 0.5,
            "speaker_buffer_ms": 3.5,
        },
        {
            "event": "speaker.playback",
            "trial": 2,
            "total_ms": 4.4,
            "queue_to_buffer_ms": 0.6,
            "speaker_buffer_ms": 3.8,
        },
    ]

    _write_speaker_artifacts(
        events,
        out_dir=tmp_path,
        title="speaker bench",
        source_note="numeric-only",
        output_block_ms=10,
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    jsonl = (tmp_path / "latency-events.jsonl").read_text(encoding="utf-8")
    ET.parse(tmp_path / "latency.svg")

    assert summary["title"] == "speaker bench"
    assert summary["sample_count"] == 2
    assert summary["output_block_ms"] == 10
    assert "queue to first playback callback p50" in readme
    assert "speaker buffer p50" in readme
    assert jsonl.count("\n") == 2
    assert "private" not in readme
    assert "transcript" not in jsonl.lower()
