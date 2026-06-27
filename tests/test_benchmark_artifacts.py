"""Benchmark artifact generation tests."""

from __future__ import annotations

import json

from zemory.observability.benchmark_artifacts import write_benchmark_artifacts


def test_write_benchmark_artifacts_without_transcript_text(tmp_path) -> None:
    events = [
        {
            "event": "turn.complete",
            "turn_id": 1,
            "total_ms": 500,
            "first_tts_byte_ms": 490,
            "api_to_playback_ms": 6,
            "speaker_buffer_ms": 5,
        },
        {
            "event": "turn.complete",
            "turn_id": 2,
            "total_ms": 700,
            "first_tts_byte_ms": 690,
            "api_to_playback_ms": 10,
            "speaker_buffer_ms": 9,
        },
        {"event": "turn.complete", "turn_id": 3, "total_ms": None, "early_cutoff": True},
        {"event": "interrupt.done", "interrupt_ms": 1.3},
    ]

    outputs = write_benchmark_artifacts(
        events,
        out_dir=tmp_path,
        title="local manual session",
        source_note="numeric-only export",
    )

    summary = json.loads(outputs.summary_json.read_text(encoding="utf-8"))
    markdown = outputs.markdown.read_text(encoding="utf-8")
    svg = outputs.svg.read_text(encoding="utf-8")

    assert summary["title"] == "local manual session"
    assert summary["turn_count"] == 2
    assert summary["total_event_count"] == 4
    assert summary["invalid_latency_count"] == 1
    assert summary["early_cutoff_count"] == 1
    assert summary["turn_p50_ms"] == 500
    assert summary["turn_p95_ms"] == 700
    assert summary["turn_representative_max_ms"] == 700
    assert summary["api_to_playback_p50_ms"] == 6
    assert summary["api_to_playback_representative_max_ms"] == 10
    assert summary["speaker_buffer_p50_ms"] == 5
    assert outputs.events_jsonl.read_text(encoding="utf-8").count("\n") == 4
    assert "numeric-only export" in markdown
    assert "turn p50" in markdown
    assert "api to playback p50" in markdown
    assert "speaker buffer p50" in markdown
    assert "early cutoffs" in markdown
    assert "representative max" in markdown
    assert "turn max" not in markdown
    assert "<svg" in svg
    assert "turn max" not in svg
    assert "private" not in markdown
