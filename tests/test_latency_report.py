"""Latency benchmark report tests."""

from __future__ import annotations

import json

import pytest

from zemory.observability.latency_report import LatencyReport, load_jsonl


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_latency_report_computes_p50_p95_and_passes_thresholds(tmp_path) -> None:
    path = tmp_path / "latency.jsonl"
    _write_jsonl(
        path,
        [
            {"fixture": "ko_short", "total_ms": 420, "interrupt_ms": 80},
            {"fixture": "ko_long_ending", "total_ms": 610, "interrupt_ms": 100},
            {"fixture": "english", "total_ms": 700, "interrupt_ms": 110},
            {"fixture": "noisy_silence", "total_ms": 690, "interrupt_ms": 120},
            {"fixture": "rag_200ms", "total_ms": 650, "interrupt_ms": 130},
        ],
    )

    report = LatencyReport.from_events(load_jsonl(path))

    assert report.turn_count == 5
    assert report.turn_p50_ms == pytest.approx(650)
    assert report.turn_p95_ms == pytest.approx(700)
    assert report.interrupt_p95_ms == pytest.approx(130)
    assert report.passes(turn_p50_ms=700, turn_p95_ms=1200, interrupt_p95_ms=150)


def test_latency_report_fails_when_interrupt_budget_is_exceeded(tmp_path) -> None:
    path = tmp_path / "latency.jsonl"
    _write_jsonl(
        path,
        [
            {"fixture": "barge_in", "total_ms": 500, "interrupt_ms": 80},
            {"fixture": "barge_in", "total_ms": 520, "interrupt_ms": 250},
        ],
    )

    report = LatencyReport.from_events(load_jsonl(path))

    assert not report.passes(
        turn_p50_ms=700,
        turn_p95_ms=1200,
        interrupt_p95_ms=150,
    )


def test_latency_report_requires_turn_samples() -> None:
    with pytest.raises(ValueError, match="turn latency"):
        LatencyReport.from_events([{"fixture": "empty"}])
