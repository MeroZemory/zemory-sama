"""Benchmark artifact CLI tests."""

from __future__ import annotations

import json
import subprocess
import sys

from zemory.observability.latency_report import (
    INTERRUPT_CHAIN_METRIC_TARGET,
    LATENCY_SCHEMA_VERSION,
)


def test_build_benchmark_artifacts_script(tmp_path) -> None:
    log = tmp_path / "run.log"
    out = tmp_path / "bench"
    log.write_text(
        "2026 [info] turn.complete total_ms=500.0 first_tts_byte_ms=490.0 "
        "interrupted=False profile=realtime_audio turn_id=1\n"
        "2026 [info] interrupt.done elapsed_ms=1.2\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_benchmark_artifacts.py",
            "--log",
            str(log),
            "--out",
            str(out),
            "--title",
            "test benchmark",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["title"] == "test benchmark"
    assert summary["turn_count"] == 1
    assert (out / "latency.svg").exists()


def test_build_benchmark_artifacts_script_accepts_json_structlog(tmp_path) -> None:
    log = tmp_path / "run-json.log"
    out = tmp_path / "bench-json"
    log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "turn.complete",
                        "total_ms": 500.0,
                        "first_tts_byte_ms": 490.0,
                        "interrupted": False,
                        "profile": "realtime_audio",
                        "turn_id": 1,
                    }
                ),
                json.dumps({"event": "interrupt.done", "elapsed_ms": 1.2}),
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_benchmark_artifacts.py",
            "--log",
            str(log),
            "--out",
            str(out),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    events = [
        json.loads(line)
        for line in (out / "latency-events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1] == {
        "event": "interrupt.done",
        "interrupt_ms": 1.2,
        "schema_version": LATENCY_SCHEMA_VERSION,
        "metric_origin": "runtime_interrupt_bus",
        "metric_target": INTERRUPT_CHAIN_METRIC_TARGET,
    }
