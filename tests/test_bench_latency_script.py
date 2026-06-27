"""CLI smoke test for scripts/bench_latency.py."""

from __future__ import annotations

import json
import subprocess
import sys


def test_bench_latency_script_exits_zero_when_thresholds_pass(tmp_path) -> None:
    path = tmp_path / "latency.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"total_ms": 500, "interrupt_ms": 80},
                {"total_ms": 650, "interrupt_ms": 100},
                {"total_ms": 690, "interrupt_ms": 120},
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "scripts/bench_latency.py", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stdout
    assert "turn_p50_ms" in result.stdout


def test_bench_latency_script_exits_nonzero_when_thresholds_fail(tmp_path) -> None:
    path = tmp_path / "latency.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"total_ms": 500, "interrupt_ms": 80},
                {"total_ms": 900, "interrupt_ms": 250},
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/bench_latency.py",
            str(path),
            "--turn-p50-ms",
            "700",
            "--turn-p95-ms",
            "1200",
            "--interrupt-p95-ms",
            "150",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "FAIL" in result.stdout
