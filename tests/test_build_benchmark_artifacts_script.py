"""Benchmark artifact CLI tests."""

from __future__ import annotations

import json
import subprocess
import sys


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
