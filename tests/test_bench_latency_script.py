"""CLI smoke test for scripts/bench_latency.py."""

from __future__ import annotations

import json
import subprocess
import sys

from zemory.observability.latency_report import (
    INTERRUPT_CHAIN_METRIC_TARGET,
    INTERRUPT_RELEASE_METRIC_TARGET,
    LATENCY_SCHEMA_VERSION,
    canonical_benchmark_config_hash,
)

TEST_BENCHMARK_CONFIG = {
    "schema_version": LATENCY_SCHEMA_VERSION,
    "model": "test-model",
}
TEST_CONFIG_HASH = canonical_benchmark_config_hash(TEST_BENCHMARK_CONFIG)


def _strict_events(*, slow: bool = False) -> list[dict]:
    turns = [
        {
            "event": "turn.complete",
            "run_id": "run-cli",
            "config_hash": TEST_CONFIG_HASH,
            "benchmark_config": TEST_BENCHMARK_CONFIG,
            "schema_version": LATENCY_SCHEMA_VERSION,
            "turn_id": index,
            "metric_origin": "speaker_probe",
            "metric_target": "device_playback",
            "total_ms": 900 if slow and index >= 4 else 500 + index,
            "api_first_audio_ms": 490 + index,
            "api_to_playback_ms": 6,
            "speaker_buffer_ms": 5,
        }
        for index in range(8)
    ]
    interrupts = [
        {
            "event": "interrupt.done",
            "run_id": "run-cli",
            "config_hash": TEST_CONFIG_HASH,
            "benchmark_config": TEST_BENCHMARK_CONFIG,
            "schema_version": LATENCY_SCHEMA_VERSION,
            "interrupt_id": index,
            "metric_origin": "audible_probe",
            "metric_target": INTERRUPT_RELEASE_METRIC_TARGET,
            "interrupt_ms": 250 if slow and index == 7 else 80 + index,
        }
        for index in range(8)
    ]
    return turns + interrupts


def test_bench_latency_script_exits_zero_when_thresholds_pass(tmp_path) -> None:
    path = tmp_path / "latency.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row) for row in _strict_events()
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
            json.dumps(row) for row in _strict_events(slow=True)
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


def test_bench_latency_script_rejects_previous_false_pass_shape(tmp_path) -> None:
    path = tmp_path / "false-pass.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"event": "turn.complete", "total_ms": 1},
                {
                    "event": "turn.complete",
                    "total_ms": None,
                    "early_cutoff": True,
                },
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

    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["status"] == "FAIL"
    assert any(
        reason.startswith("insufficient_turn_samples:")
        for reason in payload["failure_reasons"]
    )
    assert any(
        reason.startswith("insufficient_interrupt_samples:")
        for reason in payload["failure_reasons"]
    )
    assert any(
        reason.startswith("early_cutoff_events:")
        for reason in payload["failure_reasons"]
    )


def test_bench_latency_script_can_explicitly_relax_sample_minimums(tmp_path) -> None:
    path = tmp_path / "legacy-small-probe.jsonl"
    path.write_text(json.dumps({"total_ms": 500}) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/bench_latency.py",
            str(path),
            "--min-turn-samples",
            "1",
            "--min-interrupt-samples",
            "0",
            "--allow-legacy-provenance",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0, result.stderr
    assert payload["status"] == "PASS"
    assert payload["gate"]["min_turn_samples"] == 1
    assert payload["gate"]["min_interrupt_samples"] == 0
    assert payload["gate"]["require_provenance"] is False


def test_bench_latency_script_allows_strict_turn_only_gate(tmp_path) -> None:
    path = tmp_path / "strict-turn-only.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in _strict_events()[:8]) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/bench_latency.py",
            str(path),
            "--min-interrupt-samples",
            "0",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0, payload["failure_reasons"]
    assert payload["status"] == "PASS"
    assert payload["interrupt_count"] == 0
    assert payload["gate"]["require_provenance"] is True
    assert payload["gate"]["interrupt_metric_target"] is None
    assert not any(
        reason.startswith("unexpected_interrupt_metric_target:")
        for reason in payload["failure_reasons"]
    )


def test_turn_only_minimum_still_validates_present_interrupt_target(
    tmp_path,
) -> None:
    events = _strict_events()[:8]
    events.append(
        {
            "event": "interrupt.done",
            "run_id": "run-cli",
            "config_hash": TEST_CONFIG_HASH,
            "benchmark_config": TEST_BENCHMARK_CONFIG,
            "schema_version": LATENCY_SCHEMA_VERSION,
            "interrupt_id": 0,
            "metric_origin": "chain_probe",
            "metric_target": INTERRUPT_CHAIN_METRIC_TARGET,
            "interrupt_ms": 80,
        }
    )
    path = tmp_path / "wrong-present-interrupt-target.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in events) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/bench_latency.py",
            str(path),
            "--min-interrupt-samples",
            "0",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["status"] == "FAIL"
    assert payload["interrupt_count"] == 1
    assert (
        payload["gate"]["interrupt_metric_target"]
        == INTERRUPT_RELEASE_METRIC_TARGET
    )
    assert any(
        reason.startswith("unexpected_interrupt_metric_target:")
        for reason in payload["failure_reasons"]
    )
