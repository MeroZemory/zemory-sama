"""Latency benchmark report and strict release-gate tests."""

from __future__ import annotations

import hashlib
import json

import pytest

from zemory.observability.latency_report import (
    INTERRUPT_CHAIN_METRIC_TARGET,
    INTERRUPT_RELEASE_METRIC_TARGET,
    LATENCY_SCHEMA_VERSION,
    LatencyGate,
    LatencyReport,
    canonical_benchmark_config_hash,
    load_jsonl,
    parse_structlog_latency_events,
    sanitize_latency_event,
)

TEST_BENCHMARK_CONFIG = {
    "schema_version": LATENCY_SCHEMA_VERSION,
    "model": "test-model",
    "session_config": {"max_output_tokens": 512},
}
TEST_CONFIG_HASH = canonical_benchmark_config_hash(TEST_BENCHMARK_CONFIG)


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _turn(index: int, total_ms: float = 500, **overrides) -> dict:
    event = {
        "event": "turn.complete",
        "run_id": "run-1",
        "config_hash": TEST_CONFIG_HASH,
        "benchmark_config": TEST_BENCHMARK_CONFIG,
        "schema_version": LATENCY_SCHEMA_VERSION,
        "turn_id": index,
        "profile": "realtime_audio",
        "sample_source": "synthetic_pcm",
        "metric_origin": "speaker_callback_probe",
        "metric_target": "device_playback",
        "model": "test-model",
        "turn_detection": "server_vad",
        "server_vad_threshold": 0.5,
        "total_ms": total_ms,
        "api_first_audio_ms": None if total_ms is None else max(0, total_ms - 10),
        "api_to_playback_ms": 6,
        "speaker_buffer_ms": 5,
        "early_cutoff": False,
        "interrupted": False,
    }
    event.update(overrides)
    return event


def _interrupt(index: int, interrupt_ms: float = 80, **overrides) -> dict:
    event = {
        "event": "interrupt.done",
        "run_id": "run-1",
        "config_hash": TEST_CONFIG_HASH,
        "benchmark_config": TEST_BENCHMARK_CONFIG,
        "schema_version": LATENCY_SCHEMA_VERSION,
        "interrupt_id": index,
        "metric_origin": "audible_silence_probe",
        "metric_target": INTERRUPT_RELEASE_METRIC_TARGET,
        "interrupt_ms": interrupt_ms,
    }
    event.update(overrides)
    return event


def _gate() -> LatencyGate:
    return LatencyGate(
        turn_p50_ms=700,
        turn_p95_ms=1200,
        interrupt_p95_ms=150,
    )


def test_sanitize_latency_event_keeps_safe_provenance_without_prompt_text() -> None:
    event = _turn(0)
    event["transcript"] = "private transcript"
    event["benchmark_config"] = {
        "model": "test-model",
        "fixture_corpus_hash": "fixture-hash",
        "private_note": "do not publish",
        "session_config": {
            "instructions": "private system prompt",
            "instructions_sha256": "caller-controlled-wrong-digest",
            "max_output_tokens": 512,
            "api_key": "secret-key",
            "audio": {
                "input": {
                    "authorization": "Bearer nested-secret",
                    "transcription": {"prompt": "private nested prompt"},
                }
            },
        },
    }

    sanitized = sanitize_latency_event(event)

    assert "transcript" not in sanitized
    benchmark = sanitized["benchmark_config"]
    assert "private_note" not in benchmark
    session = benchmark["session_config"]
    assert "instructions" not in session
    assert "api_key" not in session
    assert session["max_output_tokens"] == 512
    assert session["instructions_sha256"] == hashlib.sha256(
        b"private system prompt"
    ).hexdigest()
    nested_input = session["audio"]["input"]
    assert "authorization" not in nested_input
    transcription = nested_input["transcription"]
    assert "prompt" not in transcription
    assert transcription["prompt_sha256"] == hashlib.sha256(
        b"private nested prompt"
    ).hexdigest()


def test_release_gate_rejects_missing_malformed_or_mismatched_config_hash() -> None:
    missing = [_turn(index) for index in range(8)]
    malformed = [_turn(index, config_hash="not-a-sha256") for index in range(8)]
    mismatched = [_turn(index, config_hash="0" * 64) for index in range(8)]
    for event in missing:
        event.pop("benchmark_config")

    missing_decision = LatencyReport.from_events(missing).evaluate_gate(
        LatencyGate(
            turn_p50_ms=700,
            turn_p95_ms=1200,
            interrupt_p95_ms=150,
            min_interrupt_samples=0,
            interrupt_metric_target=None,
        )
    )
    malformed_decision = LatencyReport.from_events(malformed).evaluate_gate(
        LatencyGate(
            turn_p50_ms=700,
            turn_p95_ms=1200,
            interrupt_p95_ms=150,
            min_interrupt_samples=0,
            interrupt_metric_target=None,
        )
    )
    mismatched_decision = LatencyReport.from_events(mismatched).evaluate_gate(
        LatencyGate(
            turn_p50_ms=700,
            turn_p95_ms=1200,
            interrupt_p95_ms=150,
            min_interrupt_samples=0,
            interrupt_metric_target=None,
        )
    )

    assert any(
        reason.startswith("invalid_turn_benchmark_config:")
        for reason in missing_decision.failure_reasons
    )
    assert any(
        reason.startswith("invalid_turn_provenance:")
        for reason in malformed_decision.failure_reasons
    )
    assert any(
        reason.startswith("invalid_turn_benchmark_config:")
        for reason in mismatched_decision.failure_reasons
    )


def test_release_gate_allows_missing_benchmark_config_only_in_legacy_mode() -> None:
    events = [_turn(index) for index in range(8)]
    for event in events:
        event.pop("benchmark_config")

    report = LatencyReport.from_events(events)
    decision = report.evaluate_gate(
        LatencyGate(
            turn_p50_ms=700,
            turn_p95_ms=1200,
            interrupt_p95_ms=150,
            min_interrupt_samples=0,
            interrupt_metric_target=None,
            require_provenance=False,
        )
    )

    assert decision.passed


def test_legacy_mode_still_rejects_a_present_tampered_benchmark_config() -> None:
    events = [_turn(index, config_hash="0" * 64) for index in range(8)]

    decision = LatencyReport.from_events(events).evaluate_gate(
        LatencyGate(
            turn_p50_ms=700,
            turn_p95_ms=1200,
            interrupt_p95_ms=150,
            min_interrupt_samples=0,
            interrupt_metric_target=None,
            require_provenance=False,
        )
    )

    assert not decision.passed
    assert any(
        reason.startswith("invalid_turn_benchmark_config:")
        for reason in decision.failure_reasons
    )


def test_latency_report_computes_percentiles_and_passes_strict_gate(tmp_path) -> None:
    totals = [420, 610, 700, 690, 650, 660, 680, 620]
    interrupts = [80, 100, 110, 120, 130, 135, 140, 145]
    path = tmp_path / "latency.jsonl"
    _write_jsonl(
        path,
        [_turn(index, total) for index, total in enumerate(totals)]
        + [_interrupt(index, value) for index, value in enumerate(interrupts)],
    )

    report = LatencyReport.from_events(load_jsonl(path))

    assert report.turn_count == 8
    assert report.turn_min_ms == pytest.approx(420)
    assert report.turn_p50_ms == pytest.approx(650)
    assert report.turn_p90_ms == pytest.approx(700)
    assert report.turn_p95_ms == pytest.approx(700)
    assert report.turn_representative_max_ms == pytest.approx(700)
    assert report.turn_max_ms == pytest.approx(700)
    assert report.turn_extreme_outlier_count == 0
    assert report.interrupt_p95_ms == pytest.approx(145)
    assert report.evaluate_gate(_gate()).passed
    assert report.passes(turn_p50_ms=700, turn_p95_ms=1200, interrupt_p95_ms=150)


def test_latency_report_fails_when_interrupt_budget_is_exceeded() -> None:
    report = LatencyReport.from_events(
        [_turn(index, 500 + index) for index in range(8)]
        + [_interrupt(index, 250 if index == 7 else 80) for index in range(8)]
    )

    decision = report.evaluate_gate(_gate())

    assert not decision.passed
    assert any(reason.startswith("interrupt_p95_exceeded:") for reason in decision.failure_reasons)


def test_latency_report_requires_turn_samples() -> None:
    with pytest.raises(ValueError, match="turn latency"):
        LatencyReport.from_events([{"fixture": "empty"}])


def test_latency_report_keeps_extreme_max_out_of_representative_tail() -> None:
    report = LatencyReport.from_events(
        [{"total_ms": value} for value in [500, 520, 540, 560, 580, 600, 620, 5000]]
    )

    assert report.turn_max_ms == pytest.approx(5000)
    assert report.turn_representative_max_ms == pytest.approx(620)
    assert report.turn_extreme_outlier_count == 1


def test_release_gate_rejects_invalid_early_cutoff_and_missing_interrupt() -> None:
    report = LatencyReport.from_events(
        [_turn(index, 500 + index) for index in range(8)]
        + [_turn(8, total_ms=None, early_cutoff=True)]
    )

    decision = report.evaluate_gate(_gate())

    assert not decision.passed
    assert any(reason.startswith("insufficient_interrupt_samples:") for reason in decision.failure_reasons)
    assert any(reason.startswith("invalid_turn_events:") for reason in decision.failure_reasons)
    assert any(reason.startswith("early_cutoff_events:") for reason in decision.failure_reasons)


def test_release_gate_rejects_nan_negative_and_overflow_values() -> None:
    report = LatencyReport.from_events(
        [_turn(index, 500 + index) for index in range(8)]
        + [
            _turn(8, total_ms=float("nan")),
            _turn(9, total_ms=10**10000),
            _interrupt(0, 80),
            _interrupt(1, -1),
        ]
    )

    decision = report.evaluate_gate(_gate())

    assert report.invalid_turn_count == 2
    assert report.invalid_interrupt_count == 1
    assert any(reason.startswith("invalid_turn_events:") for reason in decision.failure_reasons)
    assert any(reason.startswith("invalid_interrupt_events:") for reason in decision.failure_reasons)


@pytest.mark.parametrize("marker", ["yes", 1, None])
def test_release_gate_rejects_non_boolean_early_cutoff(marker) -> None:
    events = [_turn(index, 500 + index) for index in range(8)]
    events[0]["early_cutoff"] = marker
    report = LatencyReport.from_events(events + [_interrupt(index) for index in range(8)])

    decision = report.evaluate_gate(_gate())

    assert report.early_cutoff_count == 0
    assert any(reason.startswith("invalid_turn_events:") for reason in decision.failure_reasons)


def test_release_gate_rejects_missing_provenance_and_unsupported_schema() -> None:
    report = LatencyReport.from_events(
        [{"total_ms": 500 + index, "interrupt_ms": 80 + index} for index in range(8)]
    )

    decision = report.evaluate_gate(_gate())

    assert any(reason.startswith("invalid_turn_provenance:") for reason in decision.failure_reasons)
    assert any(reason.startswith("unsupported_turn_schema:") for reason in decision.failure_reasons)
    assert any(reason.startswith("missing_turn_identities:") for reason in decision.failure_reasons)
    assert any(reason.startswith("invalid_interrupt_provenance:") for reason in decision.failure_reasons)


def test_release_gate_rejects_metric_target_without_required_measurements() -> None:
    turns = [_turn(index) for index in range(8)]
    for event in turns:
        event.pop("api_first_audio_ms")
        event.pop("api_to_playback_ms")
        event.pop("speaker_buffer_ms")
    report = LatencyReport.from_events(turns + [_interrupt(index) for index in range(8)])

    decision = report.evaluate_gate(_gate())

    assert any(reason.startswith("unsupported_turn_schema:") for reason in decision.failure_reasons)


def test_release_gate_rejects_duplicate_sample_identities() -> None:
    report = LatencyReport.from_events(
        [_turn(1) for _ in range(8)] + [_interrupt(1) for _ in range(8)]
    )

    decision = report.evaluate_gate(_gate())

    assert report.duplicate_turn_count == 7
    assert report.duplicate_interrupt_count == 7
    assert any(reason.startswith("duplicate_turn_samples:") for reason in decision.failure_reasons)
    assert any(reason.startswith("duplicate_interrupt_samples:") for reason in decision.failure_reasons)


@pytest.mark.parametrize(
    ("field", "values", "reason_prefix"),
    [
        ("config_hash", ["config-1", "config-2"], "mixed_turn_metric_origins:"),
        ("schema_version", [LATENCY_SCHEMA_VERSION, "other"], "mixed_turn_metric_schemas:"),
        ("server_vad_threshold", [0.5, 0.7], "mixed_turn_metric_origins:"),
        ("play_output", [True, False], "mixed_turn_metric_origins:"),
    ],
)
def test_release_gate_rejects_mixed_cohort_or_schema(
    field: str,
    values: list,
    reason_prefix: str,
) -> None:
    turns = [_turn(index, **{field: values[index % 2]}) for index in range(8)]
    report = LatencyReport.from_events(turns + [_interrupt(index) for index in range(8)])

    decision = report.evaluate_gate(_gate())

    assert any(reason.startswith(reason_prefix) for reason in decision.failure_reasons)


def test_interrupt_chain_execution_is_not_audible_release_metric() -> None:
    interrupts = [
        _interrupt(
            index,
            1.2,
            metric_origin="runtime_interrupt_bus",
            metric_target=INTERRUPT_CHAIN_METRIC_TARGET,
        )
        for index in range(8)
    ]
    report = LatencyReport.from_events(
        [_turn(index) for index in range(8)] + interrupts
    )

    decision = report.evaluate_gate(_gate())

    assert any(
        reason.startswith("unexpected_interrupt_metric_target:")
        for reason in decision.failure_reasons
    )


def test_release_gate_rejects_interrupt_samples_from_another_run() -> None:
    report = LatencyReport.from_events(
        [_turn(index) for index in range(8)]
        + [
            _interrupt(index, run_id="run-2", config_hash="config-2")
            for index in range(8)
        ]
    )

    decision = report.evaluate_gate(_gate())

    assert any(
        reason.startswith("incompatible_turn_interrupt_cohorts:")
        for reason in decision.failure_reasons
    )


def test_release_gate_rejects_too_few_turn_samples() -> None:
    report = LatencyReport.from_events([_turn(1), _interrupt(1)])

    assert not report.passes(
        turn_p50_ms=700,
        turn_p95_ms=1200,
        interrupt_p95_ms=150,
    )


def test_latency_gate_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="turn_p50_ms"):
        LatencyGate(
            turn_p50_ms=float("nan"),
            turn_p95_ms=1200,
            interrupt_p95_ms=150,
        )
    with pytest.raises(ValueError, match="min_turn_samples"):
        LatencyGate(
            turn_p50_ms=700,
            turn_p95_ms=1200,
            interrupt_p95_ms=150,
            min_turn_samples=-1,
        )
    with pytest.raises(ValueError, match="turn_metric_target"):
        LatencyGate(
            turn_p50_ms=700,
            turn_p95_ms=1200,
            interrupt_p95_ms=150,
            turn_metric_target="",
        )


def test_parse_structlog_latency_events_labels_runtime_metric_origins() -> None:
    log = """
assistant text that should not be published
2026 [info] turn.complete first_tts_byte_ms=704.9 interrupted=False profile=realtime_audio total_ms=705.1 turn_id=1
private user transcript
2026 [info] interrupt.done elapsed_ms=1.3
"""

    events = parse_structlog_latency_events(log)

    assert events[0]["metric_origin"] == "runtime_turn_complete"
    assert events[0]["metric_target"] == "device_playback"
    assert events[0]["schema_version"] == LATENCY_SCHEMA_VERSION
    assert events[1] == {
        "event": "interrupt.done",
        "interrupt_ms": 1.3,
        "schema_version": LATENCY_SCHEMA_VERSION,
        "metric_origin": "runtime_interrupt_bus",
        "metric_target": INTERRUPT_CHAIN_METRIC_TARGET,
    }


def test_parse_structlog_latency_events_supports_json_renderer() -> None:
    log = "\n".join(
        [
            json.dumps(
                {
                    "event": "turn.complete",
                    "turn_id": 7,
                    "profile": "realtime_audio",
                    "total_ms": 612.3,
                    "first_tts_byte_ms": 600.1,
                    "early_cutoff": False,
                    "interrupted": False,
                    "transcript": "must not be exported",
                }
            ),
            json.dumps({"event": "interrupt.done", "elapsed_ms": 12.4}),
            json.dumps({"event": "user.text", "text": "private transcript"}),
        ]
    )

    events = parse_structlog_latency_events(log)

    assert events[0]["turn_id"] == 7
    assert events[0]["total_ms"] == 612.3
    assert events[0]["metric_origin"] == "runtime_turn_complete"
    assert "transcript" not in events[0]
    assert events[1]["interrupt_ms"] == 12.4
    assert events[1]["metric_target"] == INTERRUPT_CHAIN_METRIC_TARGET


def test_load_jsonl_rejects_non_object_rows(tmp_path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_jsonl(path)
