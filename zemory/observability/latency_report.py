"""Latency benchmark parsing, reporting, and release-gate utilities."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LATENCY_SCHEMA_VERSION = "zemory.latency.v1"
TURN_RELEASE_METRIC_TARGET = "device_playback"
INTERRUPT_RELEASE_METRIC_TARGET = "server_speech_started_to_audible_silence"
INTERRUPT_CHAIN_METRIC_TARGET = "interrupt_chain_execution"
DEFAULT_MIN_TURN_SAMPLES = 8
DEFAULT_MIN_INTERRUPT_SAMPLES = 8

TURN_LATENCY_FIELDS = (
    "total_ms",
    "first_tts_byte_ms",
    "api_first_audio_ms",
    "api_to_playback_ms",
    "speaker_buffer_ms",
    "vad_wait_ms",
    "first_audio_after_speech_stopped_ms",
)
INTERRUPT_LATENCY_FIELDS = ("interrupt_ms",)
PROVENANCE_FIELDS = (
    "run_id",
    "config_hash",
    "schema_version",
    "profile",
    "sample_source",
    "metric_origin",
    "metric_target",
    "model",
    "turn_detection",
    "eagerness",
    "server_vad_threshold",
    "server_vad_silence_ms",
    "input_chunk_ms",
    "local_endpoint_required_misses",
    "mode",
    "play_output",
    "measure_interrupt",
    "response_length",
    "runtime_version",
    "openai_sdk_version",
)
REQUIRED_PROVENANCE_FIELDS = (
    "run_id",
    "config_hash",
    "schema_version",
    "metric_origin",
    "metric_target",
)
TURN_TARGET_REQUIRED_FIELDS = {
    "api_first_audio": ("api_first_audio_ms",),
    "device_playback": (
        "api_first_audio_ms",
        "api_to_playback_ms",
        "speaker_buffer_ms",
    ),
}
INTERRUPT_TARGET_REQUIRED_FIELDS = {
    INTERRUPT_RELEASE_METRIC_TARGET: (),
    INTERRUPT_CHAIN_METRIC_TARGET: (),
}
SAFE_EVENT_FIELDS = {
    "event",
    "run_id",
    "config_hash",
    "schema_version",
    "turn_id",
    "interrupt_id",
    "fixture",
    "trial",
    "voice",
    "profile",
    "sample_source",
    "metric_origin",
    "metric_target",
    "model",
    "turn_detection",
    "eagerness",
    "server_vad_threshold",
    "server_vad_silence_ms",
    "input_chunk_ms",
    "local_endpoint_required_misses",
    "response_length",
    "mode",
    "play_output",
    "measure_interrupt",
    "runtime_version",
    "openai_sdk_version",
    "openai_base_url_kind",
    "openai_base_url_sha256",
    # The live benchmark producer removes instruction text before attaching
    # this object; retaining it makes config_hash independently auditable.
    "benchmark_config",
    "interrupted",
    "early_cutoff",
    "invalid_reason",
    *TURN_LATENCY_FIELDS,
    *INTERRUPT_LATENCY_FIELDS,
}
PUBLIC_BENCHMARK_CONFIG_FIELDS = {
    "schema_version",
    "runtime_version",
    "openai_sdk_version",
    "openai_base_url_kind",
    "openai_base_url_sha256",
    "model",
    "response_length",
    "mode",
    "turn_detection",
    "eagerness",
    "server_vad_threshold",
    "server_vad_silence_ms",
    "local_endpoint_required_misses",
    "input_chunk_ms",
    "play_output",
    "measure_interrupt",
    "trials",
    "timeout_s",
    "samples",
    "fixture_corpus_hash",
    "fixture_pcm_sha256",
    "session_config",
    "platform",
    "git_commit",
    "git_dirty",
    "git_diff_sha256",
    "source_tree_sha256",
}

_SECRET_CONFIG_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "client_secret",
    "password",
    "refresh_token",
    "secret",
    "token",
}
_PROMPT_CONFIG_KEYS = {"instructions", "prompt", "system_prompt"}
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")


@dataclass(frozen=True)
class LatencyGate:
    """Explicit release criteria for a latency benchmark sample set."""

    turn_p50_ms: float
    turn_p95_ms: float
    interrupt_p95_ms: float
    min_turn_samples: int = DEFAULT_MIN_TURN_SAMPLES
    min_interrupt_samples: int = DEFAULT_MIN_INTERRUPT_SAMPLES
    turn_metric_target: str | None = TURN_RELEASE_METRIC_TARGET
    interrupt_metric_target: str | None = INTERRUPT_RELEASE_METRIC_TARGET
    reject_invalid: bool = True
    reject_early_cutoff: bool = True
    reject_mixed_metric_origins: bool = True
    reject_mixed_schemas: bool = True
    require_provenance: bool = True
    reject_duplicate_samples: bool = True

    def __post_init__(self) -> None:
        for field_name in ("turn_p50_ms", "turn_p95_ms", "interrupt_p95_ms"):
            value = getattr(self, field_name)
            if not _is_finite_non_negative(value):
                raise ValueError(f"{field_name} must be finite and non-negative")
        for field_name in ("min_turn_samples", "min_interrupt_samples"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        for field_name in ("turn_metric_target", "interrupt_metric_target"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be a non-empty string or None")

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_p50_ms": self.turn_p50_ms,
            "turn_p95_ms": self.turn_p95_ms,
            "interrupt_p95_ms": self.interrupt_p95_ms,
            "min_turn_samples": self.min_turn_samples,
            "min_interrupt_samples": self.min_interrupt_samples,
            "turn_metric_target": self.turn_metric_target,
            "interrupt_metric_target": self.interrupt_metric_target,
            "reject_invalid": self.reject_invalid,
            "reject_early_cutoff": self.reject_early_cutoff,
            "reject_mixed_metric_origins": self.reject_mixed_metric_origins,
            "reject_mixed_schemas": self.reject_mixed_schemas,
            "require_provenance": self.require_provenance,
            "reject_duplicate_samples": self.reject_duplicate_samples,
        }


@dataclass(frozen=True)
class LatencyGateResult:
    """A release-gate decision with stable, machine-readable failure prefixes."""

    failure_reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failure_reasons

    def as_dict(self) -> dict[str, bool | list[str]]:
        return {
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True)
class LatencyReport:
    turn_count: int
    turn_min_ms: float
    turn_mean_ms: float
    turn_p50_ms: float
    turn_p90_ms: float
    turn_p95_ms: float
    turn_representative_max_ms: float
    turn_max_ms: float
    turn_extreme_outlier_count: int
    interrupt_count: int
    interrupt_p95_ms: float | None
    turn_event_count: int
    invalid_turn_count: int
    early_cutoff_count: int
    interrupt_event_count: int
    invalid_interrupt_count: int
    turn_provenance_error_count: int
    interrupt_provenance_error_count: int
    turn_benchmark_config_error_count: int
    interrupt_benchmark_config_error_count: int
    turn_present_benchmark_config_integrity_error_count: int
    interrupt_present_benchmark_config_integrity_error_count: int
    turn_schema_error_count: int
    interrupt_schema_error_count: int
    missing_turn_identity_count: int
    missing_interrupt_identity_count: int
    duplicate_turn_count: int
    duplicate_interrupt_count: int
    metric_origins: tuple[str, ...]
    metric_schemas: tuple[str, ...]
    interrupt_metric_origins: tuple[str, ...]
    interrupt_metric_schemas: tuple[str, ...]
    turn_metric_targets: tuple[str, ...]
    interrupt_metric_targets: tuple[str, ...]
    turn_cohort_keys: tuple[str, ...]
    interrupt_cohort_keys: tuple[str, ...]

    @classmethod
    def from_events(cls, events: list[dict[str, Any]]) -> LatencyReport:
        if any(not isinstance(event, dict) for event in events):
            raise ValueError("Latency events must be objects")
        turn_events = [event for event in events if _is_turn_event(event)]
        interrupt_events = [event for event in events if _is_interrupt_event(event)]

        turn_latencies: list[float] = []
        invalid_turn_count = 0
        for event in turn_events:
            total_ms = _valid_latency_or_none(event.get("total_ms"))
            if (
                total_ms is None
                or _has_invalid_optional_latency(event, TURN_LATENCY_FIELDS[1:])
                or _has_invalid_boolean_marker(event, "early_cutoff")
                or _has_invalid_boolean_marker(event, "interrupted")
            ):
                invalid_turn_count += 1
            if total_ms is not None:
                turn_latencies.append(total_ms)

        if not turn_latencies:
            raise ValueError("No valid turn latency samples found")

        interrupt_latencies: list[float] = []
        invalid_interrupt_count = 0
        for event in interrupt_events:
            interrupt_ms = _valid_latency_or_none(event.get("interrupt_ms"))
            if interrupt_ms is None:
                invalid_interrupt_count += 1
            else:
                interrupt_latencies.append(interrupt_ms)

        turn_identities = [_sample_identity(event, "turn") for event in turn_events]
        interrupt_identities = [
            _sample_identity(event, "interrupt") for event in interrupt_events
        ]
        turn_latencies_sorted = sorted(turn_latencies)
        representative_latencies = _exclude_extreme_outliers(turn_latencies_sorted)
        return cls(
            turn_count=len(turn_latencies),
            turn_min_ms=turn_latencies_sorted[0],
            turn_mean_ms=sum(turn_latencies) / len(turn_latencies),
            turn_p50_ms=_percentile_nearest_rank(turn_latencies, 50),
            turn_p90_ms=_percentile_nearest_rank(turn_latencies, 90),
            turn_p95_ms=_percentile_nearest_rank(turn_latencies, 95),
            turn_representative_max_ms=representative_latencies[-1],
            turn_max_ms=turn_latencies_sorted[-1],
            turn_extreme_outlier_count=(
                len(turn_latencies_sorted) - len(representative_latencies)
            ),
            interrupt_count=len(interrupt_latencies),
            interrupt_p95_ms=(
                _percentile_nearest_rank(interrupt_latencies, 95)
                if interrupt_latencies
                else None
            ),
            turn_event_count=len(turn_events),
            invalid_turn_count=invalid_turn_count,
            early_cutoff_count=sum(
                event.get("early_cutoff") is True for event in turn_events
            ),
            interrupt_event_count=len(interrupt_events),
            invalid_interrupt_count=invalid_interrupt_count,
            turn_provenance_error_count=sum(
                _has_provenance_error(event) for event in turn_events
            ),
            interrupt_provenance_error_count=sum(
                _has_provenance_error(event) for event in interrupt_events
            ),
            turn_benchmark_config_error_count=sum(
                _has_benchmark_config_error(event) for event in turn_events
            ),
            interrupt_benchmark_config_error_count=sum(
                _has_benchmark_config_error(event) for event in interrupt_events
            ),
            turn_present_benchmark_config_integrity_error_count=sum(
                _has_present_benchmark_config_integrity_error(event)
                for event in turn_events
            ),
            interrupt_present_benchmark_config_integrity_error_count=sum(
                _has_present_benchmark_config_integrity_error(event)
                for event in interrupt_events
            ),
            turn_schema_error_count=sum(
                not _has_supported_metric_schema(event, "turn") for event in turn_events
            ),
            interrupt_schema_error_count=sum(
                not _has_supported_metric_schema(event, "interrupt")
                for event in interrupt_events
            ),
            missing_turn_identity_count=sum(identity is None for identity in turn_identities),
            missing_interrupt_identity_count=sum(
                identity is None for identity in interrupt_identities
            ),
            duplicate_turn_count=_duplicate_count(turn_identities),
            duplicate_interrupt_count=_duplicate_count(interrupt_identities),
            metric_origins=tuple(sorted({_metric_origin(event) for event in turn_events})),
            metric_schemas=tuple(
                sorted({_metric_schema(event, "turn") for event in turn_events})
            ),
            interrupt_metric_origins=tuple(
                sorted({_metric_origin(event) for event in interrupt_events})
            ),
            interrupt_metric_schemas=tuple(
                sorted({_metric_schema(event, "interrupt") for event in interrupt_events})
            ),
            turn_metric_targets=tuple(
                sorted({_field_token(event, "metric_target") for event in turn_events})
            ),
            interrupt_metric_targets=tuple(
                sorted({_field_token(event, "metric_target") for event in interrupt_events})
            ),
            turn_cohort_keys=tuple(
                sorted({_cohort_key(event) for event in turn_events})
            ),
            interrupt_cohort_keys=tuple(
                sorted({_cohort_key(event) for event in interrupt_events})
            ),
        )

    def evaluate_gate(self, gate: LatencyGate) -> LatencyGateResult:
        failures: list[str] = []
        if self.turn_count < gate.min_turn_samples:
            failures.append(
                "insufficient_turn_samples: "
                f"expected at least {gate.min_turn_samples}, got {self.turn_count}"
            )
        if self.interrupt_count < gate.min_interrupt_samples:
            failures.append(
                "insufficient_interrupt_samples: "
                f"expected at least {gate.min_interrupt_samples}, got {self.interrupt_count}"
            )
        if gate.reject_invalid and self.invalid_turn_count:
            failures.append(
                f"invalid_turn_events: found {self.invalid_turn_count} invalid event(s)"
            )
        if gate.reject_invalid and self.invalid_interrupt_count:
            failures.append(
                "invalid_interrupt_events: "
                f"found {self.invalid_interrupt_count} invalid event(s)"
            )
        if gate.reject_early_cutoff and self.early_cutoff_count:
            failures.append(
                f"early_cutoff_events: found {self.early_cutoff_count} early cutoff(s)"
            )
        if gate.require_provenance:
            failures.extend(self._provenance_failures(gate))
        else:
            for kind in ("turn", "interrupt"):
                integrity_errors = getattr(
                    self,
                    f"{kind}_present_benchmark_config_integrity_error_count",
                )
                if integrity_errors:
                    failures.append(
                        f"invalid_{kind}_benchmark_config: "
                        f"found {integrity_errors} event(s)"
                    )
        if gate.reject_duplicate_samples and self.duplicate_turn_count:
            failures.append(
                f"duplicate_turn_samples: found {self.duplicate_turn_count} duplicate(s)"
            )
        if gate.reject_duplicate_samples and self.duplicate_interrupt_count:
            failures.append(
                "duplicate_interrupt_samples: "
                f"found {self.duplicate_interrupt_count} duplicate(s)"
            )
        if gate.reject_mixed_metric_origins and len(self.metric_origins) > 1:
            failures.append("mixed_turn_metric_origins: " + ", ".join(self.metric_origins))
        if gate.reject_mixed_metric_origins and len(self.interrupt_metric_origins) > 1:
            failures.append(
                "mixed_interrupt_metric_origins: "
                + ", ".join(self.interrupt_metric_origins)
            )
        if gate.reject_mixed_schemas and len(self.metric_schemas) > 1:
            failures.append("mixed_turn_metric_schemas: " + ", ".join(self.metric_schemas))
        if gate.reject_mixed_schemas and len(self.interrupt_metric_schemas) > 1:
            failures.append(
                "mixed_interrupt_metric_schemas: "
                + ", ".join(self.interrupt_metric_schemas)
            )
        if self.turn_p50_ms > gate.turn_p50_ms:
            failures.append(
                f"turn_p50_exceeded: {self.turn_p50_ms} > {gate.turn_p50_ms} ms"
            )
        if self.turn_p95_ms > gate.turn_p95_ms:
            failures.append(
                f"turn_p95_exceeded: {self.turn_p95_ms} > {gate.turn_p95_ms} ms"
            )
        if (
            self.interrupt_p95_ms is not None
            and self.interrupt_p95_ms > gate.interrupt_p95_ms
        ):
            failures.append(
                "interrupt_p95_exceeded: "
                f"{self.interrupt_p95_ms} > {gate.interrupt_p95_ms} ms"
            )
        return LatencyGateResult(tuple(failures))

    def _provenance_failures(self, gate: LatencyGate) -> list[str]:
        failures: list[str] = []
        for kind in ("turn", "interrupt"):
            provenance_errors = getattr(self, f"{kind}_provenance_error_count")
            benchmark_config_errors = getattr(
                self, f"{kind}_benchmark_config_error_count"
            )
            schema_errors = getattr(self, f"{kind}_schema_error_count")
            missing_identities = getattr(self, f"missing_{kind}_identity_count")
            targets = getattr(self, f"{kind}_metric_targets")
            expected_target = getattr(gate, f"{kind}_metric_target")
            if provenance_errors:
                failures.append(
                    f"invalid_{kind}_provenance: found {provenance_errors} event(s)"
                )
            if benchmark_config_errors:
                failures.append(
                    f"invalid_{kind}_benchmark_config: "
                    f"found {benchmark_config_errors} event(s)"
                )
            if schema_errors:
                failures.append(f"unsupported_{kind}_schema: found {schema_errors} event(s)")
            if missing_identities:
                failures.append(
                    f"missing_{kind}_identities: found {missing_identities} event(s)"
                )
            if expected_target is not None and targets != (expected_target,):
                failures.append(
                    f"unexpected_{kind}_metric_target: "
                    f"expected {expected_target}, got {', '.join(targets) or '<none>'}"
                )
        if (
            self.turn_cohort_keys
            and self.interrupt_cohort_keys
            and self.turn_cohort_keys != self.interrupt_cohort_keys
        ):
            failures.append(
                "incompatible_turn_interrupt_cohorts: turn and interrupt samples "
                "must share run_id, config_hash, and schema_version"
            )
        return failures

    def passes(
        self,
        *,
        turn_p50_ms: float,
        turn_p95_ms: float,
        interrupt_p95_ms: float,
        min_turn_samples: int = DEFAULT_MIN_TURN_SAMPLES,
        min_interrupt_samples: int = DEFAULT_MIN_INTERRUPT_SAMPLES,
        turn_metric_target: str | None = TURN_RELEASE_METRIC_TARGET,
        interrupt_metric_target: str | None = INTERRUPT_RELEASE_METRIC_TARGET,
        reject_invalid: bool = True,
        reject_early_cutoff: bool = True,
        reject_mixed_metric_origins: bool = True,
        reject_mixed_schemas: bool = True,
        require_provenance: bool = True,
        reject_duplicate_samples: bool = True,
    ) -> bool:
        """Return the strict release-gate decision while preserving the old call shape."""
        gate = LatencyGate(
            turn_p50_ms=turn_p50_ms,
            turn_p95_ms=turn_p95_ms,
            interrupt_p95_ms=interrupt_p95_ms,
            min_turn_samples=min_turn_samples,
            min_interrupt_samples=min_interrupt_samples,
            turn_metric_target=turn_metric_target,
            interrupt_metric_target=interrupt_metric_target,
            reject_invalid=reject_invalid,
            reject_early_cutoff=reject_early_cutoff,
            reject_mixed_metric_origins=reject_mixed_metric_origins,
            reject_mixed_schemas=reject_mixed_schemas,
            require_provenance=require_provenance,
            reject_duplicate_samples=reject_duplicate_samples,
        )
        return self.evaluate_gate(gate).passed

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_count": self.turn_count,
            "turn_event_count": self.turn_event_count,
            "invalid_turn_count": self.invalid_turn_count,
            "early_cutoff_count": self.early_cutoff_count,
            "turn_min_ms": self.turn_min_ms,
            "turn_mean_ms": self.turn_mean_ms,
            "turn_p50_ms": self.turn_p50_ms,
            "turn_p90_ms": self.turn_p90_ms,
            "turn_p95_ms": self.turn_p95_ms,
            "turn_representative_max_ms": self.turn_representative_max_ms,
            "turn_max_ms": self.turn_max_ms,
            "turn_extreme_outlier_count": self.turn_extreme_outlier_count,
            "interrupt_count": self.interrupt_count,
            "interrupt_event_count": self.interrupt_event_count,
            "invalid_interrupt_count": self.invalid_interrupt_count,
            "interrupt_p95_ms": self.interrupt_p95_ms,
            "turn_provenance_error_count": self.turn_provenance_error_count,
            "interrupt_provenance_error_count": self.interrupt_provenance_error_count,
            "turn_benchmark_config_error_count": (
                self.turn_benchmark_config_error_count
            ),
            "interrupt_benchmark_config_error_count": (
                self.interrupt_benchmark_config_error_count
            ),
            "turn_present_benchmark_config_integrity_error_count": (
                self.turn_present_benchmark_config_integrity_error_count
            ),
            "interrupt_present_benchmark_config_integrity_error_count": (
                self.interrupt_present_benchmark_config_integrity_error_count
            ),
            "turn_schema_error_count": self.turn_schema_error_count,
            "interrupt_schema_error_count": self.interrupt_schema_error_count,
            "missing_turn_identity_count": self.missing_turn_identity_count,
            "missing_interrupt_identity_count": self.missing_interrupt_identity_count,
            "duplicate_turn_count": self.duplicate_turn_count,
            "duplicate_interrupt_count": self.duplicate_interrupt_count,
            "metric_origins": list(self.metric_origins),
            "metric_schemas": list(self.metric_schemas),
            "interrupt_metric_origins": list(self.interrupt_metric_origins),
            "interrupt_metric_schemas": list(self.interrupt_metric_schemas),
            "turn_metric_targets": list(self.turn_metric_targets),
            "interrupt_metric_targets": list(self.interrupt_metric_targets),
            "turn_cohort_keys": list(self.turn_cohort_keys),
            "interrupt_cohort_keys": list(self.interrupt_cohort_keys),
        }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            events.append(row)
    return events


def sanitize_benchmark_session_config(
    session_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Recursively redact credentials and hash prompt-like configuration values."""
    if session_config is None:
        return None
    cloned = json.loads(
        json.dumps(session_config, ensure_ascii=False, allow_nan=False)
    )
    if not isinstance(cloned, dict):  # pragma: no cover - type contract guard
        raise TypeError("session_config must serialize to an object")
    return _sanitize_private_config_value(cloned)


def canonical_benchmark_config_hash(benchmark_config: dict[str, Any]) -> str:
    """Hash the exact public benchmark configuration retained in artifacts."""
    public_config = _sanitize_benchmark_config(benchmark_config)
    payload = json.dumps(
        public_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sanitize_latency_event(event: dict[str, Any]) -> dict[str, Any]:
    """Return the public benchmark schema without transcript or arbitrary log fields."""
    sanitized = {
        key: value for key, value in event.items() if key in SAFE_EVENT_FIELDS
    }
    benchmark_config = event.get("benchmark_config")
    if isinstance(benchmark_config, dict):
        sanitized["benchmark_config"] = _sanitize_benchmark_config(benchmark_config)
    else:
        sanitized.pop("benchmark_config", None)
    return sanitized


def parse_structlog_latency_events(text: str) -> list[dict[str, Any]]:
    """Extract benchmark events from console or JSON structlog output."""
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        json_event = _parse_json_structlog_event(stripped)
        if json_event is not None:
            events.append(json_event)
            continue

        if "turn.complete" in line:
            events.append(_normalize_console_turn_event(_parse_key_values(line)))
        elif "interrupt.done" in line:
            values = _parse_key_values(line)
            if "elapsed_ms" in values or "interrupt_ms" in values:
                events.append(_normalize_console_interrupt_event(values))
    return events


def _parse_json_structlog_event(line: str) -> dict[str, Any] | None:
    if not line.startswith("{"):
        return None
    try:
        raw = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None

    event_type = raw.get("event")
    if event_type == "turn.complete":
        return _normalize_json_turn_event(raw)
    if event_type == "interrupt.done":
        return _normalize_json_interrupt_event(raw)
    return None


def _normalize_console_turn_event(values: dict[str, str]) -> dict[str, Any]:
    event: dict[str, Any] = {"event": "turn.complete"}
    for field in SAFE_EVENT_FIELDS - {"event", *TURN_LATENCY_FIELDS, *INTERRUPT_LATENCY_FIELDS}:
        if field in values:
            event[field] = _coerce_integer(values[field]) if field in {"turn_id", "trial"} else values[field]
    for field in TURN_LATENCY_FIELDS:
        if field in values:
            event[field] = _coerce_number(values[field])
    for field in ("interrupted", "early_cutoff"):
        if field in values:
            event[field] = _coerce_bool(values[field])
    event.setdefault("schema_version", LATENCY_SCHEMA_VERSION)
    event.setdefault("metric_origin", "runtime_turn_complete")
    event.setdefault("metric_target", TURN_RELEASE_METRIC_TARGET)
    return event


def _normalize_console_interrupt_event(values: dict[str, str]) -> dict[str, Any]:
    event: dict[str, Any] = {"event": "interrupt.done"}
    for field in SAFE_EVENT_FIELDS - {"event", *TURN_LATENCY_FIELDS, *INTERRUPT_LATENCY_FIELDS}:
        if field in values:
            event[field] = values[field]
    event["interrupt_ms"] = _coerce_number(
        values.get("interrupt_ms", values.get("elapsed_ms"))
    )
    event.setdefault("schema_version", LATENCY_SCHEMA_VERSION)
    event.setdefault("metric_origin", "runtime_interrupt_bus")
    event.setdefault("metric_target", INTERRUPT_CHAIN_METRIC_TARGET)
    return event


def _normalize_json_turn_event(raw: dict[str, Any]) -> dict[str, Any]:
    event = sanitize_latency_event(raw)
    event.setdefault("schema_version", LATENCY_SCHEMA_VERSION)
    event.setdefault("metric_origin", "runtime_turn_complete")
    event.setdefault("metric_target", TURN_RELEASE_METRIC_TARGET)
    return event


def _normalize_json_interrupt_event(raw: dict[str, Any]) -> dict[str, Any]:
    event = sanitize_latency_event(raw)
    if "interrupt_ms" not in event and "elapsed_ms" in raw:
        event["interrupt_ms"] = raw["elapsed_ms"]
    event.setdefault("schema_version", LATENCY_SCHEMA_VERSION)
    event.setdefault("metric_origin", "runtime_interrupt_bus")
    event.setdefault("metric_target", INTERRUPT_CHAIN_METRIC_TARGET)
    return event


def _parse_key_values(line: str) -> dict[str, str]:
    return {
        match.group("key"): match.group("value")
        for match in re.finditer(
            r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^ ]+)",
            line,
        )
    }


def _is_turn_event(event: dict[str, Any]) -> bool:
    event_type = event.get("event")
    return event_type == "turn.complete" or (event_type is None and "total_ms" in event)


def _is_interrupt_event(event: dict[str, Any]) -> bool:
    event_type = event.get("event")
    return event_type == "interrupt.done" or (
        event_type is None and "interrupt_ms" in event
    )


def _is_finite_non_negative(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    return math.isfinite(numeric) and value >= 0


def _valid_latency_or_none(value: Any) -> float | None:
    if not _is_finite_non_negative(value):
        return None
    try:
        return float(value)
    except (OverflowError, TypeError, ValueError):
        return None


def _has_invalid_optional_latency(
    event: dict[str, Any],
    fields: tuple[str, ...],
) -> bool:
    return any(
        field in event
        and event[field] is not None
        and not _is_finite_non_negative(event[field])
        for field in fields
    )


def _has_invalid_boolean_marker(event: dict[str, Any], field: str) -> bool:
    return field in event and not isinstance(event[field], bool)


def _has_provenance_error(event: dict[str, Any]) -> bool:
    if any(
        not isinstance(event.get(field), str) or not event[field].strip()
        for field in REQUIRED_PROVENANCE_FIELDS
    ):
        return True
    return _SHA256_RE.fullmatch(event["config_hash"]) is None


def _has_benchmark_config_error(event: dict[str, Any]) -> bool:
    benchmark_config = event.get("benchmark_config")
    config_hash = event.get("config_hash")
    if not isinstance(benchmark_config, dict) or not isinstance(config_hash, str):
        return True
    try:
        expected_hash = canonical_benchmark_config_hash(benchmark_config)
    except (TypeError, ValueError):
        return True
    return expected_hash != config_hash.lower()


def _has_present_benchmark_config_integrity_error(event: dict[str, Any]) -> bool:
    if "benchmark_config" not in event:
        return False
    return _has_benchmark_config_error(event)


def _sanitize_benchmark_config(benchmark_config: dict[str, Any]) -> dict[str, Any]:
    public_config = {
        key: value
        for key, value in benchmark_config.items()
        if key in PUBLIC_BENCHMARK_CONFIG_FIELDS
    }
    session_config = public_config.get("session_config")
    if isinstance(session_config, dict):
        public_config["session_config"] = sanitize_benchmark_session_config(
            session_config
        )
    elif session_config is not None:
        public_config.pop("session_config", None)
    return public_config


def _sanitize_private_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        prompt_hashes: dict[str, str] = {}
        for key, nested_value in value.items():
            normalized = key.casefold().replace("-", "_")
            if _is_secret_config_key(normalized):
                continue
            if _is_prompt_config_key(normalized):
                prompt_hashes[f"{key}_sha256"] = _private_value_hash(nested_value)
                continue
            sanitized[key] = _sanitize_private_config_value(nested_value)
        # A caller-supplied ``*_sha256`` sibling must not overwrite the digest
        # derived from the private prompt value.
        sanitized.update(prompt_hashes)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_private_config_value(item) for item in value]
    return value


def _is_secret_config_key(normalized: str) -> bool:
    return normalized in _SECRET_CONFIG_KEYS or any(
        normalized.endswith(f"_{suffix}") for suffix in _SECRET_CONFIG_KEYS
    )


def _is_prompt_config_key(normalized: str) -> bool:
    return normalized in _PROMPT_CONFIG_KEYS or any(
        normalized.endswith(f"_{suffix}") for suffix in _PROMPT_CONFIG_KEYS
    )


def _private_value_hash(value: Any) -> str:
    if isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _has_supported_metric_schema(event: dict[str, Any], kind: str) -> bool:
    if event.get("schema_version") != LATENCY_SCHEMA_VERSION:
        return False
    requirements = (
        TURN_TARGET_REQUIRED_FIELDS if kind == "turn" else INTERRUPT_TARGET_REQUIRED_FIELDS
    )
    target = event.get("metric_target")
    if not isinstance(target, str) or target not in requirements:
        return False
    return all(_valid_latency_or_none(event.get(field)) is not None for field in requirements[target])


def _sample_identity(event: dict[str, Any], kind: str) -> str | None:
    run_id = event.get("run_id")
    if not _is_non_empty(run_id):
        return None
    if kind == "turn":
        if _is_non_empty(event.get("turn_id")):
            return f"run={run_id}|turn={event['turn_id']}"
        if _is_non_empty(event.get("fixture")) and _is_non_empty(event.get("trial")):
            return f"run={run_id}|fixture={event['fixture']}|trial={event['trial']}"
        return None
    if _is_non_empty(event.get("interrupt_id")):
        return f"run={run_id}|interrupt={event['interrupt_id']}"
    if _is_non_empty(event.get("turn_id")):
        return f"run={run_id}|turn={event['turn_id']}"
    return None


def _duplicate_count(identities: list[str | None]) -> int:
    return sum(count - 1 for count in Counter(identity for identity in identities if identity).values())


def _has_non_empty_value(event: dict[str, Any], field: str) -> bool:
    return field in event and _is_non_empty(event[field])


def _is_non_empty(value: Any) -> bool:
    return value is not None and not isinstance(value, bool) and (
        not isinstance(value, str) or bool(value.strip())
    )


def _coerce_number(value: Any) -> Any:
    if value is None or isinstance(value, int | float):
        return value
    try:
        return float(value)
    except (OverflowError, TypeError, ValueError):
        return value


def _coerce_integer(value: Any) -> Any:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    try:
        return int(value)
    except (OverflowError, TypeError, ValueError):
        return value


def _coerce_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


def _field_token(event: dict[str, Any], field: str) -> str:
    if field not in event or event[field] is None:
        return "<missing>"
    value = event[field]
    if isinstance(value, str) and not value.strip():
        return "<missing>"
    return str(value)


def _metric_origin(event: dict[str, Any]) -> str:
    return "|".join(f"{field}={_field_token(event, field)}" for field in PROVENANCE_FIELDS)


def _metric_schema(event: dict[str, Any], kind: str) -> str:
    latency_fields = TURN_LATENCY_FIELDS if kind == "turn" else INTERRUPT_LATENCY_FIELDS
    fields = sorted(field for field in latency_fields if field in event)
    return (
        f"{_field_token(event, 'schema_version')}:"
        f"{_field_token(event, 'metric_target')}:"
        f"{','.join(fields)}"
    )


def _cohort_key(event: dict[str, Any]) -> str:
    return "|".join(
        f"{field}={_field_token(event, field)}"
        for field in ("run_id", "config_hash", "schema_version")
    )


def _exclude_extreme_outliers(ordered: list[float]) -> list[float]:
    if len(ordered) < 4:
        return ordered
    q1 = _percentile_linear(ordered, 25)
    q3 = _percentile_linear(ordered, 75)
    iqr = q3 - q1
    upper_fence = q3 + 3 * iqr
    filtered = [value for value in ordered if value <= upper_fence]
    return filtered or ordered


def _percentile_nearest_rank(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("No values")
    rank = max(0, int(-(-percentile * len(ordered) // 100)) - 1)
    return ordered[min(rank, len(ordered) - 1)]


def _percentile_linear(ordered: list[float], percentile: int) -> float:
    if not ordered:
        raise ValueError("No values")
    if len(ordered) == 1:
        return ordered[0]

    rank = (percentile / 100) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
