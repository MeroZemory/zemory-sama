#!/usr/bin/env python3
"""Check latency JSONL against realtime-voice-runtime release thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    from zemory.observability.latency_report import (
        DEFAULT_MIN_INTERRUPT_SAMPLES,
        DEFAULT_MIN_TURN_SAMPLES,
        INTERRUPT_RELEASE_METRIC_TARGET,
        TURN_RELEASE_METRIC_TARGET,
        LatencyGate,
        LatencyReport,
        load_jsonl,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="Latency event JSONL file")
    parser.add_argument("--turn-p50-ms", type=float, default=700)
    parser.add_argument("--turn-p95-ms", type=float, default=1200)
    parser.add_argument("--interrupt-p95-ms", type=float, default=150)
    parser.add_argument("--min-turn-samples", type=int, default=DEFAULT_MIN_TURN_SAMPLES)
    parser.add_argument(
        "--min-interrupt-samples",
        type=int,
        default=DEFAULT_MIN_INTERRUPT_SAMPLES,
    )
    parser.add_argument("--turn-metric-target", default=TURN_RELEASE_METRIC_TARGET)
    parser.add_argument(
        "--interrupt-metric-target",
        default=INTERRUPT_RELEASE_METRIC_TARGET,
    )
    parser.add_argument("--allow-invalid-events", action="store_true")
    parser.add_argument("--allow-early-cutoffs", action="store_true")
    parser.add_argument("--allow-mixed-metric-origins", action="store_true")
    parser.add_argument("--allow-mixed-schemas", action="store_true")
    parser.add_argument("--allow-legacy-provenance", action="store_true")
    parser.add_argument("--allow-duplicate-samples", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = LatencyReport.from_events(load_jsonl(args.jsonl))
        # A zero minimum permits an absent interrupt stream. If interrupt
        # samples are present, keep validating their target provenance.
        interrupt_metric_target = (
            None
            if args.min_interrupt_samples == 0 and report.interrupt_count == 0
            else args.interrupt_metric_target
        )
        gate = LatencyGate(
            turn_p50_ms=args.turn_p50_ms,
            turn_p95_ms=args.turn_p95_ms,
            interrupt_p95_ms=args.interrupt_p95_ms,
            min_turn_samples=args.min_turn_samples,
            min_interrupt_samples=args.min_interrupt_samples,
            turn_metric_target=args.turn_metric_target,
            interrupt_metric_target=interrupt_metric_target,
            reject_invalid=not args.allow_invalid_events,
            reject_early_cutoff=not args.allow_early_cutoffs,
            reject_mixed_metric_origins=not args.allow_mixed_metric_origins,
            reject_mixed_schemas=not args.allow_mixed_schemas,
            require_provenance=not args.allow_legacy_provenance,
            reject_duplicate_samples=not args.allow_duplicate_samples,
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "failure_reasons": [f"invalid_benchmark_input: {exc}"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1

    decision = report.evaluate_gate(gate)
    payload = {
        "status": "PASS" if decision.passed else "FAIL",
        "failure_reasons": list(decision.failure_reasons),
        **report.as_dict(),
        "thresholds": {
            "turn_p50_ms": args.turn_p50_ms,
            "turn_p95_ms": args.turn_p95_ms,
            "interrupt_p95_ms": args.interrupt_p95_ms,
        },
        "gate": gate.as_dict(),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if decision.passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
