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
    from zemory.observability.latency_report import LatencyReport, load_jsonl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="Latency event JSONL file")
    parser.add_argument("--turn-p50-ms", type=float, default=700)
    parser.add_argument("--turn-p95-ms", type=float, default=1200)
    parser.add_argument("--interrupt-p95-ms", type=float, default=150)
    args = parser.parse_args(argv)

    report = LatencyReport.from_events(load_jsonl(args.jsonl))
    passed = report.passes(
        turn_p50_ms=args.turn_p50_ms,
        turn_p95_ms=args.turn_p95_ms,
        interrupt_p95_ms=args.interrupt_p95_ms,
    )
    payload = {
        "status": "PASS" if passed else "FAIL",
        **report.as_dict(),
        "thresholds": {
            "turn_p50_ms": args.turn_p50_ms,
            "turn_p95_ms": args.turn_p95_ms,
            "interrupt_p95_ms": args.interrupt_p95_ms,
        },
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
