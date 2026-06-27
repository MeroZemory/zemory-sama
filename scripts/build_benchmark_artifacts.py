#!/usr/bin/env python3
"""Build README-ready benchmark artifacts from a zemory runtime log."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    from zemory.observability.benchmark_artifacts import write_benchmark_artifacts
    from zemory.observability.latency_report import parse_structlog_latency_events

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=Path(".zemory/run.log"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="zemory-sama manual realtime audio benchmark")
    parser.add_argument(
        "--source-note",
        default=(
            "Numeric-only export from local manual realtime_audio runtime logs; "
            "raw transcripts are intentionally not committed."
        ),
    )
    args = parser.parse_args(argv)

    events = parse_structlog_latency_events(args.log.read_text(encoding="utf-8"))
    outputs = write_benchmark_artifacts(
        events,
        out_dir=args.out,
        title=args.title,
        source_note=args.source_note,
    )
    print(json.dumps({key: str(value) for key, value in outputs.__dict__.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
