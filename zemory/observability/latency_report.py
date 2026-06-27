"""Latency benchmark report utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LatencyReport:
    turn_count: int
    turn_p50_ms: float
    turn_p95_ms: float
    interrupt_count: int
    interrupt_p95_ms: float | None

    @classmethod
    def from_events(cls, events: list[dict[str, Any]]) -> LatencyReport:
        turn_latencies = [
            float(event["total_ms"])
            for event in events
            if event.get("total_ms") is not None
        ]
        if not turn_latencies:
            raise ValueError("No turn latency samples found")

        interrupt_latencies = [
            float(event["interrupt_ms"])
            for event in events
            if event.get("interrupt_ms") is not None
        ]
        return cls(
            turn_count=len(turn_latencies),
            turn_p50_ms=_percentile_nearest_rank(turn_latencies, 50),
            turn_p95_ms=_percentile_nearest_rank(turn_latencies, 95),
            interrupt_count=len(interrupt_latencies),
            interrupt_p95_ms=(
                _percentile_nearest_rank(interrupt_latencies, 95)
                if interrupt_latencies
                else None
            ),
        )

    def passes(
        self,
        *,
        turn_p50_ms: float,
        turn_p95_ms: float,
        interrupt_p95_ms: float,
    ) -> bool:
        if self.turn_p50_ms > turn_p50_ms:
            return False
        if self.turn_p95_ms > turn_p95_ms:
            return False
        if self.interrupt_p95_ms is not None and self.interrupt_p95_ms > interrupt_p95_ms:
            return False
        return True

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "turn_count": self.turn_count,
            "turn_p50_ms": self.turn_p50_ms,
            "turn_p95_ms": self.turn_p95_ms,
            "interrupt_count": self.interrupt_count,
            "interrupt_p95_ms": self.interrupt_p95_ms,
        }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _percentile_nearest_rank(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("No values")
    # Use ceil(p/100*n)-1, with a lower bound of 0.
    rank = max(0, int(-(-percentile * len(ordered) // 100)) - 1)
    return ordered[min(rank, len(ordered) - 1)]
