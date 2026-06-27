"""Latency benchmark report utilities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LatencyReport:
    turn_count: int
    turn_min_ms: float
    turn_mean_ms: float
    turn_p50_ms: float
    turn_p95_ms: float
    turn_max_ms: float
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

        turn_latencies_sorted = sorted(turn_latencies)
        interrupt_latencies = [
            float(event["interrupt_ms"])
            for event in events
            if event.get("interrupt_ms") is not None
        ]
        return cls(
            turn_count=len(turn_latencies),
            turn_min_ms=turn_latencies_sorted[0],
            turn_mean_ms=sum(turn_latencies) / len(turn_latencies),
            turn_p50_ms=_percentile_nearest_rank(turn_latencies, 50),
            turn_p95_ms=_percentile_nearest_rank(turn_latencies, 95),
            turn_max_ms=turn_latencies_sorted[-1],
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
            "turn_min_ms": self.turn_min_ms,
            "turn_mean_ms": self.turn_mean_ms,
            "turn_p50_ms": self.turn_p50_ms,
            "turn_p95_ms": self.turn_p95_ms,
            "turn_max_ms": self.turn_max_ms,
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


def parse_structlog_latency_events(text: str) -> list[dict[str, Any]]:
    """Extract numeric benchmark events from human-readable structlog output."""
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        if "turn.complete" in line:
            values = _parse_key_values(line)
            events.append(
                {
                    "event": "turn.complete",
                    "turn_id": int(values["turn_id"]),
                    "profile": values.get("profile"),
                    "total_ms": float(values["total_ms"]),
                    "first_tts_byte_ms": float(values["first_tts_byte_ms"]),
                    "interrupted": values.get("interrupted") == "True",
                }
            )
        elif "interrupt.done" in line:
            values = _parse_key_values(line)
            if "elapsed_ms" in values:
                events.append(
                    {
                        "event": "interrupt.done",
                        "interrupt_ms": float(values["elapsed_ms"]),
                    }
                )
    return events


def _parse_key_values(line: str) -> dict[str, str]:
    return {
        match.group("key"): match.group("value")
        for match in re.finditer(
            r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^ ]+)",
            line,
        )
    }


def _percentile_nearest_rank(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("No values")
    # Use ceil(p/100*n)-1, with a lower bound of 0.
    rank = max(0, int(-(-percentile * len(ordered) // 100)) - 1)
    return ordered[min(rank, len(ordered) - 1)]
