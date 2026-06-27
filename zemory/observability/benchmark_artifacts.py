"""Benchmark artifact writers for README-ready reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zemory.observability.latency_report import LatencyReport


@dataclass(frozen=True)
class BenchmarkOutputs:
    events_jsonl: Path
    summary_json: Path
    markdown: Path
    svg: Path


def write_benchmark_artifacts(
    events: list[dict[str, Any]],
    *,
    out_dir: str | Path,
    title: str,
    source_note: str,
) -> BenchmarkOutputs:
    """Write numeric-only benchmark artifacts and return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = LatencyReport.from_events(events)

    events_jsonl = out / "latency-events.jsonl"
    summary_json = out / "summary.json"
    markdown = out / "README.md"
    svg = out / "latency.svg"

    _write_jsonl(events_jsonl, events)
    summary = {
        "title": title,
        "source_note": source_note,
        **report.as_dict(),
    }
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown.write_text(_markdown(summary), encoding="utf-8")
    svg.write_text(_svg(summary), encoding="utf-8")

    return BenchmarkOutputs(
        events_jsonl=events_jsonl,
        summary_json=summary_json,
        markdown=markdown,
        svg=svg,
    )


def _write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    rows = [json.dumps(event, ensure_ascii=False, sort_keys=True) for event in events]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _markdown(summary: dict[str, Any]) -> str:
    interrupt_p95 = summary["interrupt_p95_ms"]
    interrupt_value = f"{interrupt_p95:.1f} ms" if interrupt_p95 is not None else "n/a"
    return f"""# {summary["title"]}

{summary["source_note"]}

| Metric | Value |
| --- | ---: |
| turn count | {summary["turn_count"]} |
| turn min | {summary["turn_min_ms"]:.1f} ms |
| turn mean | {summary["turn_mean_ms"]:.1f} ms |
| turn p50 | {summary["turn_p50_ms"]:.1f} ms |
| turn p90 | {summary["turn_p90_ms"]:.1f} ms |
| turn p95 | {summary["turn_p95_ms"]:.1f} ms |
| representative max | {summary["turn_representative_max_ms"]:.1f} ms |
| extreme outliers | {summary["turn_extreme_outlier_count"]} |
| observed max, diagnostic | {summary["turn_max_ms"]:.1f} ms |
| interrupt count | {summary["interrupt_count"]} |
| interrupt p95 | {interrupt_value} |

![Latency chart](latency.svg)
"""


def _svg(summary: dict[str, Any]) -> str:
    metrics = [
        ("turn p50", float(summary["turn_p50_ms"]), 700.0),
        ("turn p90", float(summary["turn_p90_ms"]), 1000.0),
        ("turn p95", float(summary["turn_p95_ms"]), 1200.0),
        (
            "representative max",
            float(summary["turn_representative_max_ms"]),
            1500.0,
        ),
    ]
    if summary["interrupt_p95_ms"] is not None:
        metrics.append(("interrupt p95", float(summary["interrupt_p95_ms"]), 150.0))

    width = 760
    row_h = 48
    height = 72 + row_h * len(metrics)
    max_value = max(max(value, target) for _, value, target in metrics)
    scale = 420 / max_value
    rows: list[str] = []
    for idx, (label, value, target) in enumerate(metrics):
        y = 52 + idx * row_h
        bar_w = max(1, value * scale)
        target_x = 220 + target * scale
        color = "#0f766e" if value <= target else "#b91c1c"
        rows.append(
            f'<text x="24" y="{y + 16}" font-size="14" fill="#111827">{label}</text>'
        )
        rows.append(
            f'<rect x="220" y="{y}" width="{bar_w:.1f}" height="22" fill="{color}" rx="3" />'
        )
        rows.append(
            f'<line x1="{target_x:.1f}" x2="{target_x:.1f}" y1="{y - 4}" y2="{y + 28}" '
            'stroke="#6b7280" stroke-width="1.5" stroke-dasharray="4 3" />'
        )
        rows.append(
            f'<text x="{230 + bar_w:.1f}" y="{y + 16}" font-size="13" fill="#111827">'
            f"{value:.1f} ms</text>"
        )
    rows_text = "\n  ".join(rows)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img" aria-label="Latency benchmark chart">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="24" y="28" font-size="18" font-weight="700" fill="#111827">zemory-sama latency benchmark</text>
  <text x="220" y="28" font-size="12" fill="#6b7280">dashed line = release target</text>
  {rows_text}
</svg>
"""
