#!/usr/bin/env python3
"""Measure local SpeakerStream queue-to-playback callback latency."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zemory.audio import SpeakerStream, generate_beep_pcm, output_block_size
from zemory.config import SAMPLE_RATE
from zemory.observability.latency_report import (
    _exclude_extreme_outliers,
    _percentile_nearest_rank,
)


def _values(events: list[dict[str, Any]], key: str) -> list[float]:
    return [float(event[key]) for event in events if event.get(key) is not None]


def _metric_summary(events: list[dict[str, Any]], key: str, prefix: str) -> dict[str, Any]:
    values = _values(events, key)
    if not values:
        raise ValueError(f"No {key} samples found")
    ordered = sorted(values)
    representative = _exclude_extreme_outliers(ordered)
    return {
        f"{prefix}_count": len(values),
        f"{prefix}_min_ms": ordered[0],
        f"{prefix}_mean_ms": sum(values) / len(values),
        f"{prefix}_p50_ms": _percentile_nearest_rank(values, 50),
        f"{prefix}_p90_ms": _percentile_nearest_rank(values, 90),
        f"{prefix}_p95_ms": _percentile_nearest_rank(values, 95),
        f"{prefix}_representative_max_ms": representative[-1],
        f"{prefix}_max_ms": ordered[-1],
        f"{prefix}_extreme_outlier_count": len(ordered) - len(representative),
    }


def _make_summary(
    events: list[dict[str, Any]],
    *,
    title: str,
    source_note: str,
    output_block_ms: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "title": title,
        "source_note": source_note,
        "sample_count": len(events),
        "output_block_ms": output_block_ms,
    }
    summary.update(_metric_summary(events, "total_ms", "queue_to_play"))
    summary.update(_metric_summary(events, "queue_to_buffer_ms", "queue_to_buffer"))
    summary.update(_metric_summary(events, "speaker_buffer_ms", "speaker_buffer"))
    return summary


def _write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    rows = [json.dumps(event, ensure_ascii=False, sort_keys=True) for event in events]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_speaker_artifacts(
    events: list[dict[str, Any]],
    *,
    out_dir: Path,
    title: str,
    source_note: str,
    output_block_ms: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _make_summary(
        events,
        title=title,
        source_note=source_note,
        output_block_ms=output_block_ms,
    )
    _write_jsonl(out_dir / "latency-events.jsonl", events)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(_markdown(summary), encoding="utf-8")
    (out_dir / "latency.svg").write_text(_svg(summary), encoding="utf-8")


def _markdown(summary: dict[str, Any]) -> str:
    return f"""# {summary["title"]}

{summary["source_note"]}

| Metric | Value |
| --- | ---: |
| sample count | {summary["sample_count"]} |
| output callback block | {summary["output_block_ms"]} ms |
| queue to first playback callback min | {summary["queue_to_play_min_ms"]:.2f} ms |
| queue to first playback callback mean | {summary["queue_to_play_mean_ms"]:.2f} ms |
| queue to first playback callback p50 | {summary["queue_to_play_p50_ms"]:.2f} ms |
| queue to first playback callback p90 | {summary["queue_to_play_p90_ms"]:.2f} ms |
| queue to first playback callback representative max | {summary["queue_to_play_representative_max_ms"]:.2f} ms |
| queue to first playback callback observed max, diagnostic | {summary["queue_to_play_max_ms"]:.2f} ms |
| queue to first playback callback outliers | {summary["queue_to_play_extreme_outlier_count"]} |
| queue to speaker buffer p50 | {summary["queue_to_buffer_p50_ms"]:.2f} ms |
| speaker buffer p50 | {summary["speaker_buffer_p50_ms"]:.2f} ms |
| speaker buffer representative max | {summary["speaker_buffer_representative_max_ms"]:.2f} ms |

![Latency chart](latency.svg)
"""


def _svg(summary: dict[str, Any]) -> str:
    metrics = [
        (
            "queue -> playback callback p50",
            float(summary["queue_to_play_p50_ms"]),
            10.0,
        ),
        (
            "queue -> playback callback rep max",
            float(summary["queue_to_play_representative_max_ms"]),
            15.0,
        ),
        ("queue -> buffer p50", float(summary["queue_to_buffer_p50_ms"]), 3.0),
        ("buffer -> playback p50", float(summary["speaker_buffer_p50_ms"]), 10.0),
    ]
    max_value = max(max(value, target) for _, value, target in metrics)
    scale = 410 / max_value
    rows: list[str] = []
    for idx, (label, value, target) in enumerate(metrics):
        y = 56 + idx * 48
        width = max(1, value * scale)
        target_x = 260 + target * scale
        color = "#0f766e" if value <= target else "#b91c1c"
        rows.append(f'<text x="24" y="{y + 16}" font-size="14" fill="#111827">{label}</text>')
        rows.append(f'<rect x="260" y="{y}" width="{width:.1f}" height="22" fill="{color}" rx="3"/>')
        rows.append(
            f'<line x1="{target_x:.1f}" x2="{target_x:.1f}" y1="{y - 4}" y2="{y + 28}" '
            'stroke="#6b7280" stroke-width="1.5" stroke-dasharray="4 3"/>'
        )
        rows.append(
            f'<text x="{270 + width:.1f}" y="{y + 16}" font-size="13" fill="#111827">'
            f"{value:.2f} ms</text>"
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="760" height="280" role="img" aria-label="Speaker output latency benchmark">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="24" y="30" font-size="18" font-weight="700" fill="#111827">Speaker output latency benchmark</text>
  <text x="260" y="30" font-size="12" fill="#6b7280">dashed line = local target</text>
  {"\n  ".join(rows)}
</svg>
"""


async def _measure_once(
    speaker: SpeakerStream,
    pcm: bytes,
    *,
    trial: int,
    output_block_ms: int,
    timeout_s: float,
) -> dict[str, Any]:
    speaker.arm()
    queued_at = time.monotonic()
    await speaker.queue.put(pcm)
    deadline = queued_at + timeout_s
    while speaker.first_play_at is None:
        if time.monotonic() > deadline:
            raise TimeoutError("speaker playback callback did not consume audio")
        await asyncio.sleep(0.001)
    first_write_at = speaker.first_write_at or speaker.first_play_at
    first_play_at = speaker.first_play_at
    event = {
        "event": "speaker.playback",
        "trial": trial,
        "sample_rate": SAMPLE_RATE,
        "output_block_ms": output_block_ms,
        "total_ms": round((first_play_at - queued_at) * 1000, 3),
        "queue_to_buffer_ms": round((first_write_at - queued_at) * 1000, 3),
        "speaker_buffer_ms": round((first_play_at - first_write_at) * 1000, 3),
    }
    await speaker.wait_until_done()
    return event


async def _run(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    speaker = SpeakerStream(loop)
    output_block_ms = round(output_block_size() / SAMPLE_RATE * 1000)
    pcm = generate_beep_pcm(
        frequency_hz=args.frequency_hz,
        duration_ms=args.tone_ms,
        sample_rate=SAMPLE_RATE,
        volume=args.volume,
    )
    feed_task: asyncio.Task | None = None
    events: list[dict[str, Any]] = []
    try:
        speaker.start()
        feed_task = asyncio.create_task(speaker.feed())
        for trial in range(1, args.trials + 1):
            events.append(
                await _measure_once(
                    speaker,
                    pcm,
                    trial=trial,
                    output_block_ms=output_block_ms,
                    timeout_s=args.timeout_s,
                )
            )
            await asyncio.sleep(args.settle_ms / 1000)
    finally:
        if feed_task is not None:
            feed_task.cancel()
            try:
                await feed_task
            except asyncio.CancelledError:
                pass
        speaker.stop()

    source_note = (
        f"{len(events)} local SpeakerStream output samples. "
        f"Output callback block={output_block_ms} ms; sounddevice latency=low. "
        "Metric is queue insertion to first output callback that consumes non-zero PCM. "
        "Numeric-only export; no transcripts are recorded."
    )
    _write_speaker_artifacts(
        events,
        out_dir=args.out,
        title=args.title,
        source_note=source_note,
        output_block_ms=output_block_ms,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="zemory-sama speaker output benchmark")
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--tone-ms", type=int, default=35)
    parser.add_argument("--frequency-hz", type=float, default=880.0)
    parser.add_argument("--volume", type=float, default=0.01)
    parser.add_argument("--settle-ms", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
