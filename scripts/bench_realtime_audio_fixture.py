#!/usr/bin/env python3
"""Run a live OpenAI Realtime audio benchmark with generated PCM fixtures."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

Mode = Literal["semantic_vad", "forced_commit"]
Eagerness = Literal["low", "medium", "high", "auto"]
TurnDetection = Literal["semantic_vad", "server_vad"]


@dataclass(frozen=True)
class Sample:
    name: str
    voice: str
    text: str


SAMPLES = [
    Sample("en_short", "Samantha", "Quick test. Please answer briefly."),
    Sample(
        "en_interrupt",
        "Samantha",
        "Can you respond as soon as I finish this short sentence?",
    ),
    Sample("ko_short", "Yuna", "짧게 대답해 주세요. 지금 속도 테스트입니다."),
    Sample(
        "ko_summary",
        "Yuna",
        "제가 방금 말한 내용을 한 문장으로 아주 짧게 요약해 주세요.",
    ),
]


def _chunk_pcm(pcm: bytes, *, chunk_size: int) -> Iterator[bytes]:
    for offset in range(0, len(pcm), chunk_size):
        yield pcm[offset : offset + chunk_size]


def _event_from_timings(
    *,
    fixture: str,
    voice: str,
    eagerness: Eagerness,
    turn_detection: TurnDetection,
    server_vad_threshold: float,
    input_chunk_ms: int,
    mode: Mode,
    audio_end_at: float,
    speech_stopped_at: float | None,
    first_audio_at: float,
) -> dict[str, float | str | bool | None]:
    early_cutoff = first_audio_at < audio_end_at or (
        speech_stopped_at is not None and speech_stopped_at < audio_end_at
    )
    total_ms = None if early_cutoff else round((first_audio_at - audio_end_at) * 1000, 1)
    event = {
        "event": "turn.complete",
        "fixture": fixture,
        "voice": voice,
        "profile": "realtime_audio",
        "eagerness": eagerness,
        "turn_detection": turn_detection,
        "server_vad_threshold": server_vad_threshold,
        "input_chunk_ms": input_chunk_ms,
        "interrupted": False,
        "early_cutoff": early_cutoff,
        "sample_source": f"macos_say_{mode}",
        "total_ms": total_ms,
        "first_tts_byte_ms": total_ms,
        "vad_wait_ms": None,
        "first_audio_after_speech_stopped_ms": None,
    }
    if speech_stopped_at is not None:
        event["vad_wait_ms"] = round((speech_stopped_at - audio_end_at) * 1000, 1)
        event["first_audio_after_speech_stopped_ms"] = round(
            (first_audio_at - speech_stopped_at) * 1000,
            1,
        )
    return event


def _require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Required binary not found: {name}")


def _write_jsonl(path: Path, events: list[dict[str, float | str | bool | None]]) -> None:
    rows = [json.dumps(event, ensure_ascii=False, sort_keys=True) for event in events]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_live_benchmark_artifacts(
    events: list[dict[str, float | str | bool | None]],
    *,
    out_dir: Path,
    title: str,
    source_note: str,
) -> None:
    from zemory.observability.benchmark_artifacts import write_benchmark_artifacts

    valid_events = [event for event in events if event.get("total_ms") is not None]
    if valid_events:
        write_benchmark_artifacts(
            events,
            out_dir=out_dir,
            title=title,
            source_note=source_note,
        )
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "latency-events.jsonl", events)
    early_cutoff_count = sum(1 for event in events if event.get("early_cutoff"))
    summary = {
        "title": title,
        "source_note": source_note,
        "turn_count": 0,
        "total_event_count": len(events),
        "invalid_latency_count": len(events),
        "early_cutoff_count": early_cutoff_count,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        f"""# {title}

{source_note}

No valid latency samples were recorded. This probe is rejected because all
{len(events)} event(s) were invalid, including {early_cutoff_count} early cutoff
event(s).
""",
        encoding="utf-8",
    )
    (out_dir / "latency.svg").write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="760" height="96" role="img" aria-label="Invalid latency benchmark">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="24" y="34" font-size="18" font-weight="700" fill="#111827">{title}</text>
  <text x="24" y="64" font-size="14" fill="#b91c1c">No valid latency samples; rejected as early cutoff probe.</text>
</svg>
""",
        encoding="utf-8",
    )


def _render_sample(sample: Sample, out_dir: Path) -> bytes:
    aiff = out_dir / f"{sample.name}.aiff"
    pcm = out_dir / f"{sample.name}.pcm"
    subprocess.run(
        ["say", "-v", sample.voice, "-o", str(aiff), sample.text],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(aiff),
            "-ac",
            "1",
            "-ar",
            "24000",
            "-f",
            "s16le",
            str(pcm),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return pcm.read_bytes()


async def _stream_pcm_realtime(
    llm,
    pcm: bytes,
    *,
    sample_rate: int,
    input_chunk_ms: int,
    sleep=asyncio.sleep,
) -> float:
    chunk_duration_s = input_chunk_ms / 1000
    chunk_size = int(sample_rate * chunk_duration_s) * 2
    for chunk in _chunk_pcm(pcm, chunk_size=chunk_size):
        await llm.push_audio(chunk)
        maybe_sleep = sleep(chunk_duration_s)
        if maybe_sleep is not None:
            await maybe_sleep
    return time.monotonic()


async def _send_silence(llm, *, sample_rate: int, duration_s: float) -> None:
    chunk_duration_s = 0.02
    chunk_size = int(sample_rate * chunk_duration_s) * 2
    silence = b"\x00" * chunk_size
    chunks = int(duration_s / chunk_duration_s)
    for _ in range(chunks):
        await llm.push_audio(silence)
        await asyncio.sleep(chunk_duration_s)


async def _wait_for_first_audio(llm, *, timeout_s: float) -> tuple[float | None, float]:
    speech_stopped_at: float | None = None

    async def consume() -> tuple[float | None, float]:
        nonlocal speech_stopped_at
        async for event in llm.events():
            event_type = event.get("type")
            if event_type == "input.speech_stopped" and speech_stopped_at is None:
                speech_stopped_at = time.monotonic()
            elif event_type == "audio.delta":
                return speech_stopped_at, time.monotonic()
            elif event_type == "error":
                raise RuntimeError(str(event.get("error")))
        raise RuntimeError("Realtime event stream ended before audio")

    return await asyncio.wait_for(consume(), timeout=timeout_s)


async def _measure_sample(
    sample: Sample,
    pcm: bytes,
    *,
    eagerness: Eagerness,
    turn_detection: TurnDetection,
    server_vad_threshold: float,
    server_vad_silence_ms: int,
    input_chunk_ms: int,
    mode: Mode,
    timeout_s: float,
) -> dict[str, float | str | bool | None]:
    from zemory import config as cfg
    from zemory.providers.llm.openai_realtime import OpenAIRealtimeLLM

    cfg.settings.profile = "realtime_audio"
    cfg.settings.realtime.semantic_vad_eagerness = eagerness
    cfg.settings.realtime.turn_detection = turn_detection
    cfg.settings.realtime.server_vad_threshold = server_vad_threshold
    cfg.settings.realtime.server_vad_silence_duration_ms = server_vad_silence_ms

    llm = OpenAIRealtimeLLM(cfg.OPENAI_API_KEY)
    await llm.open_session()
    try:
        waiter = asyncio.create_task(_wait_for_first_audio(llm, timeout_s=timeout_s))
        audio_end_at = await _stream_pcm_realtime(
            llm,
            pcm,
            sample_rate=cfg.SAMPLE_RATE,
            input_chunk_ms=input_chunk_ms,
        )
        if mode == "forced_commit":
            if llm._conn is None:  # pragma: no cover - defensive live-only guard
                raise RuntimeError("Realtime connection was not opened")
            await llm._conn.input_audio_buffer.commit()
            audio_end_at = time.monotonic()
            await llm.trigger_response()
        else:
            await _send_silence(llm, sample_rate=cfg.SAMPLE_RATE, duration_s=1.6)
        speech_stopped_at, first_audio_at = await waiter
    finally:
        await llm.close()

    return _event_from_timings(
        fixture=sample.name,
        voice=sample.voice,
        eagerness=eagerness,
        turn_detection=turn_detection,
        server_vad_threshold=server_vad_threshold,
        input_chunk_ms=input_chunk_ms,
        mode=mode,
        audio_end_at=audio_end_at,
        speech_stopped_at=speech_stopped_at,
        first_audio_at=first_audio_at,
    )


async def _run(args: argparse.Namespace) -> None:
    _require_binary("say")
    _require_binary("ffmpeg")

    selected = [sample for sample in SAMPLES if sample.name in set(args.samples)]
    if not selected:
        raise SystemExit("No matching samples selected")

    events: list[dict[str, float | str | bool | None]] = []
    with tempfile.TemporaryDirectory(prefix="zemory-bench-audio-") as tmp:
        tmp_dir = Path(tmp)
        rendered = {sample.name: _render_sample(sample, tmp_dir) for sample in selected}
        for trial in range(args.trials):
            for sample in selected:
                event = await _measure_sample(
                    sample,
                    rendered[sample.name],
                    eagerness=args.eagerness,
                    turn_detection=args.turn_detection,
                    server_vad_threshold=args.server_vad_threshold,
                    server_vad_silence_ms=args.server_vad_silence_ms,
                    input_chunk_ms=args.input_chunk_ms,
                    mode=args.mode,
                    timeout_s=args.timeout_s,
                )
                event["trial"] = trial + 1
                events.append(event)
                print(event, flush=True)

    source_note = (
        f"{len(events)} macOS say fixtures streamed as realtime 24 kHz PCM. "
        f"Mode={args.mode}; turn_detection={args.turn_detection}; "
        f"semantic_vad eagerness={args.eagerness}; "
        f"server_vad threshold={args.server_vad_threshold}; "
        f"server_vad silence={args.server_vad_silence_ms} ms. "
        f"Input chunk={args.input_chunk_ms} ms. "
        "Metric is final source-audio chunk sent to first response audio delta; "
        "raw transcripts are not recorded."
    )
    _write_live_benchmark_artifacts(
        events,
        out_dir=args.out,
        title=args.title,
        source_note=source_note,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="zemory-sama live realtime audio benchmark")
    parser.add_argument(
        "--eagerness",
        choices=["low", "medium", "high", "auto"],
        default="high",
    )
    parser.add_argument(
        "--mode",
        choices=["semantic_vad", "forced_commit"],
        default="semantic_vad",
    )
    parser.add_argument(
        "--turn-detection",
        choices=["semantic_vad", "server_vad"],
        default="semantic_vad",
    )
    parser.add_argument("--server-vad-silence-ms", type=int, default=300)
    parser.add_argument("--server-vad-threshold", type=float, default=0.5)
    parser.add_argument("--input-chunk-ms", type=int, default=20)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument(
        "--samples",
        nargs="+",
        default=[sample.name for sample in SAMPLES],
        choices=[sample.name for sample in SAMPLES],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
