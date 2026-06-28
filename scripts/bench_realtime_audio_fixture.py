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

Mode = Literal["semantic_vad", "forced_commit", "local_endpoint_commit"]
Eagerness = Literal["low", "medium", "high", "auto"]
TurnDetection = Literal["semantic_vad", "server_vad", "none"]


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
    local_endpoint_required_misses: int | None,
    audio_end_at: float,
    speech_stopped_at: float | None,
    first_audio_at: float,
    first_speaker_write_at: float | None = None,
    first_playback_at: float | None = None,
) -> dict[str, float | str | bool | None]:
    metric_at = first_playback_at or first_audio_at
    early_cutoff = first_audio_at < audio_end_at or (
        speech_stopped_at is not None and speech_stopped_at < audio_end_at
    )
    total_ms = None if early_cutoff else round((metric_at - audio_end_at) * 1000, 1)
    api_first_audio_ms = (
        None if first_audio_at < audio_end_at
        else round((first_audio_at - audio_end_at) * 1000, 1)
    )
    event = {
        "event": "turn.complete",
        "fixture": fixture,
        "voice": voice,
        "profile": "realtime_audio",
        "eagerness": eagerness,
        "turn_detection": turn_detection,
        "server_vad_threshold": server_vad_threshold,
        "input_chunk_ms": input_chunk_ms,
        "local_endpoint_required_misses": local_endpoint_required_misses,
        "interrupted": False,
        "early_cutoff": early_cutoff,
        "sample_source": f"macos_say_{mode}",
        "metric_target": "device_playback" if first_playback_at is not None else "api_first_audio",
        "api_first_audio_ms": api_first_audio_ms,
        "total_ms": total_ms,
        "first_tts_byte_ms": None if total_ms is None else api_first_audio_ms,
        "vad_wait_ms": None,
        "first_audio_after_speech_stopped_ms": None,
        "api_to_playback_ms": None,
        "speaker_buffer_ms": None,
    }
    if speech_stopped_at is not None:
        event["vad_wait_ms"] = round((speech_stopped_at - audio_end_at) * 1000, 1)
        event["first_audio_after_speech_stopped_ms"] = round(
            (first_audio_at - speech_stopped_at) * 1000,
            1,
        )
    if first_playback_at is not None:
        event["api_to_playback_ms"] = round((first_playback_at - first_audio_at) * 1000, 1)
        if first_speaker_write_at is not None:
            event["speaker_buffer_ms"] = round(
                (first_playback_at - first_speaker_write_at) * 1000,
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
        else:
            await asyncio.sleep(0)
    return time.monotonic()


async def _stream_pcm_until_local_endpoint(
    turn,
    pcm: bytes,
    *,
    sample_rate: int,
    input_chunk_ms: int,
    silence_timeout_s: float,
    sleep=asyncio.sleep,
) -> tuple[float, float | None]:
    chunk_duration_s = input_chunk_ms / 1000
    chunk_size = int(sample_rate * chunk_duration_s) * 2

    async def wait_for_speech_end() -> float:
        while True:
            event = await turn.events.get()
            if event == "speech_end":
                return time.monotonic()

    speech_end_task = asyncio.create_task(wait_for_speech_end())
    for chunk in _chunk_pcm(pcm, chunk_size=chunk_size):
        await turn.feed(chunk)
        maybe_sleep = sleep(chunk_duration_s)
        if maybe_sleep is not None:
            await maybe_sleep
        else:
            await asyncio.sleep(0)

    audio_end_at = time.monotonic()
    silence = b"\x00" * chunk_size
    deadline = audio_end_at + silence_timeout_s
    while not speech_end_task.done():
        if time.monotonic() > deadline:
            speech_end_task.cancel()
            try:
                await speech_end_task
            except asyncio.CancelledError:
                pass
            return audio_end_at, None
        await turn.feed(silence)
        maybe_sleep = sleep(chunk_duration_s)
        if maybe_sleep is not None:
            await maybe_sleep
        else:
            await asyncio.sleep(0)

    return audio_end_at, speech_end_task.result()


async def _send_silence(llm, *, sample_rate: int, duration_s: float) -> None:
    chunk_duration_s = 0.02
    chunk_size = int(sample_rate * chunk_duration_s) * 2
    silence = b"\x00" * chunk_size
    chunks = int(duration_s / chunk_duration_s)
    for _ in range(chunks):
        await llm.push_audio(silence)
        await asyncio.sleep(chunk_duration_s)


async def _commit_and_trigger_response(llm, *, audio_end_at: float) -> float:
    commit = getattr(llm, "commit_input_audio_buffer", None)
    if callable(commit):
        await commit()
    else:  # pragma: no cover - compatibility fallback for ad hoc fakes
        if llm._conn is None:
            raise RuntimeError("Realtime connection was not opened")
        await llm._conn.input_audio_buffer.commit()
    await llm.trigger_response()
    return audio_end_at


async def _wait_for_first_audio(
    llm,
    *,
    timeout_s: float,
    speaker=None,
) -> tuple[float | None, float, float | None, float | None]:
    speech_stopped_at: float | None = None

    async def consume() -> tuple[float | None, float, float | None, float | None]:
        nonlocal speech_stopped_at
        async for event in llm.events():
            event_type = event.get("type")
            if event_type == "input.speech_stopped" and speech_stopped_at is None:
                speech_stopped_at = time.monotonic()
            elif event_type == "audio.delta":
                first_audio_at = time.monotonic()
                if speaker is None:
                    return speech_stopped_at, first_audio_at, None, None
                await speaker.queue.put(event["audio"])
                deadline = first_audio_at + timeout_s
                while speaker.first_play_at is None:
                    if time.monotonic() > deadline:
                        raise TimeoutError("speaker playback callback did not consume audio")
                    await asyncio.sleep(0.001)
                return (
                    speech_stopped_at,
                    first_audio_at,
                    speaker.first_write_at,
                    speaker.first_play_at,
                )
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
    local_endpoint_required_misses: int,
    input_chunk_ms: int,
    mode: Mode,
    timeout_s: float,
    play_output: bool,
) -> dict[str, float | str | bool | None]:
    from zemory import config as cfg
    from zemory.providers.llm.openai_realtime import OpenAIRealtimeLLM
    from zemory.providers.turn.realtime_manual import RealtimeManualTurnDetector

    cfg.settings.profile = "realtime_audio"
    cfg.settings.realtime.semantic_vad_eagerness = eagerness
    cfg.settings.realtime.turn_detection = turn_detection
    cfg.settings.realtime.server_vad_threshold = server_vad_threshold
    cfg.settings.realtime.server_vad_silence_duration_ms = server_vad_silence_ms
    cfg.settings.realtime.local_endpoint_required_misses = local_endpoint_required_misses

    llm = OpenAIRealtimeLLM(cfg.OPENAI_API_KEY)
    manual_turn = None
    local_speech_stopped_at: float | None = None
    if mode == "local_endpoint_commit":
        manual_turn = RealtimeManualTurnDetector(llm=llm)
    speaker = None
    feed_task: asyncio.Task | None = None
    if play_output:
        from zemory.audio import SpeakerStream

        speaker = SpeakerStream(asyncio.get_running_loop())
        speaker.start()
        speaker.arm()
        feed_task = asyncio.create_task(speaker.feed())
    await llm.open_session()
    try:
        waiter = asyncio.create_task(
            _wait_for_first_audio(llm, timeout_s=timeout_s, speaker=speaker)
        )
        if mode == "local_endpoint_commit":
            if manual_turn is None:  # pragma: no cover - defensive guard
                raise RuntimeError("manual turn detector was not initialized")
            audio_end_at, local_speech_stopped_at = await _stream_pcm_until_local_endpoint(
                manual_turn,
                pcm,
                sample_rate=cfg.SAMPLE_RATE,
                input_chunk_ms=input_chunk_ms,
                silence_timeout_s=timeout_s,
            )
            if local_speech_stopped_at is None:
                raise TimeoutError("local endpoint detector did not emit speech_end")
            audio_end_at = await _commit_and_trigger_response(
                llm,
                audio_end_at=audio_end_at,
            )
        else:
            audio_end_at = await _stream_pcm_realtime(
                llm,
                pcm,
                sample_rate=cfg.SAMPLE_RATE,
                input_chunk_ms=input_chunk_ms,
            )
            if mode == "forced_commit":
                audio_end_at = await _commit_and_trigger_response(
                    llm,
                    audio_end_at=audio_end_at,
                )
            else:
                await _send_silence(llm, sample_rate=cfg.SAMPLE_RATE, duration_s=1.6)
        (
            api_speech_stopped_at,
            first_audio_at,
            first_speaker_write_at,
            first_playback_at,
        ) = await waiter
        speech_stopped_at = local_speech_stopped_at or api_speech_stopped_at
    finally:
        await llm.close()
        if feed_task is not None:
            feed_task.cancel()
            try:
                await feed_task
            except asyncio.CancelledError:
                pass
        if speaker is not None:
            speaker.stop()
        if manual_turn is not None:
            await manual_turn.close()

    return _event_from_timings(
        fixture=sample.name,
        voice=sample.voice,
        eagerness=eagerness,
        turn_detection=turn_detection,
        server_vad_threshold=server_vad_threshold,
        input_chunk_ms=input_chunk_ms,
        mode=mode,
        local_endpoint_required_misses=(
            local_endpoint_required_misses
            if mode == "local_endpoint_commit"
            else None
        ),
        audio_end_at=audio_end_at,
        speech_stopped_at=speech_stopped_at,
        first_audio_at=first_audio_at,
        first_speaker_write_at=first_speaker_write_at,
        first_playback_at=first_playback_at,
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
                    local_endpoint_required_misses=args.local_endpoint_required_misses,
                    input_chunk_ms=args.input_chunk_ms,
                    mode=args.mode,
                    timeout_s=args.timeout_s,
                    play_output=args.play_output,
                )
                event["trial"] = trial + 1
                events.append(event)
                print(event, flush=True)

    from zemory import config as cfg

    source_note = _source_note(
        args,
        event_count=len(events),
        response_length=cfg.settings.response_length,
    )
    _write_live_benchmark_artifacts(
        events,
        out_dir=args.out,
        title=args.title,
        source_note=source_note,
    )


def _source_note(
    args: argparse.Namespace,
    *,
    event_count: int,
    response_length: str,
) -> str:
    metric_note = (
        "Metric is final source-audio chunk sent to first local speaker playback callback; "
        "api_first_audio_ms retains the API first-audio delta timestamp."
        if args.play_output
        else "Metric is final source-audio chunk sent to first response audio delta."
    )
    local_endpoint_note = (
        f"local endpoint misses={args.local_endpoint_required_misses}. "
        if args.mode == "local_endpoint_commit"
        else ""
    )
    return (
        f"{event_count} macOS say fixtures streamed as realtime 24 kHz PCM. "
        f"Mode={args.mode}; turn_detection={args.turn_detection}; "
        f"semantic_vad eagerness={args.eagerness}; "
        f"server_vad threshold={args.server_vad_threshold}; "
        f"server_vad silence={args.server_vad_silence_ms} ms. "
        f"{local_endpoint_note}"
        f"Input chunk={args.input_chunk_ms} ms. "
        f"response length={response_length}. "
        f"play_output={args.play_output}. "
        f"{metric_note} "
        "raw transcripts are not recorded."
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="realtime-voice-runtime live realtime audio benchmark")
    parser.add_argument(
        "--eagerness",
        choices=["low", "medium", "high", "auto"],
        default="high",
    )
    parser.add_argument(
        "--mode",
        choices=["semantic_vad", "forced_commit", "local_endpoint_commit"],
        default="semantic_vad",
    )
    parser.add_argument(
        "--turn-detection",
        choices=["semantic_vad", "server_vad", "none"],
        default="semantic_vad",
    )
    parser.add_argument("--server-vad-silence-ms", type=int, default=300)
    parser.add_argument("--server-vad-threshold", type=float, default=0.5)
    parser.add_argument("--local-endpoint-required-misses", type=int, default=7)
    parser.add_argument("--input-chunk-ms", type=int, default=20)
    parser.add_argument(
        "--play-output",
        action="store_true",
        help="Route the first response audio delta through SpeakerStream and measure playback.",
    )
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
