#!/usr/bin/env python3
"""Run a live OpenAI Realtime audio benchmark with generated PCM fixtures."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

Mode = Literal["semantic_vad", "forced_commit", "local_endpoint_commit"]
Eagerness = Literal["low", "medium", "high", "auto"]
TurnDetection = Literal["semantic_vad", "server_vad", "none"]
SOURCE_IDENTITY_PATHS = ("zemory", "scripts", "pyproject.toml", "uv.lock")


def _package_version(package: str, fallback: str = "unknown") -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return fallback


RUNTIME_VERSION = _package_version("realtime-voice-runtime", "0.2.0")
OPENAI_SDK_VERSION = _package_version("openai")


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


def _config_hash(config: dict[str, object]) -> str:
    payload = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _public_session_config(
    session_config: dict[str, object] | None,
) -> dict[str, object] | None:
    """Keep reproducibility fields without publishing the instruction text."""
    from zemory.observability.latency_report import (
        sanitize_benchmark_session_config,
    )

    return sanitize_benchmark_session_config(session_config)


def _git_source_identity() -> dict[str, object]:
    """Return content hashes for relevant source state without publishing diffs."""
    identity: dict[str, object] = {
        "git_commit": "unknown",
        "git_dirty": None,
        "git_diff_sha256": None,
        "source_tree_sha256": None,
    }
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
        diff = subprocess.run(
            [
                "git",
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--binary",
                "HEAD",
                "--",
                *SOURCE_IDENTITY_PATHS,
            ],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
        listed = subprocess.run(
            [
                "git",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                *SOURCE_IDENTITY_PATHS,
            ],
            cwd=ROOT,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return identity

    source_digest = hashlib.sha256()
    for relative_bytes in sorted(filter(None, listed.stdout.split(b"\0"))):
        relative = relative_bytes.decode("utf-8", errors="surrogateescape")
        source_path = (ROOT / relative).resolve()
        try:
            source_path.relative_to(ROOT)
        except ValueError:  # pragma: no cover - git paths cannot escape the worktree
            continue
        if not source_path.is_file():
            continue
        content = source_path.read_bytes()
        source_digest.update(len(relative_bytes).to_bytes(8, "big"))
        source_digest.update(relative_bytes)
        source_digest.update(len(content).to_bytes(8, "big"))
        source_digest.update(content)

    identity.update(
        {
            "git_commit": revision.stdout.decode("ascii").strip(),
            "git_dirty": bool(status.stdout),
            "git_diff_sha256": hashlib.sha256(diff.stdout).hexdigest(),
            "source_tree_sha256": source_digest.hexdigest(),
        }
    )
    return identity


def _openai_base_url_identity(base_url: str | None) -> dict[str, str | None]:
    if base_url is None:
        return {
            "openai_base_url_kind": "unknown",
            "openai_base_url_sha256": None,
        }
    hostname = (urlsplit(base_url).hostname or "").casefold()
    return {
        "openai_base_url_kind": "official" if hostname == "api.openai.com" else "custom",
        "openai_base_url_sha256": hashlib.sha256(base_url.encode("utf-8")).hexdigest(),
    }


def _apply_realtime_session_settings(
    settings,
    *,
    eagerness: Eagerness,
    turn_detection: TurnDetection,
    server_vad_threshold: float,
    server_vad_silence_ms: int,
) -> None:
    settings.profile = "realtime_audio"
    settings.realtime.semantic_vad_eagerness = eagerness
    settings.realtime.turn_detection = turn_detection
    settings.realtime.server_vad_threshold = server_vad_threshold
    settings.realtime.server_vad_silence_duration_ms = server_vad_silence_ms


def _benchmark_config(
    args: argparse.Namespace,
    *,
    model: str,
    response_length: str,
    session_config: dict[str, object] | None = None,
    openai_base_url: str | None = None,
    rendered_pcm: dict[str, bytes] | None = None,
) -> dict[str, object]:
    selected_names = set(args.samples)
    fixture_payload = [
        {"name": sample.name, "voice": sample.voice, "text": sample.text}
        for sample in SAMPLES
        if sample.name in selected_names
    ]
    fixture_hash = _config_hash({"fixtures": fixture_payload})
    source_identity = _git_source_identity()
    endpoint_identity = _openai_base_url_identity(openai_base_url)
    return {
        "schema_version": "zemory.latency.v1",
        "runtime_version": RUNTIME_VERSION,
        "openai_sdk_version": OPENAI_SDK_VERSION,
        **endpoint_identity,
        "model": model,
        "response_length": response_length,
        "mode": args.mode,
        "turn_detection": args.turn_detection,
        "eagerness": args.eagerness,
        "server_vad_threshold": args.server_vad_threshold,
        "server_vad_silence_ms": args.server_vad_silence_ms,
        "local_endpoint_required_misses": args.local_endpoint_required_misses,
        "input_chunk_ms": args.input_chunk_ms,
        "play_output": args.play_output,
        "measure_interrupt": args.measure_interrupt,
        "trials": args.trials,
        "timeout_s": args.timeout_s,
        "samples": sorted(args.samples),
        "fixture_corpus_hash": fixture_hash,
        "fixture_pcm_sha256": {
            name: hashlib.sha256(pcm).hexdigest()
            for name, pcm in sorted((rendered_pcm or {}).items())
        },
        "session_config": _public_session_config(session_config),
        "platform": platform.platform(),
        **source_identity,
    }


def _annotate_turn_event(
    event: dict[str, object],
    *,
    run_id: str,
    config_hash: str,
    turn_id: str,
    model: str,
    response_length: str,
    server_vad_silence_ms: int,
    measure_interrupt: bool,
    benchmark_config: dict[str, object] | None = None,
) -> dict[str, object]:
    event.update(
        {
            "run_id": run_id,
            "config_hash": config_hash,
            "schema_version": "zemory.latency.v1",
            "turn_id": turn_id,
            "metric_origin": "bench_realtime_audio_fixture",
            "model": model,
            "response_length": response_length,
            "server_vad_silence_ms": server_vad_silence_ms,
            "measure_interrupt": measure_interrupt,
            "runtime_version": RUNTIME_VERSION,
            "openai_sdk_version": OPENAI_SDK_VERSION,
        }
    )
    if benchmark_config is not None:
        event["benchmark_config"] = benchmark_config
    return event


def _interrupt_event_from_timings(
    *,
    run_id: str,
    config_hash: str,
    interrupt_id: str,
    fixture: str,
    trial: int,
    model: str,
    response_length: str,
    mode: Mode,
    turn_detection: TurnDetection,
    eagerness: Eagerness,
    server_vad_threshold: float,
    server_vad_silence_ms: int,
    input_chunk_ms: int,
    speech_started_at: float,
    audible_silence_at: float,
) -> dict[str, object]:
    if audible_silence_at < speech_started_at:
        raise ValueError("audible silence cannot precede speech start")
    from zemory.observability.latency_report import (
        INTERRUPT_RELEASE_METRIC_TARGET,
        LATENCY_SCHEMA_VERSION,
    )

    return {
        "event": "interrupt.done",
        "run_id": run_id,
        "config_hash": config_hash,
        "schema_version": LATENCY_SCHEMA_VERSION,
        "interrupt_id": interrupt_id,
        "fixture": fixture,
        "trial": trial,
        "profile": "realtime_audio",
        "sample_source": "macos_say_barge_in",
        "metric_origin": "portaudio_output_dac_schedule",
        "metric_target": INTERRUPT_RELEASE_METRIC_TARGET,
        "model": model,
        "response_length": response_length,
        "mode": mode,
        "play_output": True,
        "measure_interrupt": True,
        "turn_detection": turn_detection,
        "eagerness": eagerness,
        "server_vad_threshold": server_vad_threshold,
        "server_vad_silence_ms": server_vad_silence_ms,
        "input_chunk_ms": input_chunk_ms,
        "runtime_version": RUNTIME_VERSION,
        "openai_sdk_version": OPENAI_SDK_VERSION,
        "interrupt_ms": round((audible_silence_at - speech_started_at) * 1000, 1),
    }


def _invalid_interrupt_event(*, reason: str, **metadata: object) -> dict[str, object]:
    event = _interrupt_event_from_timings(
        **metadata,
        speech_started_at=0.0,
        audible_silence_at=0.0,
    )
    event["interrupt_ms"] = None
    event["invalid_reason"] = reason
    return event


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
        "mode": mode,
        "play_output": first_playback_at is not None,
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


def _write_jsonl(path: Path, events: list[dict[str, object]]) -> None:
    rows = [
        json.dumps(event, ensure_ascii=False, sort_keys=True, allow_nan=False)
        for event in events
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_live_benchmark_artifacts(
    events: list[dict[str, object]],
    *,
    out_dir: Path,
    title: str,
    source_note: str,
) -> None:
    from zemory.observability.benchmark_artifacts import write_benchmark_artifacts
    from zemory.observability.latency_report import sanitize_latency_event

    safe_events = [sanitize_latency_event(event) for event in events]
    valid_events = [event for event in safe_events if event.get("total_ms") is not None]
    if valid_events:
        write_benchmark_artifacts(
            safe_events,
            out_dir=out_dir,
            title=title,
            source_note=source_note,
        )
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "latency-events.jsonl", safe_events)
    early_cutoff_count = sum(1 for event in safe_events if event.get("early_cutoff"))
    summary = {
        "title": title,
        "source_note": source_note,
        "turn_count": 0,
        "total_event_count": len(safe_events),
        "invalid_latency_count": len(safe_events),
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
{len(safe_events)} event(s) were invalid, including {early_cutoff_count} early cutoff
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


def _scheduled_dac_at(time_info: object, *, callback_at: float) -> float:
    """Translate PortAudio's next-buffer DAC delay into monotonic time."""

    def read(name: str) -> float | None:
        value = getattr(time_info, name, None)
        if value is None and isinstance(time_info, dict):
            value = time_info.get(name)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    current = read("currentTime")
    output_dac = read("outputBufferDacTime")
    if current is None or output_dac is None:
        return callback_at
    return callback_at + max(0.0, output_dac - current)


def _new_speaker_probe(loop: asyncio.AbstractEventLoop):
    """Create a SpeakerStream that reports the scheduled DAC silence boundary."""
    from zemory.audio import SpeakerStream

    class SpeakerProbe(SpeakerStream):
        def __init__(self) -> None:
            super().__init__(loop)
            self._silence_probe_armed = False
            self._silence_event = asyncio.Event()
            self.audible_silence_at: float | None = None
            self.output_active = False

        def arm_silence_probe(self) -> None:
            if not self.output_active:
                raise RuntimeError("assistant audio is not active at barge-in detection")
            self.audible_silence_at = None
            self._silence_event.clear()
            self._silence_probe_armed = True

        async def wait_for_audible_silence(self, *, timeout_s: float) -> float:
            await asyncio.wait_for(self._silence_event.wait(), timeout=timeout_s)
            if self.audible_silence_at is None:  # pragma: no cover - event invariant
                raise RuntimeError("silence callback did not record a timestamp")
            return self.audible_silence_at

        def _record_audible_silence(self, timestamp: float) -> None:
            if not self._silence_probe_armed:
                return
            self._silence_probe_armed = False
            self.audible_silence_at = timestamp
            self._silence_event.set()

        def _callback(self, outdata, frames, time_info, status) -> None:
            super()._callback(outdata, frames, time_info, status)
            self.output_active = bool(outdata.any())
            if self._silence_probe_armed and not self.output_active:
                callback_at = time.monotonic()
                dac_at = _scheduled_dac_at(time_info, callback_at=callback_at)
                self._loop.call_soon_threadsafe(self._record_audible_silence, dac_at)

    return SpeakerProbe()


async def _consume_until_audible_interrupt(
    llm,
    speaker,
    *,
    barge_in_started: asyncio.Event,
    timeout_s: float,
) -> tuple[float, float]:
    """Feed response audio and measure server speech-start to DAC-scheduled silence."""
    async for event in llm.events():
        event_type = event.get("type")
        if event_type == "audio.delta":
            await speaker.queue.put(event["audio"])
        elif event_type == "input.speech_started" and barge_in_started.is_set():
            speech_started_at = time.monotonic()
            speaker.arm_silence_probe()
            speaker.clear()
            await llm.cancel_current()
            audible_silence_at = await speaker.wait_for_audible_silence(
                timeout_s=timeout_s
            )
            return speech_started_at, audible_silence_at
        elif event_type == "error":
            raise RuntimeError(str(event.get("error")))
    raise RuntimeError("Realtime event stream ended before audible interrupt")


async def _wait_for_speaker_start(speaker, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while speaker.first_play_at is None:
        if time.monotonic() > deadline:
            raise TimeoutError("assistant audio did not reach the speaker callback")
        await asyncio.sleep(0.001)


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
    trigger_on_transcript: bool = False,
) -> tuple[float | None, float, float | None, float | None]:
    speech_stopped_at: float | None = None
    response_requested = False

    async def consume() -> tuple[float | None, float, float | None, float | None]:
        nonlocal speech_stopped_at, response_requested
        async for event in llm.events():
            event_type = event.get("type")
            if event_type == "input.speech_stopped" and speech_stopped_at is None:
                speech_stopped_at = time.monotonic()
            elif (
                event_type == "input.transcript"
                and trigger_on_transcript
                and not response_requested
                and str(event.get("text", "")).strip()
            ):
                # Runtime sessions deliberately configure create_response=false
                # to prevent empty/noise transcripts from self-triggering. The
                # live benchmark must exercise that same application-owned
                # response gate instead of waiting for obsolete server auto-VAD.
                response_requested = True
                await llm.trigger_response()
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

    _apply_realtime_session_settings(
        cfg.settings,
        eagerness=eagerness,
        turn_detection=turn_detection,
        server_vad_threshold=server_vad_threshold,
        server_vad_silence_ms=server_vad_silence_ms,
    )
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
            _wait_for_first_audio(
                llm,
                timeout_s=timeout_s,
                speaker=speaker,
                trigger_on_transcript=mode == "semantic_vad",
            )
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


async def _measure_interrupt_sample(
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
    run_id: str,
    config_hash: str,
    interrupt_id: str,
    trial: int,
    model: str,
    response_length: str,
) -> dict[str, object]:
    """Measure live barge-in detection to the first DAC-scheduled silent buffer."""
    if turn_detection == "none" or mode == "local_endpoint_commit":
        raise ValueError(
            "audible interrupt measurement requires server turn detection"
        )

    from zemory import config as cfg
    from zemory.providers.llm.openai_realtime import OpenAIRealtimeLLM

    _apply_realtime_session_settings(
        cfg.settings,
        eagerness=eagerness,
        turn_detection=turn_detection,
        server_vad_threshold=server_vad_threshold,
        server_vad_silence_ms=server_vad_silence_ms,
    )

    llm = OpenAIRealtimeLLM(cfg.OPENAI_API_KEY)
    speaker = _new_speaker_probe(asyncio.get_running_loop())
    speaker.start()
    speaker.arm()
    feed_task = asyncio.create_task(speaker.feed())
    barge_in_started = asyncio.Event()
    consume_task: asyncio.Task | None = None
    stream_task: asyncio.Task | None = None
    try:
        await llm.open_session()
        consume_task = asyncio.create_task(
            _consume_until_audible_interrupt(
                llm,
                speaker,
                barge_in_started=barge_in_started,
                timeout_s=timeout_s,
            )
        )
        await llm.send_user_text(
            "In one sentence, slowly count aloud from one to thirty without abbreviating."
        )
        await _wait_for_speaker_start(speaker, timeout_s=timeout_s)

        barge_in_started.set()
        stream_task = asyncio.create_task(
            _stream_pcm_realtime(
                llm,
                pcm,
                sample_rate=cfg.SAMPLE_RATE,
                input_chunk_ms=input_chunk_ms,
            )
        )
        speech_started_at, audible_silence_at = await asyncio.wait_for(
            consume_task,
            timeout=timeout_s,
        )
    finally:
        for task in (stream_task, consume_task, feed_task):
            if task is not None and not task.done():
                task.cancel()
            if task is not None:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await llm.close()
        speaker.stop()

    return _interrupt_event_from_timings(
        run_id=run_id,
        config_hash=config_hash,
        interrupt_id=interrupt_id,
        fixture=sample.name,
        trial=trial,
        model=model,
        response_length=response_length,
        mode=mode,
        turn_detection=turn_detection,
        eagerness=eagerness,
        server_vad_threshold=server_vad_threshold,
        server_vad_silence_ms=server_vad_silence_ms,
        input_chunk_ms=input_chunk_ms,
        speech_started_at=speech_started_at,
        audible_silence_at=audible_silence_at,
    )


async def _run(args: argparse.Namespace) -> None:
    _require_binary("say")
    _require_binary("ffmpeg")

    selected = [sample for sample in SAMPLES if sample.name in set(args.samples)]
    if not selected:
        raise SystemExit("No matching samples selected")

    from zemory import config as cfg

    # Capture the same effective session that each sample opens. In particular,
    # forced-commit runs must not retain the process default VAD configuration.
    _apply_realtime_session_settings(
        cfg.settings,
        eagerness=args.eagerness,
        turn_detection=args.turn_detection,
        server_vad_threshold=args.server_vad_threshold,
        server_vad_silence_ms=args.server_vad_silence_ms,
    )
    cfg.settings.realtime.local_endpoint_required_misses = (
        args.local_endpoint_required_misses
    )
    model = cfg.settings.realtime.model
    response_length = cfg.settings.response_length
    run_id = uuid.uuid4().hex
    events: list[dict[str, object]] = []
    completed = False
    try:
        with tempfile.TemporaryDirectory(prefix="zemory-bench-audio-") as tmp:
            tmp_dir = Path(tmp)
            rendered = {sample.name: _render_sample(sample, tmp_dir) for sample in selected}
            benchmark_config = _benchmark_config(
                args,
                model=model,
                response_length=response_length,
                session_config=cfg.build_session_config(),
                openai_base_url=cfg.settings.openai_base_url,
                rendered_pcm=rendered,
            )
            from zemory.observability.latency_report import (
                canonical_benchmark_config_hash,
            )

            config_hash = canonical_benchmark_config_hash(benchmark_config)
            for trial in range(args.trials):
                for sample in selected:
                    event = await _measure_sample(
                        sample,
                        rendered[sample.name],
                        eagerness=args.eagerness,
                        turn_detection=args.turn_detection,
                        server_vad_threshold=args.server_vad_threshold,
                        server_vad_silence_ms=args.server_vad_silence_ms,
                        local_endpoint_required_misses=(
                            args.local_endpoint_required_misses
                        ),
                        input_chunk_ms=args.input_chunk_ms,
                        mode=args.mode,
                        timeout_s=args.timeout_s,
                        play_output=args.play_output,
                    )
                    event["trial"] = trial + 1
                    _annotate_turn_event(
                        event,
                        run_id=run_id,
                        config_hash=config_hash,
                        turn_id=f"{trial + 1}:{sample.name}",
                        model=model,
                        response_length=response_length,
                        server_vad_silence_ms=args.server_vad_silence_ms,
                        measure_interrupt=args.measure_interrupt,
                        benchmark_config=benchmark_config,
                    )
                    events.append(event)
                    print(event, flush=True)
                    if args.measure_interrupt:
                        interrupt_metadata = {
                            "run_id": run_id,
                            "config_hash": config_hash,
                            "interrupt_id": f"{trial + 1}:{sample.name}:barge-in",
                            "trial": trial + 1,
                            "model": model,
                            "response_length": response_length,
                            "mode": args.mode,
                            "turn_detection": args.turn_detection,
                            "eagerness": args.eagerness,
                            "server_vad_threshold": args.server_vad_threshold,
                            "server_vad_silence_ms": args.server_vad_silence_ms,
                            "input_chunk_ms": args.input_chunk_ms,
                        }
                        try:
                            interrupt_event = await _measure_interrupt_sample(
                                sample,
                                rendered[sample.name],
                                timeout_s=args.timeout_s,
                                **interrupt_metadata,
                            )
                        except (TimeoutError, RuntimeError) as exc:
                            interrupt_event = _invalid_interrupt_event(
                                reason=type(exc).__name__,
                                fixture=sample.name,
                                **interrupt_metadata,
                            )
                        events.append(interrupt_event)
                        interrupt_event["benchmark_config"] = benchmark_config
                        print(interrupt_event, flush=True)
        completed = True
    finally:
        if events:
            source_note = _source_note(
                args,
                event_count=len(events),
                response_length=response_length,
            )
            if not completed:
                source_note += " Run terminated early; artifacts contain partial samples."
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
    interrupt_note = (
        "Interrupt metric is server speech_started receipt to the first PortAudio "
        "silent buffer's scheduled DAC time. "
        if args.measure_interrupt
        else "No audible interrupt samples were requested. "
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
        f"{interrupt_note}"
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
    parser.add_argument(
        "--measure-interrupt",
        action="store_true",
        help=(
            "Run a second live barge-in probe per fixture and measure speech_started "
            "to the PortAudio scheduled DAC silence boundary."
        ),
    )
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument(
        "--samples",
        nargs="+",
        default=[sample.name for sample in SAMPLES],
        choices=[sample.name for sample in SAMPLES],
    )
    args = parser.parse_args(argv)
    if args.measure_interrupt and not args.play_output:
        parser.error("--measure-interrupt requires --play-output")
    if args.measure_interrupt and (
        args.turn_detection == "none" or args.mode == "local_endpoint_commit"
    ):
        parser.error("--measure-interrupt requires server turn detection")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
