"""Realtime audio benchmark script helper tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.bench_realtime_audio_fixture import (
    Sample,
    _annotate_turn_event,
    _apply_realtime_session_settings,
    _benchmark_config,
    _chunk_pcm,
    _commit_and_trigger_response,
    _config_hash,
    _consume_until_audible_interrupt,
    _event_from_timings,
    _interrupt_event_from_timings,
    _measure_interrupt_sample,
    _measure_sample,
    _new_speaker_probe,
    _parse_args,
    _public_session_config,
    _scheduled_dac_at,
    _source_note,
    _stream_pcm_realtime,
    _stream_pcm_until_local_endpoint,
    _wait_for_first_audio,
    _write_live_benchmark_artifacts,
)
from zemory.observability.latency_report import (
    LatencyGate,
    LatencyReport,
    canonical_benchmark_config_hash,
    parse_structlog_latency_events,
    sanitize_latency_event,
)


def test_chunk_pcm_splits_pcm_without_losing_tail() -> None:
    assert list(_chunk_pcm(b"abcdefg", chunk_size=3)) == [b"abc", b"def", b"g"]


def test_source_note_records_response_length_without_irrelevant_endpoint_setting() -> None:
    args = _parse_args(
        [
            "--out",
            "out",
            "--mode",
            "semantic_vad",
            "--turn-detection",
            "server_vad",
        ]
    )

    note = _source_note(
        args,
        event_count=8,
        response_length="one short sentence",
    )

    assert "response length=one short sentence" in note
    assert "local endpoint misses" not in note


def test_source_note_records_local_endpoint_miss_window() -> None:
    args = _parse_args(
        [
            "--out",
            "out",
            "--mode",
            "local_endpoint_commit",
            "--turn-detection",
            "none",
            "--local-endpoint-required-misses",
            "14",
        ]
    )

    note = _source_note(
        args,
        event_count=4,
        response_length="one short sentence",
    )

    assert "response length=one short sentence" in note
    assert "local endpoint misses=14" in note


def test_benchmark_config_hash_is_canonical_and_configuration_sensitive() -> None:
    assert _config_hash({"b": 2, "a": 1}) == _config_hash({"a": 1, "b": 2})
    assert _config_hash({"a": 1}) != _config_hash({"a": 2})


def test_benchmark_config_captures_protocol_endpoint_source_and_pcm_identity() -> None:
    first_args = _parse_args(
        [
            "--out",
            "out",
            "--turn-detection",
            "none",
            "--trials",
            "1",
            "--timeout-s",
            "5",
        ]
    )
    second_args = _parse_args(
        [
            "--out",
            "out",
            "--turn-detection",
            "none",
            "--trials",
            "2",
            "--timeout-s",
            "15",
        ]
    )
    private_endpoint = "https://private-gateway.example/internal/v1"
    first = _benchmark_config(
        first_args,
        model="test-model",
        response_length="short",
        openai_base_url=private_endpoint,
        rendered_pcm={"en_short": b"pcm-one"},
    )
    second = _benchmark_config(
        second_args,
        model="test-model",
        response_length="short",
        openai_base_url=private_endpoint,
        rendered_pcm={"en_short": b"pcm-two"},
    )

    assert first["trials"] == 1
    assert first["timeout_s"] == 5.0
    assert first["openai_base_url_kind"] == "custom"
    assert first["openai_base_url_sha256"] == hashlib.sha256(
        private_endpoint.encode("utf-8")
    ).hexdigest()
    assert private_endpoint not in json.dumps(first)
    assert first["fixture_pcm_sha256"]["en_short"] == hashlib.sha256(
        b"pcm-one"
    ).hexdigest()
    assert len(first["git_diff_sha256"]) == 64
    assert len(first["source_tree_sha256"]) == 64
    assert canonical_benchmark_config_hash(first) != canonical_benchmark_config_hash(
        second
    )


def test_apply_realtime_session_settings_captures_forced_commit_effective_config() -> None:
    settings = SimpleNamespace(
        profile="realtime_text_external_tts",
        realtime=SimpleNamespace(
            semantic_vad_eagerness="low",
            turn_detection="server_vad",
            server_vad_threshold=0.2,
            server_vad_silence_duration_ms=900,
        ),
    )

    _apply_realtime_session_settings(
        settings,
        eagerness="high",
        turn_detection="none",
        server_vad_threshold=0.5,
        server_vad_silence_ms=200,
    )

    assert settings.profile == "realtime_audio"
    assert settings.realtime.semantic_vad_eagerness == "high"
    assert settings.realtime.turn_detection == "none"
    assert settings.realtime.server_vad_threshold == 0.5
    assert settings.realtime.server_vad_silence_duration_ms == 200


def test_public_session_config_hashes_instructions_without_mutating_source() -> None:
    session = {
        "model": "test-model",
        "instructions": "private benchmark instructions",
        "max_output_tokens": 512,
        "reasoning": {"effort": "low"},
    }

    public = _public_session_config(session)

    assert public is not None
    assert "instructions" not in public
    assert public["instructions_sha256"] == hashlib.sha256(
        b"private benchmark instructions"
    ).hexdigest()
    assert public["max_output_tokens"] == 512
    assert session["instructions"] == "private benchmark instructions"


def test_sanitized_event_retains_auditable_config_hash() -> None:
    args = _parse_args(["--out", "out", "--turn-detection", "none"])
    benchmark_config = _benchmark_config(
        args,
        model="test-model",
        response_length="one short sentence",
        session_config={
            "instructions": "private benchmark instructions",
            "max_output_tokens": 512,
        },
    )
    config_hash = _config_hash(benchmark_config)
    event = _annotate_turn_event(
        {"event": "turn.complete", "total_ms": 500.0},
        run_id="run-auditable",
        config_hash=config_hash,
        turn_id="turn-1",
        model="test-model",
        response_length="one short sentence",
        server_vad_silence_ms=300,
        measure_interrupt=False,
        benchmark_config=benchmark_config,
    )

    safe_event = sanitize_latency_event(event)

    assert _config_hash(safe_event["benchmark_config"]) == safe_event["config_hash"]
    assert "instructions" not in safe_event["benchmark_config"]["session_config"]


def test_live_producer_schema_survives_parser_and_passes_strict_gate() -> None:
    args = _parse_args(
        [
            "--out",
            "out",
            "--turn-detection",
            "server_vad",
            "--play-output",
            "--measure-interrupt",
            "--trials",
            "2",
        ]
    )
    benchmark_config = _benchmark_config(
        args,
        model="test-model",
        response_length="one short sentence",
    )
    config_hash = canonical_benchmark_config_hash(benchmark_config)
    events: list[dict] = []
    for index in range(8):
        turn = _event_from_timings(
            fixture=f"fixture-{index % 4}",
            voice="Samantha",
            eagerness="high",
            turn_detection="server_vad",
            server_vad_threshold=0.5,
            input_chunk_ms=20,
            mode="semantic_vad",
            local_endpoint_required_misses=None,
            audio_end_at=100.0,
            speech_stopped_at=100.2,
            first_audio_at=100.5,
            first_speaker_write_at=100.501,
            first_playback_at=100.507,
        )
        turn["trial"] = index // 4 + 1
        events.append(
            _annotate_turn_event(
                turn,
                run_id="run-live",
                config_hash=config_hash,
                turn_id=f"turn-{index}",
                model="test-model",
                response_length="one short sentence",
                server_vad_silence_ms=300,
                measure_interrupt=True,
                benchmark_config=benchmark_config,
            )
        )
        interrupt = _interrupt_event_from_timings(
            run_id="run-live",
            config_hash=config_hash,
            interrupt_id=f"interrupt-{index}",
            fixture=f"fixture-{index % 4}",
            trial=index // 4 + 1,
            model="test-model",
            response_length="one short sentence",
            mode="semantic_vad",
            turn_detection="server_vad",
            eagerness="high",
            server_vad_threshold=0.5,
            server_vad_silence_ms=300,
            input_chunk_ms=20,
            speech_started_at=200.0,
            audible_silence_at=200.08,
        )
        interrupt["benchmark_config"] = benchmark_config
        events.append(interrupt)

    parsed = parse_structlog_latency_events(
        "\n".join(json.dumps(event) for event in events)
    )
    report = LatencyReport.from_events(parsed)
    decision = report.evaluate_gate(
        LatencyGate(
            turn_p50_ms=700,
            turn_p95_ms=1200,
            interrupt_p95_ms=150,
        )
    )

    assert decision.passed, decision.failure_reasons
    assert report.turn_count == 8
    assert report.interrupt_count == 8


def test_event_from_timings_records_vad_and_model_audio_segments() -> None:
    event = _event_from_timings(
        fixture="ko_short",
        voice="Yuna",
        eagerness="high",
        turn_detection="semantic_vad",
        server_vad_threshold=0.5,
        input_chunk_ms=10,
        mode="semantic_vad",
        local_endpoint_required_misses=None,
        audio_end_at=100.0,
        speech_stopped_at=100.25,
        first_audio_at=100.75,
    )

    assert event["fixture"] == "ko_short"
    assert event["voice"] == "Yuna"
    assert event["eagerness"] == "high"
    assert event["turn_detection"] == "semantic_vad"
    assert event["server_vad_threshold"] == 0.5
    assert event["input_chunk_ms"] == 10
    assert event["local_endpoint_required_misses"] is None
    assert event["sample_source"] == "macos_say_semantic_vad"
    assert event["total_ms"] == pytest.approx(750.0)
    assert event["vad_wait_ms"] == pytest.approx(250.0)
    assert event["first_audio_after_speech_stopped_ms"] == pytest.approx(500.0)
    assert event["metric_target"] == "api_first_audio"


def test_event_from_timings_can_measure_device_playback() -> None:
    event = _event_from_timings(
        fixture="ko_short",
        voice="Yuna",
        eagerness="high",
        turn_detection="server_vad",
        server_vad_threshold=0.5,
        input_chunk_ms=20,
        mode="semantic_vad",
        local_endpoint_required_misses=None,
        audio_end_at=100.0,
        speech_stopped_at=100.2,
        first_audio_at=100.7,
        first_speaker_write_at=100.701,
        first_playback_at=100.709,
    )

    assert event["metric_target"] == "device_playback"
    assert event["total_ms"] == pytest.approx(709.0)
    assert event["first_tts_byte_ms"] == pytest.approx(700.0)
    assert event["api_first_audio_ms"] == pytest.approx(700.0)
    assert event["api_to_playback_ms"] == pytest.approx(9.0)
    assert event["speaker_buffer_ms"] == pytest.approx(8.0)


def test_event_from_timings_excludes_early_cutoff_from_latency_samples() -> None:
    event = _event_from_timings(
        fixture="en_short",
        voice="Samantha",
        eagerness="high",
        turn_detection="server_vad",
        server_vad_threshold=0.5,
        input_chunk_ms=20,
        mode="semantic_vad",
        local_endpoint_required_misses=None,
        audio_end_at=100.0,
        speech_stopped_at=99.5,
        first_audio_at=99.9,
    )

    assert event["early_cutoff"] is True
    assert event["total_ms"] is None
    assert event["first_tts_byte_ms"] is None
    assert event["first_audio_after_speech_stopped_ms"] == pytest.approx(400.0)


def test_event_from_timings_excludes_speech_stop_before_audio_end() -> None:
    event = _event_from_timings(
        fixture="en_short",
        voice="Samantha",
        eagerness="high",
        turn_detection="server_vad",
        server_vad_threshold=0.5,
        input_chunk_ms=20,
        mode="semantic_vad",
        local_endpoint_required_misses=None,
        audio_end_at=100.0,
        speech_stopped_at=99.7,
        first_audio_at=100.2,
    )

    assert event["early_cutoff"] is True
    assert event["total_ms"] is None


def test_write_live_benchmark_artifacts_handles_invalid_only_probe(tmp_path) -> None:
    event = _event_from_timings(
        fixture="en_short",
        voice="Samantha",
        eagerness="high",
        turn_detection="server_vad",
        server_vad_threshold=0.6,
        input_chunk_ms=20,
        mode="semantic_vad",
        local_endpoint_required_misses=None,
        audio_end_at=100.0,
        speech_stopped_at=99.5,
        first_audio_at=100.1,
    )
    event["transcript"] = "PRIVATE transcript"

    _write_live_benchmark_artifacts(
        [event],
        out_dir=tmp_path,
        title="invalid probe",
        source_note="probe note",
    )

    summary = (tmp_path / "summary.json").read_text(encoding="utf-8")
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert '"turn_count": 0' in summary
    assert '"total_event_count": 1' in summary
    assert '"invalid_latency_count": 1' in summary
    assert '"early_cutoff_count": 1' in summary
    assert "No valid latency samples" in readme
    assert "PRIVATE" not in (tmp_path / "latency-events.jsonl").read_text(
        encoding="utf-8"
    )


def test_parse_args_accepts_input_chunk_ms_and_play_output() -> None:
    args = _parse_args(
        [
            "--out",
            "out",
            "--input-chunk-ms",
            "10",
            "--mode",
            "local_endpoint_commit",
            "--turn-detection",
            "none",
            "--local-endpoint-required-misses",
            "14",
            "--play-output",
        ]
    )

    assert args.input_chunk_ms == 10
    assert args.mode == "local_endpoint_commit"
    assert args.turn_detection == "none"
    assert args.local_endpoint_required_misses == 14
    assert args.play_output is True
    assert args.trials == 2


def test_parse_args_rejects_interrupt_measurement_without_server_vad() -> None:
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--out",
                "out",
                "--mode",
                "local_endpoint_commit",
                "--turn-detection",
                "none",
                "--measure-interrupt",
            ]
        )


def test_scheduled_dac_at_uses_portaudio_buffer_deadline() -> None:
    time_info = SimpleNamespace(currentTime=10.0, outputBufferDacTime=10.012)

    assert _scheduled_dac_at(time_info, callback_at=100.0) == pytest.approx(100.012)


async def test_consume_until_audible_interrupt_measures_silence_not_chain_time(
    monkeypatch,
) -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.cancelled = False

        async def events(self):
            yield {"type": "audio.delta", "audio": b"audio"}
            yield {"type": "input.speech_started"}

        async def cancel_current(self) -> None:
            self.cancelled = True

    class FakeSpeaker:
        def __init__(self) -> None:
            self.queue: asyncio.Queue[bytes] = asyncio.Queue()
            self.armed = False
            self.cleared = False

        def arm_silence_probe(self) -> None:
            self.armed = True

        def clear(self) -> None:
            self.cleared = True

        async def wait_for_audible_silence(self, *, timeout_s: float) -> float:
            return 100.08

    llm = FakeLLM()
    speaker = FakeSpeaker()
    barge_in_started = asyncio.Event()
    barge_in_started.set()
    monkeypatch.setattr(
        "scripts.bench_realtime_audio_fixture.time.monotonic",
        lambda: 100.0,
    )

    speech_started_at, audible_silence_at = await _consume_until_audible_interrupt(
        llm,
        speaker,
        barge_in_started=barge_in_started,
        timeout_s=1.0,
    )

    assert speech_started_at == pytest.approx(100.0)
    assert audible_silence_at == pytest.approx(100.08)
    assert speaker.armed and speaker.cleared and llm.cancelled
    assert await speaker.queue.get() == b"audio"


async def test_speaker_probe_reports_scheduled_dac_silence() -> None:
    probe = _new_speaker_probe(asyncio.get_running_loop())
    frames = 240
    with probe._lock:
        probe._buffer.extend(np.ones(frames, dtype=np.int16).tobytes())
    outdata = np.zeros((frames, 1), dtype=np.int16)
    time_info = SimpleNamespace(currentTime=10.0, outputBufferDacTime=10.012)

    probe._callback(outdata, frames, time_info, 0)
    assert probe.output_active is True
    probe.arm_silence_probe()
    probe.clear()
    probe._callback(outdata, frames, time_info, 0)
    audible_at = await probe.wait_for_audible_silence(timeout_s=1.0)

    assert probe.output_active is False
    assert audible_at >= 0.012


async def test_measure_interrupt_sample_uses_long_text_then_emits_audible_event(
    monkeypatch,
) -> None:
    from zemory.providers.llm import openai_realtime

    class FakeLLM:
        instance = None

        def __init__(self, _api_key: str) -> None:
            type(self).instance = self
            self.opened = False
            self.closed = False
            self.texts: list[str] = []

        async def open_session(self) -> None:
            self.opened = True

        async def close(self) -> None:
            self.closed = True

        async def send_user_text(self, text: str) -> None:
            self.texts.append(text)

    class FakeSpeaker:
        def __init__(self) -> None:
            self.started = False
            self.stopped = False

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

        def arm(self) -> None:
            pass

        async def feed(self) -> None:
            await asyncio.Event().wait()

    speaker = FakeSpeaker()

    async def fake_consume(*_args, **_kwargs):
        return 100.0, 100.025

    async def fake_wait(*_args, **_kwargs):
        return None

    async def fake_stream(*_args, **_kwargs):
        return 101.0

    monkeypatch.setattr(openai_realtime, "OpenAIRealtimeLLM", FakeLLM)
    monkeypatch.setattr(
        "scripts.bench_realtime_audio_fixture._new_speaker_probe",
        lambda _loop: speaker,
    )
    monkeypatch.setattr(
        "scripts.bench_realtime_audio_fixture._consume_until_audible_interrupt",
        fake_consume,
    )
    monkeypatch.setattr(
        "scripts.bench_realtime_audio_fixture._wait_for_speaker_start",
        fake_wait,
    )
    monkeypatch.setattr(
        "scripts.bench_realtime_audio_fixture._stream_pcm_realtime",
        fake_stream,
    )

    event = await _measure_interrupt_sample(
        Sample("unit", "Samantha", "test"),
        b"pcm",
        eagerness="high",
        turn_detection="server_vad",
        server_vad_threshold=0.5,
        server_vad_silence_ms=200,
        input_chunk_ms=20,
        mode="semantic_vad",
        timeout_s=1.0,
        run_id="run-live",
        config_hash="config-live",
        interrupt_id="interrupt-1",
        trial=1,
        model="test-model",
        response_length="one short sentence",
    )

    assert event["interrupt_ms"] == pytest.approx(25.0)
    assert event["metric_origin"] == "portaudio_output_dac_schedule"
    assert FakeLLM.instance.opened and FakeLLM.instance.closed
    assert "count aloud" in FakeLLM.instance.texts[0]
    assert speaker.started and speaker.stopped


async def test_stream_pcm_realtime_uses_requested_chunk_duration() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.chunks: list[bytes] = []

        async def push_audio(self, chunk: bytes) -> None:
            self.chunks.append(chunk)

    llm = FakeLLM()

    await _stream_pcm_realtime(
        llm,
        b"a" * 960,
        sample_rate=24_000,
        input_chunk_ms=10,
        sleep=lambda _duration: None,
    )

    assert [len(chunk) for chunk in llm.chunks] == [480, 480]


async def test_forced_commit_keeps_source_audio_end_timestamp() -> None:
    class FakeInputAudioBuffer:
        def __init__(self) -> None:
            self.committed = False

        async def commit(self) -> None:
            self.committed = True

    class FakeConnection:
        def __init__(self) -> None:
            self.input_audio_buffer = FakeInputAudioBuffer()

    class FakeLLM:
        def __init__(self) -> None:
            self._conn = FakeConnection()
            self.triggered = False

        async def trigger_response(self) -> None:
            self.triggered = True

    llm = FakeLLM()

    audio_end_at = await _commit_and_trigger_response(llm, audio_end_at=123.4)

    assert audio_end_at == pytest.approx(123.4)
    assert llm._conn.input_audio_buffer.committed is True
    assert llm.triggered is True


async def test_wait_for_first_audio_owns_response_after_nonempty_transcript() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.triggered = 0

        async def events(self):
            yield {"type": "input.speech_stopped"}
            yield {"type": "input.transcript", "text": "benchmark turn"}
            yield {"type": "audio.delta", "audio": b"pcm"}

        async def trigger_response(self) -> None:
            self.triggered += 1

    llm = FakeLLM()

    speech_stopped_at, first_audio_at, _, _ = await _wait_for_first_audio(
        llm,
        timeout_s=0.1,
        trigger_on_transcript=True,
    )

    assert speech_stopped_at is not None
    assert first_audio_at >= speech_stopped_at
    assert llm.triggered == 1


async def test_stream_pcm_until_local_endpoint_waits_after_source_audio() -> None:
    class FakeTurnDetector:
        def __init__(self) -> None:
            self.events: asyncio.Queue[str] = asyncio.Queue()
            self.fed: list[bytes] = []

        async def feed(self, chunk: bytes) -> None:
            self.fed.append(chunk)
            if len(self.fed) == 3:
                await self.events.put("speech_end")

    turn = FakeTurnDetector()

    audio_end_at, speech_stopped_at = await _stream_pcm_until_local_endpoint(
        turn,
        b"a" * 960,
        sample_rate=24_000,
        input_chunk_ms=10,
        silence_timeout_s=1.0,
        sleep=lambda _duration: None,
    )

    assert audio_end_at is not None
    assert speech_stopped_at is not None
    assert speech_stopped_at >= audio_end_at
    assert [len(chunk) for chunk in turn.fed] == [480, 480, 480]


async def test_measure_local_endpoint_preserves_local_speech_end(monkeypatch) -> None:
    from zemory import config as cfg
    from zemory.providers.llm import openai_realtime
    from zemory.providers.turn import realtime_manual

    class FakeLLM:
        def __init__(self, _api_key: str) -> None:
            self.opened = False
            self.closed = False

        async def open_session(self) -> None:
            self.opened = True

        async def close(self) -> None:
            self.closed = True

    class FakeManualTurnDetector:
        def __init__(self, llm: FakeLLM) -> None:
            self.llm = llm
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    async def fake_stream_until_local_endpoint(*_args, **_kwargs):
        return 100.0, 100.25

    async def fake_commit_and_trigger_response(_llm, *, audio_end_at: float) -> float:
        return audio_end_at

    async def fake_wait_for_first_audio(
        _llm,
        *,
        timeout_s: float,
        speaker=None,
        trigger_on_transcript: bool = False,
    ):
        return None, 101.0, None, None

    monkeypatch.setattr(openai_realtime, "OpenAIRealtimeLLM", FakeLLM)
    monkeypatch.setattr(realtime_manual, "RealtimeManualTurnDetector", FakeManualTurnDetector)
    monkeypatch.setattr(cfg.settings, "profile", cfg.settings.profile)
    monkeypatch.setattr(cfg.settings.realtime, "semantic_vad_eagerness", cfg.settings.realtime.semantic_vad_eagerness)
    monkeypatch.setattr(cfg.settings.realtime, "turn_detection", cfg.settings.realtime.turn_detection)
    monkeypatch.setattr(cfg.settings.realtime, "server_vad_threshold", cfg.settings.realtime.server_vad_threshold)
    monkeypatch.setattr(
        cfg.settings.realtime,
        "local_endpoint_required_misses",
        cfg.settings.realtime.local_endpoint_required_misses,
    )
    monkeypatch.setattr(
        cfg.settings.realtime,
        "server_vad_silence_duration_ms",
        cfg.settings.realtime.server_vad_silence_duration_ms,
    )
    monkeypatch.setattr(
        "scripts.bench_realtime_audio_fixture._stream_pcm_until_local_endpoint",
        fake_stream_until_local_endpoint,
    )
    monkeypatch.setattr(
        "scripts.bench_realtime_audio_fixture._commit_and_trigger_response",
        fake_commit_and_trigger_response,
    )
    monkeypatch.setattr(
        "scripts.bench_realtime_audio_fixture._wait_for_first_audio",
        fake_wait_for_first_audio,
    )

    event = await _measure_sample(
        Sample("unit", "Samantha", "test"),
        b"audio",
        eagerness="high",
        turn_detection="none",
        server_vad_threshold=0.5,
        server_vad_silence_ms=300,
        local_endpoint_required_misses=7,
        input_chunk_ms=20,
        mode="local_endpoint_commit",
        timeout_s=1.0,
        play_output=False,
    )

    assert event["vad_wait_ms"] == pytest.approx(250.0)
    assert event["first_audio_after_speech_stopped_ms"] == pytest.approx(750.0)
    assert event["local_endpoint_required_misses"] == 7
