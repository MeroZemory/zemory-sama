"""Realtime audio benchmark script helper tests."""

from __future__ import annotations

import asyncio

import pytest

from scripts.bench_realtime_audio_fixture import (
    Sample,
    _chunk_pcm,
    _commit_and_trigger_response,
    _event_from_timings,
    _measure_sample,
    _parse_args,
    _stream_pcm_realtime,
    _stream_pcm_until_local_endpoint,
    _write_live_benchmark_artifacts,
)


def test_chunk_pcm_splits_pcm_without_losing_tail() -> None:
    assert list(_chunk_pcm(b"abcdefg", chunk_size=3)) == [b"abc", b"def", b"g"]


def test_event_from_timings_records_vad_and_model_audio_segments() -> None:
    event = _event_from_timings(
        fixture="ko_short",
        voice="Yuna",
        eagerness="high",
        turn_detection="semantic_vad",
        server_vad_threshold=0.5,
        input_chunk_ms=10,
        mode="semantic_vad",
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
        audio_end_at=100.0,
        speech_stopped_at=99.5,
        first_audio_at=100.1,
    )

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
            "--play-output",
        ]
    )

    assert args.input_chunk_ms == 10
    assert args.mode == "local_endpoint_commit"
    assert args.turn_detection == "none"
    assert args.play_output is True


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

    async def fake_wait_for_first_audio(_llm, *, timeout_s: float, speaker=None):
        return None, 101.0, None, None

    monkeypatch.setattr(openai_realtime, "OpenAIRealtimeLLM", FakeLLM)
    monkeypatch.setattr(realtime_manual, "RealtimeManualTurnDetector", FakeManualTurnDetector)
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
        input_chunk_ms=20,
        mode="local_endpoint_commit",
        timeout_s=1.0,
        play_output=False,
    )

    assert event["vad_wait_ms"] == pytest.approx(250.0)
    assert event["first_audio_after_speech_stopped_ms"] == pytest.approx(750.0)
