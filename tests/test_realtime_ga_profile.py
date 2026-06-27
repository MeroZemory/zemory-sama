"""Realtime GA audio profile contract tests.

These tests pin the code to the 2026-06 design: the default runtime is
OpenAI Realtime GA audio-in/audio-out, with external TTS kept as an
explicit profile rather than the default fast path.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from types import SimpleNamespace

from zemory import config as cfg
from zemory.providers.base import build_pipeline
from zemory.providers.llm.openai_realtime import OpenAIRealtimeLLM


def test_default_session_config_is_audio_native_realtime_ga() -> None:
    assert cfg.settings.profile == "realtime_audio"

    session = cfg.build_session_config()

    assert session["type"] == "realtime"
    assert session["model"] == "gpt-realtime-2"
    assert session["output_modalities"] == ["audio"]
    assert session["audio"]["input"]["format"] == {
        "type": "audio/pcm",
        "rate": 24_000,
    }
    assert session["audio"]["input"]["turn_detection"] == {
        "type": "semantic_vad",
        "eagerness": "high",
        "create_response": True,
        "interrupt_response": False,
    }
    assert session["audio"]["output"]["format"] == {
        "type": "audio/pcm",
        "rate": 24_000,
    }
    assert session["audio"]["output"]["voice"] == "marin"
    assert session["reasoning"]["effort"] == "low"
    assert "temperature" not in session
    assert "modalities" not in session


def test_external_tts_profile_requests_text_output(monkeypatch) -> None:
    monkeypatch.setattr(cfg.settings, "profile", "realtime_text_external_tts")

    session = cfg.build_session_config()

    assert session["type"] == "realtime"
    assert session["model"] == "gpt-realtime-2"
    assert session["output_modalities"] == ["text"]
    assert session["audio"]["input"]["turn_detection"]["type"] == "semantic_vad"
    assert "output" not in session["audio"]


def test_local_cascade_profile_disables_server_turn_detection(monkeypatch) -> None:
    monkeypatch.setattr(cfg.settings, "profile", "local_cascade")

    session = cfg.build_session_config()

    assert session["type"] == "realtime"
    assert session["output_modalities"] == ["text"]
    assert session["audio"]["input"]["turn_detection"] is None


def test_realtime_audio_pipeline_does_not_require_external_tts() -> None:
    bundle = build_pipeline(
        "realtime_audio",
        openai_api_key="test-openai",
        elevenlabs_api_key="",
    )

    assert type(bundle.tts).__name__ == "NullTTS"


def test_realtime_audio_config_import_does_not_require_elevenlabs_key() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["OPENAI_API_KEY"] = "test-openai"
    env["ELEVENLABS_API_KEY"] = ""
    env["ZEMORY_PROFILE"] = "realtime_audio"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from zemory.config import settings; print(settings.profile)",
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "realtime_audio"


def test_normalizes_ga_audio_and_text_events() -> None:
    audio = b"pcm-bytes"
    audio_event = SimpleNamespace(
        type="response.output_audio.delta",
        delta=base64.b64encode(audio).decode("ascii"),
    )
    text_event = SimpleNamespace(type="response.output_text.delta", delta="hello")
    transcript_event = SimpleNamespace(
        type="response.output_audio_transcript.delta",
        delta="안녕",
    )

    assert OpenAIRealtimeLLM._normalize(audio_event) == {
        "type": "audio.delta",
        "audio": audio,
    }
    assert OpenAIRealtimeLLM._normalize(text_event) == {
        "type": "text.delta",
        "delta": "hello",
    }
    assert OpenAIRealtimeLLM._normalize(transcript_event) == {
        "type": "audio.transcript.delta",
        "delta": "안녕",
    }
