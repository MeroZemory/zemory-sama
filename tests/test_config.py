"""Configuration source, validation, and credential-boundary contracts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from zemory import config as cfg


def _clear_zemory_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith("ZEMORY_"):
            monkeypatch.delenv(name)


def test_nested_environment_values_use_zemory_prefix(monkeypatch) -> None:
    _clear_zemory_environment(monkeypatch)
    monkeypatch.setenv("ZEMORY_REALTIME__MODEL", "nested-model")
    monkeypatch.setenv("ZEMORY_REALTIME__VOICE", "nested-voice")
    monkeypatch.setenv("ZEMORY_VAD__SMOOTHING_WINDOW", "7")
    monkeypatch.setenv("ZEMORY_TTS__MAX_CONCURRENT", "4")

    settings = cfg.Settings(_env_file=None)

    assert settings.realtime.model == "nested-model"
    assert settings.realtime.voice == "nested-voice"
    assert settings.vad.smoothing_window == 7
    assert settings.tts.max_concurrent == 4


def test_unprefixed_nested_names_do_not_change_settings(monkeypatch) -> None:
    _clear_zemory_environment(monkeypatch)
    monkeypatch.setenv("MODEL", "ambient-model")
    monkeypatch.setenv("VOICE", "ambient-voice")
    monkeypatch.setenv("SMOOTHING_WINDOW", "99")
    monkeypatch.setenv("MAX_CONCURRENT", "99")
    monkeypatch.setenv("ZEMORY_MODEL", "wrong-level-model")
    monkeypatch.setenv("ZEMORY_VOICE", "wrong-level-voice")

    settings = cfg.Settings(_env_file=None)

    assert settings.realtime.model == "gpt-realtime-2.1"
    assert settings.realtime.voice == "marin"
    assert settings.vad.smoothing_window == 5
    assert settings.tts.max_concurrent == 3


def test_prefixed_provider_key_wins_over_legacy_name(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "legacy")
    monkeypatch.setenv("ZEMORY_OPENAI_API_KEY", "prefixed")

    settings = cfg.Settings(_env_file=None)

    assert settings.openai_api_key.get_secret_value() == "prefixed"


def test_ambient_openai_base_url_is_ignored(monkeypatch) -> None:
    _clear_zemory_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://untrusted.example/v1")

    configured = cfg.Settings(_env_file=None)

    assert configured.openai_base_url == "https://api.openai.com/v1"


def test_namespaced_openai_base_url_override_is_accepted(monkeypatch) -> None:
    _clear_zemory_environment(monkeypatch)
    monkeypatch.setenv("ZEMORY_OPENAI_BASE_URL", "http://127.0.0.1:8080/v1/")

    configured = cfg.Settings(_env_file=None)

    assert configured.openai_base_url == "http://127.0.0.1:8080/v1"


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.openai.com/v1",
        "ftp://api.openai.com/v1",
        "https://user:password@api.openai.com/v1",
        "https://api.openai.com/v1?redirect=1",
        "https://api.openai.com:99999/v1",
        "not-a-url",
    ],
)
def test_openai_base_url_rejects_unsafe_or_invalid_values(base_url) -> None:
    with pytest.raises(ValidationError, match="openai_base_url"):
        cfg.Settings(_env_file=None, openai_base_url=base_url)


def test_toml_tables_are_loaded_and_legacy_profile_is_normalized(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_zemory_environment(monkeypatch)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
safety_delay_s = 0.25

[profile]
name = "local"

[audio]
sample_rate = 24000
vad_sample_rate = 16000
chunk_duration_ms = 10

[memory]
enabled = false
path = "memory/test.sqlite3"
recall_deadline_ms = 45
recall_limit = 2

[context]
tool_deadline_ms = 125

[realtime]
voice = "cedar"
turn_detection = "none"

[vad]
smoothing_window = 9

[tts]
max_concurrent = 2
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "_CONFIG_PATH", config_path)

    settings = cfg.Settings(_env_file=None)

    assert settings.profile == "local_cascade"
    assert settings.safety_delay_s == 0.25
    assert settings.sample_rate == 24_000
    assert settings.vad_sample_rate == 16_000
    assert settings.chunk_duration_ms == 10
    assert settings.memory_enabled is False
    assert settings.memory_path == "memory/test.sqlite3"
    assert settings.memory_recall_deadline_ms == 45
    assert settings.memory_recall_limit == 2
    assert settings.context_tool_deadline_ms == 125
    assert settings.realtime.voice == "cedar"
    assert settings.realtime.turn_detection == "none"
    assert settings.vad.smoothing_window == 9
    assert settings.tts.max_concurrent == 2


def test_environment_nested_value_overrides_toml_table(tmp_path: Path, monkeypatch) -> None:
    _clear_zemory_environment(monkeypatch)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[realtime]\nvoice = "toml-voice"\nmodel = "toml-model"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(cfg, "_CONFIG_PATH", config_path)
    monkeypatch.setenv("ZEMORY_REALTIME__VOICE", "env-voice")

    settings = cfg.Settings(_env_file=None)

    assert settings.realtime.voice == "env-voice"
    assert settings.realtime.model == "toml-model"


@pytest.mark.parametrize(
    "invalid_values",
    [
        {"tts": {"max_concurrent": 0}},
        {"vad": {"smoothing_window": 0}},
        {"vad": {"required_hits": 0}},
        {"realtime": {"server_vad_silence_duration_ms": 0}},
        {"realtime": {"truncation_retention_ratio": 0}},
        {"realtime": {"truncation_retention_ratio": 1.01}},
        {"realtime": {"truncation_post_instructions": 0}},
        {"realtime": {"max_output_tokens": 0}},
        {"realtime": {"max_output_tokens": 4097}},
        {"sample_rate": 48_000},
        {"vad_sample_rate": 8_000},
        {"memory_recall_deadline_ms": -1},
        {"memory_recall_limit": -1},
        {"context_tool_deadline_ms": -1},
        {"transcript_correction_timeout_s": 0},
        {"transcript_correction_timeout_s": 60.01},
        {"ready_beep_volume": 1.01},
        {"profile": "local_cascade", "enable_barge_in": True},
    ],
)
def test_downstream_invalid_ranges_fail_during_settings_load(invalid_values) -> None:
    with pytest.raises(ValidationError):
        cfg.Settings(_env_file=None, **invalid_values)


def test_config_import_does_not_require_provider_keys() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["OPENAI_API_KEY"] = ""
    env["ELEVENLABS_API_KEY"] = ""
    env["ZEMORY_OPENAI_API_KEY"] = ""
    env["ZEMORY_ELEVENLABS_API_KEY"] = ""

    result = subprocess.run(
        [sys.executable, "-c", "import zemory.config; print('imported')"],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imported"


def test_runtime_credentials_are_profile_aware() -> None:
    missing_openai = cfg.Settings(
        _env_file=None,
        profile="realtime_audio",
        openai_api_key="",
        elevenlabs_api_key="",
    )
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        cfg.validate_runtime_credentials(missing_openai)

    audio_native = cfg.Settings(
        _env_file=None,
        profile="realtime_audio",
        openai_api_key="openai",
        elevenlabs_api_key="",
    )
    cfg.validate_runtime_credentials(audio_native)

    external_tts = cfg.Settings(
        _env_file=None,
        profile="realtime_text_external_tts",
        openai_api_key="openai",
        elevenlabs_api_key="",
    )
    with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
        cfg.validate_runtime_credentials(external_tts)


@pytest.mark.parametrize(
    ("profile", "openai_key", "elevenlabs_key", "missing_name"),
    [
        ("realtime_audio", " \t\n", "", "OPENAI_API_KEY"),
        (
            "realtime_text_external_tts",
            "openai",
            " \t\n",
            "ELEVENLABS_API_KEY",
        ),
    ],
)
def test_runtime_credentials_reject_whitespace_only_values(
    profile,
    openai_key,
    elevenlabs_key,
    missing_name,
) -> None:
    configured = cfg.Settings(
        _env_file=None,
        profile=profile,
        openai_api_key=openai_key,
        elevenlabs_api_key=elevenlabs_key,
    )

    with pytest.raises(cfg.RuntimeCredentialError, match=missing_name):
        cfg.validate_runtime_credentials(configured)


def test_provider_keys_are_trimmed_before_validation_and_use() -> None:
    configured = cfg.Settings(
        _env_file=None,
        profile="realtime_text_external_tts",
        openai_api_key="  openai-key\n",
        elevenlabs_api_key="\televenlabs-key  ",
    )

    cfg.validate_runtime_credentials(configured)

    assert configured.openai_api_key.get_secret_value() == "openai-key"
    assert configured.elevenlabs_api_key.get_secret_value() == "elevenlabs-key"
