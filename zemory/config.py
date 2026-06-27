"""Unified configuration with profile switching.

Loads from, in order of precedence:
  1. Environment variables (ZEMORY_* prefix, plus legacy OPENAI_API_KEY / ELEVENLABS_API_KEY)
  2. config.toml at repo root (if present)
  3. Defaults in this module

Canonical profiles:
- ``realtime_audio``: OpenAI Realtime GA audio-in/audio-out (default)
- ``realtime_text_external_tts``: Realtime text output + external TTS
- ``local_cascade``: Local Silero VAD + Whisper STT + Realtime text LLM

Legacy aliases ``realtime`` and ``local`` are accepted and normalized.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")


ProfileName = Literal[
    "realtime_audio",
    "realtime_text_external_tts",
    "local_cascade",
    "research_full_duplex",
]

_PROFILE_ALIASES = {
    "realtime": "realtime_audio",
    "local": "local_cascade",
}


def canonical_profile(profile: str) -> ProfileName:
    """Return the canonical profile name, accepting legacy aliases."""
    normalized = _PROFILE_ALIASES.get(profile, profile)
    if normalized not in {
        "realtime_audio",
        "realtime_text_external_tts",
        "local_cascade",
        "research_full_duplex",
    }:
        raise ValueError(f"Unknown profile: {profile!r}")
    return normalized  # type: ignore[return-value]


def _load_toml() -> dict:
    path = _REPO_ROOT / "config.toml"
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


_TOML = _load_toml()


class VADSettings(BaseSettings):
    """Local Silero VAD parameters (used in local profile only)."""

    prob_threshold: float = 0.15
    db_threshold: float = 45.0
    required_hits: int = 3                    # 3 × 32 ms = 96 ms
    required_misses: int = 28                 # 28 × 32 ms = 896 ms
    smoothing_window: int = 5
    pre_buffer_chunks: int = 32               # 32 × 20 ms = 640 ms


class TTSSettings(BaseSettings):
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    model_id: str = "eleven_flash_v2_5"
    quick_latency_level: int = 4              # first sentence (RVC-inspired Quick)
    final_latency_level: int = 2              # subsequent sentences (Final)
    max_concurrent: int = 3                   # OLV TTSTaskManager semaphore cap


class RealtimeSession(BaseSettings):
    """OpenAI Realtime API session parameters."""

    model: str = "gpt-realtime-2"
    voice: str = "marin"
    reasoning_effort: Literal["low", "medium", "high"] = "low"

    # Fast path: server_vad with a short silence window wins local live fixtures.
    # semantic_vad remains available when more conservative turn-taking is desired.
    turn_detection: Literal["semantic_vad", "server_vad"] = "server_vad"
    semantic_vad_eagerness: Literal["low", "medium", "high", "auto"] = "high"
    server_vad_threshold: float = 0.5
    server_vad_prefix_padding_ms: int = 300
    server_vad_silence_duration_ms: int = 200
    server_vad_idle_timeout_ms: int | None = None

    # Local profile: Realtime STT disabled (we inject text)
    transcription_model: str = "gpt-4o-mini-transcribe"


class Settings(BaseSettings):
    """Root settings object. All values overridable via env or config.toml."""

    model_config = SettingsConfigDict(env_prefix="ZEMORY_", extra="ignore")

    # --- API keys ---
    openai_api_key: SecretStr = Field(
        default=SecretStr(os.environ.get("OPENAI_API_KEY", ""))
    )
    elevenlabs_api_key: SecretStr = Field(
        default=SecretStr(os.environ.get("ELEVENLABS_API_KEY", ""))
    )

    # --- profile ---
    profile: ProfileName = Field(
        default=_TOML.get("profile", {}).get("name", "realtime_audio")
    )

    @field_validator("profile", mode="before")
    @classmethod
    def _canonicalize_profile(cls, value: object) -> str:
        return canonical_profile(str(value))

    # --- audio ---
    sample_rate: int = 24_000
    vad_sample_rate: int = 16_000
    chunk_duration_ms: int = 20

    # --- turn-taking (applies to ACTIVE → LISTENING transitions) ---
    safety_delay_s: float = 0.5

    # --- context ---
    max_context_turns: int = 10
    memory_enabled: bool = Field(
        default=_TOML.get("memory", {}).get("enabled", True)
    )
    memory_path: str = Field(
        default=_TOML.get("memory", {}).get("path", ".zemory/memory.sqlite3")
    )
    memory_recall_deadline_ms: int = Field(
        default=_TOML.get("memory", {}).get("recall_deadline_ms", 80)
    )
    memory_recall_limit: int = Field(
        default=_TOML.get("memory", {}).get("recall_limit", 5)
    )
    context_tool_deadline_ms: int = Field(
        default=_TOML.get("context", {}).get("tool_deadline_ms", 200)
    )

    # --- STT (local profile) ---
    stt_model: str = "gpt-4o-transcribe"

    # --- behaviour ---
    response_length: str = "1-2 sentences"

    # --- barge-in / interrupt handling ---
    # Default off: speaker audio leaking into the mic on laptops without
    # hardware AEC produces echo-triggered false interrupts. Enable only
    # with a headset or a device that has proper echo cancellation.
    enable_barge_in: bool = False

    # --- transcript correction (context-aware rewrite of raw ASR output) ---
    # Optional extra LLM hop that fixes likely transcription errors in the
    # user's utterance using recent conversation context. Adds 200-500 ms
    # per turn. Default off; set ZEMORY_TRANSCRIPT_CORRECTION=1 to enable.
    transcript_correction_enabled: bool = False
    transcript_correction_model: str = "gpt-5-mini"
    transcript_correction_history_turns: int = 5

    # --- "ready to speak" beep ---
    # Short tone played through the speaker the moment the mic becomes live
    # (RESPONDING → LISTENING transition, plus once at startup). Played
    # while mic is still muted so the beep itself doesn't leak into the
    # transcript; a small post-gap lets echo decay before listening resumes.
    enable_ready_beep: bool = True
    ready_beep_frequency_hz: float = 880.0   # A5 — subtle, not alarming
    ready_beep_duration_ms: int = 80
    ready_beep_volume: float = 0.15          # 0.0 – 1.0
    ready_beep_post_gap_s: float = 0.1

    # --- sub-sections ---
    vad: VADSettings = Field(default_factory=VADSettings)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    realtime: RealtimeSession = Field(default_factory=RealtimeSession)


settings = Settings()

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if not settings.openai_api_key.get_secret_value():
    raise SystemExit(
        "Error: OPENAI_API_KEY is not set.\n"
        "Copy .env.example to .env and add your API key."
    )

if (
    canonical_profile(settings.profile)
    in {"realtime_text_external_tts", "local_cascade"}
    and not settings.elevenlabs_api_key.get_secret_value()
):
    raise SystemExit("Error: ELEVENLABS_API_KEY is not set.")


# ---------------------------------------------------------------------------
# System instructions (language-mirroring, length-enforcing)
# ---------------------------------------------------------------------------
INSTRUCTIONS = (
    "You are Zemory, a friendly and enthusiastic AI assistant. "
    "You speak naturally and conversationally in Korean, English, and any "
    "other language the user chooses. "
    "Unless the user explicitly requests a different language, "
    "you MUST respond in the same language the user spoke in. "
    f"STRICT RULE: Your response MUST be {settings.response_length} maximum. "
    "Never exceed this limit. This is a real-time voice conversation — "
    "long responses feel unnatural. Be direct and concise."
)


def build_session_config() -> dict:
    """Build Realtime session config based on active profile."""
    realtime = settings.realtime
    profile = canonical_profile(settings.profile)

    def _turn_detection() -> dict:
        if realtime.turn_detection == "semantic_vad":
            return {
                "type": "semantic_vad",
                "eagerness": realtime.semantic_vad_eagerness,
                "create_response": True,
                "interrupt_response": settings.enable_barge_in,
            }

        server_vad = {
                "type": "server_vad",
                "threshold": realtime.server_vad_threshold,
                "prefix_padding_ms": realtime.server_vad_prefix_padding_ms,
                "silence_duration_ms": realtime.server_vad_silence_duration_ms,
                "create_response": True,
                "interrupt_response": settings.enable_barge_in,
        }
        if realtime.server_vad_idle_timeout_ms is not None:
            server_vad["idle_timeout_ms"] = realtime.server_vad_idle_timeout_ms
        return server_vad

    output_modalities = ["audio"] if profile == "realtime_audio" else ["text"]
    audio: dict = {
        "input": {
            "format": {"type": "audio/pcm", "rate": settings.sample_rate},
            "turn_detection": (
                _turn_detection()
                if profile in {"realtime_audio", "realtime_text_external_tts"}
                else None
            ),
        }
    }
    if profile == "realtime_audio":
        audio["output"] = {
            "format": {"type": "audio/pcm", "rate": settings.sample_rate},
            "voice": realtime.voice,
        }

    return {
        "type": "realtime",
        "model": realtime.model,
        "instructions": INSTRUCTIONS,
        "output_modalities": output_modalities,
        "audio": audio,
        "reasoning": {"effort": realtime.reasoning_effort},
    }


# ---------------------------------------------------------------------------
# Backwards-compatible module-level names (kept so legacy imports work)
# ---------------------------------------------------------------------------
OPENAI_API_KEY = settings.openai_api_key.get_secret_value()
ELEVENLABS_API_KEY = settings.elevenlabs_api_key.get_secret_value()
MODEL = settings.realtime.model
SAMPLE_RATE = settings.sample_rate
CHUNK_DURATION_MS = settings.chunk_duration_ms
ELEVENLABS_VOICE_ID = settings.tts.voice_id
ELEVENLABS_MODEL_ID = settings.tts.model_id
SESSION_CONFIG = build_session_config()
