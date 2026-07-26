"""Protocol definitions for pluggable pipeline components."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

TurnEvent = Literal["speech_start", "speech_end"]


@dataclass
class Injection:
    """A priority-tagged piece of context to include in the next prompt.

    Priorities follow the Neuro convention: low = early in prompt,
    high = late (LLM recency bias). Values used in-tree::

        system=10  history=50  memory=60  chat=150  patience=180  custom=200
    """

    source: str
    priority: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # Memory/tool output is user-controlled or externally sourced data by
    # default. Only static, code-curated instructions may opt into system
    # authority explicitly.
    trust: Literal["untrusted_data", "trusted_instruction"] = "untrusted_data"


@runtime_checkable
class TurnDetector(Protocol):
    """Consumes audio or external events and emits speech_start / speech_end."""

    events: asyncio.Queue  # Queue[TurnEvent]

    async def feed(self, pcm24k: bytes) -> None:
        """Push one mic frame (PCM 24 kHz int16 bytes)."""

    async def close(self) -> None:
        ...


@runtime_checkable
class STTProvider(Protocol):
    """Speech-to-text. In the Realtime profile this is a no-op (NullSTT)."""

    async def transcribe(self, pcm_chunks: list[bytes]) -> str:
        ...


@runtime_checkable
class LLMProvider(Protocol):
    """Streaming LLM abstraction."""

    async def open_session(self) -> None:
        ...

    async def send_user_text(
        self,
        text: str,
        injections: list[Injection],
        *,
        generation_id: int | None = None,
    ) -> None:
        """Push user text + optional context injections.

        In the Realtime profile with server_vad, audio is sent continuously
        and ``send_user_text`` is a no-op. In the local profile we inject
        the Whisper transcription.
        """

    async def push_audio(self, pcm_bytes: bytes) -> None:
        """Stream mic audio to the LLM (Realtime profile only; else no-op)."""

    async def cancel_current(self, response_id: str | None = None) -> None:
        """Abort the in-flight response (barge-in)."""

    async def commit_input_audio_buffer(
        self,
        *,
        generation_id: int | None = None,
    ) -> None:
        """Commit streamed audio for a manual Realtime turn."""

    async def clear_input_buffer(
        self,
        *,
        generation_id: int | None = None,
    ) -> None:
        """Clear a failed manual Realtime input generation."""

    async def trigger_response(self, *, generation_id: int | None = None) -> None:
        """Request a response after a manual Realtime commit."""

    async def events(self) -> AsyncIterator[dict]:
        """Yield normalized events::

            {"type": "text.delta", "delta": str}
            {"type": "text.done"}
            {"type": "response.done", "usage": {...}}
            {"type": "input.speech_started"}        # Realtime only
            {"type": "input.speech_stopped"}        # Realtime only
            {"type": "input.transcript", "text": str}
            {"type": "error", "error_code": str | None,
             "error_type": str | None}
        """
        if False:
            yield {}

    async def close(self) -> None:
        ...


@runtime_checkable
class TTSProvider(Protocol):
    """Text-to-speech; yields PCM 24 kHz audio chunks."""

    async def synthesize(self, text: str, quick: bool = False) -> AsyncIterator[bytes]:
        ...


# ---------------------------------------------------------------------------
# Profile → concrete provider wiring
# ---------------------------------------------------------------------------


@dataclass
class PipelineBundle:
    turn: TurnDetector
    stt: STTProvider
    llm: LLMProvider
    tts: TTSProvider


def build_pipeline(
    profile: str,
    *,
    openai_api_key: str,
    elevenlabs_api_key: str,
) -> PipelineBundle:
    """Instantiate the four providers for a profile.

    Kept intentionally simple — a registry was considered but explicit
    if/else is clearer given only two profiles.
    """
    # Late imports break a circular dependency (providers → config → ...).
    from zemory.config import canonical_profile, settings
    from zemory.providers.llm.openai_realtime import OpenAIRealtimeLLM
    from zemory.providers.stt.null import NullSTT
    from zemory.providers.stt.openai_whisper import WhisperSTT
    from zemory.providers.tts.elevenlabs import ElevenLabsTTS
    from zemory.providers.tts.null import NullTTS
    from zemory.providers.turn.realtime_manual import RealtimeManualTurnDetector
    from zemory.providers.turn.server_vad import ServerVADTurnDetector
    from zemory.providers.turn.silero import SileroTurnDetector

    profile = canonical_profile(profile)
    llm = OpenAIRealtimeLLM(api_key=openai_api_key)

    if profile == "realtime_audio":
        return PipelineBundle(
            turn=(
                RealtimeManualTurnDetector(llm=llm)
                if settings.realtime.turn_detection == "none"
                else ServerVADTurnDetector(llm=llm)
            ),
            stt=NullSTT(),
            llm=llm,
            tts=NullTTS(),
        )
    if profile == "realtime_text_external_tts":
        return PipelineBundle(
            turn=(
                RealtimeManualTurnDetector(llm=llm)
                if settings.realtime.turn_detection == "none"
                else ServerVADTurnDetector(llm=llm)
            ),
            stt=NullSTT(),
            llm=llm,
            tts=ElevenLabsTTS(api_key=elevenlabs_api_key),
        )
    if profile == "local_cascade":
        return PipelineBundle(
            turn=SileroTurnDetector(),
            stt=WhisperSTT(api_key=openai_api_key),
            llm=llm,
            tts=ElevenLabsTTS(api_key=elevenlabs_api_key),
        )
    if profile == "research_full_duplex":
        raise NotImplementedError("research_full_duplex is a design placeholder")
    raise ValueError(f"Unknown profile: {profile!r}")
