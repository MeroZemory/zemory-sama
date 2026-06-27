"""Realtime audio turn detector with local endpoint commits.

This experimental path keeps audio-native Realtime output, but disables
server-side turn detection. Microphone PCM is still streamed to Realtime while
a local endpoint detector emits ``speech_start`` / ``speech_end`` events. The
orchestrator commits the Realtime input buffer on local ``speech_end``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zemory.providers.llm.openai_realtime import OpenAIRealtimeLLM


class RealtimeManualTurnDetector:
    def __init__(
        self,
        *,
        llm: OpenAIRealtimeLLM,
        endpoint_detector: Any | None = None,
    ) -> None:
        self._llm = llm
        self._endpoint_detector = endpoint_detector
        self.events: asyncio.Queue = (
            endpoint_detector.events if endpoint_detector is not None else asyncio.Queue()
        )

    def _endpoint(self) -> Any:
        if self._endpoint_detector is None:
            from zemory.providers.turn.silero import SileroTurnDetector

            self._endpoint_detector = SileroTurnDetector()
            self._endpoint_detector.events = self.events
        return self._endpoint_detector

    async def feed(self, pcm24k: bytes) -> None:
        await self._llm.push_audio(pcm24k)
        await self._endpoint().feed(pcm24k)

    def consume_audio(self) -> list[bytes]:
        return []

    def reset(self) -> None:
        reset = getattr(self._endpoint_detector, "reset", None)
        if callable(reset):
            reset()

    async def close(self) -> None:
        close = getattr(self._endpoint_detector, "close", None)
        if callable(close):
            await close()
