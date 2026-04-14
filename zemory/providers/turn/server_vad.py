"""Realtime API server_vad turn detector.

In the realtime profile, VAD runs on OpenAI's server. This adapter's
:meth:`feed` forwards mic audio to the LLM (which drives VAD), and
speech_start / speech_end events are populated by the orchestrator when
it receives the corresponding Realtime events.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zemory.providers.llm.openai_realtime import OpenAIRealtimeLLM


class ServerVADTurnDetector:
    def __init__(self, llm: OpenAIRealtimeLLM) -> None:
        self._llm = llm
        self.events: asyncio.Queue = asyncio.Queue()

    async def feed(self, pcm24k: bytes) -> None:
        await self._llm.push_audio(pcm24k)

    def consume_audio(self) -> list[bytes]:
        # Realtime server sees audio; no local replay buffer.
        return []

    async def notify(self, event: str) -> None:
        await self.events.put(event)

    async def close(self) -> None:
        return None
