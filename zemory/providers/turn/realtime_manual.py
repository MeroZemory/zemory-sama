"""Realtime audio turn detector with local endpoint commits.

This experimental path keeps audio-native Realtime output, but disables
server-side turn detection. Microphone PCM is still streamed to Realtime while
a local endpoint detector emits ``speech_start`` / ``speech_end`` events. The
orchestrator commits the Realtime input buffer on local ``speech_end``.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING, Any

from zemory.config import settings

if TYPE_CHECKING:
    from zemory.providers.llm.openai_realtime import OpenAIRealtimeLLM


class RealtimeEndpointStateMachine:
    """Endpoint-only VAD state machine for manual Realtime commits."""

    def __init__(
        self,
        *,
        prob_threshold: float,
        db_threshold: float,
        required_hits: int,
        required_misses: int,
        smoothing_window: int,
    ) -> None:
        self.prob_threshold = prob_threshold
        self.db_threshold = db_threshold
        self.required_hits = required_hits
        self.required_misses = required_misses
        self._prob_win: deque[float] = deque(maxlen=smoothing_window)
        self._db_win: deque[float] = deque(maxlen=smoothing_window)
        self._speaking = False
        self._hit = 0
        self._miss = 0

    def reset(self) -> None:
        self._prob_win.clear()
        self._db_win.clear()
        self._speaking = False
        self._hit = 0
        self._miss = 0

    def process(self, prob: float, db: float) -> str | None:
        self._prob_win.append(prob)
        self._db_win.append(db)
        sp = sum(self._prob_win) / len(self._prob_win)
        sd = sum(self._db_win) / len(self._db_win)
        is_speech = sp >= self.prob_threshold and sd >= self.db_threshold

        if not self._speaking:
            if is_speech:
                self._hit += 1
                if self._hit >= self.required_hits:
                    self._speaking = True
                    self._hit = 0
                    self._miss = 0
                    return "speech_start"
            else:
                self._hit = 0
            return None

        if is_speech:
            self._miss = 0
            return None

        self._miss += 1
        if self._miss >= self.required_misses:
            self._speaking = False
            self._miss = 0
            self._hit = 0
            return "speech_end"
        return None


def _build_endpoint_state_machine() -> RealtimeEndpointStateMachine:
    return RealtimeEndpointStateMachine(
        prob_threshold=settings.vad.prob_threshold,
        db_threshold=settings.vad.db_threshold,
        required_hits=settings.vad.required_hits,
        required_misses=settings.realtime.local_endpoint_required_misses,
        smoothing_window=settings.vad.smoothing_window,
    )


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

            self._endpoint_detector = SileroTurnDetector(
                state_machine=_build_endpoint_state_machine(),
            )
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
