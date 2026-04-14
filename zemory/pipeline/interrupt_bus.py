"""Barge-in abort chain.

When the user starts speaking during ``Phase.RESPONDING`` the orchestrator
fires :meth:`InterruptBus.trigger`. This runs the following sequence
(target: ≤ 150 ms p95):

1. ``SpeakerStream.clear()`` — drop in-flight audio
2. ``TTSTaskManager.abort()`` — cancel synthesis tasks, drop queued bytes
3. ``LLMProvider.cancel_current()`` — tell Realtime to abort response
4. record partial assistant text into history (OLV-inspired)
5. transition state to ACTIVE
6. ``LLMProvider.clear_input_buffer()`` (Realtime only)

Debounce: ignore triggers fired within 250 ms of the last one.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from zemory.observability import get_logger, metrics
from zemory.state import Phase

if TYPE_CHECKING:
    from zemory.audio import SpeakerStream
    from zemory.pipeline.tts_manager import TTSTaskManager
    from zemory.providers.base import LLMProvider
    from zemory.state import StateMachine

_log = get_logger(__name__)

_DEBOUNCE_S = 0.25


class InterruptBus:
    def __init__(
        self,
        state: StateMachine,
        speaker: SpeakerStream,
        on_partial: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._state = state
        self._speaker = speaker
        self._tts_manager: TTSTaskManager | None = None
        self._llm: LLMProvider | None = None
        self._on_partial = on_partial
        self._last_trigger_ts = 0.0
        self._partial_text = ""
        self._lock = asyncio.Lock()

    def bind(self, tts_manager: TTSTaskManager, llm: LLMProvider) -> None:
        self._tts_manager = tts_manager
        self._llm = llm

    def record_partial(self, delta: str) -> None:
        """Accumulate assistant deltas so we can record what was said on abort."""
        self._partial_text += delta

    def reset_partial(self) -> None:
        self._partial_text = ""

    async def trigger(self, reason: str) -> bool:
        """Run the abort chain. Returns True if trigger fired, False if debounced.

        Only effective when ``phase == RESPONDING`` — harmless no-op otherwise.
        """
        now = time.monotonic()
        if now - self._last_trigger_ts < _DEBOUNCE_S:
            return False
        if self._state.phase != Phase.RESPONDING:
            return False

        async with self._lock:
            # re-check under lock
            if self._state.phase != Phase.RESPONDING:
                return False
            self._last_trigger_ts = now
            _log.warning("interrupt.trigger", reason=reason,
                         partial_len=len(self._partial_text))

            # [1] Drop audio in the speaker immediately.
            self._speaker.clear()

            # [2] Cancel all TTS synthesis tasks.
            if self._tts_manager is not None:
                await self._tts_manager.abort()

            # [3] Tell LLM to cancel its response.
            if self._llm is not None:
                await self._llm.cancel_current()

            # [4] Preserve what the assistant actually said so the next turn
            #     has context (OLV pattern).
            if self._on_partial and self._partial_text:
                try:
                    await self._on_partial(self._partial_text)
                except Exception as e:  # pragma: no cover
                    _log.warning("interrupt.partial_record_failed", error=str(e))
            self._partial_text = ""

            # [5] State transition — mark user as speaking.
            await self._state.transition(Phase.ACTIVE)

            # [6] Clear any server-side queued audio (Realtime only).
            #     Duck-typed check avoids importing OpenAIRealtimeLLM inside
            #     the hot path (openai SDK lazy-import overhead).
            clear = getattr(self._llm, "clear_input_buffer", None)
            if callable(clear):
                await clear()

            elapsed_ms = (time.monotonic() - now) * 1000
            metrics.observe("interrupt.chain_total_ms", elapsed_ms)
            _log.info("interrupt.done", elapsed_ms=round(elapsed_ms, 1))
            return True
