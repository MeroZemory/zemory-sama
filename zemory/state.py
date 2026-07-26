"""Unified turn-taking state machine.

Phase-1 (`zemory/`) used a single `speaking: asyncio.Event` gate.
Phase-2 (`zemory_vad/`) introduced `Phase.{LISTENING,ACTIVE,RESPONDING}`.

This module consolidates both under one enum protected by an ``asyncio.Lock``
so transitions are atomic. Actual microphone forwarding is profile-specific:
LISTENING audio must reach turn detection, while RESPONDING audio is forwarded
only when the selected Realtime profile enables barge-in.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from enum import Enum, auto


class Phase(Enum):
    LISTENING = auto()   # awaiting user speech
    ACTIVE = auto()      # user is speaking
    RESPONDING = auto()  # LLM + TTS pipeline is producing output


class StateMachine:
    """Task-safe holder for the current Phase.

    Transitions are mediated by :meth:`transition` which acquires a lock so
    two tasks cannot race. A set of listeners can be notified on each
    transition (used by InterruptBus, metrics, structlog binders).
    """

    def __init__(self, initial: Phase = Phase.LISTENING) -> None:
        self._phase = initial
        self._lock = asyncio.Lock()
        self._listeners: list[Callable[[Phase, Phase], None]] = []
        self.speech_end_ts: float | None = None  # set by TurnDetector on speech_end

    @property
    def phase(self) -> Phase:
        return self._phase

    def add_listener(self, cb: Callable[[Phase, Phase], None]) -> None:
        self._listeners.append(cb)

    async def transition(self, new: Phase) -> Phase:
        """Atomically move to ``new`` phase. Returns the previous phase."""
        async with self._lock:
            old = self._phase
            self._phase = new
        for cb in self._listeners:
            try:
                cb(old, new)
            except Exception:
                pass
        return old

    def mark_speech_end(self) -> None:
        self.speech_end_ts = time.monotonic()
