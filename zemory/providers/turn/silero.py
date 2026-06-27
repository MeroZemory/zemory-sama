"""Local Silero VAD turn detector (local profile).

Wraps :class:`zemory.vad.SileroVAD` + :class:`VADStateMachine` and exposes
a :class:`TurnDetector` interface. Additionally maintains a rolling
pre-buffer so transcription includes the ~640 ms of audio captured
immediately before ``speech_start``.
"""

from __future__ import annotations

import asyncio
from collections import deque

import numpy as np

from zemory.config import settings
from zemory.observability import get_logger
from zemory.vad import CHUNK_SAMPLES, SileroVAD, VADStateMachine, calc_db, resample_24k_to_16k

_log = get_logger(__name__)


class SileroTurnDetector:
    def __init__(
        self,
        *,
        vad: SileroVAD | None = None,
        state_machine: VADStateMachine | None = None,
    ) -> None:
        self._vad = vad or SileroVAD(onnx=True)
        self._sm = state_machine or VADStateMachine()
        self._vad_buf = np.array([], dtype=np.int16)
        self._pre_buffer: deque[bytes] = deque(maxlen=settings.vad.pre_buffer_chunks)
        self._audio_chunks: list[bytes] = []
        self._speaking = False

        self.events: asyncio.Queue = asyncio.Queue()

    @property
    def captured_audio(self) -> list[bytes]:
        """Audio accumulated since last ``speech_start``. Caller consumes + clears."""
        return self._audio_chunks

    def consume_audio(self) -> list[bytes]:
        out, self._audio_chunks = self._audio_chunks, []
        return out

    async def feed(self, pcm24k: bytes) -> None:
        """Push a 20 ms PCM-24k frame from the mic."""
        pcm_24k = np.frombuffer(pcm24k, dtype=np.int16)
        pcm_16k = resample_24k_to_16k(pcm_24k)
        self._vad_buf = np.concatenate([self._vad_buf, pcm_16k])

        speech_started = False
        speech_ended = False
        while len(self._vad_buf) >= CHUNK_SAMPLES:
            chunk = self._vad_buf[:CHUNK_SAMPLES]
            self._vad_buf = self._vad_buf[CHUNK_SAMPLES:]
            prob = self._vad(chunk)
            db = calc_db(chunk)
            sig = self._sm.process(prob, db)
            if sig == "speech_start":
                speech_started = True
            elif sig == "speech_end":
                speech_ended = True
                break

        if not self._speaking:
            self._pre_buffer.append(pcm24k)
            if speech_started:
                self._speaking = True
                self._audio_chunks.clear()
                self._audio_chunks.extend(self._pre_buffer)
                self._pre_buffer.clear()
                await self.events.put("speech_start")
                _log.info("vad.speech_start")
        else:
            self._audio_chunks.append(pcm24k)
            if speech_ended:
                self._speaking = False
                await self.events.put("speech_end")
                _log.info("vad.speech_end", samples=len(self._audio_chunks))

    def reset(self) -> None:
        self._vad.reset()
        self._sm.reset()
        self._vad_buf = np.array([], dtype=np.int16)
        self._pre_buffer.clear()
        self._speaking = False

    async def close(self) -> None:
        self.reset()
