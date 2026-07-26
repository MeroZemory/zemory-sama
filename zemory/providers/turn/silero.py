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
from zemory.vad import (
    CHUNK_SAMPLES,
    SileroVAD,
    StreamingResampler24To16,
    VADStateMachine,
    calc_db,
)

_log = get_logger(__name__)

_MIC_FRAME_DURATION_MS = 20
_PCM16_BYTES_PER_SAMPLE = 2
_EVENT_QUEUE_MAXSIZE = 4


class SileroTurnDetector:
    def __init__(
        self,
        *,
        vad: SileroVAD | None = None,
        state_machine: VADStateMachine | None = None,
        capture_audio: bool = True,
    ) -> None:
        self._vad = vad or SileroVAD(onnx=True)
        self._sm = state_machine or VADStateMachine()
        self._resampler = StreamingResampler24To16()
        self._vad_buf = np.array([], dtype=np.int16)
        self._pre_buffer: deque[bytes] = deque()
        self._pre_buffer_bytes = 0
        self._capture_audio = capture_audio
        self._pre_buffer_limit_bytes = (
            settings.sample_rate
            * _PCM16_BYTES_PER_SAMPLE
            * settings.vad.pre_buffer_chunks
            * _MIC_FRAME_DURATION_MS
            // 1000
        ) if capture_audio else 0
        self._max_utterance_bytes = (
            settings.sample_rate
            * _PCM16_BYTES_PER_SAMPLE
            * settings.vad.max_utterance_ms
            // 1000
        )
        self._captured_audio_limit_bytes = (
            self._max_utterance_bytes + self._pre_buffer_limit_bytes
        )
        self._audio_chunks: list[bytes] = []
        self._captured_audio_bytes = 0
        self._utterance_bytes = 0
        self._speaking = False

        self.events: asyncio.Queue[str] = asyncio.Queue(maxsize=_EVENT_QUEUE_MAXSIZE)

    @property
    def captured_audio(self) -> list[bytes]:
        """Audio accumulated since last ``speech_start``. Caller consumes + clears."""
        return self._audio_chunks

    def consume_audio(self) -> list[bytes]:
        out, self._audio_chunks = self._audio_chunks, []
        self._captured_audio_bytes = 0
        return out

    def _capture_pcm(self, pcm24k: bytes) -> None:
        if not self._capture_audio or not pcm24k:
            return
        remaining = self._captured_audio_limit_bytes - self._captured_audio_bytes
        if remaining <= 0:
            return
        bounded = pcm24k[:remaining]
        if len(bounded) % _PCM16_BYTES_PER_SAMPLE:
            bounded = bounded[:-1]
        if bounded:
            self._audio_chunks.append(bounded)
            self._captured_audio_bytes += len(bounded)

    def _append_pre_buffer(self, pcm24k: bytes) -> None:
        if not pcm24k or self._pre_buffer_limit_bytes == 0:
            return

        self._pre_buffer.append(pcm24k)
        self._pre_buffer_bytes += len(pcm24k)
        overflow = self._pre_buffer_bytes - self._pre_buffer_limit_bytes
        while overflow > 0:
            oldest = self._pre_buffer[0]
            if len(oldest) <= overflow:
                self._pre_buffer.popleft()
                self._pre_buffer_bytes -= len(oldest)
                overflow -= len(oldest)
                continue
            self._pre_buffer[0] = oldest[overflow:]
            self._pre_buffer_bytes -= overflow
            overflow = 0

    def _clear_pre_buffer(self) -> None:
        self._pre_buffer.clear()
        self._pre_buffer_bytes = 0

    async def feed(self, pcm24k: bytes) -> None:
        """Push a 20 ms PCM-24k frame from the mic."""
        pcm_24k = np.frombuffer(pcm24k, dtype=np.int16)
        pcm_16k = self._resampler.process(pcm_24k)
        self._vad_buf = np.concatenate([self._vad_buf, pcm_16k])

        signals: list[str] = []
        while len(self._vad_buf) >= CHUNK_SAMPLES:
            chunk = self._vad_buf[:CHUNK_SAMPLES]
            self._vad_buf = self._vad_buf[CHUNK_SAMPLES:]
            prob = self._vad(chunk)
            db = calc_db(chunk)
            sig = self._sm.process(prob, db)
            if sig == "speech_start":
                signals.append(sig)
            elif sig == "speech_end":
                signals.append(sig)
                break

        was_speaking = self._speaking
        if was_speaking:
            self._capture_pcm(pcm24k)
            self._utterance_bytes += len(pcm24k)
        else:
            self._append_pre_buffer(pcm24k)

        for signal in signals:
            if signal == "speech_start" and not self._speaking:
                self._speaking = True
                self._audio_chunks.clear()
                self._captured_audio_bytes = 0
                if self._capture_audio:
                    if self._pre_buffer:
                        for chunk in self._pre_buffer:
                            self._capture_pcm(chunk)
                    else:
                        # With pre-buffer disabled, retain the frame that
                        # actually crossed the speech-start threshold.
                        self._capture_pcm(pcm24k)
                self._clear_pre_buffer()
                self._utterance_bytes = len(pcm24k)
                await self.events.put("speech_start")
                _log.info("vad.speech_start")
            elif signal == "speech_end" and self._speaking:
                self._speaking = False
                self._utterance_bytes = 0
                await self.events.put("speech_end")
                captured_samples = sum(len(chunk) for chunk in self._audio_chunks) // 2
                _log.info("vad.speech_end", samples=captured_samples, forced=False)

        if self._speaking and self._utterance_bytes >= self._max_utterance_bytes:
            self._speaking = False
            self._utterance_bytes = 0
            self._sm.reset()
            self._vad.reset()
            self._resampler.reset()
            self._vad_buf = np.array([], dtype=np.int16)
            await self.events.put("speech_end")
            captured_samples = self._captured_audio_bytes // _PCM16_BYTES_PER_SAMPLE
            _log.warning(
                "vad.speech_end",
                samples=captured_samples,
                forced=True,
                max_utterance_ms=settings.vad.max_utterance_ms,
            )

    def reset(self) -> None:
        self._vad.reset()
        self._sm.reset()
        self._resampler.reset()
        self._vad_buf = np.array([], dtype=np.int16)
        self._clear_pre_buffer()
        self._audio_chunks.clear()
        self._captured_audio_bytes = 0
        self._utterance_bytes = 0
        self._speaking = False

    async def close(self) -> None:
        self.reset()
