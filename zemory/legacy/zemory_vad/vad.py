"""Silero VAD with state machine — using official silero-vad package."""

from __future__ import annotations

from collections import deque
from enum import Enum

import numpy as np
import torch
from silero_vad import load_silero_vad
from zemory_vad.config import (
    VAD_DB_THRESHOLD,
    VAD_PROB_THRESHOLD,
    VAD_REQUIRED_HITS,
    VAD_REQUIRED_MISSES,
    VAD_SAMPLE_RATE,
    VAD_SMOOTHING_WINDOW,
)

CHUNK_SAMPLES = 512  # 32ms at 16kHz


# ---------------------------------------------------------------------------
# Silero VAD wrapper (official package)
# ---------------------------------------------------------------------------

class SileroVAD:
    """Thin wrapper around the official silero-vad model."""

    def __init__(self) -> None:
        self._model = load_silero_vad(onnx=False)

    def reset(self) -> None:
        self._model.reset_states()

    def __call__(self, audio_int16: np.ndarray) -> float:
        """Run inference on a 512-sample chunk at 16 kHz.

        Returns speech probability in [0.0, 1.0].
        """
        audio_f32 = torch.FloatTensor(audio_int16.astype(np.float32) / 32768.0)
        prob = self._model(audio_f32, VAD_SAMPLE_RATE).item()
        return prob


# ---------------------------------------------------------------------------
# State machine (mirrors Open-LLM-VTuber silero.py)
# ---------------------------------------------------------------------------

class VADState(Enum):
    IDLE = 1      # Waiting for speech
    ACTIVE = 2    # Speech detected, accumulating
    INACTIVE = 3  # Silence after speech, may resume or end


class VADStateMachine:
    """Three-state VAD with smoothing and consecutive-frame thresholds.

    Signals returned by :meth:`process`:
    * ``"speech_start"`` – IDLE → ACTIVE
    * ``"speech_end"``   – INACTIVE → IDLE
    * ``None``           – no transition
    """

    def __init__(self) -> None:
        self.state = VADState.IDLE
        self._hit = 0
        self._miss = 0
        self._prob_win: deque[float] = deque(maxlen=VAD_SMOOTHING_WINDOW)
        self._db_win: deque[float] = deque(maxlen=VAD_SMOOTHING_WINDOW)

    def reset(self) -> None:
        self.state = VADState.IDLE
        self._hit = 0
        self._miss = 0
        self._prob_win.clear()
        self._db_win.clear()

    def process(self, prob: float, db: float) -> str | None:
        self._prob_win.append(prob)
        self._db_win.append(db)
        sp = float(np.mean(self._prob_win))
        sd = float(np.mean(self._db_win))
        is_speech = sp >= VAD_PROB_THRESHOLD and sd >= VAD_DB_THRESHOLD

        if self.state == VADState.IDLE:
            if is_speech:
                self._hit += 1
                if self._hit >= VAD_REQUIRED_HITS:
                    self.state = VADState.ACTIVE
                    self._hit = 0
                    return "speech_start"
            else:
                self._hit = 0

        elif self.state == VADState.ACTIVE:
            if is_speech:
                self._miss = 0
            else:
                self._miss += 1
                if self._miss >= VAD_REQUIRED_MISSES:
                    self.state = VADState.INACTIVE
                    self._miss = 0

        elif self.state == VADState.INACTIVE:
            if is_speech:
                self._hit += 1
                if self._hit >= VAD_REQUIRED_HITS:
                    self.state = VADState.ACTIVE
                    self._hit = 0
                    self._miss = 0
            else:
                self._hit = 0
                self._miss += 1
                if self._miss >= VAD_REQUIRED_MISSES:
                    self.state = VADState.IDLE
                    self._miss = 0
                    return "speech_end"

        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def calc_db(audio_int16: np.ndarray) -> float:
    rms = np.sqrt(np.mean(audio_int16.astype(np.float64) ** 2))
    if rms < 1:
        return 0.0
    return 20.0 * np.log10(rms)


def resample_24k_to_16k(audio_24k: np.ndarray) -> np.ndarray:
    """Down-sample 24 kHz int16 → 16 kHz int16 via linear interpolation."""
    n_in = len(audio_24k)
    n_out = int(n_in * 16000 / 24000)
    if n_out == 0:
        return np.array([], dtype=np.int16)
    idx = np.linspace(0, n_in - 1, n_out)
    return np.interp(idx, np.arange(n_in), audio_24k.astype(np.float64)).astype(
        np.int16
    )
