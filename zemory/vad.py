"""Silero VAD wrapper + three-state machine.

Moved from ``zemory_vad/vad.py`` to consolidate into the single ``zemory``
package. Uses ONNX by default (~2× faster on CPU than Torch path) to keep
Torch off the VAD hot path — falls back to Torch if ONNX model load fails.
"""

from __future__ import annotations

from collections import deque
from enum import Enum

import numpy as np
from silero_vad import load_silero_vad

from zemory.config import settings

CHUNK_SAMPLES = 512  # 32 ms at 16 kHz


class SileroVAD:
    """Thin wrapper around the official silero-vad model."""

    def __init__(self, onnx: bool = True) -> None:
        try:
            self._model = load_silero_vad(onnx=onnx)
            self._onnx = onnx
        except Exception:  # pragma: no cover
            # Fallback to torch path if onnx runtime unavailable
            self._model = load_silero_vad(onnx=False)
            self._onnx = False

    def reset(self) -> None:
        self._model.reset_states()

    def __call__(self, audio_int16: np.ndarray) -> float:
        """Run inference on a 512-sample chunk at 16 kHz. Returns [0, 1]."""
        audio_f32 = audio_int16.astype(np.float32) / 32768.0
        if self._onnx:
            try:
                prob = self._model(audio_f32, settings.vad_sample_rate).item()
            except AttributeError as exc:
                if "dim" not in str(exc):
                    raise
                import torch
                audio_t = torch.from_numpy(audio_f32)
                prob = self._model(audio_t, settings.vad_sample_rate).item()
        else:
            import torch
            audio_t = torch.FloatTensor(audio_f32)
            prob = self._model(audio_t, settings.vad_sample_rate).item()
        return prob


class VADState(Enum):
    IDLE = 1
    ACTIVE = 2
    INACTIVE = 3


class VADStateMachine:
    """Three-state VAD with smoothing and consecutive-frame thresholds.

    Signals returned by :meth:`process`:
    * ``"speech_start"`` — IDLE → ACTIVE
    * ``"speech_end"``   — INACTIVE → IDLE
    * ``None``           — no transition
    """

    def __init__(self) -> None:
        self.state = VADState.IDLE
        self._hit = 0
        self._miss = 0
        self._prob_win: deque[float] = deque(maxlen=settings.vad.smoothing_window)
        self._db_win: deque[float] = deque(maxlen=settings.vad.smoothing_window)

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
        is_speech = sp >= settings.vad.prob_threshold and sd >= settings.vad.db_threshold

        if self.state == VADState.IDLE:
            if is_speech:
                self._hit += 1
                if self._hit >= settings.vad.required_hits:
                    self.state = VADState.ACTIVE
                    self._hit = 0
                    return "speech_start"
            else:
                self._hit = 0

        elif self.state == VADState.ACTIVE:
            if is_speech:
                self._miss = 0
            else:
                self.state = VADState.INACTIVE
                self._miss = 1
                if self._miss >= settings.vad.required_misses:
                    self.state = VADState.IDLE
                    self._miss = 0
                    return "speech_end"

        elif self.state == VADState.INACTIVE:
            if is_speech:
                self.state = VADState.ACTIVE
                self._hit = 0
                self._miss = 0
            else:
                self._miss += 1
                if self._miss >= settings.vad.required_misses:
                    self.state = VADState.IDLE
                    self._miss = 0
                    return "speech_end"

        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StreamingResampler24To16:
    """Boundary-independent linear PCM16 resampler for the fixed 3:2 ratio.

    Output samples lie at source positions ``0, 1.5, 3, 4.5, ...``.  At most
    one source sample is retained between calls, and the cumulative output
    count is always ``floor(total_input_samples * 2 / 3)``.
    """

    def __init__(self) -> None:
        self._phase = 0
        self._held_sample = 0

    def reset(self) -> None:
        self._phase = 0
        self._held_sample = 0

    def process(self, audio_24k: np.ndarray) -> np.ndarray:
        source = np.asarray(audio_24k, dtype=np.int16).reshape(-1)
        if source.size == 0:
            return np.array([], dtype=np.int16)

        outputs: list[np.ndarray] = []
        offset = 0

        # Phase 1 holds x[3k]. Once x[3k+1] arrives, x[3k] is safe to emit
        # and x[3k+1] becomes the left side of the half-sample interpolation.
        if self._phase == 1:
            outputs.append(np.array([self._held_sample], dtype=np.int16))
            self._held_sample = int(source[0])
            self._phase = 2
            offset = 1

        # Phase 2 holds x[3k+1] and waits for x[3k+2]. Cast before summing to
        # avoid int16 overflow at full-scale input.
        if self._phase == 2 and offset < source.size:
            interpolated = int((self._held_sample + int(source[offset])) / 2)
            outputs.append(np.array([interpolated], dtype=np.int16))
            self._held_sample = 0
            self._phase = 0
            offset += 1

        if self._phase == 0:
            remaining = source[offset:]
            complete_groups = remaining.size // 3
            if complete_groups:
                grouped = remaining[: complete_groups * 3].reshape(-1, 3)
                resampled = np.empty(complete_groups * 2, dtype=np.int16)
                resampled[0::2] = grouped[:, 0]
                sums = grouped[:, 1].astype(np.int32) + grouped[:, 2].astype(np.int32)
                resampled[1::2] = np.trunc(sums / 2).astype(np.int16)
                outputs.append(resampled)
                offset += complete_groups * 3

            tail = source[offset:]
            if tail.size == 1:
                self._held_sample = int(tail[0])
                self._phase = 1
            elif tail.size == 2:
                outputs.append(tail[:1].copy())
                self._held_sample = int(tail[1])
                self._phase = 2

        if not outputs:
            return np.array([], dtype=np.int16)
        if len(outputs) == 1:
            return outputs[0]
        return np.concatenate(outputs)


def calc_db(audio_int16: np.ndarray) -> float:
    rms = np.sqrt(np.mean(audio_int16.astype(np.float64) ** 2))
    if rms < 1:
        return 0.0
    return 20.0 * np.log10(rms)


def resample_24k_to_16k(audio_24k: np.ndarray) -> np.ndarray:
    """Down-sample one PCM16 buffer from 24 kHz to 16 kHz."""
    return StreamingResampler24To16().process(audio_24k)
