"""Local VAD fallback tests without loading the Silero model."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from zemory.providers.turn.silero import SileroTurnDetector
from zemory.vad import CHUNK_SAMPLES, SileroVAD, VADStateMachine, calc_db, resample_24k_to_16k


class SequenceVAD:
    def __init__(self, probs: list[float]) -> None:
        self._probs = probs
        self.reset_called = 0

    def __call__(self, audio_int16: np.ndarray) -> float:
        if not self._probs:
            return 0.0
        return self._probs.pop(0)

    def reset(self) -> None:
        self.reset_called += 1


@pytest.mark.asyncio
async def test_silero_turn_detector_supports_fake_vad_and_prebuffer(monkeypatch) -> None:
    from zemory import config as cfg

    monkeypatch.setattr(cfg.settings.vad, "required_hits", 1)
    monkeypatch.setattr(cfg.settings.vad, "required_misses", 1)
    monkeypatch.setattr(cfg.settings.vad, "prob_threshold", 0.5)
    monkeypatch.setattr(cfg.settings.vad, "db_threshold", 1.0)
    monkeypatch.setattr(cfg.settings.vad, "smoothing_window", 1)
    monkeypatch.setattr(cfg.settings.vad, "pre_buffer_chunks", 2)

    detector = SileroTurnDetector(vad=SequenceVAD([0.0, 0.9, 0.0, 0.0]))
    loud = (np.ones(768, dtype=np.int16) * 1000).tobytes()
    quiet = np.zeros(768, dtype=np.int16).tobytes()

    await detector.feed(quiet)
    await detector.feed(loud)
    assert await asyncio.wait_for(detector.events.get(), timeout=0.1) == "speech_start"

    await detector.feed(quiet)
    await detector.feed(quiet)
    assert await asyncio.wait_for(detector.events.get(), timeout=0.1) == "speech_end"

    captured = detector.consume_audio()
    assert len(captured) >= 2
    assert captured[0] == quiet
    assert captured[1] == loud


def test_vad_state_machine_transitions_with_thresholds(monkeypatch) -> None:
    from zemory import config as cfg

    monkeypatch.setattr(cfg.settings.vad, "required_hits", 1)
    monkeypatch.setattr(cfg.settings.vad, "required_misses", 1)
    monkeypatch.setattr(cfg.settings.vad, "prob_threshold", 0.5)
    monkeypatch.setattr(cfg.settings.vad, "db_threshold", 10.0)
    monkeypatch.setattr(cfg.settings.vad, "smoothing_window", 1)

    sm = VADStateMachine()
    assert sm.process(0.9, 30.0) == "speech_start"
    assert sm.process(0.0, 0.0) is None
    assert sm.process(0.0, 0.0) == "speech_end"


def test_vad_helpers_compute_db_and_resample() -> None:
    silence = np.zeros(10, dtype=np.int16)
    loud = np.ones(10, dtype=np.int16) * 1000
    source = np.arange(240, dtype=np.int16)

    assert calc_db(silence) == 0.0
    assert calc_db(loud) > 0
    assert len(resample_24k_to_16k(source)) == 160


def test_silero_vad_onnx_wrapper_accepts_tensor_only_model() -> None:
    from zemory import config as cfg

    class TensorOnlyModel:
        def __call__(self, audio, sample_rate: int):
            assert sample_rate == cfg.settings.vad_sample_rate
            assert audio.dim() == 1
            return np.array(0.42, dtype=np.float32)

    vad = SileroVAD.__new__(SileroVAD)
    vad._model = TensorOnlyModel()
    vad._onnx = True

    assert vad(np.zeros(CHUNK_SAMPLES, dtype=np.int16)) == pytest.approx(0.42)
