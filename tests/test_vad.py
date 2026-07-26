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


class RecordingVAD:
    def __init__(self) -> None:
        self.chunks: list[np.ndarray] = []

    def __call__(self, audio_int16: np.ndarray) -> float:
        self.chunks.append(audio_int16.copy())
        return 0.0

    def reset(self) -> None:
        pass


class ScriptedStateMachine:
    def __init__(self, signals: list[str | None]) -> None:
        self._signals = signals

    def process(self, prob: float, db: float) -> str | None:
        del prob, db
        return self._signals.pop(0) if self._signals else None

    def reset(self) -> None:
        pass


@pytest.mark.asyncio
async def test_silero_turn_detector_supports_fake_vad_and_prebuffer(monkeypatch) -> None:
    from zemory import config as cfg

    monkeypatch.setattr(cfg.settings.vad, "required_hits", 1)
    monkeypatch.setattr(cfg.settings.vad, "required_misses", 1)
    monkeypatch.setattr(cfg.settings.vad, "prob_threshold", 0.5)
    monkeypatch.setattr(cfg.settings.vad, "db_threshold", 1.0)
    monkeypatch.setattr(cfg.settings.vad, "smoothing_window", 1)
    monkeypatch.setattr(cfg.settings.vad, "pre_buffer_chunks", 4)

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
    assert sm.process(0.0, 0.0) == "speech_end"


def test_vad_end_requires_one_consecutive_miss_window(monkeypatch) -> None:
    from zemory import config as cfg

    monkeypatch.setattr(cfg.settings.vad, "required_hits", 1)
    monkeypatch.setattr(cfg.settings.vad, "required_misses", 3)
    monkeypatch.setattr(cfg.settings.vad, "prob_threshold", 0.5)
    monkeypatch.setattr(cfg.settings.vad, "db_threshold", 10.0)
    monkeypatch.setattr(cfg.settings.vad, "smoothing_window", 1)

    sm = VADStateMachine()
    assert sm.process(0.9, 30.0) == "speech_start"
    assert sm.process(0.0, 0.0) is None
    assert sm.process(0.9, 30.0) is None
    assert sm.process(0.0, 0.0) is None
    assert sm.process(0.0, 0.0) is None
    assert sm.process(0.0, 0.0) == "speech_end"


async def _resampled_detector_audio(
    source: np.ndarray,
    chunk_sizes: list[int],
) -> np.ndarray:
    vad = RecordingVAD()
    detector = SileroTurnDetector(vad=vad)
    offset = 0
    size_index = 0
    while offset < len(source):
        size = chunk_sizes[size_index % len(chunk_sizes)]
        await detector.feed(source[offset : offset + size].tobytes())
        offset += size
        size_index += 1
    pieces = [*vad.chunks, detector._vad_buf]
    return np.concatenate(pieces) if pieces else np.array([], dtype=np.int16)


@pytest.mark.asyncio
async def test_streaming_resample_is_independent_of_microphone_chunk_boundaries() -> None:
    source = ((np.arange(24_001, dtype=np.int32) * 37) % 60_000 - 30_000).astype(np.int16)

    whole = await _resampled_detector_audio(source, [len(source)])
    fragmented = await _resampled_detector_audio(source, [1, 2, 7, 11, 480, 5, 31])

    assert len(whole) == len(fragmented) == len(source) * 2 // 3
    np.testing.assert_array_equal(fragmented, whole)


@pytest.mark.asyncio
async def test_feed_emits_start_and_end_detected_in_the_same_large_chunk() -> None:
    detector = SileroTurnDetector(
        vad=SequenceVAD([0.9, 0.0]),
        state_machine=ScriptedStateMachine(["speech_start", "speech_end"]),
    )

    await detector.feed((np.ones(1_536, dtype=np.int16) * 1000).tobytes())

    assert detector.events.get_nowait() == "speech_start"
    assert detector.events.get_nowait() == "speech_end"
    assert detector._speaking is False


@pytest.mark.asyncio
async def test_prebuffer_duration_is_stable_across_smaller_input_chunks(monkeypatch) -> None:
    from zemory import config as cfg

    monkeypatch.setattr(cfg.settings.vad, "pre_buffer_chunks", 2)
    detector = SileroTurnDetector(vad=SequenceVAD([]))
    source = np.arange(6 * 240, dtype=np.int16)

    for frame in np.split(source, 6):
        await detector.feed(frame.tobytes())

    buffered = np.frombuffer(b"".join(detector._pre_buffer), dtype=np.int16)
    assert buffered.size == 24_000 * 40 // 1000
    np.testing.assert_array_equal(buffered, source[-buffered.size :])
    assert detector.events.maxsize > 0


@pytest.mark.asyncio
async def test_reset_clears_captured_audio(monkeypatch) -> None:
    from zemory import config as cfg

    monkeypatch.setattr(cfg.settings.vad, "required_hits", 1)
    monkeypatch.setattr(cfg.settings.vad, "prob_threshold", 0.5)
    monkeypatch.setattr(cfg.settings.vad, "db_threshold", 1.0)
    monkeypatch.setattr(cfg.settings.vad, "smoothing_window", 1)
    detector = SileroTurnDetector(vad=SequenceVAD([0.9]))

    await detector.feed((np.ones(768, dtype=np.int16) * 1000).tobytes())
    assert detector.captured_audio

    detector.reset()

    assert detector.captured_audio == []


@pytest.mark.asyncio
async def test_missing_endpoint_forces_bounded_utterance_end(monkeypatch) -> None:
    from zemory import config as cfg

    monkeypatch.setattr(cfg.settings.vad, "pre_buffer_chunks", 0)
    monkeypatch.setattr(cfg.settings.vad, "max_utterance_ms", 40)
    detector = SileroTurnDetector(
        vad=SequenceVAD([0.9, 0.9]),
        state_machine=ScriptedStateMachine(["speech_start", None]),
    )
    frame = (np.ones(768, dtype=np.int16) * 1000).tobytes()  # 32 ms at 24 kHz

    await detector.feed(frame)
    assert detector.events.get_nowait() == "speech_start"
    await detector.feed(frame)

    assert detector.events.get_nowait() == "speech_end"
    assert detector._speaking is False
    assert sum(len(chunk) for chunk in detector.captured_audio) <= (
        24_000 * 2 * 40 // 1000
    )


@pytest.mark.asyncio
async def test_endpoint_only_mode_never_accumulates_transcription_audio(
    monkeypatch,
) -> None:
    from zemory import config as cfg

    monkeypatch.setattr(cfg.settings.vad, "pre_buffer_chunks", 32)
    detector = SileroTurnDetector(
        vad=SequenceVAD([0.9] * 20),
        state_machine=ScriptedStateMachine(["speech_start"] + [None] * 19),
        capture_audio=False,
    )
    frame = (np.ones(768, dtype=np.int16) * 1000).tobytes()

    for _ in range(20):
        await detector.feed(frame)

    assert detector.captured_audio == []
    assert detector._pre_buffer_bytes == 0


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
