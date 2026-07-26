"""Deterministic microphone callback-health tests without audio hardware."""

from __future__ import annotations

import asyncio
import threading
import time

import numpy as np

from zemory import audio
from zemory.audio import MicrophoneStream


class FakeInputStream:
    def __init__(self, **kwargs) -> None:
        self.callback = kwargs["callback"]
        self.finished_callback = kwargs["finished_callback"]
        self.active = False
        self.closed = False

    def start(self) -> None:
        self.active = True

    def stop(self) -> None:
        self.active = False
        self.finished_callback()

    def close(self) -> None:
        self.closed = True


def _started_microphone(monkeypatch) -> tuple[MicrophoneStream, FakeInputStream]:
    loop = asyncio.new_event_loop()
    captured: list[FakeInputStream] = []

    def build_stream(**kwargs) -> FakeInputStream:
        stream = FakeInputStream(**kwargs)
        captured.append(stream)
        return stream

    monkeypatch.setattr(audio.sd, "InputStream", build_stream)
    microphone = MicrophoneStream(loop)
    microphone.start()
    return microphone, captured[0]


def _close_microphone(microphone: MicrophoneStream) -> None:
    microphone.stop()
    microphone._loop.close()


def test_microphone_health_reports_inactive_stream(monkeypatch) -> None:
    microphone, stream = _started_microphone(monkeypatch)
    try:
        stream.active = False
        with microphone._health_lock:
            microphone._started_at = (
                time.monotonic() - audio._MIC_STARTUP_GRACE_S - 0.1
            )

        health = microphone.capture_health()

        assert health.started is True
        assert health.finished is False
        assert health.active is False
        assert health.failure_reason == "stream_inactive"
    finally:
        _close_microphone(microphone)


def test_transient_inactive_state_is_masked_during_startup_grace(
    monkeypatch,
) -> None:
    microphone, stream = _started_microphone(monkeypatch)
    try:
        stream.active = False

        health = microphone.capture_health()

        assert health.active is False
        assert health.last_callback_at is None
        assert health.failure_reason is None
    finally:
        _close_microphone(microphone)


def test_microphone_health_reports_finished_callback(monkeypatch) -> None:
    microphone, stream = _started_microphone(monkeypatch)
    try:
        stream.finished_callback()

        health = microphone.capture_health()

        assert health.finished is True
        assert health.failure_reason == "stream_finished"
    finally:
        _close_microphone(microphone)


def test_active_microphone_without_callbacks_stalls_after_startup_grace(
    monkeypatch,
) -> None:
    microphone, _ = _started_microphone(monkeypatch)
    try:
        assert microphone.capture_health().failure_reason is None

        with microphone._health_lock:
            microphone._started_at = (
                time.monotonic() - audio._MIC_STARTUP_GRACE_S - 0.1
            )

        health = microphone.capture_health()

        assert health.active is True
        assert health.last_callback_at is None
        assert health.failure_reason == "callback_stalled"
    finally:
        _close_microphone(microphone)


def test_silent_callbacks_refresh_health_and_enqueue_silent_frames(monkeypatch) -> None:
    microphone, stream = _started_microphone(monkeypatch)
    try:
        silent_frame = np.zeros((microphone.chunk_size, 1), dtype=np.int16)
        stream.callback(silent_frame, microphone.chunk_size, None, None)
        microphone._loop.run_until_complete(asyncio.sleep(0))

        health = microphone.capture_health()

        assert health.active is True
        assert health.last_callback_at is not None
        assert health.failure_reason is None
        assert microphone.queue.get_nowait() == silent_frame.tobytes()

        with microphone._health_lock:
            microphone._last_callback_at = (
                time.monotonic() - audio._MIC_CALLBACK_STALL_TIMEOUT_S - 0.1
            )
        assert microphone.capture_health().failure_reason == "callback_stalled"
    finally:
        _close_microphone(microphone)


def test_microphone_stop_masks_finished_and_inactive_shutdown_race(
    monkeypatch,
) -> None:
    microphone, stream = _started_microphone(monkeypatch)
    stop_entered = threading.Event()
    release_stop = threading.Event()

    def blocking_stop() -> None:
        stream.active = False
        stream.finished_callback()
        stop_entered.set()
        assert release_stop.wait(timeout=1.0)

    stream.stop = blocking_stop  # type: ignore[method-assign]
    stop_thread = threading.Thread(target=microphone.stop)
    stop_thread.start()
    try:
        assert stop_entered.wait(timeout=1.0)

        health_during_stop = microphone.capture_health()

        assert health_during_stop.stopping is True
        assert health_during_stop.failure_reason is None
    finally:
        release_stop.set()
        stop_thread.join(timeout=1.0)
        microphone._loop.close()

    assert stop_thread.is_alive() is False
    health_after_stop = microphone.capture_health()
    assert health_after_stop.started is False
    assert health_after_stop.failure_reason is None
