"""Full-task orchestration regressions with deterministic in-memory providers."""

from __future__ import annotations

import asyncio
import time
import traceback
from types import SimpleNamespace

import pytest

from zemory import orchestrator as orch


def _exception_tree_contains(error: BaseException, message: str) -> bool:
    if message in str(error):
        return True
    if isinstance(error, BaseExceptionGroup):
        return any(
            _exception_tree_contains(child, message)
            for child in error.exceptions
        )
    return False


class FakeMic:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.started = False
        self.stopped = False
        self.failure_reason: str | None = None
        self.clear_count = 0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def clear(self) -> None:
        self.clear_count += 1
        while not self.queue.empty():
            self.queue.get_nowait()

    def capture_health(self):
        return SimpleNamespace(
            failure_reason=(
                None if not self.started or self.stopped else self.failure_reason
            )
        )


class BlockingSpeaker:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.buffered = asyncio.Event()
        self.wait_started = asyncio.Event()
        self._drained = asyncio.Event()
        self._drained.set()
        self.cleared = 0
        self.first_write_at: float | None = None
        self.first_play_at: float | None = None
        self.played_audio_ms = 120

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def arm(self) -> None:
        self._drained.clear()
        self.first_write_at = None
        self.first_play_at = None

    async def feed(self) -> None:
        while True:
            payload = await self.queue.get()
            if payload:
                now = asyncio.get_running_loop().time()
                self.first_write_at = self.first_write_at or now
                self.first_play_at = self.first_play_at or now
            self.buffered.set()

    def clear(self) -> None:
        self.cleared += 1
        self._drained.set()
        while not self.queue.empty():
            self.queue.get_nowait()

    async def wait_until_done(self) -> None:
        self.wait_started.set()
        await self._drained.wait()


class StalledSpeaker(BlockingSpeaker):
    """Speaker whose playback task never consumes its bounded queue."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__(loop)
        self.queue = asyncio.Queue(maxsize=1)

    async def feed(self) -> None:
        await asyncio.Event().wait()


class TurnEventQueue(asyncio.Queue[str]):
    """Keep local/manual fixtures faithful to speech-start/end ordering."""

    def __init__(self) -> None:
        super().__init__()
        self._speech_active = False

    async def put(self, item: str) -> None:
        if item == "speech_end" and not self._speech_active:
            await super().put("speech_start")
        if item == "speech_start":
            self._speech_active = True
        elif item == "speech_end":
            self._speech_active = False
        await super().put(item)

    async def put_raw(self, item: str) -> None:
        await super().put(item)


class FakeTurn:
    def __init__(self) -> None:
        self.events = TurnEventQueue()
        self.closed = False
        self.fed: list[bytes] = []
        self.feed_event = asyncio.Event()

    async def feed(self, pcm: bytes) -> None:
        self.fed.append(pcm)
        self.feed_event.set()

    async def close(self) -> None:
        self.closed = True


class FakeSTT:
    async def transcribe(self, chunks: list[bytes]) -> str:
        return ""


class FakeTTS:
    async def synthesize(self, text: str, quick: bool = False):
        if False:
            yield b""


class RealtimeEventQueue(asyncio.Queue[dict]):
    """Keep ordinary test turns faithful to the server VAD event contract."""

    def __init__(self) -> None:
        super().__init__()
        self._speech_active = False

    async def put(self, item: dict) -> None:
        event_type = item.get("type")
        if event_type == "input.speech_stopped" and not self._speech_active:
            await super().put(
                {
                    "type": "input.speech_started",
                    "item_id": item.get("item_id"),
                }
            )
        if event_type == "input.speech_started":
            self._speech_active = True
        elif event_type == "input.speech_stopped":
            self._speech_active = False
        await super().put(item)

    async def put_raw(self, item: dict) -> None:
        """Inject a deliberately malformed/out-of-order provider event."""
        await super().put(item)


class QueueLLM:
    def __init__(self) -> None:
        self.queue = RealtimeEventQueue()
        self.opened = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.cancelled_response_ids: list[str | None] = []
        self.closed = False
        self.truncated: list[dict] = []
        self.committed = 0
        self.input_buffer_cleared = 0
        self.responses_created = 0
        self.user_texts: list[str] = []
        self.deleted_items: list[str] = []

    async def open_session(self) -> None:
        self.opened.set()

    async def close(self) -> None:
        self.closed = True

    async def events(self):
        while True:
            yield await self.queue.get()

    async def cancel_current(self, response_id: str | None = None) -> None:
        self.cancelled_response_ids.append(response_id)
        self.cancelled.set()

    async def commit_input_audio_buffer(
        self,
        *,
        generation_id: int | None = None,
    ) -> None:
        self.committed += 1

    async def clear_input_buffer(
        self,
        *,
        generation_id: int | None = None,
    ) -> None:
        self.input_buffer_cleared += 1

    async def trigger_response(self, *, generation_id: int | None = None) -> None:
        self.responses_created += 1

    async def send_user_text(
        self,
        text: str,
        injections=None,
        *,
        generation_id: int | None = None,
    ) -> None:
        self.user_texts.append(text)

    async def truncate_item(
        self, item_id: str, *, content_index: int, audio_end_ms: int
    ) -> None:
        self.truncated.append(
            {
                "item_id": item_id,
                "content_index": content_index,
                "audio_end_ms": audio_end_ms,
            }
        )

    async def delete_item(self, item_id: str) -> bool:
        self.deleted_items.append(item_id)
        return True


class EofLLM(QueueLLM):
    async def events(self):
        if False:
            yield {}


def _configure_runtime(monkeypatch, llm, speaker: BlockingSpeaker):
    turn = FakeTurn()
    pipeline = SimpleNamespace(
        turn=turn,
        stt=FakeSTT(),
        llm=llm,
        tts=FakeTTS(),
    )
    mic = FakeMic(asyncio.get_running_loop())

    monkeypatch.setattr(orch, "build_pipeline", lambda *args, **kwargs: pipeline)
    monkeypatch.setattr(orch, "MicrophoneStream", lambda loop: mic)
    monkeypatch.setattr(orch, "SpeakerStream", lambda loop: speaker)
    monkeypatch.setattr(orch, "validate_runtime_credentials", lambda: None)
    monkeypatch.setattr(orch.settings, "profile", "realtime_audio")
    monkeypatch.setattr(orch.settings.realtime, "turn_detection", "server_vad")
    monkeypatch.setattr(orch.settings, "enable_barge_in", True)
    monkeypatch.setattr(orch.settings, "enable_ready_beep", False)
    monkeypatch.setattr(orch.settings, "safety_delay_s", 0.0)
    monkeypatch.setattr(orch.settings, "transcript_correction_enabled", False)
    monkeypatch.setattr(orch.settings, "memory_enabled", False)
    monkeypatch.setattr(
        orch.settings,
        "openai_base_url",
        "https://api.openai.com/v1",
    )
    return pipeline, mic


@pytest.mark.parametrize("profile", ["realtime_audio", "local_cascade"])
def test_correction_deadline_is_identical_across_runtime_profiles(
    monkeypatch,
    profile,
) -> None:
    client_options: dict[str, object] = {}
    corrector_options: dict[str, object] = {}
    client = object()

    def build_client(**kwargs):
        client_options.update(kwargs)
        return client

    class CapturingCorrector:
        def __init__(self, **kwargs) -> None:
            corrector_options.update(kwargs)

    monkeypatch.setattr(orch.settings, "profile", profile)
    monkeypatch.setattr(orch.settings, "transcript_correction_enabled", True)
    monkeypatch.setattr(orch.settings, "transcript_correction_timeout_s", 2.5)
    monkeypatch.setattr(orch.settings, "openai_base_url", "https://api.openai.com/v1")
    monkeypatch.setattr(orch, "AsyncOpenAI", build_client)
    monkeypatch.setattr(orch, "TranscriptCorrector", CapturingCorrector)

    built = orch.build_transcript_corrector()

    assert built is not None
    assert client_options["timeout"] == 2.5
    assert client_options["max_retries"] == 0
    assert corrector_options["client"] is client
    assert corrector_options["timeout_s"] == 2.5


@pytest.mark.asyncio
async def test_response_done_drain_does_not_block_barge_in(monkeypatch) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, mic = _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-1",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {
                "type": "response.output_item",
                "response_id": "resp-1",
                "item_id": "item-1",
            }
        )
        await llm.queue.put(
            {
                "type": "audio.delta",
                "response_id": "resp-1",
                "item_id": "item-1",
                "content_index": 0,
                "audio": b"pcm",
            }
        )
        await asyncio.wait_for(speaker.buffered.wait(), timeout=0.2)
        await llm.queue.put(
            {"type": "response.done", "response_id": "resp-1", "status": "completed"}
        )
        await asyncio.wait_for(speaker.wait_started.wait(), timeout=0.2)

        # The finalizer is deliberately blocked on playback drain. Event
        # consumption must continue so this interruption still fires.
        await llm.queue.put({"type": "input.speech_started"})
        await asyncio.wait_for(llm.cancelled.wait(), timeout=0.2)

        assert speaker.cleared == 1
        assert llm.truncated == [
            {"item_id": "item-1", "content_index": 0, "audio_end_ms": 120}
        ]
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)

    assert mic.stopped is True
    assert pipeline.turn.closed is True
    assert llm.closed is True


@pytest.mark.asyncio
async def test_realtime_event_eof_fails_runtime_and_closes_resources(monkeypatch) -> None:
    llm = EofLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, mic = _configure_runtime(monkeypatch, llm, speaker)

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await asyncio.wait_for(orch.run(), timeout=0.2)

    assert any(
        isinstance(error, ConnectionError)
        for error in exc_info.value.exceptions
    )
    assert mic.stopped is True
    assert pipeline.turn.closed is True
    assert llm.closed is True


@pytest.mark.asyncio
async def test_startup_never_invokes_billable_tts_warmup(monkeypatch) -> None:
    class BillableWarmupTTS(FakeTTS):
        def __init__(self) -> None:
            self.warmup_calls = 0
            self.warmup_started = asyncio.Event()

        async def warmup(self) -> None:
            self.warmup_calls += 1
            self.warmup_started.set()

    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, mic = _configure_runtime(monkeypatch, llm, speaker)
    tts = BillableWarmupTTS()
    pipeline.tts = tts
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    # Give every startup task a deterministic scheduling opportunity. Under
    # the old implementation, warmup_started is set at this point.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert tts.warmup_calls == 0
    assert not tts.warmup_started.is_set()

    run_task.cancel()
    await asyncio.gather(run_task, return_exceptions=True)
    assert mic.stopped is True
    assert pipeline.turn.closed is True
    assert llm.closed is True


@pytest.mark.asyncio
async def test_microphone_failure_fails_taskgroup_and_closes_resources(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, mic = _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch, "_MIC_HEALTH_POLL_S", 0.005)
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    mic.failure_reason = "stream_inactive"

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await asyncio.wait_for(run_task, timeout=0.2)

    assert _exception_tree_contains(
        exc_info.value,
        "Microphone capture failed (stream_inactive); restart Zemory",
    )
    assert mic.stopped is True
    assert pipeline.turn.closed is True
    assert llm.closed is True


@pytest.mark.asyncio
async def test_server_vad_sender_failure_fails_taskgroup_and_closes_resources(
    monkeypatch,
) -> None:
    from zemory.providers.turn.server_vad import ServerVADTurnDetector

    class FailingAudioLLM(QueueLLM):
        async def push_audio(self, pcm: bytes) -> None:
            del pcm
            raise RuntimeError("sensitive provider failure payload")

    llm = FailingAudioLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, mic = _configure_runtime(monkeypatch, llm, speaker)
    pipeline.turn = ServerVADTurnDetector(llm=llm)  # type: ignore[arg-type]
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    await mic.queue.put(b"pcm")

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await asyncio.wait_for(run_task, timeout=0.2)

    assert _exception_tree_contains(
        exc_info.value,
        "Realtime server VAD sender failed",
    )
    assert "sensitive provider failure payload" not in str(exc_info.value)
    assert mic.stopped is True
    assert pipeline.turn._closed is True
    assert llm.closed is True


@pytest.mark.asyncio
async def test_cleanup_failure_cannot_skip_remaining_resource_closes(monkeypatch) -> None:
    llm = EofLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, mic = _configure_runtime(monkeypatch, llm, speaker)

    def fail_mic_stop() -> None:
        mic.stopped = True
        raise RuntimeError("private cleanup payload")

    mic.stop = fail_mic_stop  # type: ignore[method-assign]

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await asyncio.wait_for(orch.run(), timeout=0.2)

    formatted = "".join(traceback.format_exception(exc_info.value))
    assert "private cleanup payload" not in formatted
    assert mic.stopped is True
    assert pipeline.turn.closed is True
    assert llm.closed is True


@pytest.mark.asyncio
async def test_cleanup_timeouts_cannot_block_later_resource_closes(monkeypatch) -> None:
    class ClosingTTS(FakeTTS):
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    class StubbornCloseLLM(QueueLLM):
        def __init__(self) -> None:
            super().__init__()
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()
            self.close_finished = asyncio.Event()

        async def close(self) -> None:
            self.close_started.set()
            try:
                await self.release_close.wait()
            except asyncio.CancelledError:
                await self.release_close.wait()
            finally:
                self.closed = True
                self.close_finished.set()

    llm = StubbornCloseLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, mic = _configure_runtime(monkeypatch, llm, speaker)
    closing_tts = ClosingTTS()
    pipeline.tts = closing_tts
    monkeypatch.setattr(orch, "_CLEANUP_TIMEOUT_S", 0.01)
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    await llm.queue.put(
        {"type": "error", "error": SimpleNamespace(code="force_shutdown")}
    )

    with pytest.raises(BaseExceptionGroup):
        await asyncio.wait_for(run_task, timeout=0.2)

    # The cancellation-resistant close timed out, yet subsequent providers
    # were still closed and run() returned control to its caller.
    assert llm.close_started.is_set()
    assert mic.stopped is True
    assert pipeline.turn.closed is True
    assert closing_tts.closed is True

    llm.release_close.set()
    await asyncio.wait_for(llm.close_finished.wait(), timeout=0.2)


@pytest.mark.asyncio
async def test_manual_turn_speech_start_uses_interrupt_bus(monkeypatch) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, _ = _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings.realtime, "turn_detection", "none")
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await pipeline.turn.events.put("speech_end")
        for _ in range(20):
            if llm.committed:
                break
            await asyncio.sleep(0)
        assert llm.committed == 1
        assert llm.responses_created == 0

        await llm.queue.put(
            {
                "type": "input.committed",
                "item_id": "user-m",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {"type": "input.transcript", "text": "manual turn", "item_id": "user-m"}
        )
        for _ in range(20):
            if llm.responses_created:
                break
            await asyncio.sleep(0)
        assert llm.responses_created == 1

        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-manual",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {
                "type": "response.output_item",
                "response_id": "resp-manual",
                "item_id": "item-manual",
            }
        )
        await llm.queue.put(
            {
                "type": "audio.delta",
                "response_id": "resp-manual",
                "item_id": "item-manual",
                "content_index": 0,
                "audio": b"pcm",
            }
        )
        await asyncio.wait_for(speaker.buffered.wait(), timeout=0.2)

        await pipeline.turn.events.put("speech_start")
        await asyncio.wait_for(llm.cancelled.wait(), timeout=0.2)

        assert speaker.cleared == 1
        assert llm.truncated[0]["item_id"] == "item-manual"
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_startup_beep_frames_are_dropped_before_mic_opens(monkeypatch) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    speaker._drained.clear()
    pipeline, mic = _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings, "enable_ready_beep", True)
    monkeypatch.setattr(orch.settings, "ready_beep_post_gap_s", 0.0)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(speaker.wait_started.wait(), timeout=0.2)
        await mic.queue.put(b"beep-echo-frame")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert pipeline.turn.fed == []

        speaker._drained.set()
        await asyncio.sleep(0)
        await mic.queue.put(b"user-frame")
        await asyncio.wait_for(pipeline.turn.feed_event.wait(), timeout=0.2)
        assert pipeline.turn.fed == [b"user-frame"]
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_zero_duration_beep_uses_safety_delay_before_mic_reopens(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, mic = _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings, "enable_barge_in", False)
    monkeypatch.setattr(orch.settings, "enable_ready_beep", True)
    monkeypatch.setattr(orch.settings, "ready_beep_duration_ms", 0)
    monkeypatch.setattr(orch.settings, "safety_delay_s", 0.1)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "hello", "item_id": "user-1"}
        )
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-1",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {
                "type": "audio.delta",
                "response_id": "resp-1",
                "item_id": "assistant-1",
                "audio": b"assistant-pcm",
            }
        )
        await asyncio.wait_for(speaker.buffered.wait(), timeout=0.2)
        await llm.queue.put(
            {"type": "response.done", "response_id": "resp-1", "status": "completed"}
        )
        await asyncio.wait_for(speaker.wait_started.wait(), timeout=0.2)
        speaker._drained.set()
        await asyncio.sleep(0)

        await mic.queue.put(b"speaker-tail")
        await asyncio.sleep(0.02)
        assert pipeline.turn.fed == []

        await asyncio.sleep(0.1)
        await mic.queue.put(b"user-frame")
        await asyncio.wait_for(pipeline.turn.feed_event.wait(), timeout=0.2)
        assert pipeline.turn.fed == [b"user-frame"]
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_stale_transcript_correction_cannot_replace_next_turn(monkeypatch) -> None:
    class ControlledCorrector:
        instance: ControlledCorrector | None = None

        def __init__(self, **kwargs) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.cancelled = asyncio.Event()
            ControlledCorrector.instance = self

        async def correct(self, raw: str) -> tuple[str, float]:
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            return f"corrected:{raw}", 10.0

        def record_user(self, text: str) -> None:
            return None

        def record_assistant(self, text: str) -> None:
            return None

    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings, "transcript_correction_enabled", True)
    monkeypatch.setattr(orch.settings, "transcript_correction_timeout_s", 5.0)
    monkeypatch.setattr(orch, "TranscriptCorrector", ControlledCorrector)
    client_options: dict[str, object] = {}

    def build_correction_client(**kwargs):
        client_options.update(kwargs)
        return object()

    monkeypatch.setattr(orch, "AsyncOpenAI", build_correction_client)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-a"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "turn-a", "item_id": "user-a"}
        )
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-a",
                "generation_id": 1,
            }
        )
        corrector = ControlledCorrector.instance
        assert corrector is not None
        assert client_options["base_url"] == "https://api.openai.com/v1"
        await asyncio.wait_for(corrector.started.wait(), timeout=0.2)

        await llm.queue.put({"type": "input.speech_started"})
        await asyncio.wait_for(corrector.cancelled.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-b"})
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-b",
                "generation_id": 3,
            }
        )
        corrector.release.set()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert llm.user_texts == []
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_empty_realtime_transcript_cannot_start_autonomous_response_loop(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put(
            {"type": "input.speech_stopped", "item_id": "empty-user"}
        )
        await llm.queue.put(
            {"type": "input.transcript", "text": "   ", "item_id": "empty-user"}
        )

        for _ in range(30):
            if llm.deleted_items:
                break
            await asyncio.sleep(0.002)
        assert llm.responses_created == 0
        assert llm.deleted_items == ["empty-user"]

        # The ignored noise turn must recover to listening and allow a real
        # subsequent utterance to create exactly one response.
        await llm.queue.put(
            {"type": "input.speech_stopped", "item_id": "real-user"}
        )
        await llm.queue.put(
            {"type": "input.transcript", "text": "안녕", "item_id": "real-user"}
        )
        for _ in range(30):
            if llm.responses_created:
                break
            await asyncio.sleep(0.002)
        assert llm.responses_created == 1
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_stalled_audio_output_cannot_block_barge_in_control_event(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = StalledSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "hello", "item_id": "user-1"}
        )
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-1",
                "generation_id": 1,
            }
        )
        for payload in (b"first", b"second", b"third"):
            await llm.queue.put(
                {
                    "type": "audio.delta",
                    "response_id": "resp-1",
                    "item_id": "assistant-1",
                    "content_index": 0,
                    "audio": payload,
                }
            )

        # The audio relay worker is now blocked behind the stalled speaker,
        # but the independent event consumer must still see speech_started.
        await llm.queue.put({"type": "input.speech_started"})
        await asyncio.wait_for(llm.cancelled.wait(), timeout=0.2)
        assert speaker.cleared >= 1
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_interrupted_assistant_text_never_enters_corrector_history(
    monkeypatch,
) -> None:
    class RecordingCorrector:
        instance: RecordingCorrector | None = None

        def __init__(self, **kwargs) -> None:
            self.assistant_history: list[str] = []
            RecordingCorrector.instance = self

        async def correct(self, raw: str) -> tuple[str, float]:
            return raw, 0.0

        def record_user(self, text: str) -> None:
            return None

        def record_assistant(self, text: str) -> None:
            self.assistant_history.append(text)

        async def aclose(self) -> None:
            return None

    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings, "transcript_correction_enabled", True)
    monkeypatch.setattr(orch, "TranscriptCorrector", RecordingCorrector)
    monkeypatch.setattr(orch, "AsyncOpenAI", lambda **kwargs: object())
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "turn", "item_id": "user-1"}
        )
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-1",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {
                "type": "audio.transcript.delta",
                "response_id": "resp-1",
                "delta": "FULL_UNHEARD",
            }
        )
        await llm.queue.put(
            {"type": "audio.transcript.done", "response_id": "resp-1"}
        )
        await llm.queue.put({"type": "input.speech_started"})
        await asyncio.wait_for(llm.cancelled.wait(), timeout=0.2)

        corrector = RecordingCorrector.instance
        assert corrector is not None
        assert corrector.assistant_history == []
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_unknown_realtime_error_fails_explicitly_instead_of_stalling(
    monkeypatch,
) -> None:
    class SensitiveProviderError(RuntimeError):
        code = "session_failed"

    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, mic = _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    await llm.queue.put(
        {"type": "input.speech_stopped", "item_id": "user-error"}
    )
    await llm.queue.put(
        {
            "type": "error",
            "error": SensitiveProviderError("private transcript in provider body"),
        }
    )

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await asyncio.wait_for(run_task, timeout=0.2)

    assert any(
        str(error) == "Realtime provider error (session_failed)"
        for error in exc_info.value.exceptions
    )
    formatted = "".join(traceback.format_exception(exc_info.value))
    assert "private transcript in provider body" not in formatted
    assert mic.stopped is True
    assert pipeline.turn.closed is True
    assert llm.closed is True


@pytest.mark.asyncio
async def test_local_response_send_timeout_fails_closed(monkeypatch) -> None:
    class HangingSendLLM(QueueLLM):
        def __init__(self) -> None:
            super().__init__()
            self.send_started = asyncio.Event()

        async def send_user_text(
            self,
            text: str,
            injections=None,
            *,
            generation_id: int | None = None,
        ) -> None:
            self.send_started.set()
            await asyncio.Event().wait()

    class LocalSTT(FakeSTT):
        async def transcribe(self, chunks: list[bytes]) -> str:
            return "local turn"

    llm = HangingSendLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, mic = _configure_runtime(monkeypatch, llm, speaker)
    pipeline.stt = LocalSTT()
    pipeline.turn.consume_audio = lambda: [b"pcm"]
    monkeypatch.setattr(orch.settings, "profile", "local_cascade")
    monkeypatch.setattr(orch, "_PROVIDER_CONTROL_TIMEOUT_S", 0.01)
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    await pipeline.turn.events.put("speech_end")
    await asyncio.wait_for(llm.send_started.wait(), timeout=0.2)

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await asyncio.wait_for(run_task, timeout=0.2)

    assert _exception_tree_contains(
        exc_info.value,
        "Local response request timed out",
    )
    assert mic.stopped is True
    assert pipeline.turn.closed is True
    assert llm.closed is True


@pytest.mark.asyncio
async def test_local_response_send_failure_drops_provider_payload_from_traceback(
    monkeypatch,
) -> None:
    class FailingSendLLM(QueueLLM):
        async def send_user_text(
            self,
            text: str,
            injections=None,
            *,
            generation_id: int | None = None,
        ) -> None:
            raise RuntimeError(f"private provider body containing {text}")

    class LocalSTT(FakeSTT):
        async def transcribe(self, chunks: list[bytes]) -> str:
            return "private local transcript"

    llm = FailingSendLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, mic = _configure_runtime(monkeypatch, llm, speaker)
    pipeline.stt = LocalSTT()
    pipeline.turn.consume_audio = lambda: [b"pcm"]
    monkeypatch.setattr(orch.settings, "profile", "local_cascade")
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    await pipeline.turn.events.put("speech_end")

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await asyncio.wait_for(run_task, timeout=0.2)

    formatted = "".join(traceback.format_exception(exc_info.value))
    assert "Local response request failed" in formatted
    assert "private provider body" not in formatted
    assert "private local transcript" not in formatted
    assert mic.stopped is True
    assert pipeline.turn.closed is True
    assert llm.closed is True


@pytest.mark.asyncio
async def test_local_turns_reset_generation_text_between_responses(monkeypatch) -> None:
    class SequenceSTT(FakeSTT):
        def __init__(self) -> None:
            self._texts = iter(("first user", "second user"))

        async def transcribe(self, chunks: list[bytes]) -> str:
            return next(self._texts)

    class AudibleTTS(FakeTTS):
        async def synthesize(self, text: str, quick: bool = False):
            yield b"pcm"

    class RecordingCorrector:
        instance: RecordingCorrector | None = None

        def __init__(self, **kwargs) -> None:
            self.assistant_history: list[str] = []
            RecordingCorrector.instance = self

        async def correct(self, raw: str) -> tuple[str, float]:
            return raw, 0.0

        def record_user(self, text: str) -> None:
            return None

        def record_assistant(self, text: str) -> None:
            self.assistant_history.append(text)

        async def aclose(self) -> None:
            return None

    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, _ = _configure_runtime(monkeypatch, llm, speaker)
    pipeline.stt = SequenceSTT()
    pipeline.tts = AudibleTTS()
    pipeline.turn.consume_audio = lambda: [b"pcm"]
    monkeypatch.setattr(orch.settings, "profile", "local_cascade")
    monkeypatch.setattr(orch.settings, "transcript_correction_enabled", True)
    monkeypatch.setattr(orch, "TranscriptCorrector", RecordingCorrector)
    monkeypatch.setattr(orch, "AsyncOpenAI", lambda **kwargs: object())
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        corrector = RecordingCorrector.instance
        assert corrector is not None

        for generation, text in ((1, "FIRST."), (2, "SECOND.")):
            speaker.buffered.clear()
            await pipeline.turn.events.put("speech_start")
            await pipeline.turn.events.put("speech_end")
            for _ in range(30):
                if len(llm.user_texts) == generation:
                    break
                await asyncio.sleep(0.002)
            await llm.queue.put(
                {
                    "type": "response.created",
                    "response_id": f"resp-{generation}",
                    "generation_id": generation,
                }
            )
            await llm.queue.put(
                {
                    "type": "response.output_item",
                    "response_id": f"resp-{generation}",
                    "item_id": f"assistant-{generation}",
                }
            )
            await llm.queue.put(
                {
                    "type": "text.delta",
                    "response_id": f"resp-{generation}",
                    "delta": text,
                }
            )
            await llm.queue.put(
                {"type": "text.done", "response_id": f"resp-{generation}"}
            )
            await llm.queue.put(
                {
                    "type": "response.done",
                    "response_id": f"resp-{generation}",
                    "status": "completed",
                }
            )
            await asyncio.wait_for(speaker.buffered.wait(), timeout=0.2)
            speaker._drained.set()
            for _ in range(50):
                if len(corrector.assistant_history) == generation:
                    break
                await asyncio.sleep(0.002)

        assert corrector.assistant_history == ["FIRST.", "SECOND."]
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_duplicate_local_speech_end_cannot_replace_active_response(
    monkeypatch,
) -> None:
    class SequenceSTT(FakeSTT):
        def __init__(self) -> None:
            self._texts = iter(("first", "must-not-send"))

        async def transcribe(self, chunks: list[bytes]) -> str:
            return next(self._texts)

    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, _ = _configure_runtime(monkeypatch, llm, speaker)
    pipeline.stt = SequenceSTT()
    pipeline.turn.consume_audio = lambda: [b"pcm"]
    monkeypatch.setattr(orch.settings, "profile", "local_cascade")
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await pipeline.turn.events.put("speech_end")
        for _ in range(30):
            if llm.user_texts == ["first"]:
                break
            await asyncio.sleep(0.002)
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-current",
                "generation_id": 1,
            }
        )

        await pipeline.turn.events.put_raw("speech_end")
        await asyncio.sleep(0.02)

        assert llm.user_texts == ["first"]
        assert llm.cancelled_response_ids == []
        assert run_task.done() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.parametrize(
    ("operation", "item_create_purpose"),
    [
        ("response.create", None),
        ("conversation.item.create", "user_input"),
    ],
)
@pytest.mark.asyncio
async def test_partial_user_turn_control_error_is_session_fatal(
    monkeypatch,
    operation: str,
    item_create_purpose: str | None,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    await llm.queue.put(
        {"type": "input.speech_stopped", "item_id": "partial-user"}
    )
    await llm.queue.put(
        {
            "type": "input.transcript",
            "text": "must not recover a partial provider transaction",
            "item_id": "partial-user",
        }
    )
    for _ in range(30):
        if llm.responses_created == 1:
            break
        await asyncio.sleep(0.002)
    await llm.queue.put(
        {
            "type": "error",
            "operation": operation,
            "generation_id": 1,
            "item_create_purpose": item_create_purpose,
            "error": SimpleNamespace(code="invalid_request_error"),
        }
    )

    with pytest.raises(BaseExceptionGroup) as raised:
        await asyncio.wait_for(run_task, timeout=0.3)
    assert _exception_tree_contains(
        raised.value,
        "Provider user-turn transaction failed",
    )


@pytest.mark.asyncio
async def test_late_audio_delta_from_cancelled_response_is_dropped(monkeypatch) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put(
            {"type": "input.speech_stopped", "item_id": "user-old"}
        )
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-old",
                "generation_id": 1,
            }
        )
        await llm.queue.put({"type": "input.speech_started"})
        await asyncio.wait_for(llm.cancelled.wait(), timeout=0.2)

        # The server normally acknowledges cancellation before any straggling
        # deltas already in transit arrive. The cancelled response identity
        # must remain quarantined after that acknowledgement.
        await llm.queue.put(
            {"type": "response.done", "response_id": "resp-old", "status": "cancelled"}
        )

        await llm.queue.put(
            {
                "type": "audio.delta",
                "response_id": "resp-old",
                "item_id": "item-old",
                "content_index": 0,
                "audio": b"must-not-play",
            }
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert speaker.buffered.is_set() is False
        assert speaker.queue.empty()
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_response_created_after_unbound_barge_in_cannot_play_ghost_audio(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put(
            {"type": "input.speech_stopped", "item_id": "user-ghost"}
        )
        # Interrupt before response.created binds an ID to this generation.
        await llm.queue.put({"type": "input.speech_started"})
        await asyncio.sleep(0)
        assert llm.cancelled.is_set() is False

        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-ghost",
                "generation_id": 1,
            }
        )
        await asyncio.wait_for(llm.cancelled.wait(), timeout=0.2)
        await llm.queue.put(
            {
                "type": "audio.delta",
                "response_id": "resp-ghost",
                "item_id": "item-ghost",
                "content_index": 0,
                "audio": b"ghost-audio",
            }
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert speaker.buffered.is_set() is False
        assert speaker.queue.empty()
        assert llm.cancelled_response_ids == ["resp-ghost"]
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_late_response_scoped_cancel_failure_stops_session(monkeypatch) -> None:
    class CancelFailureLLM(QueueLLM):
        async def cancel_current(self, response_id: str | None = None) -> None:
            self.cancelled_response_ids.append(response_id)
            self.cancelled.set()
            raise RuntimeError("synthetic scoped cancel failure")

    llm = CancelFailureLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    await llm.queue.put(
        {"type": "input.speech_stopped", "item_id": "user-ghost"}
    )
    await llm.queue.put({"type": "input.speech_started"})
    await llm.queue.put(
        {
            "type": "response.created",
            "response_id": "resp-ghost",
            "generation_id": 1,
        }
    )

    with pytest.raises(BaseExceptionGroup) as raised:
        await asyncio.wait_for(run_task, timeout=0.3)
    assert _exception_tree_contains(
        raised.value,
        "Scoped response cancellation failed",
    )
    assert llm.cancelled_response_ids == ["resp-ghost"]


@pytest.mark.parametrize(
    "terminal_status",
    ["cancelled", "completed", "failed", "incomplete"],
)
@pytest.mark.asyncio
async def test_next_response_waits_for_late_old_response_cancel_ack(
    monkeypatch,
    terminal_status: str,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "first", "item_id": "user-1"}
        )
        for _ in range(30):
            if llm.responses_created == 1:
                break
            await asyncio.sleep(0.002)
        assert llm.responses_created == 1

        # Barge in before the first response ID arrives, then finish the new
        # utterance. The old response.created is delivered late.
        await llm.queue.put({"type": "input.speech_started"})
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-2"})
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-old",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {"type": "input.transcript", "text": "second", "item_id": "user-2"}
        )
        await asyncio.wait_for(llm.cancelled.wait(), timeout=0.2)
        await asyncio.sleep(0.01)

        # response.create for the second utterance is held until the server
        # confirms that resp-old is no longer active.
        assert llm.responses_created == 1
        assert llm.cancelled_response_ids == ["resp-old"]

        await llm.queue.put(
            {
                "type": "response.done",
                "response_id": "resp-old",
                "status": terminal_status,
            }
        )
        for _ in range(100):
            if llm.responses_created == 2:
                break
            await asyncio.sleep(0.002)
        assert llm.responses_created == 2
        assert run_task.done() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_active_response_conflict_retries_once_without_crashing(monkeypatch) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "turn", "item_id": "user-1"}
        )
        for _ in range(30):
            if llm.responses_created == 1:
                break
            await asyncio.sleep(0.002)

        await llm.queue.put(
            {
                "type": "error",
                "error": SimpleNamespace(
                    code="conversation_already_has_active_response"
                ),
            }
        )
        for _ in range(30):
            if llm.cancelled_response_ids == [None]:
                break
            await asyncio.sleep(0.002)
        assert llm.responses_created == 1
        await llm.queue.put(
            {
                "type": "error",
                "operation": "response.cancel",
                "response_id": None,
                "error": SimpleNamespace(code="response_cancel_not_active"),
            }
        )
        for _ in range(30):
            if llm.responses_created == 2:
                break
            await asyncio.sleep(0.002)

        assert llm.responses_created == 2
        assert llm.cancelled_response_ids == [None]
        assert run_task.done() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_active_response_conflict_retry_gets_fresh_terminal_deadline(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch, "_RESPONSE_TERMINAL_TIMEOUT_S", 0.04)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "turn", "item_id": "user-1"}
        )
        for _ in range(30):
            if llm.responses_created == 1:
                break
            await asyncio.sleep(0.002)
        await asyncio.sleep(0.03)
        await llm.queue.put(
            {
                "type": "error",
                "operation": "response.create",
                "generation_id": 1,
                "error": SimpleNamespace(
                    code="conversation_already_has_active_response"
                ),
            }
        )
        await asyncio.wait_for(llm.cancelled.wait(), timeout=0.2)
        await llm.queue.put(
            {
                "type": "error",
                "operation": "response.cancel",
                "response_id": None,
                "error": SimpleNamespace(code="response_cancel_not_active"),
            }
        )
        for _ in range(30):
            if llm.responses_created == 2:
                break
            await asyncio.sleep(0.002)

        # This is beyond the first request's remaining deadline but still
        # within the retry's fresh one. The stale watchdog must not cancel it.
        await asyncio.sleep(0.02)
        assert llm.responses_created == 2
        assert llm.cancelled_response_ids == [None]
        assert run_task.done() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_unrelated_response_done_cannot_release_unscoped_cancel_barrier(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "turn", "item_id": "user-1"}
        )
        for _ in range(30):
            if llm.responses_created == 1:
                break
            await asyncio.sleep(0.002)

        await llm.queue.put(
            {
                "type": "error",
                "operation": "response.create",
                "generation_id": 1,
                "error": SimpleNamespace(
                    code="conversation_already_has_active_response"
                ),
            }
        )
        for _ in range(30):
            if llm.cancelled_response_ids == [None]:
                break
            await asyncio.sleep(0.002)
        assert llm.responses_created == 1
        assert llm.cancelled_response_ids == [None]

        # A delayed terminal from some earlier response does not acknowledge
        # the unscoped cancellation that recovery just sent for the currently
        # active provider response.
        await llm.queue.put(
            {
                "type": "response.done",
                "response_id": "unrelated-old-response",
                "status": "completed",
            }
        )
        await asyncio.sleep(0.01)
        assert llm.responses_created == 1

        # Status alone is not correlation either: a delayed cancelled terminal
        # may also belong to an older request.
        await llm.queue.put(
            {
                "type": "response.done",
                "response_id": "unrelated-old-cancelled-response",
                "status": "cancelled",
            }
        )
        await asyncio.sleep(0.01)
        assert llm.responses_created == 1

        # Only the error carrying the original cancel event ID is safely
        # attributable for an unscoped request.
        await llm.queue.put(
            {
                "type": "error",
                "operation": "response.cancel",
                "response_id": None,
                "error": SimpleNamespace(code="response_cancel_not_active"),
            }
        )
        for _ in range(30):
            if llm.responses_created == 2:
                break
            await asyncio.sleep(0.002)
        assert llm.responses_created == 2
        assert run_task.done() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_stale_generation_active_response_conflict_is_ignored(monkeypatch) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "first", "item_id": "user-1"}
        )
        for _ in range(30):
            if llm.responses_created == 1:
                break
            await asyncio.sleep(0.002)

        await llm.queue.put({"type": "input.speech_started"})
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-2"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "second", "item_id": "user-2"}
        )
        for _ in range(30):
            if llm.responses_created == 2:
                break
            await asyncio.sleep(0.002)

        await llm.queue.put(
            {
                "type": "error",
                "operation": "response.create",
                "generation_id": 1,
                "error": SimpleNamespace(
                    code="conversation_already_has_active_response"
                ),
            }
        )
        await asyncio.sleep(0.01)

        assert llm.responses_created == 2
        assert llm.cancelled_response_ids == []
        assert run_task.done() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_duplicate_active_response_conflicts_share_one_recovery(monkeypatch) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "turn", "item_id": "user-1"}
        )
        for _ in range(30):
            if llm.responses_created == 1:
                break
            await asyncio.sleep(0.002)

        conflict = {
            "type": "error",
            "operation": "response.create",
            "generation_id": 1,
            "error": SimpleNamespace(code="conversation_already_has_active_response"),
        }
        await llm.queue.put(conflict.copy())
        await llm.queue.put(conflict.copy())
        for _ in range(30):
            if llm.cancelled_response_ids:
                break
            await asyncio.sleep(0.002)

        assert llm.responses_created == 1
        assert llm.cancelled_response_ids == [None]

        await llm.queue.put(
            {
                "type": "error",
                "operation": "response.cancel",
                "response_id": None,
                "error": SimpleNamespace(code="response_cancel_not_active"),
            }
        )
        for _ in range(30):
            if llm.responses_created == 2:
                break
            await asyncio.sleep(0.002)
        await asyncio.sleep(0.01)

        assert llm.responses_created == 2
        assert llm.cancelled_response_ids == [None]
        assert run_task.done() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_late_duplicate_active_conflict_event_is_ignored(monkeypatch) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "turn", "item_id": "user-1"}
        )
        for _ in range(30):
            if llm.responses_created == 1:
                break
            await asyncio.sleep(0.002)

        await llm.queue.put(
            {
                "type": "error",
                "client_event_id": "create-event-1",
                "operation": "response.create",
                "generation_id": 1,
                "error": SimpleNamespace(
                    code="conversation_already_has_active_response"
                ),
            }
        )
        for _ in range(30):
            if llm.cancelled_response_ids == [None]:
                break
            await asyncio.sleep(0.002)
        await llm.queue.put(
            {
                "type": "error",
                "operation": "response.cancel",
                "response_id": None,
                "error": SimpleNamespace(code="response_cancel_not_active"),
            }
        )
        for _ in range(30):
            if llm.responses_created == 2:
                break
            await asyncio.sleep(0.002)

        # Once the adapter has consumed operation metadata, a duplicated raw
        # provider error can retain only the original client event identity.
        await llm.queue.put(
            {
                "type": "error",
                "client_event_id": "create-event-1",
                "error": SimpleNamespace(
                    code="conversation_already_has_active_response"
                ),
            }
        )
        await asyncio.sleep(0.01)
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-retry",
                "generation_id": 1,
            }
        )
        await asyncio.sleep(0.01)

        assert llm.responses_created == 2
        assert llm.cancelled_response_ids == [None]
        assert run_task.done() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_delayed_system_note_create_error_cannot_abandon_new_turn(
    monkeypatch,
) -> None:
    class NoteLLM(QueueLLM):
        def __init__(self) -> None:
            super().__init__()
            self.notes: list[str] = []

        async def record_system_note(self, text: str) -> None:
            self.notes.append(text)

    llm = NoteLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings, "profile", "realtime_text_external_tts")
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "first", "item_id": "user-1"}
        )
        for _ in range(30):
            if llm.responses_created == 1:
                break
            await asyncio.sleep(0.002)
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-old",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {
                "type": "response.output_item",
                "response_id": "resp-old",
                "item_id": "assistant-old",
            }
        )
        await llm.queue.put({"type": "input.speech_started"})
        for _ in range(100):
            if (
                llm.cancelled_response_ids == ["resp-old"]
                and "assistant-old" in llm.deleted_items
            ):
                break
            await asyncio.sleep(0.002)
        assert llm.cancelled_response_ids == ["resp-old"]
        assert "assistant-old" in llm.deleted_items

        await llm.queue.put(
            {
                "type": "response.done",
                "response_id": "resp-old",
                "status": "cancelled",
            }
        )
        await llm.queue.put(
            {"type": "conversation.item.deleted", "item_id": "assistant-old"}
        )
        for _ in range(100):
            if llm.notes:
                break
            await asyncio.sleep(0.002)
        assert llm.notes == ["(The previous assistant response was interrupted.)"]

        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-2"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "second", "item_id": "user-2"}
        )
        for _ in range(30):
            if llm.responses_created == 2:
                break
            await asyncio.sleep(0.002)

        await llm.queue.put(
            {
                "type": "error",
                "operation": "conversation.item.create",
                "item_create_purpose": "system_note",
                "error": SimpleNamespace(code="invalid_request_error"),
            }
        )
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-new",
                "generation_id": 3,
            }
        )
        await asyncio.sleep(0.02)

        assert llm.responses_created == 2
        assert llm.cancelled_response_ids == ["resp-old"]
        assert run_task.done() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("profile", "ack_type"),
    [
        ("realtime_audio", "conversation.item.truncated"),
        ("realtime_text_external_tts", "conversation.item.deleted"),
    ],
)
async def test_barge_in_waits_for_history_mutation_ack_before_next_response(
    monkeypatch,
    profile: str,
    ack_type: str,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings, "profile", profile)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "first", "item_id": "user-1"}
        )
        for _ in range(30):
            if llm.responses_created == 1:
                break
            await asyncio.sleep(0.002)
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-old",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {
                "type": "response.output_item",
                "response_id": "resp-old",
                "item_id": "assistant-old",
            }
        )
        await llm.queue.put({"type": "input.speech_started"})
        for _ in range(100):
            mutation_sent = (
                bool(llm.truncated)
                if profile == "realtime_audio"
                else "assistant-old" in llm.deleted_items
            )
            if llm.cancelled_response_ids == ["resp-old"] and mutation_sent:
                break
            await asyncio.sleep(0.002)

        await llm.queue.put(
            {
                "type": "response.done",
                "response_id": "resp-old",
                "status": "cancelled",
            }
        )
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-2"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "second", "item_id": "user-2"}
        )
        await asyncio.sleep(0.01)
        assert llm.responses_created == 1

        await llm.queue.put({"type": ack_type, "item_id": "assistant-old"})
        for _ in range(100):
            if llm.responses_created == 2:
                break
            await asyncio.sleep(0.002)
        assert llm.responses_created == 2
        assert run_task.done() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("profile", "operation"),
    [
        ("realtime_audio", "conversation.item.truncate"),
        ("realtime_text_external_tts", "conversation.item.delete"),
    ],
)
async def test_barge_in_history_mutation_rejection_stops_unsafe_session(
    monkeypatch,
    profile: str,
    operation: str,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings, "profile", profile)
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
    await llm.queue.put(
        {"type": "input.transcript", "text": "first", "item_id": "user-1"}
    )
    for _ in range(30):
        if llm.responses_created == 1:
            break
        await asyncio.sleep(0.002)
    await llm.queue.put(
        {
            "type": "response.created",
            "response_id": "resp-old",
            "generation_id": 1,
        }
    )
    await llm.queue.put(
        {
            "type": "response.output_item",
            "response_id": "resp-old",
            "item_id": "assistant-old",
        }
    )
    await llm.queue.put({"type": "input.speech_started"})
    for _ in range(100):
        mutation_sent = (
            bool(llm.truncated)
            if profile == "realtime_audio"
            else "assistant-old" in llm.deleted_items
        )
        if mutation_sent:
            break
        await asyncio.sleep(0.002)

    await llm.queue.put(
        {
            "type": "error",
            "operation": operation,
            "item_id": "assistant-old",
            "error": SimpleNamespace(code="invalid_request_error"),
        }
    )
    with pytest.raises(BaseExceptionGroup) as raised:
        await asyncio.wait_for(run_task, timeout=0.3)
    assert _exception_tree_contains(
        raised.value,
        "Provider history mutation was rejected",
    )


@pytest.mark.asyncio
async def test_barge_in_cancel_send_failure_stops_unsafe_session(monkeypatch) -> None:
    class CancelFailureLLM(QueueLLM):
        async def cancel_current(self, response_id: str | None = None) -> None:
            raise RuntimeError("synthetic cancel send failure")

    llm = CancelFailureLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
    await llm.queue.put(
        {"type": "input.transcript", "text": "first", "item_id": "user-1"}
    )
    for _ in range(30):
        if llm.responses_created == 1:
            break
        await asyncio.sleep(0.002)
    assert llm.responses_created == 1
    await llm.queue.put(
        {
            "type": "response.created",
            "response_id": "resp-old",
            "generation_id": 1,
        }
    )
    await llm.queue.put({"type": "input.speech_started"})

    with pytest.raises(BaseExceptionGroup) as raised:
        await asyncio.wait_for(run_task, timeout=0.3)
    assert _exception_tree_contains(
        raised.value,
        "Interrupt cancel synchronization failed",
    )


@pytest.mark.asyncio
async def test_barge_in_cancel_ack_timeout_stops_before_next_response(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch, "_PROVIDER_CONTROL_TIMEOUT_S", 0.01)
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
    await llm.queue.put(
        {"type": "input.transcript", "text": "first", "item_id": "user-1"}
    )
    for _ in range(30):
        if llm.responses_created == 1:
            break
        await asyncio.sleep(0.002)
    assert llm.responses_created == 1
    await llm.queue.put(
        {
            "type": "response.created",
            "response_id": "resp-old",
            "generation_id": 1,
        }
    )
    await llm.queue.put({"type": "input.speech_started"})
    await asyncio.wait_for(llm.cancelled.wait(), timeout=0.2)
    await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-2"})
    await llm.queue.put(
        {"type": "input.transcript", "text": "second", "item_id": "user-2"}
    )

    with pytest.raises(BaseExceptionGroup) as raised:
        await asyncio.wait_for(run_task, timeout=0.3)
    assert _exception_tree_contains(
        raised.value,
        "Response cancellation was not acknowledged",
    )
    assert llm.responses_created == 1


@pytest.mark.asyncio
async def test_uncorrelated_not_active_error_cannot_release_scoped_cancel(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch, "_PROVIDER_CONTROL_TIMEOUT_S", 0.01)
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
    await llm.queue.put(
        {"type": "input.transcript", "text": "first", "item_id": "user-1"}
    )
    for _ in range(30):
        if llm.responses_created == 1:
            break
        await asyncio.sleep(0.002)
    assert llm.responses_created == 1
    await llm.queue.put(
        {
            "type": "response.created",
            "response_id": "resp-old",
            "generation_id": 1,
        }
    )
    await llm.queue.put({"type": "input.speech_started"})
    await asyncio.wait_for(llm.cancelled.wait(), timeout=0.2)

    # A stale provider error without the original operation metadata cannot
    # acknowledge the scoped cancellation merely because it is the only one.
    await llm.queue.put(
        {
            "type": "error",
            "error": SimpleNamespace(code="response_cancel_not_active"),
        }
    )
    await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-2"})
    await llm.queue.put(
        {"type": "input.transcript", "text": "second", "item_id": "user-2"}
    )

    with pytest.raises(BaseExceptionGroup) as raised:
        await asyncio.wait_for(run_task, timeout=0.3)
    assert _exception_tree_contains(
        raised.value,
        "Response cancellation was not acknowledged",
    )
    assert llm.responses_created == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["realtime_audio", "realtime_text_external_tts"])
async def test_barge_in_mutation_send_failure_stops_unsafe_session(
    monkeypatch,
    profile: str,
) -> None:
    class MutationFailureLLM(QueueLLM):
        async def truncate_item(
            self,
            item_id: str,
            *,
            content_index: int,
            audio_end_ms: int,
        ) -> None:
            raise RuntimeError("synthetic truncate send failure")

        async def delete_item(self, item_id: str) -> bool:
            raise RuntimeError("synthetic delete send failure")

    llm = MutationFailureLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings, "profile", profile)
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
    await llm.queue.put(
        {"type": "input.transcript", "text": "first", "item_id": "user-1"}
    )
    await llm.queue.put(
        {
            "type": "response.created",
            "response_id": "resp-old",
            "generation_id": 1,
        }
    )
    await llm.queue.put(
        {
            "type": "response.output_item",
            "response_id": "resp-old",
            "item_id": "assistant-old",
        }
    )
    await llm.queue.put({"type": "input.speech_started"})

    with pytest.raises(BaseExceptionGroup) as raised:
        await asyncio.wait_for(run_task, timeout=0.3)
    assert _exception_tree_contains(
        raised.value,
        "Interrupted response history mutation failed",
    )


@pytest.mark.asyncio
async def test_generic_assistant_item_never_binds_output_generation(monkeypatch) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "turn", "item_id": "user-1"}
        )
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-current",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {
                "type": "conversation.item.created",
                "item_id": "unscoped-old-assistant",
                "role": "assistant",
            }
        )
        await llm.queue.put({"type": "input.speech_started"})
        await asyncio.wait_for(llm.cancelled.wait(), timeout=0.2)

        assert llm.truncated == []
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_late_old_transcript_cannot_claim_new_turn(monkeypatch) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put(
            {"type": "input.speech_stopped", "item_id": "user-old"}
        )
        await llm.queue.put(
            {"type": "input.speech_started", "item_id": "user-new"}
        )
        await llm.queue.put(
            {"type": "input.speech_stopped", "item_id": "user-new"}
        )
        await llm.queue.put(
            {
                "type": "input.transcript",
                "text": "OLD TRANSCRIPT",
                "item_id": "user-old",
            }
        )
        await llm.queue.put(
            {
                "type": "input.transcript",
                "text": "NEW TRANSCRIPT",
                "item_id": "user-new",
            }
        )

        for _ in range(30):
            if "user-old" in llm.deleted_items:
                break
            await asyncio.sleep(0.002)
        await llm.queue.put(
            {"type": "conversation.item.deleted", "item_id": "user-old"}
        )
        for _ in range(30):
            if llm.responses_created == 1:
                break
            await asyncio.sleep(0.002)

        assert llm.responses_created == 1
        assert "user-old" in llm.deleted_items
        assert run_task.done() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stopped_item_id", "transcript_item_id", "expected_error"),
    [
        (None, "old-item", "Realtime input item identity was unavailable"),
        ("current-item", None, "Realtime transcript item identity was unavailable"),
    ],
)
async def test_missing_input_item_identity_stops_before_response(
    monkeypatch,
    stopped_item_id: str | None,
    transcript_item_id: str | None,
    expected_error: str,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    await llm.queue.put(
        {"type": "input.speech_stopped", "item_id": stopped_item_id}
    )
    await llm.queue.put(
        {
            "type": "input.transcript",
            "text": "must not claim this turn",
            "item_id": transcript_item_id,
        }
    )

    with pytest.raises(BaseExceptionGroup) as raised:
        await asyncio.wait_for(run_task, timeout=0.2)
    assert _exception_tree_contains(raised.value, expected_error)
    assert llm.responses_created == 0


@pytest.mark.asyncio
async def test_duplicate_speech_stop_cannot_replace_active_response_ownership(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_started", "item_id": "user-1"})
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "first", "item_id": "user-1"}
        )
        for _ in range(30):
            if llm.responses_created == 1:
                break
            await asyncio.sleep(0.002)
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-current",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {
                "type": "response.output_item",
                "response_id": "resp-current",
                "item_id": "assistant-current",
            }
        )

        await llm.queue.put_raw(
            {"type": "input.speech_stopped", "item_id": "foreign-user"}
        )
        await llm.queue.put(
            {
                "type": "input.transcript",
                "text": "foreign transcript",
                "item_id": "foreign-user",
            }
        )
        await asyncio.sleep(0.01)
        await llm.queue.put(
            {
                "type": "audio.delta",
                "response_id": "resp-current",
                "item_id": "assistant-current",
                "content_index": 0,
                "audio": b"still-current",
            }
        )
        await asyncio.wait_for(speaker.buffered.wait(), timeout=0.2)

        assert llm.responses_created == 1
        assert llm.cancelled_response_ids == []
        assert run_task.done() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_transcript_timeout_deletes_committed_input_before_reuse(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch, "_INPUT_TRANSCRIPT_TIMEOUT_S", 0.01)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put(
            {"type": "input.speech_stopped", "item_id": "timed-out-user"}
        )
        for _ in range(50):
            if llm.deleted_items:
                break
            await asyncio.sleep(0.002)

        assert llm.deleted_items == ["timed-out-user"]
        assert llm.responses_created == 0
        assert run_task.done() is False

        await llm.queue.put(
            {"type": "conversation.item.deleted", "item_id": "timed-out-user"}
        )
        await asyncio.sleep(0.01)
        await llm.queue.put(
            {"type": "input.speech_stopped", "item_id": "next-user"}
        )
        await llm.queue.put(
            {"type": "input.transcript", "text": "next", "item_id": "next-user"}
        )
        for _ in range(30):
            if llm.responses_created == 1:
                break
            await asyncio.sleep(0.002)
        assert llm.responses_created == 1
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_completed_response_does_not_bust_cache_by_deleting_oldest_item(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put(
            {"type": "input.speech_stopped", "item_id": "current-user"}
        )
        for item_id in ("old-user", "old-assistant", "current-user"):
            await llm.queue.put(
                {
                    "type": "conversation.item.created",
                    "item_id": item_id,
                    "role": "user",
                }
            )
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-trim",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {"type": "response.done", "response_id": "resp-trim", "status": "completed"}
        )
        await asyncio.wait_for(speaker.wait_started.wait(), timeout=0.2)
        speaker._drained.set()

        await asyncio.sleep(0.01)
        assert llm.deleted_items == []
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_zero_byte_external_tts_never_commits_unheard_assistant_history(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings, "profile", "realtime_text_external_tts")
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "hello", "item_id": "user-1"}
        )
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-1",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {
                "type": "response.output_item",
                "response_id": "resp-1",
                "item_id": "assistant-1",
            }
        )
        await llm.queue.put(
            {"type": "text.delta", "response_id": "resp-1", "delta": "UNHEARD."}
        )
        await llm.queue.put({"type": "text.done", "response_id": "resp-1"})
        await llm.queue.put(
            {"type": "response.done", "response_id": "resp-1", "status": "completed"}
        )
        # The deterministic fake speaker uses this event as its drain signal;
        # the zero-byte TTS provider never writes or plays any audio.
        await asyncio.wait_for(speaker.wait_started.wait(), timeout=0.3)
        speaker._drained.set()

        for _ in range(100):
            if "assistant-1" in llm.deleted_items:
                break
            await asyncio.sleep(0.002)

        assert speaker.first_play_at is None
        assert "assistant-1" in llm.deleted_items
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_zero_transcript_unheard_output_item_is_deleted(monkeypatch) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "hello", "item_id": "user-1"}
        )
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-no-delta",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {
                "type": "response.output_item",
                "response_id": "resp-no-delta",
                "item_id": "assistant-no-delta",
            }
        )
        await llm.queue.put(
            {
                "type": "response.done",
                "response_id": "resp-no-delta",
                "status": "completed",
            }
        )
        await asyncio.wait_for(speaker.wait_started.wait(), timeout=0.2)
        speaker._drained.set()

        for _ in range(100):
            if "assistant-no-delta" in llm.deleted_items:
                break
            await asyncio.sleep(0.002)

        assert speaker.first_play_at is None
        assert "assistant-no-delta" in llm.deleted_items
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_unheard_output_item_waits_for_delete_ack_before_listening(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    state = orch.StateMachine()
    monkeypatch.setattr(orch, "StateMachine", lambda: state)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "hello", "item_id": "user-1"}
        )
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-no-delta",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {
                "type": "response.output_item",
                "response_id": "resp-no-delta",
                "item_id": "assistant-no-delta",
            }
        )
        await llm.queue.put(
            {
                "type": "response.done",
                "response_id": "resp-no-delta",
                "status": "completed",
            }
        )
        await asyncio.wait_for(speaker.wait_started.wait(), timeout=0.2)
        speaker._drained.set()

        for _ in range(30):
            if "assistant-no-delta" in llm.deleted_items:
                break
            await asyncio.sleep(0.002)
        assert "assistant-no-delta" in llm.deleted_items

        # Sending conversation.item.delete is not deletion success. Do not
        # accept another user turn until the server's authoritative ACK arrives.
        await asyncio.sleep(0.01)
        assert state.phase == orch.Phase.RESPONDING

        await llm.queue.put(
            {
                "type": "conversation.item.deleted",
                "item_id": "assistant-no-delta",
            }
        )
        for _ in range(30):
            if state.phase == orch.Phase.LISTENING:
                break
            await asyncio.sleep(0.002)
        assert state.phase == orch.Phase.LISTENING
        assert run_task.done() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ["reject", "timeout"])
async def test_unheard_output_delete_failure_stops_unsafe_session(
    monkeypatch,
    failure_mode: str,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, mic = _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch, "_PROVIDER_CONTROL_TIMEOUT_S", 0.01)
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
    await llm.queue.put(
        {"type": "input.transcript", "text": "hello", "item_id": "user-1"}
    )
    await llm.queue.put(
        {
            "type": "response.created",
            "response_id": "resp-no-delta",
            "generation_id": 1,
        }
    )
    await llm.queue.put(
        {
            "type": "response.output_item",
            "response_id": "resp-no-delta",
            "item_id": "assistant-no-delta",
        }
    )
    await llm.queue.put(
        {
            "type": "response.done",
            "response_id": "resp-no-delta",
            "status": "completed",
        }
    )
    await asyncio.wait_for(speaker.wait_started.wait(), timeout=0.2)
    speaker._drained.set()
    for _ in range(30):
        if "assistant-no-delta" in llm.deleted_items:
            break
        await asyncio.sleep(0.002)
    assert "assistant-no-delta" in llm.deleted_items

    if failure_mode == "reject":
        await llm.queue.put(
            {
                "type": "error",
                "operation": "conversation.item.delete",
                "item_id": "assistant-no-delta",
                "error": SimpleNamespace(code="invalid_request_error"),
            }
        )

    with pytest.raises(BaseExceptionGroup) as raised:
        await asyncio.wait_for(run_task, timeout=0.3)
    expected_failure = (
        "Provider history mutation was rejected"
        if failure_mode == "reject"
        else "Unheard response item deletion was not acknowledged"
    )
    assert _exception_tree_contains(raised.value, expected_failure)
    assert mic.stopped is True
    assert pipeline.turn.closed is True
    assert llm.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("audible", [False, True], ids=["zero_playback", "partial"])
async def test_terminal_speaker_failure_discards_remote_item_and_stops_runtime(
    monkeypatch,
    audible: bool,
) -> None:
    class FailedDrainSpeaker(BlockingSpeaker):
        def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
            super().__init__(loop)
            self.device_failed = asyncio.Event()
            self.played_audio_ms = 120 if audible else 0

        async def feed(self) -> None:
            payload = await self.queue.get()
            if payload and audible:
                now = asyncio.get_running_loop().time()
                self.first_write_at = now
                self.first_play_at = now
            self.buffered.set()
            await self.device_failed.wait()
            raise RuntimeError(
                "Speaker output failed (stream_inactive); restart Zemory"
            )

        async def wait_until_done(self) -> bool:
            self.wait_started.set()
            return False

    llm = QueueLLM()
    speaker = FailedDrainSpeaker(asyncio.get_running_loop())
    pipeline, mic = _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "hello", "item_id": "user-1"}
        )
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-1",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {
                "type": "response.output_item",
                "response_id": "resp-1",
                "item_id": "assistant-1",
            }
        )
        await llm.queue.put(
            {
                "type": "audio.transcript.delta",
                "response_id": "resp-1",
                "delta": "FULL TEXT",
            }
        )
        await llm.queue.put(
            {
                "type": "audio.delta",
                "response_id": "resp-1",
                "item_id": "assistant-1",
                "content_index": 0,
                "audio": b"partial-audio",
            }
        )
        await asyncio.wait_for(speaker.buffered.wait(), timeout=0.2)
        assert (speaker.first_play_at is not None) is audible
        await llm.queue.put(
            {"type": "response.done", "response_id": "resp-1", "status": "completed"}
        )

        for _ in range(100):
            if "assistant-1" in llm.deleted_items:
                break
            await asyncio.sleep(0.002)

        assert "assistant-1" in llm.deleted_items
        speaker.device_failed.set()

        with pytest.raises(BaseExceptionGroup) as raised:
            await asyncio.wait_for(run_task, timeout=0.3)

        assert _exception_tree_contains(
            raised.value,
            "Speaker output failed (stream_inactive); restart Zemory",
        )
        assert mic.stopped is True
        assert pipeline.turn.closed is True
        assert llm.closed is True
    finally:
        if not run_task.done():
            run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_cleanup_task_cannot_skip_later_resources(monkeypatch) -> None:
    llm = EofLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, mic = _configure_runtime(monkeypatch, llm, speaker)

    async def cancel_mic_cleanup() -> None:
        mic.stopped = True
        raise asyncio.CancelledError

    mic.stop = cancel_mic_cleanup  # type: ignore[method-assign]

    with pytest.raises(BaseExceptionGroup):
        await orch.run()

    assert mic.stopped is True
    assert pipeline.turn.closed is True
    assert llm.closed is True


@pytest.mark.asyncio
async def test_sync_cleanup_timeout_is_a_real_wall_clock_deadline(monkeypatch) -> None:
    llm = EofLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, mic = _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch, "_CLEANUP_TIMEOUT_S", 0.01)

    def block_mic_cleanup() -> None:
        time.sleep(0.3)
        mic.stopped = True

    mic.stop = block_mic_cleanup  # type: ignore[method-assign]
    started = time.perf_counter()

    with pytest.raises(BaseExceptionGroup):
        await orch.run()

    assert time.perf_counter() - started < 0.15
    assert pipeline.turn.closed is True
    assert llm.closed is True


@pytest.mark.asyncio
async def test_ambiguous_manual_commit_failure_stops_session(
    monkeypatch,
) -> None:
    class FailOnceCommitLLM(QueueLLM):
        def __init__(self) -> None:
            super().__init__()
            self.commit_attempts = 0

        async def commit_input_audio_buffer(
            self,
            *,
            generation_id: int | None = None,
        ) -> None:
            self.commit_attempts += 1
            if self.commit_attempts == 1:
                raise RuntimeError("synthetic commit failure")
            await super().commit_input_audio_buffer(generation_id=generation_id)

    llm = FailOnceCommitLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, mic = _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings.realtime, "turn_detection", "none")
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    await pipeline.turn.events.put("speech_end")
    with pytest.raises(BaseExceptionGroup) as raised:
        await asyncio.wait_for(run_task, timeout=0.2)
    assert _exception_tree_contains(
        raised.value,
        "Manual input commit failed in an ambiguous provider state",
    )
    assert llm.commit_attempts == 1
    assert llm.input_buffer_cleared == 0
    assert mic.stopped is True
    assert llm.closed is True


@pytest.mark.asyncio
async def test_correlated_manual_commit_error_requires_clear_ack_before_reuse(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, _ = _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings.realtime, "turn_detection", "none")
    state = orch.StateMachine()
    monkeypatch.setattr(orch, "StateMachine", lambda: state)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await pipeline.turn.events.put("speech_end")
        for _ in range(30):
            if llm.committed == 1:
                break
            await asyncio.sleep(0.002)
        await llm.queue.put(
            {
                "type": "error",
                "operation": "input_audio_buffer.commit",
                "generation_id": 1,
                "error": SimpleNamespace(code="invalid_request_error"),
            }
        )
        for _ in range(30):
            if llm.input_buffer_cleared == 1:
                break
            await asyncio.sleep(0.002)
        assert llm.input_buffer_cleared == 1
        assert state.phase == orch.Phase.RESPONDING

        await llm.queue.put({"type": "input.cleared", "generation_id": 1})
        for _ in range(30):
            if state.phase == orch.Phase.LISTENING:
                break
            await asyncio.sleep(0.002)
        assert state.phase == orch.Phase.LISTENING
        assert run_task.done() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ["reject", "timeout"])
async def test_manual_input_clear_failure_stops_unsafe_session(
    monkeypatch,
    failure_mode: str,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, _ = _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings.realtime, "turn_detection", "none")
    monkeypatch.setattr(orch, "_PROVIDER_CONTROL_TIMEOUT_S", 0.01)
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    await pipeline.turn.events.put("speech_end")
    for _ in range(30):
        if llm.committed == 1:
            break
        await asyncio.sleep(0.002)
    await llm.queue.put(
        {
            "type": "error",
            "operation": "input_audio_buffer.commit",
            "generation_id": 1,
            "error": SimpleNamespace(code="invalid_request_error"),
        }
    )
    for _ in range(30):
        if llm.input_buffer_cleared == 1:
            break
        await asyncio.sleep(0.002)
    assert llm.input_buffer_cleared == 1

    if failure_mode == "reject":
        await llm.queue.put(
                {
                    "type": "error",
                    "operation": "input_audio_buffer.clear",
                    "generation_id": 1,
                    "error": SimpleNamespace(code="invalid_request_error"),
                }
        )

    with pytest.raises(BaseExceptionGroup) as raised:
        await asyncio.wait_for(run_task, timeout=0.3)
    expected = (
        "Manual input buffer clear was rejected"
        if failure_mode == "reject"
        else "Manual input buffer clear was not acknowledged"
    )
    assert _exception_tree_contains(raised.value, expected)


@pytest.mark.asyncio
async def test_new_speech_before_manual_commit_ack_fails_closed(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, _ = _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings.realtime, "turn_detection", "none")
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    await pipeline.turn.events.put("speech_end")
    for _ in range(30):
        if llm.committed == 1:
            break
        await asyncio.sleep(0.002)
    assert llm.committed == 1

    # The real manual detector remains paused until the commit ACK. If an
    # out-of-order source nevertheless reports a new utterance, continuing
    # could mix its frames into the still-uncommitted server buffer.
    await pipeline.turn.events.put("speech_start")
    with pytest.raises(BaseExceptionGroup) as raised:
        await asyncio.wait_for(run_task, timeout=0.2)
    assert _exception_tree_contains(
        raised.value,
        "New speech began before the manual input boundary was acknowledged",
    )
    assert llm.input_buffer_cleared == 0


@pytest.mark.asyncio
async def test_stale_manual_commit_ack_fails_closed_without_claiming_turn(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, _ = _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings.realtime, "turn_detection", "none")
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    await pipeline.turn.events.put("speech_end")
    for _ in range(30):
        if llm.committed == 1:
            break
        await asyncio.sleep(0.002)
    await llm.queue.put(
        {
            "type": "input.committed",
            "item_id": "stale-user-item",
            "generation_id": 0,
        }
    )

    with pytest.raises(BaseExceptionGroup) as raised:
        await asyncio.wait_for(run_task, timeout=0.2)
    assert _exception_tree_contains(
        raised.value,
        "Stale manual input commit acknowledgement",
    )
    assert llm.responses_created == 0


@pytest.mark.asyncio
async def test_manual_commit_ack_before_send_return_keeps_item_scoped_guard(
    monkeypatch,
) -> None:
    class AckBeforeReturnLLM(QueueLLM):
        async def commit_input_audio_buffer(
            self,
            *,
            generation_id: int | None = None,
        ) -> None:
            self.committed += 1
            await self.queue.put(
                {
                    "type": "input.committed",
                    "item_id": "manual-early-ack",
                    "generation_id": generation_id,
                }
            )
            while not self.queue.empty():
                await asyncio.sleep(0)
            await asyncio.sleep(0)

    llm = AckBeforeReturnLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, _ = _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings.realtime, "turn_detection", "none")
    monkeypatch.setattr(orch, "_INPUT_TRANSCRIPT_TIMEOUT_S", 0.01)
    state = orch.StateMachine()
    monkeypatch.setattr(orch, "StateMachine", lambda: state)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await pipeline.turn.events.put("speech_end")
        for _ in range(50):
            if llm.deleted_items == ["manual-early-ack"]:
                break
            if run_task.done():
                break
            await asyncio.sleep(0.002)
        assert llm.deleted_items == ["manual-early-ack"]
        assert run_task.done() is False

        await llm.queue.put(
            {
                "type": "conversation.item.deleted",
                "item_id": "manual-early-ack",
            }
        )
        for _ in range(30):
            if state.phase == orch.Phase.LISTENING:
                break
            await asyncio.sleep(0.002)
        assert state.phase == orch.Phase.LISTENING
        assert run_task.done() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_new_speech_during_manual_clear_ack_wait_fails_closed(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, _ = _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings.realtime, "turn_detection", "none")
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    await pipeline.turn.events.put("speech_end")
    for _ in range(30):
        if llm.committed == 1:
            break
        await asyncio.sleep(0.002)
    await llm.queue.put(
        {
            "type": "error",
            "operation": "input_audio_buffer.commit",
            "generation_id": 1,
            "error": SimpleNamespace(code="invalid_request_error"),
        }
    )
    for _ in range(30):
        if llm.input_buffer_cleared == 1:
            break
        await asyncio.sleep(0.002)
    assert llm.input_buffer_cleared == 1

    await pipeline.turn.events.put("speech_start")
    await llm.queue.put(
        {
            "type": "input.cleared",
            "generation_id": 1,
        }
    )
    with pytest.raises(BaseExceptionGroup) as raised:
        await asyncio.wait_for(run_task, timeout=0.2)
    assert _exception_tree_contains(
        raised.value,
        "New speech began before the manual input boundary was acknowledged",
    )


@pytest.mark.asyncio
async def test_stale_manual_clear_ack_cannot_release_current_generation(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, _ = _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings.realtime, "turn_detection", "none")
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    await pipeline.turn.events.put("speech_end")
    for _ in range(30):
        if llm.committed == 1:
            break
        await asyncio.sleep(0.002)
    await llm.queue.put(
        {
            "type": "error",
            "operation": "input_audio_buffer.commit",
            "generation_id": 1,
            "error": SimpleNamespace(code="invalid_request_error"),
        }
    )
    for _ in range(30):
        if llm.input_buffer_cleared == 1:
            break
        await asyncio.sleep(0.002)
    assert llm.input_buffer_cleared == 1

    await llm.queue.put(
        {
            "type": "input.cleared",
            "generation_id": 0,
        }
    )
    with pytest.raises(BaseExceptionGroup) as raised:
        await asyncio.wait_for(run_task, timeout=0.2)
    assert _exception_tree_contains(
        raised.value,
        "Unowned manual input clear acknowledgement",
    )


@pytest.mark.asyncio
async def test_audible_raw_response_records_user_before_assistant_when_correction_loses(
    monkeypatch,
) -> None:
    class DelayedRecordingCorrector:
        instance: DelayedRecordingCorrector | None = None

        def __init__(self, **kwargs) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.history: list[tuple[str, str]] = []
            DelayedRecordingCorrector.instance = self

        async def correct(self, raw: str) -> tuple[str, float]:
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            return f"corrected:{raw}", 1.0

        def record_user(self, text: str) -> None:
            self.history.append(("user", text))

        def record_assistant(self, text: str) -> None:
            self.history.append(("assistant", text))

        async def aclose(self) -> None:
            return None

    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings, "transcript_correction_enabled", True)
    monkeypatch.setattr(orch, "TranscriptCorrector", DelayedRecordingCorrector)
    monkeypatch.setattr(orch, "AsyncOpenAI", lambda **kwargs: object())
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put(
            {"type": "input.speech_stopped", "item_id": "user-raw"}
        )
        await llm.queue.put(
            {"type": "input.transcript", "text": "raw", "item_id": "user-raw"}
        )
        corrector = DelayedRecordingCorrector.instance
        assert corrector is not None
        await asyncio.wait_for(corrector.started.wait(), timeout=0.2)

        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-raw",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {
                "type": "audio.transcript.delta",
                "response_id": "resp-raw",
                "delta": "RAW ANSWER",
            }
        )
        await llm.queue.put(
            {
                "type": "audio.delta",
                "response_id": "resp-raw",
                "item_id": "assistant-raw",
                "audio": b"pcm",
            }
        )
        await llm.queue.put(
            {
                "type": "response.done",
                "response_id": "resp-raw",
                "generation_id": 1,
                "status": "completed",
            }
        )
        await asyncio.wait_for(speaker.buffered.wait(), timeout=0.2)
        speaker._drained.set()

        for _ in range(50):
            if corrector.history:
                break
            await asyncio.sleep(0.002)
        corrector.release.set()
        for _ in range(50):
            if len(corrector.history) == 2 or corrector.cancelled.is_set():
                break
            await asyncio.sleep(0.002)

        assert corrector.history == [
            ("user", "raw"),
            ("assistant", "RAW ANSWER"),
        ]
        assert corrector.cancelled.is_set()
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_successful_correction_records_user_once_before_replacement_assistant(
    monkeypatch,
) -> None:
    class DelayedRecordingCorrector:
        instance: DelayedRecordingCorrector | None = None

        def __init__(self, **kwargs) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.history: list[tuple[str, str]] = []
            DelayedRecordingCorrector.instance = self

        async def correct(self, raw: str) -> tuple[str, float]:
            self.started.set()
            await self.release.wait()
            return f"corrected:{raw}", 1.0

        def record_user(self, text: str) -> None:
            self.history.append(("user", text))

        def record_assistant(self, text: str) -> None:
            self.history.append(("assistant", text))

        async def aclose(self) -> None:
            return None

    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings, "transcript_correction_enabled", True)
    monkeypatch.setattr(orch, "TranscriptCorrector", DelayedRecordingCorrector)
    monkeypatch.setattr(orch, "AsyncOpenAI", lambda **kwargs: object())
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put(
            {"type": "input.speech_stopped", "item_id": "user-raw"}
        )
        await llm.queue.put(
            {"type": "input.transcript", "text": "raw", "item_id": "user-raw"}
        )
        corrector = DelayedRecordingCorrector.instance
        assert corrector is not None
        await asyncio.wait_for(corrector.started.wait(), timeout=0.2)
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-raw",
                "generation_id": 1,
            }
        )
        await asyncio.sleep(0)
        corrector.release.set()
        await asyncio.wait_for(llm.cancelled.wait(), timeout=0.2)
        assert llm.cancelled_response_ids == ["resp-raw"]
        await llm.queue.put(
            {
                "type": "response.done",
                "response_id": "resp-raw",
                "generation_id": 1,
                "status": "cancelled",
            }
        )
        for _ in range(50):
            if llm.deleted_items == ["user-raw"]:
                break
            await asyncio.sleep(0.002)
        await llm.queue.put(
            {"type": "conversation.item.deleted", "item_id": "user-raw"}
        )
        for _ in range(50):
            if llm.user_texts == ["corrected:raw"]:
                break
            await asyncio.sleep(0.002)
        assert llm.user_texts == ["corrected:raw"]

        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-corrected",
                "generation_id": 2,
            }
        )
        await llm.queue.put(
            {
                "type": "audio.transcript.delta",
                "response_id": "resp-corrected",
                "delta": "CORRECTED ANSWER",
            }
        )
        await llm.queue.put(
            {
                "type": "audio.delta",
                "response_id": "resp-corrected",
                "item_id": "assistant-corrected",
                "audio": b"pcm",
            }
        )
        await llm.queue.put(
            {
                "type": "response.done",
                "response_id": "resp-corrected",
                "generation_id": 2,
                "status": "completed",
            }
        )
        await asyncio.wait_for(speaker.buffered.wait(), timeout=0.2)
        speaker._drained.set()
        for _ in range(50):
            if len(corrector.history) == 2:
                break
            await asyncio.sleep(0.002)

        assert corrector.history == [
            ("user", "corrected:raw"),
            ("assistant", "CORRECTED ANSWER"),
        ]
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.parametrize("failure_mode", ["delete", "send"])
@pytest.mark.asyncio
async def test_speculative_correction_ambiguous_failure_stops_session(
    monkeypatch,
    failure_mode: str,
) -> None:
    class ControlledCorrector:
        instance: ControlledCorrector | None = None

        def __init__(self, **kwargs) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            ControlledCorrector.instance = self

        async def correct(self, raw: str) -> tuple[str, float]:
            self.started.set()
            await self.release.wait()
            return f"corrected:{raw}", 1.0

        def record_user(self, text: str) -> None:
            return None

        def record_assistant(self, text: str) -> None:
            return None

    class ReplacementFailureLLM(QueueLLM):
        def __init__(self) -> None:
            super().__init__()
            self.failure_observed = asyncio.Event()

        async def delete_item(self, item_id: str) -> bool:
            self.deleted_items.append(item_id)
            if failure_mode == "delete":
                self.failure_observed.set()
                return False
            return True

        async def send_user_text(
            self,
            text: str,
            injections=None,
            *,
            generation_id: int | None = None,
        ) -> None:
            if failure_mode == "send":
                self.failure_observed.set()
                raise RuntimeError("synthetic replacement send failure")
            await super().send_user_text(
                text,
                injections=injections,
                generation_id=generation_id,
            )

    llm = ReplacementFailureLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings, "transcript_correction_enabled", True)
    monkeypatch.setattr(orch, "TranscriptCorrector", ControlledCorrector)
    monkeypatch.setattr(orch, "AsyncOpenAI", lambda **kwargs: object())
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put(
            {"type": "input.speech_stopped", "item_id": "user-raw"}
        )
        await llm.queue.put(
            {"type": "input.transcript", "text": "raw", "item_id": "user-raw"}
        )
        corrector = ControlledCorrector.instance
        assert corrector is not None
        await asyncio.wait_for(corrector.started.wait(), timeout=0.2)
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-raw",
                "generation_id": 1,
            }
        )
        await asyncio.sleep(0)
        corrector.release.set()
        await asyncio.wait_for(llm.cancelled.wait(), timeout=0.2)
        await llm.queue.put(
            {
                "type": "response.done",
                "response_id": "resp-raw",
                "status": "cancelled",
            }
        )
        if failure_mode == "send":
            for _ in range(50):
                if "user-raw" in llm.deleted_items:
                    break
                await asyncio.sleep(0.002)
            await llm.queue.put(
                {"type": "conversation.item.deleted", "item_id": "user-raw"}
            )
        await asyncio.wait_for(llm.failure_observed.wait(), timeout=0.2)
        await asyncio.sleep(0.01)

        with pytest.raises(BaseExceptionGroup) as raised:
            await asyncio.wait_for(run_task, timeout=0.2)
        expected = (
            "Corrected input item deletion was not acknowledged"
            if failure_mode == "delete"
            else "Corrected response request failed in an ambiguous provider state"
        )
        assert _exception_tree_contains(raised.value, expected)
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_unique_stale_response_flood_fails_at_scoped_cancel_capacity(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch, "_PENDING_PROVIDER_CONTROL_LIMIT", 4)
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    original_create_task = asyncio.create_task
    control_task_names: list[str] = []

    def tracking_create_task(coro, *, name=None, context=None):
        if isinstance(name, str) and name.startswith("cancel_stale_response_"):
            control_task_names.append(name)
        kwargs = {"name": name}
        if context is not None:
            kwargs["context"] = context
        return original_create_task(coro, **kwargs)

    monkeypatch.setattr(orch.asyncio, "create_task", tracking_create_task)
    for index in range(1_000):
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": f"stale-{index}",
                "generation_id": -1,
            }
        )

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await asyncio.wait_for(run_task, timeout=0.5)

    assert _exception_tree_contains(
        exc_info.value,
        "Pending response cancellation ACK capacity exceeded",
    )
    assert len(control_task_names) == 4
    assert len(llm.cancelled_response_ids) <= 4


@pytest.mark.asyncio
async def test_unique_stale_item_flood_fails_at_delete_task_capacity(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch, "_PENDING_PROVIDER_CONTROL_LIMIT", 4)
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    original_create_task = asyncio.create_task
    control_task_names: list[str] = []

    def tracking_create_task(coro, *, name=None, context=None):
        if isinstance(name, str) and name.startswith("delete_stale_item_"):
            control_task_names.append(name)
        kwargs = {"name": name}
        if context is not None:
            kwargs["context"] = context
        return original_create_task(coro, **kwargs)

    monkeypatch.setattr(orch.asyncio, "create_task", tracking_create_task)
    for index in range(1_000):
        await llm.queue.put(
            {
                "type": "conversation.item.created",
                "item_id": f"stale-item-{index}",
                "role": "assistant",
            }
        )

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await asyncio.wait_for(run_task, timeout=0.5)

    assert _exception_tree_contains(
        exc_info.value,
        "Pending stale item deletion capacity exceeded",
    )
    assert len(control_task_names) == 4
    assert len(llm.deleted_items) <= 4


@pytest.mark.parametrize("created_received", [False, True])
@pytest.mark.asyncio
async def test_missing_response_terminal_watchdog_recovers_to_listening(
    monkeypatch,
    created_received: bool,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, mic = _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings, "enable_barge_in", False)
    monkeypatch.setattr(orch, "_RESPONSE_TERMINAL_TIMEOUT_S", 0.02)
    monkeypatch.setattr(orch, "_PROVIDER_CONTROL_TIMEOUT_S", 0.05)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put(
            {"type": "input.speech_stopped", "item_id": "user-lost-terminal"}
        )
        await llm.queue.put(
            {
                "type": "input.transcript",
                "text": "hello",
                "item_id": "user-lost-terminal",
            }
        )
        for _ in range(50):
            if llm.responses_created == 1:
                break
            await asyncio.sleep(0.002)
        assert llm.responses_created == 1

        if created_received:
            await llm.queue.put(
                {
                    "type": "response.created",
                    "response_id": "resp-lost-terminal",
                    "generation_id": 1,
                }
            )

        await asyncio.wait_for(llm.cancelled.wait(), timeout=0.2)
        if created_received:
            await llm.queue.put(
                {
                    "type": "response.done",
                    "response_id": "resp-lost-terminal",
                    "status": "cancelled",
                }
            )
            assert llm.cancelled_response_ids == ["resp-lost-terminal"]
        else:
            # The response.create call returned, but response.created never
            # arrived, so the watchdog must use the bounded unscoped path.
            await llm.queue.put(
                {
                    "type": "error",
                    "operation": "response.cancel",
                    "error": SimpleNamespace(code="response_cancel_not_active"),
                }
            )
            assert llm.cancelled_response_ids == [None]

        for _ in range(50):
            await mic.queue.put(b"after-watchdog")
            if pipeline.turn.feed_event.is_set():
                break
            await asyncio.sleep(0.002)
        await asyncio.wait_for(pipeline.turn.feed_event.wait(), timeout=0.2)
        assert pipeline.turn.fed
        assert set(pipeline.turn.fed) == {b"after-watchdog"}
        assert speaker.cleared >= 1
        assert run_task.done() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.parametrize(
    "profile",
    ["realtime_audio", "realtime_text_external_tts"],
)
@pytest.mark.asyncio
async def test_terminal_watchdog_syncs_partial_output_before_listening(
    monkeypatch,
    profile: str,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings, "profile", profile)
    monkeypatch.setattr(orch, "_RESPONSE_TERMINAL_TIMEOUT_S", 0.01)
    monkeypatch.setattr(orch, "_PROVIDER_CONTROL_TIMEOUT_S", 0.05)
    state = orch.StateMachine()
    monkeypatch.setattr(orch, "StateMachine", lambda: state)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_started"})
        await llm.queue.put(
            {"type": "input.speech_stopped", "item_id": "user-partial"}
        )
        await llm.queue.put(
            {
                "type": "input.transcript",
                "text": "hello",
                "item_id": "user-partial",
            }
        )
        for _ in range(50):
            if llm.responses_created == 1:
                break
            await asyncio.sleep(0.002)
        assert llm.responses_created == 1
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-partial",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {
                "type": "response.output_item",
                "response_id": "resp-partial",
                "item_id": "assistant-partial",
            }
        )
        if profile == "realtime_audio":
            await llm.queue.put(
                {
                    "type": "audio.delta",
                    "response_id": "resp-partial",
                    "item_id": "assistant-partial",
                    "content_index": 2,
                    "audio": b"partial-audio",
                }
            )
            await asyncio.wait_for(speaker.buffered.wait(), timeout=0.2)
        else:
            await llm.queue.put(
                {
                    "type": "text.delta",
                    "response_id": "resp-partial",
                    "delta": "PARTIAL",
                }
            )

        await asyncio.wait_for(llm.cancelled.wait(), timeout=0.2)
        assert llm.truncated == []
        assert llm.deleted_items == []
        await llm.queue.put(
            {
                "type": "response.done",
                "response_id": "resp-partial",
                "status": "cancelled",
            }
        )
        for _ in range(50):
            mutation_sent = (
                bool(llm.truncated)
                if profile == "realtime_audio"
                else "assistant-partial" in llm.deleted_items
            )
            if mutation_sent:
                break
            await asyncio.sleep(0.002)

        assert state.phase == orch.Phase.RESPONDING
        if profile == "realtime_audio":
            assert llm.truncated == [
                {
                    "item_id": "assistant-partial",
                    "content_index": 2,
                    "audio_end_ms": 120,
                }
            ]
            await llm.queue.put(
                {
                    "type": "conversation.item.truncated",
                    "item_id": "assistant-partial",
                }
            )
        else:
            assert llm.deleted_items == ["assistant-partial"]
            await llm.queue.put(
                {
                    "type": "conversation.item.deleted",
                    "item_id": "assistant-partial",
                }
            )

        for _ in range(50):
            if state.phase == orch.Phase.LISTENING:
                break
            await asyncio.sleep(0.002)
        assert state.phase == orch.Phase.LISTENING
        assert run_task.done() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.parametrize(
    "profile",
    ["realtime_audio", "realtime_text_external_tts"],
)
@pytest.mark.parametrize("mutation_outcome", ["rejected", "timeout"])
@pytest.mark.asyncio
async def test_terminal_watchdog_partial_mutation_failure_stops_unsafe_session(
    monkeypatch,
    profile: str,
    mutation_outcome: str,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings, "profile", profile)
    monkeypatch.setattr(orch, "_RESPONSE_TERMINAL_TIMEOUT_S", 0.01)
    monkeypatch.setattr(orch, "_PROVIDER_CONTROL_TIMEOUT_S", 0.02)
    state = orch.StateMachine()
    monkeypatch.setattr(orch, "StateMachine", lambda: state)
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    await llm.queue.put({"type": "input.speech_started"})
    await llm.queue.put(
        {"type": "input.speech_stopped", "item_id": "user-partial"}
    )
    await llm.queue.put(
        {
            "type": "input.transcript",
            "text": "hello",
            "item_id": "user-partial",
        }
    )
    for _ in range(50):
        if llm.responses_created == 1:
            break
        await asyncio.sleep(0.002)
    assert llm.responses_created == 1
    await llm.queue.put(
        {
            "type": "response.created",
            "response_id": "resp-partial",
            "generation_id": 1,
        }
    )
    await llm.queue.put(
        {
            "type": "response.output_item",
            "response_id": "resp-partial",
            "item_id": "assistant-partial",
        }
    )
    if profile == "realtime_audio":
        await llm.queue.put(
            {
                "type": "audio.delta",
                "response_id": "resp-partial",
                "item_id": "assistant-partial",
                "content_index": 0,
                "audio": b"partial-audio",
            }
        )
    else:
        await llm.queue.put(
            {
                "type": "text.delta",
                "response_id": "resp-partial",
                "delta": "PARTIAL",
            }
        )

    await asyncio.wait_for(llm.cancelled.wait(), timeout=0.2)
    await llm.queue.put(
        {
            "type": "response.done",
            "response_id": "resp-partial",
            "status": "cancelled",
        }
    )
    for _ in range(50):
        mutation_sent = (
            bool(llm.truncated)
            if profile == "realtime_audio"
            else "assistant-partial" in llm.deleted_items
        )
        if mutation_sent:
            break
        await asyncio.sleep(0.002)
    assert mutation_sent

    if mutation_outcome == "rejected":
        await llm.queue.put(
            {
                "type": "error",
                "operation": (
                    "conversation.item.truncate"
                    if profile == "realtime_audio"
                    else "conversation.item.delete"
                ),
                "item_id": "assistant-partial",
                "error": SimpleNamespace(code="invalid_request_error"),
            }
        )

    with pytest.raises(BaseExceptionGroup) as raised:
        await asyncio.wait_for(run_task, timeout=0.3)
    expected = (
        "Provider history mutation was rejected"
        if mutation_outcome == "rejected"
        else "Terminal response history mutation was not acknowledged"
    )
    assert _exception_tree_contains(raised.value, expected)
    assert state.phase == orch.Phase.RESPONDING


@pytest.mark.asyncio
@pytest.mark.parametrize("created_received", [False, True])
async def test_watchdog_cancel_ack_timeout_stops_unsafe_session(
    monkeypatch,
    created_received: bool,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch, "_RESPONSE_TERMINAL_TIMEOUT_S", 0.01)
    monkeypatch.setattr(orch, "_PROVIDER_CONTROL_TIMEOUT_S", 0.01)
    run_task = asyncio.create_task(orch.run())

    await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
    await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
    await llm.queue.put(
        {"type": "input.transcript", "text": "hello", "item_id": "user-1"}
    )
    if created_received:
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-lost-terminal",
                "generation_id": 1,
            }
        )
    await asyncio.wait_for(llm.cancelled.wait(), timeout=0.2)

    with pytest.raises(BaseExceptionGroup) as raised:
        await asyncio.wait_for(run_task, timeout=0.3)
    assert _exception_tree_contains(
        raised.value,
        "Timed-out response cancellation was not acknowledged",
    )


@pytest.mark.asyncio
async def test_normal_response_done_cancels_terminal_watchdog(monkeypatch) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch, "_RESPONSE_TERMINAL_TIMEOUT_S", 0.02)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put(
            {"type": "input.speech_stopped", "item_id": "user-complete"}
        )
        await llm.queue.put(
            {
                "type": "input.transcript",
                "text": "hello",
                "item_id": "user-complete",
            }
        )
        for _ in range(50):
            if llm.responses_created == 1:
                break
            await asyncio.sleep(0.002)
        await asyncio.sleep(0)
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-complete",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {
                "type": "response.done",
                "response_id": "resp-complete",
                "status": "completed",
            }
        )
        await asyncio.sleep(0.05)

        assert llm.cancelled_response_ids == []
        assert run_task.done() is False
    finally:
        speaker._drained.set()
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_stale_terminal_watchdog_cannot_cancel_new_generation(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch, "_RESPONSE_TERMINAL_TIMEOUT_S", 0.12)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put(
            {"type": "input.speech_stopped", "item_id": "user-old"}
        )
        await llm.queue.put(
            {"type": "input.transcript", "text": "old", "item_id": "user-old"}
        )
        for _ in range(50):
            if llm.responses_created == 1:
                break
            await asyncio.sleep(0.002)
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-old",
                "generation_id": 1,
            }
        )
        await asyncio.sleep(0.07)

        await llm.queue.put({"type": "input.speech_started"})
        await asyncio.wait_for(llm.cancelled.wait(), timeout=0.2)
        await llm.queue.put(
            {
                "type": "response.done",
                "response_id": "resp-old",
                "status": "cancelled",
            }
        )
        await llm.queue.put(
            {"type": "input.speech_stopped", "item_id": "user-new"}
        )
        await llm.queue.put(
            {"type": "input.transcript", "text": "new", "item_id": "user-new"}
        )
        for _ in range(50):
            if llm.responses_created == 2:
                break
            await asyncio.sleep(0.002)
        assert llm.responses_created == 2
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-new",
                "generation_id": 3,
            }
        )

        # Cross generation 1's original deadline while remaining comfortably
        # inside generation 3's independently armed deadline.
        await asyncio.sleep(0.07)
        assert llm.cancelled_response_ids == ["resp-old"]
        assert run_task.done() is False

        await llm.queue.put(
            {
                "type": "response.done",
                "response_id": "resp-new",
                "status": "completed",
            }
        )
    finally:
        speaker._drained.set()
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_repeated_stale_item_events_schedule_only_one_delete(monkeypatch) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        for _ in range(2):
            await llm.queue.put(
                {
                    "type": "audio.delta",
                    "response_id": "resp-stale",
                    "item_id": "item-stale",
                    "audio": b"stale",
                }
            )
        for _ in range(50):
            if llm.deleted_items:
                break
            await asyncio.sleep(0.002)

        assert llm.deleted_items == ["item-stale"]
        assert speaker.buffered.is_set() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.parametrize(
    "terminal_status",
    ["completed", "cancelled", "failed", "incomplete"],
)
@pytest.mark.asyncio
async def test_every_response_terminal_records_usage_exactly_once(
    monkeypatch,
    terminal_status: str,
) -> None:
    class RecordingLog:
        def __init__(self) -> None:
            self.usage_events: list[dict] = []

        def info(self, event: str, **fields) -> None:
            if event == "response.usage":
                self.usage_events.append(fields)

        def debug(self, event: str, **fields) -> None:
            return None

        def warning(self, event: str, **fields) -> None:
            return None

        def error(self, event: str, **fields) -> None:
            return None

    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    recording_log = RecordingLog()
    observations: list[tuple[str, float]] = []
    monkeypatch.setattr(orch, "_log", recording_log)
    monkeypatch.setattr(
        orch.metrics,
        "observe",
        lambda name, value: observations.append((name, value)),
    )
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "private", "item_id": "user-1"}
        )
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-usage",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {
                "type": "response.done",
                "response_id": "resp-usage",
                "status": terminal_status,
                "usage": {
                    "total_tokens": 14,
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "cached_tokens": 3,
                },
            }
        )
        for _ in range(30):
            if recording_log.usage_events:
                break
            await asyncio.sleep(0.002)
        speaker._drained.set()
        await asyncio.sleep(0.01)

        assert len(recording_log.usage_events) == 1
        assert recording_log.usage_events[0]["status"] == terminal_status
        assert "private" not in repr(recording_log.usage_events[0])
        assert observations.count(("tokens.input", 10)) == 1
        assert observations.count(("tokens.cached", 3)) == 1
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.parametrize("terminal_status", ["failed", "incomplete"])
@pytest.mark.asyncio
async def test_noncompleted_response_never_commits_full_assistant_history(
    monkeypatch,
    terminal_status: str,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "turn", "item_id": "user-1"}
        )
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-1",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {
                "type": "response.output_item",
                "response_id": "resp-1",
                "item_id": "assistant-1",
            }
        )
        await llm.queue.put(
            {
                "type": "audio.transcript.delta",
                "response_id": "resp-1",
                "delta": "PARTIAL",
            }
        )
        await llm.queue.put(
            {
                "type": "audio.delta",
                "response_id": "resp-1",
                "item_id": "assistant-1",
                "content_index": 0,
                "audio": b"heard-partial",
            }
        )
        await asyncio.wait_for(speaker.buffered.wait(), timeout=0.2)
        await llm.queue.put(
            {
                "type": "response.done",
                "response_id": "resp-1",
                "status": terminal_status,
            }
        )
        for _ in range(50):
            if llm.truncated:
                break
            await asyncio.sleep(0.002)

        assert llm.truncated == [
            {
                "item_id": "assistant-1",
                "content_index": 0,
                "audio_end_ms": 120,
            }
        ]
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.parametrize(
    "profile",
    ["realtime_audio", "realtime_text_external_tts"],
)
@pytest.mark.asyncio
async def test_noncompleted_terminal_acks_history_and_beep_before_listening(
    monkeypatch,
    profile: str,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    pipeline, mic = _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings, "profile", profile)
    monkeypatch.setattr(orch.settings, "enable_ready_beep", True)
    monkeypatch.setattr(orch.settings, "ready_beep_duration_ms", 10)
    monkeypatch.setattr(orch.settings, "ready_beep_post_gap_s", 0.0)
    state = orch.StateMachine()
    monkeypatch.setattr(orch, "StateMachine", lambda: state)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        for _ in range(50):
            if mic.clear_count:
                break
            await asyncio.sleep(0.002)
        assert mic.clear_count == 1
        startup_clear_count = mic.clear_count

        await llm.queue.put({"type": "input.speech_started"})
        await llm.queue.put(
            {"type": "input.speech_stopped", "item_id": "user-failed"}
        )
        await llm.queue.put(
            {
                "type": "input.transcript",
                "text": "hello",
                "item_id": "user-failed",
            }
        )
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-failed",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {
                "type": "response.output_item",
                "response_id": "resp-failed",
                "item_id": "assistant-failed",
            }
        )
        if profile == "realtime_audio":
            await llm.queue.put(
                {
                    "type": "audio.delta",
                    "response_id": "resp-failed",
                    "item_id": "assistant-failed",
                    "content_index": 1,
                    "audio": b"partial-audio",
                }
            )
        else:
            await llm.queue.put(
                {
                    "type": "text.delta",
                    "response_id": "resp-failed",
                    "delta": "PARTIAL",
                }
            )
        await llm.queue.put(
            {
                "type": "response.done",
                "response_id": "resp-failed",
                "generation_id": 1,
                "status": "failed",
            }
        )

        for _ in range(50):
            mutation_sent = (
                bool(llm.truncated)
                if profile == "realtime_audio"
                else "assistant-failed" in llm.deleted_items
            )
            if mutation_sent:
                break
            await asyncio.sleep(0.002)
        assert mutation_sent
        assert state.phase == orch.Phase.RESPONDING
        assert mic.clear_count == startup_clear_count
        assert pipeline.turn.feed_event.is_set() is False

        await llm.queue.put(
            {
                "type": (
                    "conversation.item.truncated"
                    if profile == "realtime_audio"
                    else "conversation.item.deleted"
                ),
                "item_id": "assistant-failed",
            }
        )
        for _ in range(50):
            if state.phase == orch.Phase.LISTENING:
                break
            await asyncio.sleep(0.002)

        assert state.phase == orch.Phase.LISTENING
        assert mic.clear_count == startup_clear_count + 1
        assert run_task.done() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.parametrize(
    "profile",
    ["realtime_audio", "realtime_text_external_tts"],
)
@pytest.mark.asyncio
async def test_unowned_cancelled_terminal_syncs_partial_before_listening(
    monkeypatch,
    profile: str,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings, "profile", profile)
    monkeypatch.setattr(orch, "_RESPONSE_TERMINAL_TIMEOUT_S", 1.0)
    state = orch.StateMachine()
    monkeypatch.setattr(orch, "StateMachine", lambda: state)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_started"})
        await llm.queue.put(
            {"type": "input.speech_stopped", "item_id": "user-cancelled"}
        )
        await llm.queue.put(
            {
                "type": "input.transcript",
                "text": "hello",
                "item_id": "user-cancelled",
            }
        )
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-cancelled",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {
                "type": "response.output_item",
                "response_id": "resp-cancelled",
                "item_id": "assistant-cancelled",
            }
        )
        if profile == "realtime_audio":
            await llm.queue.put(
                {
                    "type": "audio.delta",
                    "response_id": "resp-cancelled",
                    "item_id": "assistant-cancelled",
                    "content_index": 3,
                    "audio": b"partial-audio",
                }
            )
            await asyncio.wait_for(speaker.buffered.wait(), timeout=0.2)
        else:
            await llm.queue.put(
                {
                    "type": "text.delta",
                    "response_id": "resp-cancelled",
                    "delta": "PARTIAL",
                }
            )
        await llm.queue.put(
            {
                "type": "response.done",
                "response_id": "resp-cancelled",
                "generation_id": 1,
                "status": "cancelled",
            }
        )

        for _ in range(50):
            mutation_sent = (
                bool(llm.truncated)
                if profile == "realtime_audio"
                else "assistant-cancelled" in llm.deleted_items
            )
            if mutation_sent:
                break
            await asyncio.sleep(0.002)
        assert mutation_sent
        assert state.phase == orch.Phase.RESPONDING
        assert llm.cancelled_response_ids == []

        if profile == "realtime_audio":
            assert llm.truncated == [
                {
                    "item_id": "assistant-cancelled",
                    "content_index": 3,
                    "audio_end_ms": 120,
                }
            ]
            await llm.queue.put(
                {
                    "type": "conversation.item.truncated",
                    "item_id": "assistant-cancelled",
                }
            )
        else:
            assert llm.deleted_items == ["assistant-cancelled"]
            await llm.queue.put(
                {
                    "type": "conversation.item.deleted",
                    "item_id": "assistant-cancelled",
                }
            )

        for _ in range(50):
            if state.phase == orch.Phase.LISTENING:
                break
            await asyncio.sleep(0.002)
        assert state.phase == orch.Phase.LISTENING
        assert run_task.done() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_foreign_cancelled_terminal_cannot_mutate_current_partial_output(
    monkeypatch,
) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch, "_RESPONSE_TERMINAL_TIMEOUT_S", 1.0)
    state = orch.StateMachine()
    monkeypatch.setattr(orch, "StateMachine", lambda: state)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_started"})
        await llm.queue.put(
            {"type": "input.speech_stopped", "item_id": "user-current"}
        )
        await llm.queue.put(
            {
                "type": "input.transcript",
                "text": "hello",
                "item_id": "user-current",
            }
        )
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-current",
                "generation_id": 1,
            }
        )
        await llm.queue.put(
            {
                "type": "response.output_item",
                "response_id": "resp-current",
                "item_id": "assistant-current",
            }
        )
        await llm.queue.put(
            {
                "type": "audio.delta",
                "response_id": "resp-current",
                "item_id": "assistant-current",
                "content_index": 0,
                "audio": b"partial-audio",
            }
        )

        await llm.queue.put(
            {
                "type": "response.done",
                "response_id": "resp-foreign",
                "generation_id": 1,
                "status": "cancelled",
            }
        )
        await llm.queue.put(
            {
                "type": "response.done",
                "response_id": "resp-current",
                "generation_id": 0,
                "status": "cancelled",
            }
        )
        await asyncio.sleep(0.02)

        assert llm.truncated == []
        assert llm.deleted_items == []
        assert state.phase == orch.Phase.RESPONDING
        assert run_task.done() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_untagged_response_is_quarantined_as_foreign(monkeypatch) -> None:
    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put({"type": "input.speech_stopped", "item_id": "user-1"})
        await llm.queue.put(
            {"type": "input.transcript", "text": "turn", "item_id": "user-1"}
        )
        await llm.queue.put(
            {"type": "response.created", "response_id": "resp-foreign"}
        )
        await asyncio.wait_for(llm.cancelled.wait(), timeout=0.2)
        await llm.queue.put(
            {
                "type": "audio.delta",
                "response_id": "resp-foreign",
                "item_id": "foreign-item",
                "audio": b"must-not-play",
            }
        )
        await asyncio.sleep(0.01)

        assert llm.cancelled_response_ids == ["resp-foreign"]
        assert speaker.buffered.is_set() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_active_conflict_recovery_owns_generation_before_correction(
    monkeypatch,
) -> None:
    class DelayedChangedCorrector:
        instance: DelayedChangedCorrector | None = None

        def __init__(self, **kwargs) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            DelayedChangedCorrector.instance = self

        async def correct(self, raw: str) -> tuple[str, float]:
            self.started.set()
            await self.release.wait()
            return f"corrected:{raw}", 1.0

        def record_user(self, text: str) -> None:
            return None

        def record_assistant(self, text: str) -> None:
            return None

        async def aclose(self) -> None:
            return None

    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings, "transcript_correction_enabled", True)
    monkeypatch.setattr(orch, "TranscriptCorrector", DelayedChangedCorrector)
    monkeypatch.setattr(orch, "AsyncOpenAI", lambda **kwargs: object())
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put(
            {"type": "input.speech_stopped", "item_id": "user-raw"}
        )
        await llm.queue.put(
            {"type": "input.transcript", "text": "raw", "item_id": "user-raw"}
        )
        corrector = DelayedChangedCorrector.instance
        assert corrector is not None
        await asyncio.wait_for(corrector.started.wait(), timeout=0.2)
        for _ in range(50):
            if llm.responses_created == 1:
                break
            await asyncio.sleep(0.002)
        assert llm.responses_created == 1

        await llm.queue.put(
            {
                "type": "error",
                "operation": "response.create",
                "generation_id": 1,
                "error": SimpleNamespace(
                    code="conversation_already_has_active_response"
                ),
            }
        )
        for _ in range(50):
            if llm.cancelled_response_ids == [None]:
                break
            await asyncio.sleep(0.002)
        assert llm.cancelled_response_ids == [None]

        # Recovery has already claimed generation 1. Releasing correction
        # must not launch a second unscoped cancel/replacement path.
        corrector.release.set()
        await asyncio.sleep(0.02)
        assert llm.cancelled_response_ids == [None]
        assert llm.deleted_items == []

        await llm.queue.put(
            {
                "type": "error",
                "operation": "response.cancel",
                "response_id": None,
                "error": SimpleNamespace(code="response_cancel_not_active"),
            }
        )
        for _ in range(50):
            if llm.responses_created == 2:
                break
            await asyncio.sleep(0.002)

        assert llm.responses_created == 2
        assert llm.user_texts == []
        assert run_task.done() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_corrected_replacement_preserves_response_recovery_bookkeeping(
    monkeypatch,
) -> None:
    class DelayedChangedCorrector:
        instance: DelayedChangedCorrector | None = None

        def __init__(self, **kwargs) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            DelayedChangedCorrector.instance = self

        async def correct(self, raw: str) -> tuple[str, float]:
            self.started.set()
            await self.release.wait()
            return f"corrected:{raw}", 1.0

        def record_user(self, text: str) -> None:
            return None

        def record_assistant(self, text: str) -> None:
            return None

        async def aclose(self) -> None:
            return None

    llm = QueueLLM()
    speaker = BlockingSpeaker(asyncio.get_running_loop())
    _configure_runtime(monkeypatch, llm, speaker)
    monkeypatch.setattr(orch.settings, "transcript_correction_enabled", True)
    monkeypatch.setattr(orch, "TranscriptCorrector", DelayedChangedCorrector)
    monkeypatch.setattr(orch, "AsyncOpenAI", lambda **kwargs: object())
    run_task = asyncio.create_task(orch.run())

    try:
        await asyncio.wait_for(llm.opened.wait(), timeout=0.2)
        await llm.queue.put(
            {"type": "input.speech_stopped", "item_id": "user-raw"}
        )
        await llm.queue.put(
            {"type": "input.transcript", "text": "raw", "item_id": "user-raw"}
        )
        corrector = DelayedChangedCorrector.instance
        assert corrector is not None
        await asyncio.wait_for(corrector.started.wait(), timeout=0.2)
        await llm.queue.put(
            {
                "type": "response.created",
                "response_id": "resp-raw",
                "generation_id": 1,
            }
        )
        corrector.release.set()
        await asyncio.wait_for(llm.cancelled.wait(), timeout=0.2)
        assert llm.cancelled_response_ids == ["resp-raw"]
        await llm.queue.put(
            {
                "type": "response.done",
                "response_id": "resp-raw",
                "status": "cancelled",
            }
        )
        for _ in range(50):
            if "user-raw" in llm.deleted_items:
                break
            await asyncio.sleep(0.002)
        assert llm.deleted_items == ["user-raw"]
        await llm.queue.put(
            {"type": "conversation.item.deleted", "item_id": "user-raw"}
        )
        for _ in range(50):
            if llm.user_texts == ["corrected:raw"]:
                break
            await asyncio.sleep(0.002)
        assert llm.user_texts == ["corrected:raw"]

        await llm.queue.put(
            {
                "type": "error",
                "operation": "response.create",
                "generation_id": 2,
                "error": SimpleNamespace(
                    code="conversation_already_has_active_response"
                ),
            }
        )
        for _ in range(50):
            if llm.cancelled_response_ids == ["resp-raw", None]:
                break
            await asyncio.sleep(0.002)
        assert llm.cancelled_response_ids == ["resp-raw", None]

        await llm.queue.put(
            {
                "type": "error",
                "operation": "response.cancel",
                "response_id": None,
                "error": SimpleNamespace(code="response_cancel_not_active"),
            }
        )
        for _ in range(50):
            if llm.responses_created == 2:
                break
            await asyncio.sleep(0.002)
        assert llm.responses_created == 2
        assert run_task.done() is False
    finally:
        run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)
