"""Central orchestrator.

Wires the five tasks of the pipeline:

1. mic pump (PCM → TurnDetector, subject to mute_mic)
2. LLM event consumer (deltas → TTSTaskManager, turn events → Phase/interrupt)
3. TurnDetector event consumer (local profile; speech_end → STT → LLM inject)
4. SpeakerStream.feed
5. (no explicit TTS worker — TTSTaskManager owns its own task set)

End-to-end latency measurement emits a ``turn.complete`` structlog entry
carrying ``speech_end_ts``, ``first_llm_delta_ts``, ``first_tts_byte_ts``,
``speaker_first_write_ts``, ``total_ms``, ``profile``, ``interrupted``.
"""

from __future__ import annotations

import asyncio
import time

from zemory.audio import MicrophoneStream, SpeakerStream
from zemory.config import ELEVENLABS_API_KEY, OPENAI_API_KEY, settings
from zemory.observability import configure_logging, get_logger, metrics
from zemory.pipeline.chunker import SentenceChunker
from zemory.pipeline.interrupt_bus import InterruptBus
from zemory.pipeline.tts_manager import TTSTaskManager
from zemory.providers.base import build_pipeline
from zemory.state import Phase, StateMachine

_log = get_logger("orchestrator")


class TurnTimer:
    """Per-turn stopwatch capturing the five latency milestones."""

    __slots__ = (
        "turn_id",
        "speech_end_ts",
        "first_llm_delta_ts",
        "first_tts_byte_ts",
        "speaker_first_write_ts",
        "interrupted",
    )

    def __init__(self, turn_id: int) -> None:
        self.turn_id = turn_id
        self.speech_end_ts: float | None = None
        self.first_llm_delta_ts: float | None = None
        self.first_tts_byte_ts: float | None = None
        self.speaker_first_write_ts: float | None = None
        self.interrupted = False

    def total_ms(self) -> float | None:
        if self.speech_end_ts is None or self.speaker_first_write_ts is None:
            return None
        return (self.speaker_first_write_ts - self.speech_end_ts) * 1000


async def run() -> None:
    configure_logging()
    loop = asyncio.get_running_loop()

    pipeline = build_pipeline(
        settings.profile,
        openai_api_key=OPENAI_API_KEY,
        elevenlabs_api_key=ELEVENLABS_API_KEY,
    )

    mic = MicrophoneStream(loop)
    speaker = SpeakerStream(loop)
    state = StateMachine()

    turn_seq = 0
    timer = TurnTimer(turn_seq)
    item_ids: list[str] = []

    async def on_partial_abort(partial: str) -> None:
        """Record interrupted assistant text as history for the next turn."""
        # We can't inject "assistant" into Realtime after a cancel without risking
        # double-responses; instead emit a system-note so the model sees context.
        note = f"(You were interrupted while saying: \"{partial[:200]}\")"
        await pipeline.llm.send_user_text(note, injections=[])
        _log.info("interrupt.partial_recorded", chars=len(partial))

    interrupt_bus = InterruptBus(state, speaker, on_partial=None)

    await pipeline.llm.open_session()

    def _on_first_tts(seq: int, ttfb_ms: float) -> None:
        # Record the first TTS chunk OF THE CURRENT TURN.
        # ``reset_for_new_turn`` wipes counters, so seq 0 here means the
        # first sentence of this turn (not the session).
        if seq == 0 and timer.first_tts_byte_ts is None:
            timer.first_tts_byte_ts = time.monotonic()

    tts_manager = TTSTaskManager(
        tts=pipeline.tts,
        speaker=speaker,
        max_concurrent=settings.tts.max_concurrent,
        on_first_chunk=_on_first_tts,
    )
    interrupt_bus.bind(tts_manager, pipeline.llm)
    tts_manager.start()

    # Pre-warm TTS provider connection to avoid 2-3 s cold start on first reply.
    warmup = getattr(pipeline.tts, "warmup", None)
    if callable(warmup):
        asyncio.create_task(warmup())

    # ------------------------------------------------------------------
    # Tasks
    # ------------------------------------------------------------------
    async def mic_pump() -> None:
        while True:
            pcm = await mic.queue.get()
            if state.phase == Phase.RESPONDING:
                # While AI is responding:
                #   - barge-in ON:  keep pushing mic so server VAD can detect
                #     interruption (but echo will cause false triggers on
                #     devices without hardware AEC)
                #   - barge-in OFF: suppress mic entirely → speaker audio
                #     leaking into mic cannot be mis-transcribed as user input
                if settings.profile == "realtime" and settings.enable_barge_in:
                    await pipeline.turn.feed(pcm)
                # else: drop frame
            else:
                await pipeline.turn.feed(pcm)

    async def turn_event_consumer() -> None:
        """Local profile only: speech_end → STT → LLM text inject."""
        if settings.profile != "local":
            return
        while True:
            event = await pipeline.turn.events.get()
            if event == "speech_start":
                await state.transition(Phase.ACTIVE)
            elif event == "speech_end":
                state.mark_speech_end()
                timer.speech_end_ts = time.monotonic()
                await state.transition(Phase.RESPONDING)
                speaker.arm()
                chunks = pipeline.turn.consume_audio()
                text = await pipeline.stt.transcribe(chunks)
                if not text or text == "[transcription failed]":
                    _log.warning("stt.empty_or_failed")
                    await state.transition(Phase.LISTENING)
                    continue
                _log.info("user.text", text=text)
                print(f"\n[You]: {text}")
                nonlocal turn_seq
                turn_seq += 1
                timer.__init__(turn_seq)
                timer.speech_end_ts = state.speech_end_ts
                tts_manager.reset_for_new_turn()
                await pipeline.llm.send_user_text(text, injections=[])
                # Reset local VAD state for the next turn
                from zemory.providers.turn.silero import SileroTurnDetector
                if isinstance(pipeline.turn, SileroTurnDetector):
                    pipeline.turn.reset()

    async def llm_event_consumer() -> None:
        chunker = SentenceChunker()
        async for event in pipeline.llm.events():
            t = event.get("type")

            if t == "session.created":
                _log.info("session.created", id=event.get("session_id"))
            elif t == "session.updated":
                _log.info("session.configured")

            elif t == "input.speech_started":
                # Realtime profile barge-in opportunity
                if state.phase == Phase.RESPONDING:
                    interrupt_bus.reset_partial()
                    await interrupt_bus.trigger("realtime_speech_started")
                else:
                    await state.transition(Phase.ACTIVE)

            elif t == "input.speech_stopped":
                # Realtime: user finished — move to RESPONDING and time turn
                state.mark_speech_end()
                nonlocal turn_seq
                turn_seq += 1
                timer.__init__(turn_seq)
                timer.speech_end_ts = state.speech_end_ts
                await state.transition(Phase.RESPONDING)
                speaker.arm()
                # Clear abort flag + reset seq counter so this turn's TTS
                # submissions are accepted (prior interrupt would have stuck).
                tts_manager.reset_for_new_turn()
                interrupt_bus.reset_partial()
                chunker = SentenceChunker()

            elif t == "input.transcript":
                _log.info("user.text", text=event.get("text"))
                print(f"\n[You]: {event.get('text')}")

            elif t == "conversation.item.created":
                item_id = event.get("item_id")
                if item_id:
                    item_ids.append(item_id)

            elif t == "text.delta":
                if timer.first_llm_delta_ts is None:
                    timer.first_llm_delta_ts = time.monotonic()
                delta = event.get("delta", "")
                print(delta, end="", flush=True)
                interrupt_bus.record_partial(delta)
                for sentence in chunker.add(delta):
                    tts_manager.submit(sentence)

            elif t == "text.done":
                print()
                tail = chunker.flush()
                if tail:
                    tts_manager.submit(tail)

            elif t == "response.done":
                await _finalize_response(timer)
                await _trim_context()

            elif t == "error":
                _log.error("llm.error", error=event.get("error"))

    async def _finalize_response(t: TurnTimer) -> None:
        # Wait for all TTS chunks to reach speaker, then for speaker to drain.
        await tts_manager.wait_until_empty()
        await speaker.wait_until_done()
        await asyncio.sleep(settings.safety_delay_s)

        # Capture speaker timing.
        if speaker.first_write_at is not None:
            t.speaker_first_write_ts = speaker.first_write_at

        # Emit the turn.complete summary.
        total = t.total_ms()
        metrics.observe("turn.total_ms", total or 0.0)
        _log.info(
            "turn.complete",
            turn_id=t.turn_id,
            total_ms=round(total, 1) if total else None,
            first_llm_delta_ms=(
                round((t.first_llm_delta_ts - t.speech_end_ts) * 1000, 1)
                if t.first_llm_delta_ts and t.speech_end_ts else None
            ),
            first_tts_byte_ms=(
                round((t.first_tts_byte_ts - t.speech_end_ts) * 1000, 1)
                if t.first_tts_byte_ts and t.speech_end_ts else None
            ),
            interrupted=t.interrupted,
            profile=settings.profile,
        )

        await state.transition(Phase.LISTENING)

    async def _trim_context() -> None:
        max_items = settings.max_context_turns * 2
        from zemory.providers.llm.openai_realtime import OpenAIRealtimeLLM
        if not isinstance(pipeline.llm, OpenAIRealtimeLLM):
            return
        while len(item_ids) > max_items:
            old = item_ids.pop(0)
            await pipeline.llm.delete_item(old)

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------
    mic.start()
    speaker.start()
    _log.info("orchestrator.started", profile=settings.profile)
    print(f"Zemory is listening… profile={settings.profile} (Ctrl+C to quit)\n")

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(mic_pump(), name="mic_pump")
            tg.create_task(turn_event_consumer(), name="turn_events")
            tg.create_task(llm_event_consumer(), name="llm_events")
            tg.create_task(speaker.feed(), name="speaker_feed")
    finally:
        await tts_manager.stop()
        await pipeline.llm.close()
        mic.stop()
        speaker.stop()
