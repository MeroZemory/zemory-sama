"""Central orchestrator.

Wires the five tasks of the pipeline:

1. mic pump (PCM → TurnDetector, subject to mute_mic)
2. LLM event consumer (deltas → TTSTaskManager, turn events → Phase/interrupt)
3. TurnDetector event consumer (local profile; speech_end → STT → LLM inject)
4. SpeakerStream.feed
5. (no explicit TTS worker — TTSTaskManager owns its own task set)

End-to-end latency measurement emits a ``turn.complete`` structlog entry
carrying ``speech_end_ts``, ``first_llm_delta_ts``, ``first_tts_byte_ts``,
``speaker_first_play_ts``, ``total_ms``, ``profile``, ``interrupted``.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from openai import AsyncOpenAI

from zemory.audio import MicrophoneStream, SpeakerStream, generate_beep_pcm
from zemory.config import (
    ELEVENLABS_API_KEY,
    OPENAI_API_KEY,
    canonical_profile,
    settings,
)
from zemory.observability import configure_logging, get_logger, metrics
from zemory.pipeline.chunker import SentenceChunker
from zemory.pipeline.context import (
    AsyncContextScheduler,
    SQLiteMemoryStore,
    TranscriptLedger,
)
from zemory.pipeline.interrupt_bus import InterruptBus
from zemory.pipeline.realtime_events import handle_speech_started
from zemory.pipeline.transcript_corrector import TranscriptCorrector
from zemory.pipeline.tts_manager import TTSTaskManager
from zemory.providers.base import build_pipeline
from zemory.state import Phase, StateMachine

_log = get_logger("orchestrator")


def build_context_scheduler() -> AsyncContextScheduler:
    """Build the runtime context scheduler from settings."""
    memory = None
    if settings.memory_enabled:
        memory_path = Path(settings.memory_path).expanduser()
        if not memory_path.is_absolute():
            memory_path = Path.cwd() / memory_path
        memory = SQLiteMemoryStore(memory_path)
    return AsyncContextScheduler(
        memory=memory,
        memory_deadline_ms=settings.memory_recall_deadline_ms,
        tool_deadline_ms=settings.context_tool_deadline_ms,
        memory_limit=settings.memory_recall_limit,
    )


class TurnTimer:
    """Per-turn stopwatch capturing the five latency milestones."""

    __slots__ = (
        "turn_id",
        "speech_end_ts",
        "first_llm_delta_ts",
        "first_tts_byte_ts",
        "speaker_first_play_ts",
        "speaker_buffer_ms",
        "correction_ms",
        "interrupted",
    )

    def __init__(self, turn_id: int) -> None:
        self.turn_id = turn_id
        self.speech_end_ts: float | None = None
        self.first_llm_delta_ts: float | None = None
        self.first_tts_byte_ts: float | None = None
        self.speaker_first_play_ts: float | None = None
        self.speaker_buffer_ms: float | None = None
        self.correction_ms: float | None = None
        self.interrupted = False

    def total_ms(self) -> float | None:
        if self.speech_end_ts is None or self.speaker_first_play_ts is None:
            return None
        return (self.speaker_first_play_ts - self.speech_end_ts) * 1000


async def run() -> None:
    configure_logging()
    loop = asyncio.get_running_loop()

    pipeline = build_pipeline(
        settings.profile,
        openai_api_key=OPENAI_API_KEY,
        elevenlabs_api_key=ELEVENLABS_API_KEY,
    )
    profile = canonical_profile(settings.profile)
    manual_realtime_turns = (
        profile in {"realtime_audio", "realtime_text_external_tts"}
        and settings.realtime.turn_detection == "none"
    )

    mic = MicrophoneStream(loop)
    speaker = SpeakerStream(loop)
    state = StateMachine()

    turn_seq = 0
    timer = TurnTimer(turn_seq)
    item_ids: list[str] = []
    ledger = TranscriptLedger(max_turns=settings.max_context_turns * 2)
    context_scheduler = build_context_scheduler()

    async def on_partial_abort(partial: str) -> None:
        """Record interrupted assistant text as history for the next turn."""
        ledger.record_assistant(partial, interrupted=True)
        note = f"(You were interrupted while saying: \"{partial[:200]}\")"
        ledger.record_system(note)
        record = getattr(pipeline.llm, "record_system_note", None)
        if callable(record):
            await record(note)
        _log.info("interrupt.partial_recorded", chars=len(partial))

    interrupt_bus = InterruptBus(state, speaker, on_partial=on_partial_abort)

    # Optional transcript corrector (context-aware ASR fix-up).
    corrector: TranscriptCorrector | None = None
    if settings.transcript_correction_enabled:
        corrector = TranscriptCorrector(
            client=AsyncOpenAI(api_key=OPENAI_API_KEY),
            model=settings.transcript_correction_model,
            history_turns=settings.transcript_correction_history_turns,
        )
        _log.info(
            "correction.enabled",
            model=settings.transcript_correction_model,
            history_turns=settings.transcript_correction_history_turns,
        )

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

    # Pre-compute the "ready to speak" beep PCM once at startup.
    ready_beep_pcm = (
        generate_beep_pcm(
            frequency_hz=settings.ready_beep_frequency_hz,
            duration_ms=settings.ready_beep_duration_ms,
            sample_rate=settings.sample_rate,
            volume=settings.ready_beep_volume,
        )
        if settings.enable_ready_beep
        else b""
    )

    async def _play_ready_beep() -> None:
        """Play the mic-ready beep while the mic is still muted.

        Call this BEFORE transitioning to ``Phase.LISTENING`` so the beep
        can never leak back into the transcript via speaker→mic echo.
        Waits for the speaker buffer to drain + a short post-gap before
        returning so the mic doesn't open on the beep's tail.
        """
        if not ready_beep_pcm:
            return
        await speaker.queue.put(ready_beep_pcm)
        await speaker.wait_until_done()
        if settings.ready_beep_post_gap_s > 0:
            await asyncio.sleep(settings.ready_beep_post_gap_s)

    # --- Per-response generation state (shared between llm_event_consumer and
    # speculative correction task so the latter can abort+replace cleanly). ---
    gen_chunker: list[SentenceChunker] = [SentenceChunker()]
    gen_assistant_text: list[str] = [""]
    # True from ``speech_stopped`` until ``response.done``. Distinct from
    # ``Phase.RESPONDING`` which stays True through the post-response
    # TTS/speaker drain + safety_delay window.
    gen_response_active: list[bool] = [False]

    def _reset_generation_state() -> None:
        gen_chunker[0] = SentenceChunker()
        gen_assistant_text[0] = ""

    async def _speculative_correction(raw: str, raw_item_id: str | None) -> None:
        """Run correction in parallel with the speculative (raw) response.

        If correction returns the same text → no-op; the already-streaming
        response runs to completion with zero added latency.

        If correction differs AND the response is still being generated
        (``Phase == RESPONDING``), we abort the speculative response, delete
        the server-transcribed user item, inject the corrected text, and
        trigger a new response. The user will hear the start of the raw
        response cut off and the corrected response begin (tradeoff vs. pure
        sequential correction's fixed latency cost).

        If the speculative response already finished before correction
        completes, we just record the user text in history and skip.
        """
        assert corrector is not None
        corrected, correction_ms = await corrector.correct(raw)
        timer.correction_ms = correction_ms
        corrector.record_user(corrected)

        if corrected == raw:
            return

        if not gen_response_active[0]:
            # The speculative response already completed before correction
            # arrived — don't kick off a duplicate response. The user's
            # corrected text is still recorded in history for the next turn.
            _log.info(
                "correction.too_late_to_replace",
                raw=raw,
                corrected=corrected,
            )
            return

        # Mark the upcoming replacement response as active before we kick it
        # off so a second speculative task (unlikely but possible) sees the
        # right state.
        gen_response_active[0] = True

        _log.info(
            "correction.speculative_replace",
            raw=raw,
            corrected=corrected,
            ms=round(correction_ms, 1),
        )
        print(f"\n[corrected]: {corrected}")

        # Abort the speculative stream (same shape as barge-in abort chain,
        # but we stay in RESPONDING because the user isn't talking).
        speaker.clear()
        await tts_manager.abort()
        try:
            await pipeline.llm.cancel_current()
        except Exception as e:
            _log.warning("correction.cancel_failed", error=str(e))

        if raw_item_id:
            try:
                await pipeline.llm.delete_item(raw_item_id)
            except Exception as e:
                _log.warning("correction.delete_failed", error=str(e))

        # Reset per-response generation state so the replacement stream is
        # clean and the latency milestones are re-captured.
        _reset_generation_state()
        tts_manager.reset_for_new_turn()
        speaker.arm()
        interrupt_bus.reset_partial()
        timer.first_llm_delta_ts = None
        timer.first_tts_byte_ts = None

        await pipeline.llm.send_user_text(corrected, injections=[])

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
                if profile in {"realtime_audio", "realtime_text_external_tts"} and settings.enable_barge_in:
                    await pipeline.turn.feed(pcm)
                # else: drop frame
            else:
                await pipeline.turn.feed(pcm)

    async def turn_event_consumer() -> None:
        """Handle local or manual-Realtime turn detector events."""
        nonlocal turn_seq
        if profile != "local_cascade" and not manual_realtime_turns:
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

                if manual_realtime_turns:
                    turn_seq += 1
                    timer.__init__(turn_seq)
                    timer.speech_end_ts = state.speech_end_ts
                    tts_manager.reset_for_new_turn()
                    interrupt_bus.reset_partial()
                    _reset_generation_state()
                    gen_response_active[0] = True
                    await pipeline.llm.commit_input_audio_buffer()
                    await pipeline.llm.trigger_response()
                    reset = getattr(pipeline.turn, "reset", None)
                    if callable(reset):
                        reset()
                    continue

                chunks = pipeline.turn.consume_audio()
                text = await pipeline.stt.transcribe(chunks)
                if not text or text == "[transcription failed]":
                    _log.warning("stt.empty_or_failed")
                    await _play_ready_beep()
                    await state.transition(Phase.LISTENING)
                    continue
                raw_text = text
                turn_seq += 1
                timer.__init__(turn_seq)
                timer.speech_end_ts = state.speech_end_ts

                if corrector is not None:
                    corrected, correction_ms = await corrector.correct(raw_text)
                    timer.correction_ms = correction_ms
                    text = corrected
                    if corrected != raw_text:
                        print(f"\n[You (raw)]:       {raw_text}")
                        print(f"[You (corrected)]: {corrected}")
                    else:
                        print(f"\n[You]: {text}")
                    corrector.record_user(text)
                else:
                    print(f"\n[You]: {text}")

                _log.info("user.text", text=text,
                          raw=raw_text if raw_text != text else None)
                tts_manager.reset_for_new_turn()
                ledger.record_user(text)
                context = await context_scheduler.gather_for_turn(text)
                await pipeline.llm.send_user_text(text, injections=context.injections)
                # Reset local VAD state for the next turn
                from zemory.providers.turn.silero import SileroTurnDetector
                if isinstance(pipeline.turn, SileroTurnDetector):
                    pipeline.turn.reset()

    async def llm_event_consumer() -> None:
        async for event in pipeline.llm.events():
            t = event.get("type")

            if t == "session.created":
                _log.info("session.created", id=event.get("session_id"))
            elif t == "session.updated":
                _log.info("session.configured")

            elif t == "input.speech_started":
                await handle_speech_started(state, interrupt_bus)

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
                _reset_generation_state()
                gen_response_active[0] = True

            elif t == "input.transcript":
                raw_text = event.get("text", "")
                raw_item_id = event.get("item_id")

                # Server already kicked off the response on speech_stopped
                # (create_response=true). No manual trigger needed.
                print(f"\n[You]: {raw_text}")

                if corrector is not None and raw_text.strip():
                    # Speculative: run correction in parallel with the
                    # already-streaming response. Abort+replace only if
                    # the correction actually differs from the raw text.
                    asyncio.create_task(
                        _speculative_correction(raw_text, raw_item_id)
                    )

                _log.info("user.text", text=raw_text)
                if raw_text:
                    ledger.record_user(raw_text)

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
                gen_assistant_text[0] += delta
                if profile != "realtime_audio":
                    for sentence in gen_chunker[0].add(delta):
                        tts_manager.submit(sentence)

            elif t == "text.done":
                print()
                if profile != "realtime_audio":
                    tail = gen_chunker[0].flush()
                    if tail:
                        tts_manager.submit(tail)
                # Feed the full assistant response into the corrector's
                # rolling history so next-turn correction has context.
                if corrector is not None and gen_assistant_text[0]:
                    corrector.record_assistant(gen_assistant_text[0])
                if gen_assistant_text[0]:
                    ledger.record_assistant(gen_assistant_text[0])

            elif t == "audio.delta":
                if timer.first_tts_byte_ts is None:
                    timer.first_tts_byte_ts = time.monotonic()
                audio = event.get("audio", b"")
                if audio:
                    await speaker.queue.put(audio)

            elif t == "audio.transcript.delta":
                delta = event.get("delta", "")
                print(delta, end="", flush=True)
                interrupt_bus.record_partial(delta)
                gen_assistant_text[0] += delta

            elif t == "audio.transcript.done":
                print()
                if corrector is not None and gen_assistant_text[0]:
                    corrector.record_assistant(gen_assistant_text[0])
                if gen_assistant_text[0]:
                    ledger.record_assistant(gen_assistant_text[0])

            elif t == "response.done":
                # If the response was cancelled (speculative abort or
                # barge-in), a *replacement* response may already be on its
                # way. Skip the finalize+transition so the next response
                # isn't prematurely cut off, and leave gen_response_active
                # unchanged (the replace logic owns it).
                if event.get("status") == "cancelled":
                    _log.info("response.cancelled_skipped_finalize",
                              turn_id=timer.turn_id)
                    continue
                gen_response_active[0] = False
                await _finalize_response(timer)
                await _trim_context()

            elif t == "error":
                err = event.get("error")
                # Expected race: speculative abort sometimes arrives after
                # the server has already finished the response. Log at info.
                code = getattr(err, "code", None) if err is not None else None
                if code == "response_cancel_not_active":
                    _log.info("llm.cancel_race_ignored")
                else:
                    _log.error("llm.error", error=err)

    async def _finalize_response(t: TurnTimer) -> None:
        # Wait for all TTS chunks to reach speaker, then for speaker to drain.
        await tts_manager.wait_until_empty()
        await speaker.wait_until_done()

        # Capture speaker timing BEFORE beep plays so the next turn's
        # ttfb measurement isn't overwritten by the beep's buffer write.
        if speaker.first_play_at is not None:
            t.speaker_first_play_ts = speaker.first_play_at
        elif speaker.first_write_at is not None:
            t.speaker_first_play_ts = speaker.first_write_at
        if speaker.first_write_at is not None and speaker.first_play_at is not None:
            t.speaker_buffer_ms = (
                speaker.first_play_at - speaker.first_write_at
            ) * 1000

        # Immediately play the "mic live" beep. Its built-in post-gap
        # (ready_beep_post_gap_s) doubles as the echo-decay window that
        # ``safety_delay_s`` used to provide. Fall back to safety_delay
        # only when the beep is disabled.
        if settings.enable_ready_beep:
            await _play_ready_beep()
        elif settings.safety_delay_s > 0:
            await asyncio.sleep(settings.safety_delay_s)

        # Emit the turn.complete summary — but only if this response
        # actually produced audio. A speculative response that was
        # replaced by correction will call _finalize_response after being
        # aborted; skipping the log avoids a noisy ``total_ms=None`` entry.
        total = t.total_ms()
        if total is not None:
            metrics.observe("turn.total_ms", total)
            _log.info(
                "turn.complete",
                turn_id=t.turn_id,
                total_ms=round(total, 1),
                first_llm_delta_ms=(
                    round((t.first_llm_delta_ts - t.speech_end_ts) * 1000, 1)
                    if t.first_llm_delta_ts and t.speech_end_ts else None
                ),
                first_tts_byte_ms=(
                    round((t.first_tts_byte_ts - t.speech_end_ts) * 1000, 1)
                    if t.first_tts_byte_ts and t.speech_end_ts else None
                ),
                speaker_buffer_ms=(
                    round(t.speaker_buffer_ms, 1)
                    if t.speaker_buffer_ms is not None else None
                ),
                correction_ms=(
                    round(t.correction_ms, 1) if t.correction_ms is not None else None
                ),
                interrupted=t.interrupted,
                profile=settings.profile,
            )
        else:
            _log.debug(
                "turn.aborted_before_audio",
                turn_id=t.turn_id,
                correction_ms=(
                    round(t.correction_ms, 1) if t.correction_ms is not None else None
                ),
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
            # Fire-and-forget startup beep so the user knows the mic is live.
            # Runs inside the TaskGroup so speaker.feed is already pumping.
            tg.create_task(_play_ready_beep(), name="startup_beep")
    finally:
        await tts_manager.stop()
        await pipeline.llm.close()
        mic.stop()
        speaker.stop()
