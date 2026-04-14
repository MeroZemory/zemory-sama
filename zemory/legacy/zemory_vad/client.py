"""Half-duplex conversation pipeline with local Silero VAD, Whisper STT,
and async web search.

Pipeline:
    mic → VAD → local audio buffer → Whisper STT → text → Realtime API → TTS

Phase diagram:
    LISTENING ──(VAD speech_start)──→ ACTIVE ──(VAD speech_end)──→ RESPONDING
        ↑                                                             │
        └──────────(TTS done + speaker empty + safety delay)──────────┘
"""

from __future__ import annotations

import asyncio
import sys
from collections import deque
from enum import Enum, auto

import httpx
import numpy as np
from openai import AsyncOpenAI
from zemory_vad.audio import MicrophoneStream, SpeakerStream
from zemory_vad.config import (
    MAX_CONTEXT_TURNS,
    MODEL,
    OPENAI_API_KEY,
    SAFETY_DELAY_S,
    SESSION_CONFIG,
    VAD_PRE_BUFFER_CHUNKS,
)
from zemory_vad.search import detect_uncertainty, search_pipeline
from zemory_vad.stt import WhisperSTT
from zemory_vad.tts import SentenceChunker, elevenlabs_tts
from zemory_vad.vad import (
    CHUNK_SAMPLES,
    SileroVAD,
    VADStateMachine,
    calc_db,
    resample_24k_to_16k,
)


class Phase(Enum):
    LISTENING = auto()
    ACTIVE = auto()
    RESPONDING = auto()


async def run() -> None:
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    loop = asyncio.get_running_loop()

    mic = MicrophoneStream(loop)
    speaker = SpeakerStream(loop)
    stt = WhisperSTT(openai_client)
    sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()

    # Shared mutable state
    phase = Phase.LISTENING

    # VAD
    vad_model = SileroVAD()
    vad_sm = VADStateMachine()
    vad_buf = np.array([], dtype=np.int16)
    pre_buffer: deque[bytes] = deque(maxlen=VAD_PRE_BUFFER_CHUNKS)

    # Audio buffer for STT (accumulated during ACTIVE phase)
    audio_chunks: list[bytes] = []

    # Context management — track item IDs for sliding window
    item_ids: list[str] = []

    # Search state
    last_user_transcript: str = ""
    full_response_text: str = ""
    pending_search_result: str | None = None
    search_in_progress: bool = False
    search_result_ready = asyncio.Event()
    skip_next_uncertainty_check: bool = False

    async with openai_client.beta.realtime.connect(model=MODEL) as conn:
        await conn.session.update(session=SESSION_CONFIG)
        print("Zemory-VAD is listening… (Ctrl+C to quit)\n")

        mic.start()
        speaker.start()

        # ------------------------------------------------------------------
        # Background search helpers
        # ------------------------------------------------------------------
        async def _check_and_search(transcript: str, response_text: str) -> None:
            nonlocal pending_search_result, search_in_progress
            try:
                uncertain = await detect_uncertainty(openai_client, response_text)
                if not uncertain:
                    return
                print("[search] LLM detected uncertainty, searching…", file=sys.stderr)
                result = await asyncio.wait_for(
                    search_pipeline(openai_client, transcript, response_text),
                    timeout=15.0,
                )
                pending_search_result = result
                search_result_ready.set()
                print("[search] ready", file=sys.stderr)
            except TimeoutError:
                print("[search] timed out (15s)", file=sys.stderr)
            except Exception as e:
                print(f"[search] failed: {e}", file=sys.stderr)
            finally:
                search_in_progress = False

        # ------------------------------------------------------------------
        # Task 1: mic → VAD → buffer audio → Whisper STT → text inject
        # ------------------------------------------------------------------
        async def mic_vad_task() -> None:
            nonlocal phase, vad_buf, last_user_transcript

            while True:
                raw = await mic.queue.get()

                if phase == Phase.RESPONDING:
                    continue

                pcm_24k = np.frombuffer(raw, dtype=np.int16)
                pcm_16k = resample_24k_to_16k(pcm_24k)
                vad_buf = np.concatenate([vad_buf, pcm_16k])

                speech_started = False
                speech_ended = False

                while len(vad_buf) >= CHUNK_SAMPLES:
                    chunk = vad_buf[:CHUNK_SAMPLES]
                    vad_buf = vad_buf[CHUNK_SAMPLES:]

                    prob = vad_model(chunk)
                    db = calc_db(chunk)
                    sig = vad_sm.process(prob, db)

                    if sig == "speech_start":
                        speech_started = True
                    elif sig == "speech_end":
                        speech_ended = True
                        break

                if phase == Phase.LISTENING:
                    pre_buffer.append(raw)
                    if speech_started:
                        phase = Phase.ACTIVE
                        print("[vad] speech started")
                        # Include pre-buffer in audio for STT
                        audio_chunks.clear()
                        audio_chunks.extend(pre_buffer)
                        pre_buffer.clear()

                elif phase == Phase.ACTIVE:
                    audio_chunks.append(raw)
                    if speech_ended:
                        print("[vad] speech ended → transcribing…")
                        phase = Phase.RESPONDING

                        # Whisper STT
                        text = await stt.transcribe(audio_chunks)
                        audio_chunks.clear()
                        print(f"\n[You]: {text}")
                        last_user_transcript = text

                        if not text:
                            # Empty transcription — go back to listening
                            phase = Phase.LISTENING
                        else:
                            # Inject text into conversation
                            await conn.conversation.item.create(
                                item={
                                    "type": "message",
                                    "role": "user",
                                    "content": [
                                        {"type": "input_text", "text": text}
                                    ],
                                }
                            )
                            await conn.response.create()

                        vad_model.reset()
                        vad_sm.reset()
                        vad_buf = np.array([], dtype=np.int16)

        # ------------------------------------------------------------------
        # Task 2: receive Realtime API events → sentence queue
        # ------------------------------------------------------------------
        async def _trim_context() -> None:
            """Delete oldest items to keep context within MAX_CONTEXT_TURNS."""
            max_items = MAX_CONTEXT_TURNS * 2  # ~2 items per turn (user+assistant)
            while len(item_ids) > max_items:
                old_id = item_ids.pop(0)
                try:
                    await conn.conversation.item.delete(item_id=old_id)
                except Exception:
                    pass

        async def recv_task() -> None:
            nonlocal full_response_text, search_in_progress
            chunker = SentenceChunker()

            async for event in conn:
                match event.type:
                    case "session.created":
                        print(f"[session] {event.session.id}")
                    case "session.updated":
                        print("[session] configured.")

                    case "conversation.item.created":
                        if event.item and event.item.id:
                            item_ids.append(event.item.id)

                    case "response.text.delta":
                        print(event.delta, end="", flush=True)
                        full_response_text += event.delta
                        for s in chunker.add(event.delta):
                            await sentence_queue.put(s)

                    case "response.text.done":
                        print()

                    case "response.done":
                        nonlocal skip_next_uncertainty_check
                        remaining = chunker.flush()
                        if remaining:
                            await sentence_queue.put(remaining)
                        await sentence_queue.put(None)

                        if skip_next_uncertainty_check:
                            skip_next_uncertainty_check = False
                        elif not search_in_progress and last_user_transcript:
                            search_in_progress = True
                            asyncio.create_task(
                                _check_and_search(
                                    last_user_transcript, full_response_text
                                )
                            )

                        full_response_text = ""

                        # Trim old context
                        await _trim_context()

                        if event.response.usage:
                            u = event.response.usage
                            print(
                                f"  [tokens: {u.total_tokens}]",
                                file=sys.stderr,
                            )

                    case "error":
                        print(f"[error] {event.error}", file=sys.stderr)

        # ------------------------------------------------------------------
        # Task 3: TTS worker — sentence queue → ElevenLabs → speaker
        # ------------------------------------------------------------------
        async def tts_task() -> None:
            nonlocal phase

            async with httpx.AsyncClient(timeout=30.0) as http:
                while True:
                    sentence = await sentence_queue.get()

                    if sentence is None:
                        await speaker.wait_until_done()
                        await asyncio.sleep(SAFETY_DELAY_S)
                        pre_buffer.clear()
                        phase = Phase.LISTENING
                        print("[state] listening")
                        continue

                    try:
                        async for chunk in elevenlabs_tts(http, sentence):
                            await speaker.queue.put(chunk)
                    except httpx.HTTPStatusError as e:
                        print(
                            f"[tts] HTTP {e.response.status_code}: {e.response.text}",
                            file=sys.stderr,
                        )
                    except httpx.RequestError as e:
                        print(f"[tts] {e}", file=sys.stderr)

        # ------------------------------------------------------------------
        # Task 4: inject search results when LISTENING
        # ------------------------------------------------------------------
        async def search_inject_task() -> None:
            nonlocal phase, pending_search_result, skip_next_uncertainty_check

            while True:
                await search_result_ready.wait()
                search_result_ready.clear()

                while phase != Phase.LISTENING:
                    await asyncio.sleep(0.1)

                result = pending_search_result
                if result is None:
                    continue
                pending_search_result = None

                print("[search] injecting results…", file=sys.stderr)
                try:
                    skip_next_uncertainty_check = True
                    phase = Phase.RESPONDING
                    await conn.conversation.item.create(
                        item={
                            "type": "message",
                            "role": "system",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": (
                                        "[Web search results] "
                                        "Share this with the user naturally and "
                                        f"concisely in the user's language: {result}"
                                    ),
                                }
                            ],
                        }
                    )
                    await conn.response.create()
                except Exception as e:
                    print(f"[search] injection failed: {e}", file=sys.stderr)
                    phase = Phase.LISTENING

        # ------------------------------------------------------------------
        # Run all tasks
        # ------------------------------------------------------------------
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(mic_vad_task())
                tg.create_task(recv_task())
                tg.create_task(tts_task())
                tg.create_task(speaker.feed())
                tg.create_task(search_inject_task())
        finally:
            mic.stop()
            speaker.stop()
