"""Half-duplex conversation pipeline with local Silero VAD.

Phase diagram:
    LISTENING ──(VAD speech_start)──→ ACTIVE ──(VAD speech_end)──→ RESPONDING
        ↑                                                             │
        └──────────(TTS done + speaker empty + safety delay)──────────┘

During LISTENING : VAD processes mic; no audio sent to API.
During ACTIVE    : mic audio streamed to API in real time.
During RESPONDING: mic completely muted; LLM text → TTS → speaker.
"""

from __future__ import annotations

import asyncio
import base64
import sys
from collections import deque
from enum import Enum, auto

import httpx
import numpy as np
from openai import AsyncOpenAI

from zemory_vad.audio import MicrophoneStream, SpeakerStream
from zemory_vad.config import (
    MODEL,
    OPENAI_API_KEY,
    SAFETY_DELAY_S,
    SESSION_CONFIG,
    VAD_PRE_BUFFER_CHUNKS,
)
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
    openai = AsyncOpenAI(api_key=OPENAI_API_KEY)
    loop = asyncio.get_running_loop()

    mic = MicrophoneStream(loop)
    speaker = SpeakerStream(loop)
    sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()

    # Shared mutable state (single-writer per field keeps this safe)
    phase = Phase.LISTENING

    # VAD
    vad_model = SileroVAD()
    vad_sm = VADStateMachine()
    vad_buf = np.array([], dtype=np.int16)  # accumulator for 512-sample chunks
    pre_buffer: deque[bytes] = deque(maxlen=VAD_PRE_BUFFER_CHUNKS)

    async with openai.beta.realtime.connect(model=MODEL) as conn:
        await conn.session.update(session=SESSION_CONFIG)
        print("Zemory-VAD is listening… (Ctrl+C to quit)\n")

        mic.start()
        speaker.start()

        # ------------------------------------------------------------------
        # Task 1: mic capture → local VAD → conditionally send to API
        # ------------------------------------------------------------------
        async def mic_vad_task() -> None:
            nonlocal phase, vad_buf

            while True:
                raw = await mic.queue.get()

                if phase == Phase.RESPONDING:
                    continue  # hard mute

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

                # --- state transitions ---
                if phase == Phase.LISTENING:
                    pre_buffer.append(raw)
                    if speech_started:
                        phase = Phase.ACTIVE
                        print("[vad] speech started")
                        for buf in pre_buffer:
                            await conn.input_audio_buffer.append(
                                audio=base64.b64encode(buf).decode("ascii")
                            )
                        pre_buffer.clear()

                elif phase == Phase.ACTIVE:
                    await conn.input_audio_buffer.append(
                        audio=base64.b64encode(raw).decode("ascii")
                    )
                    if speech_ended:
                        print("[vad] speech ended → committing")
                        phase = Phase.RESPONDING
                        await conn.input_audio_buffer.commit()
                        await conn.response.create()
                        vad_model.reset()
                        vad_sm.reset()
                        vad_buf = np.array([], dtype=np.int16)

        # ------------------------------------------------------------------
        # Task 2: receive Realtime API events → sentence queue
        # ------------------------------------------------------------------
        async def recv_task() -> None:
            chunker = SentenceChunker()

            async for event in conn:
                match event.type:
                    case "session.created":
                        print(f"[session] {event.session.id}")
                    case "session.updated":
                        print("[session] configured.")

                    case "response.text.delta":
                        print(event.delta, end="", flush=True)
                        for s in chunker.add(event.delta):
                            await sentence_queue.put(s)

                    case "response.text.done":
                        print()

                    case "response.done":
                        remaining = chunker.flush()
                        if remaining:
                            await sentence_queue.put(remaining)
                        await sentence_queue.put(None)  # sentinel
                        if event.response.usage:
                            u = event.response.usage
                            print(
                                f"  [tokens: {u.total_tokens}]",
                                file=sys.stderr,
                            )

                    case "conversation.item.input_audio_transcription.completed":
                        print(f"\n[You]: {event.transcript}")

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
                        # All sentences delivered; wait for speaker to finish
                        await speaker.wait_until_done()
                        await asyncio.sleep(SAFETY_DELAY_S)
                        await conn.input_audio_buffer.clear()
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
        # Run all tasks
        # ------------------------------------------------------------------
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(mic_vad_task())
                tg.create_task(recv_task())
                tg.create_task(tts_task())
                tg.create_task(speaker.feed())
        finally:
            mic.stop()
            speaker.stop()
