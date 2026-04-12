from __future__ import annotations

import asyncio
import base64
import sys

import httpx
from openai import AsyncOpenAI

from zemory.audio import MicrophoneStream, SpeakerStream
from zemory.config import MODEL, OPENAI_API_KEY, SESSION_CONFIG
from zemory.tts import SentenceChunker, elevenlabs_tts


async def send_mic_audio(
    conn, mic: MicrophoneStream, speaking: asyncio.Event
) -> None:
    """Send mic audio to Realtime API.

    Half-duplex: mutes while AI is responding. On unmute, clears the
    server-side input buffer so stale audio never triggers a new turn.
    """
    was_speaking = False
    while True:
        pcm_bytes = await mic.queue.get()
        if speaking.is_set():
            was_speaking = True
            continue
        if was_speaking:
            await conn.input_audio_buffer.clear()
            was_speaking = False
        encoded = base64.b64encode(pcm_bytes).decode("ascii")
        await conn.input_audio_buffer.append(audio=encoded)


async def receive_events(
    conn,
    sentence_queue: asyncio.Queue[str | None],
    speaking: asyncio.Event,
) -> None:
    """Receive Realtime API events, chunk text into sentences, queue for TTS."""
    chunker = SentenceChunker()

    async for event in conn:
        match event.type:
            case "session.created":
                print(f"[session] created: {event.session.id}")

            case "session.updated":
                print("[session] configured.")

            case "input_audio_buffer.speech_stopped":
                # User finished talking — mute mic immediately.
                # Prevents echo and stale audio during LLM + TTS phase.
                speaking.set()

            case "response.text.delta":
                print(event.delta, end="", flush=True)
                for sentence in chunker.add(event.delta):
                    await sentence_queue.put(sentence)

            case "response.text.done":
                print()

            case "response.done":
                # Flush remaining text and signal TTS worker
                remaining = chunker.flush()
                if remaining:
                    await sentence_queue.put(remaining)
                await sentence_queue.put(None)  # Sentinel: response complete
                if event.response.usage:
                    u = event.response.usage
                    print(f"  [tokens: {u.total_tokens}]", file=sys.stderr)

            case "conversation.item.input_audio_transcription.completed":
                print(f"\n[You]: {event.transcript}")

            case "error":
                print(f"[error] {event.error}", file=sys.stderr)


async def tts_worker(
    speaker: SpeakerStream,
    sentence_queue: asyncio.Queue[str | None],
    speaking: asyncio.Event,
) -> None:
    """Pick sentences from queue, stream ElevenLabs TTS audio to speaker."""
    async with httpx.AsyncClient(timeout=30.0) as http:
        while True:
            sentence = await sentence_queue.get()

            if sentence is None:
                # Response complete — wait for speaker to finish playing,
                # then unmute mic for the next user turn.
                await speaker.wait_until_done()
                speaking.clear()
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
                print(f"[tts] request error: {e}", file=sys.stderr)


async def run() -> None:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    loop = asyncio.get_running_loop()

    mic = MicrophoneStream(loop)
    speaker = SpeakerStream(loop)
    sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
    speaking = asyncio.Event()

    async with client.beta.realtime.connect(model=MODEL) as conn:
        await conn.session.update(session=SESSION_CONFIG)

        print("Zemory is listening... (Ctrl+C to quit)\n")

        mic.start()
        speaker.start()

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(send_mic_audio(conn, mic, speaking))
                tg.create_task(receive_events(conn, sentence_queue, speaking))
                tg.create_task(tts_worker(speaker, sentence_queue, speaking))
                tg.create_task(speaker.feed())
        finally:
            mic.stop()
            speaker.stop()
