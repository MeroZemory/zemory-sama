import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit(
        "Error: OPENAI_API_KEY is not set.\n"
        "Copy .env.example to .env and add your API key."
    )

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    raise SystemExit("Error: ELEVENLABS_API_KEY is not set.")

MODEL = "gpt-4o-mini-realtime-preview"
SAMPLE_RATE = 24_000
CHUNK_DURATION_MS = 20

# ElevenLabs TTS
# Voice list: https://elevenlabs.io/app/voice-library
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL_ID = "eleven_flash_v2_5"

INSTRUCTIONS = (
    "You are Zemory, a friendly and enthusiastic AI assistant. "
    "You speak naturally and conversationally in the user's language. "
    "Keep responses concise since this is a voice conversation."
)

# Realtime API: text-only output (TTS handled by ElevenLabs)
SESSION_CONFIG = {
    "modalities": ["text"],
    "instructions": INSTRUCTIONS,
    "input_audio_format": "pcm16",
    "input_audio_transcription": {
        "model": "gpt-4o-mini-transcribe",
    },
    "turn_detection": {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500,
        "create_response": True,
        "interrupt_response": False,
    },
    "temperature": 0.8,
}
