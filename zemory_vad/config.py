import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("Error: OPENAI_API_KEY is not set.")

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    raise SystemExit("Error: ELEVENLABS_API_KEY is not set.")

MODEL = "gpt-4o-mini-realtime-preview"
SAMPLE_RATE = 24_000
VAD_SAMPLE_RATE = 16_000
CHUNK_DURATION_MS = 20  # 480 samples at 24kHz

# ElevenLabs TTS
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL_ID = "eleven_flash_v2_5"

# --- VAD parameters (tuned for this setup) ---
VAD_PROB_THRESHOLD = 0.15
VAD_DB_THRESHOLD = 45
VAD_REQUIRED_HITS = 3       # 3 × 32ms = 96ms  → speech start
VAD_REQUIRED_MISSES = 24    # 24 × 32ms = 768ms → speech end
VAD_SMOOTHING_WINDOW = 5
VAD_PRE_BUFFER_CHUNKS = 32  # 32 × 20ms = 640ms of pre-speech audio

# Safety delay after speaker finishes before unmuting mic
SAFETY_DELAY_S = 0.5

INSTRUCTIONS = (
    "You are Zemory, a friendly and enthusiastic AI assistant. "
    "You speak naturally and conversationally. "
    "Unless the user explicitly requests a different language, "
    "you MUST respond in the same language the user spoke in. "
    "Keep responses concise since this is a voice conversation."
)

# Realtime API session — turn_detection disabled (local VAD)
SESSION_CONFIG = {
    "modalities": ["text"],
    "instructions": INSTRUCTIONS,
    "input_audio_format": "pcm16",
    "input_audio_transcription": {
        "model": "gpt-4o-mini-transcribe",
    },
    "turn_detection": None,
    "temperature": 0.8,
}
