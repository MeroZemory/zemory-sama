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

# STT model (for separate transcription before sending text to LLM)
STT_MODEL = "gpt-4o-transcribe"

# Response length: "1-2 sentences", "3-4 sentences", "short paragraph", etc.
RESPONSE_LENGTH = os.environ.get("ZEMORY_RESPONSE_LENGTH", "1-2 sentences")

# Max conversation turns to keep in context (older turns are deleted)
MAX_CONTEXT_TURNS = int(os.environ.get("ZEMORY_MAX_CONTEXT_TURNS", "10"))

INSTRUCTIONS = (
    "You are Zemory, a friendly and enthusiastic AI assistant. "
    "You speak naturally and conversationally. "
    "Unless the user explicitly requests a different language, "
    "you MUST respond in the same language the user spoke in. "
    f"STRICT RULE: Your response MUST be {RESPONSE_LENGTH} maximum. "
    "Never exceed this limit. This is a real-time voice conversation — "
    "long responses feel unnatural. Be direct and concise. "
    "If you do not know the answer to a factual question, or if the user asks about "
    "current events, news, or anything you are uncertain about, say '찾아볼게요' "
    "(or 'Let me look that up' if speaking English) and give a brief placeholder response. "
    "The system will search the web and provide you with the answer shortly."
)

# --- Search configuration ---
SEARCH_MAX_RESULTS = 5
SEARCH_MODEL = "gpt-5-mini"

# Realtime API session — text-only input (STT handled locally)
SESSION_CONFIG = {
    "modalities": ["text"],
    "instructions": INSTRUCTIONS,
    "turn_detection": None,
    "temperature": 0.8,
}
