"""Backwards-compatible shim.

``SentenceChunker`` moved to :mod:`zemory.pipeline.chunker`.
``elevenlabs_tts`` moved to :mod:`zemory.providers.tts.elevenlabs`.
Imports here continue to work for any external callers or tests.
"""

from __future__ import annotations

import warnings
from collections.abc import AsyncIterator

import httpx

from zemory.config import ELEVENLABS_API_KEY, settings
from zemory.pipeline.chunker import SentenceChunker
from zemory.providers.tts.elevenlabs import ElevenLabsTTS

__all__ = ["SentenceChunker", "elevenlabs_tts"]


async def elevenlabs_tts(http: httpx.AsyncClient, text: str) -> AsyncIterator[bytes]:
    """Deprecated shim. Prefer :class:`ElevenLabsTTS` directly."""
    warnings.warn(
        "zemory.tts.elevenlabs_tts is deprecated; use ElevenLabsTTS provider",
        DeprecationWarning,
        stacklevel=2,
    )
    tts = ElevenLabsTTS(api_key=ELEVENLABS_API_KEY, http=http)
    async for chunk in tts.synthesize(text, quick=False):
        yield chunk
    _ = settings  # keep import used in case of future refactors
