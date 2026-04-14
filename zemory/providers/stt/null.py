"""No-op STT: used in the Realtime profile where transcription happens inline."""

from __future__ import annotations


class NullSTT:
    async def transcribe(self, pcm_chunks: list[bytes]) -> str:  # pragma: no cover
        return ""
