"""Context-aware transcript correction.

Optional pre-LLM pass that rewrites the raw ASR transcript using the
recent conversation as context, intended to fix proper-noun mistakes
and homophones that Whisper/Realtime's built-in transcription gets
wrong. Guarded by ``settings.transcript_correction_enabled``.

Cost: one extra chat completion per user turn. Expected latency with
a small/fast model (e.g. ``gpt-4o-mini``, ``gpt-5-mini``) is 200-500 ms,
which is on top of the normal turn budget. Measured via
``ttfb.correction`` histogram and emitted in the ``turn.complete`` log.
"""

from __future__ import annotations

import time
from collections import deque

from openai import AsyncOpenAI

from zemory.observability import get_logger, metrics

_log = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a transcript post-processor for a real-time voice chat. "
    "Given the recent conversation and the user's latest utterance as "
    "transcribed by ASR, return the utterance with likely recognition "
    "errors corrected — especially proper nouns, domain-specific terms, "
    "homophones, and names the user has used before. "
    "If the transcript already looks correct, return it unchanged. "
    "Output ONLY the corrected utterance, nothing else. Preserve the "
    "user's original language."
)


class TranscriptCorrector:
    """Holds a short rolling history and applies correction on demand."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        history_turns: int,
    ) -> None:
        self._client = client
        self._model = model
        # Each turn contributes up to 2 entries (user + assistant).
        self._history: deque[tuple[str, str]] = deque(maxlen=history_turns * 2)

    def record_user(self, text: str) -> None:
        if text:
            self._history.append(("user", text))

    def record_assistant(self, text: str) -> None:
        if text:
            self._history.append(("assistant", text))

    async def correct(self, raw: str) -> tuple[str, float]:
        """Return (corrected_text, elapsed_ms). Falls back to ``raw`` on failure."""
        if not raw or not raw.strip():
            return raw, 0.0

        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        if self._history:
            convo = "\n".join(f"{role}: {text}" for role, text in self._history)
            messages.append(
                {"role": "system", "content": f"Recent conversation:\n{convo}"}
            )
        messages.append(
            {
                "role": "user",
                "content": f"Raw ASR transcript to correct:\n{raw}",
            }
        )

        started = time.monotonic()
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.0,
            )
            corrected = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            _log.warning("correction.failed", error=str(e), raw=raw)
            return raw, (time.monotonic() - started) * 1000

        elapsed_ms = (time.monotonic() - started) * 1000
        metrics.observe("ttfb.correction", elapsed_ms)

        if not corrected:
            _log.warning("correction.empty_result", raw=raw)
            return raw, elapsed_ms

        if corrected != raw:
            _log.info(
                "correction.applied",
                raw=raw,
                corrected=corrected,
                ms=round(elapsed_ms, 1),
            )
        else:
            _log.info("correction.noop", ms=round(elapsed_ms, 1))
        return corrected, elapsed_ms
