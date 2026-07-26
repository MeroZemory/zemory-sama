"""Sentence chunker with AIRIS-style early-trigger.

Hard boundaries (always split): ``. ? ! 。 ？ ！ \\n``.
Soft boundaries (split only when buffer ≥ ``_SOFT_MIN_LEN`` chars):
``, : ; 、 ，`` — enables first TTS firing earlier in long clauses without
cutting short phrases mid-sentence.

Korean- and English-safe; trims whitespace from yielded sentences.
"""

from __future__ import annotations

_HARD_BOUNDARIES = frozenset(".?!。？！\n")
_SOFT_BOUNDARIES = frozenset(",:;、，")
_SOFT_MIN_LEN = 40  # chars
_HARD_MAX_LEN = 240


class SentenceChunker:
    """Accumulates streaming text and yields complete sentences.

    Usage::

        chunker = SentenceChunker()
        for sentence in chunker.add(delta):
            ...   # dispatch to TTS
        tail = chunker.flush()   # after response.done
    """

    def __init__(self) -> None:
        self._buf = ""

    def add(self, text: str) -> list[str]:
        self._buf += text
        sentences: list[str] = []
        while True:
            idx = self._find_split()
            if idx == -1:
                break
            sentence = self._buf[: idx + 1].strip()
            self._buf = self._buf[idx + 1 :]
            if sentence:
                sentences.append(sentence)
        return sentences

    def _find_split(self) -> int:
        for i, ch in enumerate(self._buf):
            if ch in _HARD_BOUNDARIES:
                return i
            if ch in _SOFT_BOUNDARIES and (i + 1) >= _SOFT_MIN_LEN:
                # Only soft-split when we've accumulated enough context.
                return i
        if len(self._buf) >= _HARD_MAX_LEN:
            # A model can ignore the requested response length or emit a very
            # long punctuation-free token stream. Bound both memory and TTS
            # start latency, preferring a nearby word boundary when possible.
            boundary = self._buf.rfind(" ", _SOFT_MIN_LEN, _HARD_MAX_LEN)
            return boundary if boundary >= 0 else _HARD_MAX_LEN - 1
        return -1

    def flush(self) -> str | None:
        """Return any leftover buffer (e.g. after response.done)."""
        remaining = self._buf.strip()
        self._buf = ""
        return remaining or None
