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
        return -1

    def flush(self) -> str | None:
        """Return any leftover buffer (e.g. after response.done)."""
        remaining = self._buf.strip()
        self._buf = ""
        return remaining or None
