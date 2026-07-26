"""SentenceChunker unit tests — hard/soft boundaries, Korean, edge cases."""

from __future__ import annotations

from zemory.pipeline.chunker import SentenceChunker


def test_hard_boundary_period():
    c = SentenceChunker()
    assert c.add("Hello world.") == ["Hello world."]
    assert c.add(" Next.") == ["Next."]


def test_hard_boundary_korean():
    c = SentenceChunker()
    assert c.add("안녕하세요?") == ["안녕하세요?"]
    assert c.add(" 반갑습니다。") == ["반갑습니다。"]


def test_multiple_sentences_in_one_chunk():
    c = SentenceChunker()
    out = c.add("First. Second! Third?")
    assert out == ["First.", "Second!", "Third?"]


def test_flush_preserves_tail():
    c = SentenceChunker()
    assert c.add("incomplete") == []
    assert c.flush() == "incomplete"
    assert c.flush() is None


def test_whitespace_and_empty():
    c = SentenceChunker()
    assert c.add("") == []
    assert c.add("   .") == ["."]
    assert c.flush() is None


def test_soft_boundary_requires_min_length():
    """Comma splits only when buffer is ≥ 40 chars — avoids mid-phrase cuts."""
    c = SentenceChunker()
    # Under 40 chars — comma should NOT split
    assert c.add("Short, still buffering") == []
    assert c.flush() == "Short, still buffering"

    c2 = SentenceChunker()
    # 40+ chars — first comma after position 40 splits
    long_text = "A" * 40 + ", then continues"
    out = c2.add(long_text)
    # Should split at the first soft boundary found at index 40
    assert len(out) == 1 and out[0].endswith(",")
    # Rest goes in buffer
    assert c2.flush() == "then continues"


def test_korean_english_mixed():
    c = SentenceChunker()
    out = c.add("안녕 world. 반가워!")
    assert out == ["안녕 world.", "반가워!"]


def test_newline_is_hard_boundary():
    c = SentenceChunker()
    assert c.add("Line one\n") == ["Line one"]


def test_incremental_delta_accumulation():
    """Simulates token-by-token streaming from an LLM."""
    c = SentenceChunker()
    collected: list[str] = []
    for tok in ["Hel", "lo ", "wor", "ld", ". ", "Byeh"]:
        collected.extend(c.add(tok))
    assert collected == ["Hello world."]
    assert c.flush() == "Byeh"


def test_punctuation_free_model_output_is_bounded() -> None:
    c = SentenceChunker()

    chunks = c.add("word " * 250)
    tail = c.flush()

    assert chunks
    assert all(0 < len(chunk) <= 240 for chunk in chunks)
    assert tail is not None and len(tail) < 240
    assert "".join(chunks + [tail]).replace(" ", "") == "word" * 250
