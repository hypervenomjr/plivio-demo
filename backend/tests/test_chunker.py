from backend.tts.chunker import ClauseChunker


def test_chunker_emits_clause_on_sentence_end():
    chunker = ClauseChunker()
    assert chunker.push("We have that in stock. ") == ["We have that in stock."]


def test_chunker_buffers_incomplete_clause():
    chunker = ClauseChunker()
    assert chunker.push("We have that") == []


def test_chunker_assembles_clause_across_fragments():
    """The LLM streams token fragments, not whole clauses."""
    chunker = ClauseChunker()
    assert chunker.push("Yes, ") == ["Yes,"]
    assert chunker.push("we have") == []
    assert chunker.push(" headphones. ") == ["we have headphones."]


def test_chunker_emits_multiple_clauses_from_one_fragment():
    chunker = ClauseChunker()
    assert chunker.push("Yes, we do. ") == ["Yes,", "we do."]


def test_chunker_flush_returns_remaining_text():
    chunker = ClauseChunker()
    chunker.push("We have that")
    assert chunker.flush() == "We have that"


def test_chunker_flush_returns_none_when_empty():
    chunker = ClauseChunker()
    chunker.push("Done. ")
    assert chunker.flush() is None


def test_chunker_flush_clears_the_buffer():
    chunker = ClauseChunker()
    chunker.push("partial")
    assert chunker.flush() == "partial"
    assert chunker.flush() is None


def test_chunker_never_emits_empty_or_whitespace_clauses():
    chunker = ClauseChunker()
    clauses = chunker.push("   ,  .  ! ")
    assert all(clause.strip() for clause in clauses)


def test_chunker_handles_devanagari_punctuation():
    """Hindi and Marathi end sentences with danda, not a full stop.

    Without this the entire Hindi reply buffers until flush, which silently
    disables streaming for exactly the languages the demo is meant to show.
    """
    chunker = ClauseChunker()
    assert chunker.push("हाँ, हमारे पास है। ") == ["हाँ,", "हमारे पास है।"]


def test_chunker_does_not_split_decimal_prices():
    """"$89.99" must not break into "$89." and "99" — the clause would be
    spoken as two fragments with a pause in the middle of the number."""
    chunker = ClauseChunker()
    clauses = chunker.push("It costs $89.99 today. ")
    assert clauses == ["It costs $89.99 today."]


def test_chunker_emits_long_clause_without_punctuation():
    """A model that rambles without punctuation must not block playback
    until the whole answer finishes."""
    chunker = ClauseChunker(max_chars=40)
    text = "yes we have quite a lot of those available right now in stock "
    clauses = chunker.push(text)
    assert clauses
    assert all(len(clause) <= 60 for clause in clauses)
