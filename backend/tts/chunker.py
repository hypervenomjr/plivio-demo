"""Incremental clause chunking for streaming TTS.

This is the piece that makes the pipeline feel responsive. The LLM streams
token fragments; without chunking, synthesis could only start once the whole
answer existed, so the caller would wait for generation *and then* a full TTS
pass. Emitting each clause as its punctuation closes lets TTS for clause 1 run
while the LLM is still producing clause 2.

Three details that are easy to get wrong:

- **Devanagari uses danda (। / ॥), not a full stop.** Miss it and a Hindi or
  Marathi reply buffers whole, silently disabling streaming for exactly the
  languages this demo exists to show.
- **Decimals must not split.** "$89.99" breaking into "$89." and "99" would be
  spoken as two fragments with a pause mid-number.
- **A clause must eventually be emitted even without punctuation.** A model
  that rambles unpunctuated would otherwise block playback until flush.
"""
import re

# A clause ends at . ! ? , ; : or a Devanagari danda, when followed by
# whitespace — the trailing-whitespace requirement is what prevents splitting
# "89.99", since the dot there is followed by a digit.
_CLAUSE_END = re.compile(r"([.!?,;:।॥])\s+")

# Fallback: emit at a word boundary once a clause grows past this, so
# unpunctuated output still streams.
DEFAULT_MAX_CHARS = 160


class ClauseChunker:
    def __init__(self, max_chars: int = DEFAULT_MAX_CHARS):
        self._buffer = ""
        self._max_chars = max_chars

    def push(self, text_fragment: str) -> list[str]:
        """Feed an LLM fragment; return any clauses it completed."""
        self._buffer += text_fragment
        clauses = []

        while True:
            match = _CLAUSE_END.search(self._buffer)
            if match:
                end = match.end()
                clause = self._buffer[:end].strip()
                self._buffer = self._buffer[end:]
                if clause:
                    clauses.append(clause)
                continue

            forced = self._force_split()
            if forced:
                clauses.append(forced)
                continue

            break

        return clauses

    def _force_split(self) -> str | None:
        """Split an over-long unpunctuated buffer at the last word boundary."""
        if len(self._buffer) <= self._max_chars:
            return None

        window = self._buffer[:self._max_chars]
        split_at = window.rfind(" ")
        if split_at <= 0:
            return None  # a single very long token; wait for a boundary

        clause = self._buffer[:split_at].strip()
        self._buffer = self._buffer[split_at:]
        return clause or None

    def flush(self) -> str | None:
        """Return any trailing partial clause at end of generation."""
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining or None
