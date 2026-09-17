# Phase 4: Text-to-Speech (Meta MMS-TTS, CPU) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn partial LLM output into spoken WAV audio in English, Hindi or Marathi, on CPU, emitting audio clause-by-clause so playback starts before the full answer finishes generating.

**Architecture:** A `ClauseChunker` accumulates streamed LLM text fragments and emits complete clauses as soon as punctuation closes them — this is what makes the reply audible early. A `Synthesizer` lazily loads one Meta MMS-TTS (VITS) checkpoint per language and renders a clause to WAV bytes. MMS-TTS is chosen over Piper (no Marathi voice) and AI4Bharat Indic-TTS (heavier, GPU-leaning) for CPU-native fit plus Indic coverage.

**Tech Stack:** Python 3.10+, transformers (VitsModel), torch (CPU), scipy.

**Spec:** `docs/superpowers/specs/2026-09-17-voice-rag-agent-design.md`

**Index:** [README.md](README.md)

## Isolation

Depends on **nothing**. Both pieces are pure text-in / bytes-out, testable locally on CPU with no GPU, no server, no other phase.

## Contract produced (Phase 5 relies on these — do not change without updating it)

- `Synthesizer(language_model_map: dict[str, str] = None)`
- `Synthesizer.synthesize(text: str, language: str = "en") -> bytes` — returns WAV bytes; raises `ValueError` for an unsupported language code
- `ClauseChunker()`
- `ClauseChunker.push(text_fragment: str) -> list[str]` — zero or more complete clauses
- `ClauseChunker.flush() -> str | None` — trailing partial clause at end of generation

Language codes accepted match what Phase 3's `Transcriber` emits: `"en"`, `"hi"`, `"mr"`.

## Global Constraints

- TTS runs on CPU only — never allocate GPU here; the GPU is reserved for the LLM. *(spec: Constraints)*
- Must support English, Hindi and Marathi output, selected by the language code detected during transcription. *(spec: Constraints, Data flow step 3)*
- Audio must start playing before the full LLM response finishes — hence clause-level chunking rather than whole-answer synthesis. *(spec: Data flow step 6)*

---

### Task 4.1: Multi-language synthesis wrapper

**Files:**
- Create: `backend/tts/synthesize.py`
- Create: `backend/tts/__init__.py`
- Test: `backend/tests/test_synthesize.py`
- Modify: `backend/requirements.txt`

**Interfaces:**
- Consumes: a clause string plus a language code.
- Produces:
  - `class Synthesizer` with `__init__(self, language_model_map: dict[str, str] = None)` — maps `"en"`/`"hi"`/`"mr"` to `facebook/mms-tts-eng` / `facebook/mms-tts-hin` / `facebook/mms-tts-mar`
  - `Synthesizer.synthesize(self, text: str, language: str = "en") -> bytes` — WAV bytes

- [ ] **Step 1: Add dependencies**

```text
# backend/requirements.txt (append)
transformers==4.44.2
torch==2.4.0
scipy==1.14.0
```

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_synthesize.py
import pytest
from backend.tts.synthesize import Synthesizer


@pytest.fixture(scope="module")
def synth():
    return Synthesizer()


def test_synthesize_returns_wav_bytes_english(synth):
    audio = synth.synthesize("hello, we have that in stock", language="en")
    assert audio[:4] == b"RIFF"
    assert len(audio) > 1000


def test_synthesize_returns_wav_bytes_hindi(synth):
    audio = synth.synthesize("नमस्ते, यह स्टॉक में है", language="hi")
    assert audio[:4] == b"RIFF"
    assert len(audio) > 1000


def test_synthesize_returns_wav_bytes_marathi(synth):
    audio = synth.synthesize("नमस्कार, हे स्टॉकमध्ये आहे", language="mr")
    assert audio[:4] == b"RIFF"
    assert len(audio) > 1000


def test_synthesize_unknown_language_raises(synth):
    with pytest.raises(ValueError):
        synth.synthesize("test", language="xx")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest backend/tests/test_synthesize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.tts.synthesize'`

- [ ] **Step 4: Write minimal implementation**

```python
# backend/tts/synthesize.py
import io

import scipy.io.wavfile
import torch
from transformers import AutoTokenizer, VitsModel

DEFAULT_LANGUAGE_MODEL_MAP = {
    "en": "facebook/mms-tts-eng",
    "hi": "facebook/mms-tts-hin",
    "mr": "facebook/mms-tts-mar",
}


class Synthesizer:
    def __init__(self, language_model_map: dict[str, str] = None):
        self._map = language_model_map or DEFAULT_LANGUAGE_MODEL_MAP
        self._models = {}
        self._tokenizers = {}

    def _load(self, language: str):
        if language not in self._map:
            raise ValueError(f"Unsupported language: {language}")
        if language not in self._models:
            checkpoint = self._map[language]
            # CPU only — the GPU is reserved for the LLM.
            self._models[language] = VitsModel.from_pretrained(checkpoint)
            self._tokenizers[language] = AutoTokenizer.from_pretrained(checkpoint)
        return self._models[language], self._tokenizers[language]

    def synthesize(self, text: str, language: str = "en") -> bytes:
        model, tokenizer = self._load(language)
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            waveform = model(**inputs).waveform
        samples = waveform.squeeze().numpy()

        buffer = io.BytesIO()
        scipy.io.wavfile.write(buffer, rate=model.config.sampling_rate, data=samples)
        return buffer.getvalue()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pip install -r backend/requirements.txt && pytest backend/tests/test_synthesize.py -v`
Expected: PASS (4 tests). The first run downloads three MMS checkpoints, so it is slow.

- [ ] **Step 6: Commit**

```bash
git add backend/tts backend/tests/test_synthesize.py backend/requirements.txt
git commit -m "feat: add multi-language MMS-TTS CPU synthesis wrapper"
```

---

### Task 4.2: Incremental clause chunker

**Files:**
- Create: `backend/tts/chunker.py`
- Test: `backend/tests/test_chunker.py`

**Interfaces:**
- Consumes: LLM text fragments, pushed in the order they stream in.
- Produces:
  - `class ClauseChunker` with `__init__(self)`
  - `ClauseChunker.push(self, text_fragment: str) -> list[str]` — complete clauses, split on `.`, `!`, `?` or `,` followed by whitespace
  - `ClauseChunker.flush(self) -> str | None` — any trailing partial clause

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_chunker.py
from backend.tts.chunker import ClauseChunker


def test_chunker_emits_clause_on_sentence_end():
    chunker = ClauseChunker()
    assert chunker.push("We have that in stock. ") == ["We have that in stock."]


def test_chunker_buffers_incomplete_clause():
    chunker = ClauseChunker()
    assert chunker.push("We have that") == []


def test_chunker_assembles_clause_across_fragments():
    chunker = ClauseChunker()
    assert chunker.push("Yes, ") == ["Yes,"]
    assert chunker.push("we have") == []
    assert chunker.push(" headphones. ") == ["we have headphones."]


def test_chunker_flush_returns_remaining_text():
    chunker = ClauseChunker()
    chunker.push("We have that")
    assert chunker.flush() == "We have that"


def test_chunker_flush_returns_none_when_empty():
    chunker = ClauseChunker()
    chunker.push("Done. ")
    assert chunker.flush() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.tts.chunker'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/tts/chunker.py
import re

_CLAUSE_END = re.compile(r"([.!?,])\s+")


class ClauseChunker:
    def __init__(self):
        self._buffer = ""

    def push(self, text_fragment: str) -> list[str]:
        self._buffer += text_fragment
        clauses = []
        while True:
            match = _CLAUSE_END.search(self._buffer)
            if not match:
                break
            end = match.end()
            clause = self._buffer[:end].strip()
            if clause:
                clauses.append(clause)
            self._buffer = self._buffer[end:]
        return clauses

    def flush(self) -> str | None:
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_chunker.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/tts/chunker.py backend/tests/test_chunker.py
git commit -m "feat: add clause chunker for incremental streaming TTS"
```

---

## Phase 4 done when

`pytest backend/tests/test_synthesize.py backend/tests/test_chunker.py -v` passes on CPU, with no GPU and no dependency on any other phase.
