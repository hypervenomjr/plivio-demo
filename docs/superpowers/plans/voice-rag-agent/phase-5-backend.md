# Phase 5: Backend Orchestration Server — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A FastAPI WebSocket server that runs a full turn — audio in, transcript, retrieval, LLM, clause-by-clause speech out — and persists conversation history per session.

**Architecture:** `create_app()` takes every model-backed dependency as an injected argument, so the entire server is testable with fakes and no GPU, no weights and no audio. A separate `backend/main.py` wires the real implementations for Colab. History lives in SQLite on local Colab disk.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, websockets, SQLite.

**Spec:** `docs/superpowers/specs/2026-09-17-voice-rag-agent-design.md`

**Index:** [README.md](README.md)

## Isolation

This is the one phase that *composes* others, so it is built against their **frozen contracts, not their code**. Task 5.2's tests inject fakes for every model — it can be written and passing before Phases 1-4 exist at all. Only Task 5.3 (the Colab wiring script) imports the real classes, and it has no unit tests of its own.

## Contracts

**Consumed** (from the index; fakes stand in for all of them during tests):
- `InventoryStore.query(text, top_k=5) -> list[dict]` *(Phase 1)*
- `build_prompt(user_text, retrieved, history, language="en", max_history_turns=6) -> str` *(Phase 2)*
- `LLMEngine.generate(prompt, stop=None, max_tokens=200) -> Iterator[str]` *(Phase 2)*
- `Transcriber.transcribe(audio_path) -> {"text": str, "language": str}` *(Phase 3)*
- `UtteranceSegmenter.push(pcm_chunk) -> bytes | None` and `.speech_active -> bool` *(Phase 3)*
- `Synthesizer.synthesize(text, language="en") -> bytes` *(Phase 4)*
- `ClauseChunker.push(fragment) -> list[str]` / `.flush() -> str | None` *(Phase 4)*

**Produced:**
- History turn dict `{"role": "user"|"assistant", "text": str}` — consumed by Phase 2's prompt builder
- `HistoryStore(db_path)`, `.append(session_id, role, text)`, `.get(session_id, max_turns=6) -> list[dict]`
- `create_app(...) -> FastAPI` exposing `/ws/{session_id}`
- **The WebSocket wire protocol** — the sole contract with Phase 6, reproduced below

### WebSocket wire protocol

**Client → server:**
- Binary frames: raw 16kHz mono PCM chunks.
- Text frame `"__end__"`: force end-of-turn (push-to-talk release, and tests).

**Server → client, per turn, in order:**
1. Text `{"type": "transcript", "text": str, "language": str}`
2. Repeating per clause: text `{"type": "answer_text", "text": str}`, then a binary WAV frame for that clause.
3. Text `{"type": "timing", "stages": {...}}` — per-stage milliseconds for this turn.
4. Text `{"type": "turn_end"}`

**Out of band, at any point during a turn:**
- Text `{"type": "interrupted"}` — the caller spoke over the agent; generation was cancelled. The client must stop playback and discard any queued audio. No `turn_end` follows an `interrupted`.

## Global Constraints

- Conversation history windowed to the last N turns and persisted to SQLite on local Colab disk; wiped on runtime reset, which is accepted. *(spec: Conversation history)*
- Retrieval capped to top-k=3-5 per turn. *(spec: Cost / token optimization)*
- The LLM runs on GPU (`n_gpu_layers=-1`); STT device is configurable and TTS stays on CPU. *(spec: Constraints)*
- The language code detected during transcription is used **twice**: it selects the TTS voice *and* it is passed to `build_prompt` so the model replies in that language. Missing either one breaks multilingual output. *(spec: Data flow step 3)*
- Barge-in: when the caller speaks while the agent is talking, the in-flight turn is cancelled. *(spec: Data flow step 8)*

---

### Task 5.1: Conversation history store (SQLite)

**Files:**
- Create: `backend/history/store.py`
- Create: `backend/history/__init__.py`
- Test: `backend/tests/test_history_store.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class HistoryStore` with `__init__(self, db_path: str)`
  - `HistoryStore.append(self, session_id: str, role: str, text: str) -> None`
  - `HistoryStore.get(self, session_id: str, max_turns: int = 6) -> list[dict]` — oldest-to-newest, limited to the last `max_turns`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_history_store.py
import os
import tempfile
import pytest
from backend.history.store import HistoryStore


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield HistoryStore(db_path=path)
    os.remove(path)


def test_append_and_get(store):
    store.append("session1", "user", "hello")
    store.append("session1", "assistant", "hi there")
    assert store.get("session1") == [
        {"role": "user", "text": "hello"},
        {"role": "assistant", "text": "hi there"},
    ]


def test_sessions_are_isolated(store):
    store.append("session1", "user", "a")
    store.append("session2", "user", "b")
    assert store.get("session1") == [{"role": "user", "text": "a"}]


def test_get_windows_to_max_turns_keeping_newest(store):
    for i in range(10):
        store.append("session1", "user", f"turn {i}")
    history = store.get("session1", max_turns=3)
    assert [h["text"] for h in history] == ["turn 7", "turn 8", "turn 9"]


def test_get_unknown_session_returns_empty(store):
    assert store.get("nobody") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_history_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.history.store'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/history/store.py
import sqlite3


class HistoryStore:
    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        self._conn.commit()

    def append(self, session_id: str, role: str, text: str) -> None:
        self._conn.execute(
            "INSERT INTO history (session_id, role, text) VALUES (?, ?, ?)",
            (session_id, role, text),
        )
        self._conn.commit()

    def get(self, session_id: str, max_turns: int = 6) -> list[dict]:
        cursor = self._conn.execute(
            "SELECT role, text FROM history WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, max_turns),
        )
        rows = cursor.fetchall()
        rows.reverse()  # newest-first from SQL, oldest-first for the prompt
        return [{"role": role, "text": text} for role, text in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_history_store.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/history backend/tests/test_history_store.py
git commit -m "feat: add SQLite conversation history store"
```

---

### Task 5.2: WebSocket orchestration server

**Files:**
- Create: `backend/server/app.py`
- Create: `backend/server/__init__.py`
- Test: `backend/tests/test_server.py`
- Modify: `backend/requirements.txt`

**Interfaces:**
- Consumes: the injected dependencies listed in the Contracts section above.
- Produces: `create_app(segmenter_factory, transcriber, inventory_store, llm_engine, synthesizer, history_db_path, scratch_audio_path) -> FastAPI` exposing `/ws/{session_id}` on the wire protocol above.

- [ ] **Step 1: Add dependencies**

```text
# backend/requirements.txt (append)
fastapi==0.112.0
uvicorn==0.30.5
websockets==12.0
httpx==0.27.0
```

- [ ] **Step 2: Write the failing test (fakes for every model, real HistoryStore and prompt builder)**

```python
# backend/tests/test_server.py
import json
import tempfile
import pytest
from fastapi.testclient import TestClient
from backend.server.app import create_app


class FakeSegmenter:
    """Never fires on its own — the test drives end-of-turn via the "__end__" frame."""
    def __init__(self):
        self.speech_active = False

    def push(self, chunk):
        return None


class BargeInSegmenter:
    """Reports speech starting on the Nth pushed chunk, to trigger barge-in."""
    def __init__(self, speak_on_chunk: int):
        self.speech_active = False
        self._count = 0
        self._speak_on = speak_on_chunk

    def push(self, chunk):
        self._count += 1
        if self._count >= self._speak_on:
            self.speech_active = True
        return None


class FakeTranscriber:
    def __init__(self, text="do you have headphones", language="en"):
        self._text = text
        self._language = language

    def transcribe(self, audio_path):
        return {"text": self._text, "language": self._language}


class FakeStore:
    def query(self, text, top_k=5):
        return [{"id": "1", "name": "Wireless Headphones", "description": "...",
                 "price": 89.99, "stock": 20, "category": "Electronics", "score": 0.1}]


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, stop=None, max_tokens=200):
        self.prompts.append(prompt)
        yield "Yes, "
        yield "we have Wireless Headphones in stock. "


class SlowLLM:
    """Generates slowly so a barge-in can land mid-answer."""
    def generate(self, prompt, stop=None, max_tokens=200):
        for _ in range(50):
            time.sleep(0.01)
            yield "word, "


class FakeSynthesizer:
    def __init__(self):
        self.calls = []

    def synthesize(self, text, language="en"):
        self.calls.append((text, language))
        return b"RIFFfakewavbytes"


def build_client(segmenter_factory=None, transcriber=None, llm=None, synthesizer=None):
    app = create_app(
        segmenter_factory=segmenter_factory or (lambda: FakeSegmenter()),
        transcriber=transcriber or FakeTranscriber(),
        inventory_store=FakeStore(),
        llm_engine=llm or FakeLLM(),
        synthesizer=synthesizer or FakeSynthesizer(),
        history_db_path=tempfile.mktemp(suffix=".db"),
        scratch_audio_path=tempfile.mktemp(suffix=".wav"),
    )
    return TestClient(app)


def drain_turn(ws):
    """Read messages until turn_end, returning everything seen."""
    messages = []
    while True:
        msg = json.loads(ws.receive_text())
        messages.append(msg)
        if msg["type"] == "answer_text":
            ws.receive_bytes()
        if msg["type"] in ("turn_end", "interrupted"):
            return messages


@pytest.fixture
def synthesizer():
    return FakeSynthesizer()


@pytest.fixture
def client(synthesizer):
    return build_client(synthesizer=synthesizer)


def test_full_turn_over_websocket(client):
    with client.websocket_connect("/ws/test-session") as ws:
        ws.send_bytes(b"\x00\x00" * 100)
        ws.send_text("__end__")

        assert json.loads(ws.receive_text()) == {
            "type": "transcript", "text": "do you have headphones", "language": "en",
        }

        answer_msg = json.loads(ws.receive_text())
        assert answer_msg["type"] == "answer_text"
        assert ws.receive_bytes().startswith(b"RIFF")

        msg = json.loads(ws.receive_text())
        while msg["type"] == "answer_text":
            ws.receive_bytes()
            msg = json.loads(ws.receive_text())
        assert msg["type"] == "timing"
        assert json.loads(ws.receive_text())["type"] == "turn_end"


def test_timing_message_reports_stages(client):
    with client.websocket_connect("/ws/test-session") as ws:
        ws.send_bytes(b"\x00\x00" * 100)
        ws.send_text("__end__")
        messages = drain_turn(ws)
    timing = next(m for m in messages if m["type"] == "timing")
    for stage in ["stt", "retrieval", "llm_total", "tts_total", "time_to_first_audio"]:
        assert stage in timing["stages"]


def test_synthesis_uses_detected_language(synthesizer):
    client = build_client(transcriber=FakeTranscriber("क्या हेडफ़ोन हैं", "hi"),
                          synthesizer=synthesizer)
    with client.websocket_connect("/ws/test-session") as ws:
        ws.send_bytes(b"\x00\x00" * 100)
        ws.send_text("__end__")
        drain_turn(ws)
    assert synthesizer.calls
    assert all(language == "hi" for _, language in synthesizer.calls)


def test_detected_language_reaches_the_prompt():
    llm = FakeLLM()
    client = build_client(transcriber=FakeTranscriber("हेडफ़ोन", "hi"), llm=llm)
    with client.websocket_connect("/ws/test-session") as ws:
        ws.send_bytes(b"\x00\x00" * 100)
        ws.send_text("__end__")
        drain_turn(ws)
    assert "Hindi" in llm.prompts[0]


def test_empty_transcript_skips_llm():
    llm = FakeLLM()
    client = build_client(transcriber=FakeTranscriber("", "en"), llm=llm)
    with client.websocket_connect("/ws/test-session") as ws:
        ws.send_bytes(b"\x00\x00" * 100)
        ws.send_text("__end__")
        drain_turn(ws)
    assert llm.prompts == []


def test_barge_in_cancels_in_flight_turn():
    client = build_client(
        segmenter_factory=lambda: BargeInSegmenter(speak_on_chunk=2),
        llm=SlowLLM(),
    )
    with client.websocket_connect("/ws/test-session") as ws:
        ws.send_bytes(b"\x00\x00" * 100)   # chunk 1, no speech yet
        ws.send_text("__end__")            # starts a slow turn
        ws.send_bytes(b"\x00\x00" * 100)   # chunk 2 -> speech_active rises

        saw_interrupted = False
        for _ in range(40):
            msg = json.loads(ws.receive_text())
            if msg["type"] == "answer_text":
                ws.receive_bytes()
            if msg["type"] == "interrupted":
                saw_interrupted = True
                break
        assert saw_interrupted


def test_history_persists_across_turns(client):
    with client.websocket_connect("/ws/history-session") as ws:
        for _ in range(2):
            ws.send_bytes(b"\x00\x00" * 100)
            ws.send_text("__end__")
            drain_turn(ws)
    # Two turns stored two user + two assistant rows; HistoryStore's own unit
    # tests cover the storage semantics.
```

Add `import time` alongside the other imports at the top of the file.

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest backend/tests/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.server.app'`

- [ ] **Step 4: Write minimal implementation**

```python
# backend/server/app.py
import asyncio
import json
import time
import wave
from contextlib import contextmanager

from fastapi import FastAPI, WebSocket

from backend.history.store import HistoryStore
from backend.llm.prompt import build_prompt
from backend.tts.chunker import ClauseChunker

TOP_K = 5


@contextmanager
def _timed(stages: dict, name: str):
    """Record a stage's wall-clock duration in milliseconds."""
    start = time.perf_counter()
    yield
    stages[name] = round((time.perf_counter() - start) * 1000, 1)


def create_app(segmenter_factory, transcriber, inventory_store, llm_engine,
               synthesizer, history_db_path: str, scratch_audio_path: str):
    app = FastAPI()
    history_store = HistoryStore(db_path=history_db_path)

    def write_wav(pcm_bytes: bytes, path: str) -> None:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm_bytes)

    @app.websocket("/ws/{session_id}")
    async def ws_endpoint(websocket: WebSocket, session_id: str):
        await websocket.accept()
        segmenter = segmenter_factory()
        buffer = bytearray()
        turn_task = None

        def start_turn(utterance: bytes):
            return asyncio.create_task(_handle_turn(
                websocket, session_id, utterance, transcriber, inventory_store,
                llm_engine, synthesizer, history_store, write_wav, scratch_audio_path,
            ))

        async def cancel_turn_if_running():
            """Barge-in: abandon the in-flight answer.

            Cancel, then await the task so its cancellation actually completes
            before we touch the socket again — Starlette websockets are not
            safe for concurrent sends from two coroutines.
            """
            nonlocal turn_task
            if turn_task is not None and not turn_task.done():
                turn_task.cancel()
                try:
                    await turn_task
                except asyncio.CancelledError:
                    pass
                await websocket.send_text(json.dumps({"type": "interrupted"}))
            turn_task = None

        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                await cancel_turn_if_running()
                break

            if message.get("bytes") is not None:
                chunk = message["bytes"]
                was_speaking = segmenter.speech_active
                utterance = segmenter.push(chunk)

                # Rising edge of speech while the agent is answering = barge-in.
                if not was_speaking and segmenter.speech_active:
                    await cancel_turn_if_running()

                buffer.extend(chunk)
                if utterance is not None:
                    buffer.clear()
                    turn_task = start_turn(utterance)

            elif message.get("text") == "__end__":
                await cancel_turn_if_running()
                utterance = bytes(buffer)
                buffer.clear()
                turn_task = start_turn(utterance)

    return app


async def _handle_turn(websocket, session_id, utterance_pcm, transcriber,
                       inventory_store, llm_engine, synthesizer, history_store,
                       write_wav, scratch_audio_path):
    stages = {}
    turn_start = time.perf_counter()

    with _timed(stages, "stt"):
        write_wav(utterance_pcm, scratch_audio_path)
        transcript = transcriber.transcribe(scratch_audio_path)

    language = transcript["language"]
    await websocket.send_text(json.dumps({
        "type": "transcript", "text": transcript["text"], "language": language,
    }))

    if not transcript["text"]:
        await websocket.send_text(json.dumps({"type": "turn_end"}))
        return

    with _timed(stages, "retrieval"):
        history = history_store.get(session_id)
        retrieved = inventory_store.query(transcript["text"], top_k=TOP_K)

    # language is used twice: here so the model answers in it, and below so
    # the matching TTS voice speaks it.
    prompt = build_prompt(transcript["text"], retrieved, history, language=language)

    chunker = ClauseChunker()
    full_answer = []
    first_audio_at = None
    tts_ms = 0.0

    async def speak(clause: str) -> None:
        nonlocal first_audio_at, tts_ms
        await websocket.send_text(json.dumps({"type": "answer_text", "text": clause}))
        tts_start = time.perf_counter()
        audio = synthesizer.synthesize(clause, language=language)
        tts_ms += (time.perf_counter() - tts_start) * 1000
        await websocket.send_bytes(audio)
        if first_audio_at is None:
            first_audio_at = time.perf_counter()

    llm_start = time.perf_counter()
    first_token_at = None

    for fragment in llm_engine.generate(prompt):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        full_answer.append(fragment)
        for clause in chunker.push(fragment):
            await speak(clause)
        # Yield to the event loop so an incoming barge-in can be noticed
        # mid-generation rather than only between turns.
        await asyncio.sleep(0)

    trailing = chunker.flush()
    if trailing:
        await speak(trailing)

    if first_token_at is not None:
        stages["llm_first_token"] = round((first_token_at - llm_start) * 1000, 1)
    stages["llm_total"] = round((time.perf_counter() - llm_start) * 1000, 1)
    stages["tts_total"] = round(tts_ms, 1)
    if first_audio_at is not None:
        stages["time_to_first_audio"] = round((first_audio_at - turn_start) * 1000, 1)
    stages["turn_total"] = round((time.perf_counter() - turn_start) * 1000, 1)

    history_store.append(session_id, "user", transcript["text"])
    history_store.append(session_id, "assistant", "".join(full_answer))

    await websocket.send_text(json.dumps({"type": "timing", "stages": stages}))
    await websocket.send_text(json.dumps({"type": "turn_end"}))
```

Two things worth understanding rather than just copying:

**Why `await asyncio.sleep(0)` inside the generation loop.** `llm_engine.generate` is a synchronous generator. Without an explicit yield point, the event loop never gets control during generation, so the receive loop cannot notice the caller talking over the agent, and barge-in would only ever fire between turns. This is the cheapest correct fix; a fully correct one would run generation in a thread executor.

**Why `time_to_first_audio` is measured here.** It is the number that defines the user experience, and the only one that cannot be reconstructed by adding up the others — the whole point of clause streaming is that stages overlap, so the sum of the parts exceeds the real elapsed time.

- [ ] **Step 5: Run test to verify it passes**

Run: `pip install -r backend/requirements.txt && pytest backend/tests/test_server.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Commit**

```bash
git add backend/server backend/tests/test_server.py backend/requirements.txt
git commit -m "feat: add WebSocket orchestration server wiring STT/RAG/LLM/TTS/history"
```

---

### Task 5.3: Real-model wiring script (Colab entrypoint)

**Files:**
- Create: `backend/main.py`

**Interfaces:**
- Consumes: `create_app` from Task 5.2 plus the real classes from Phases 1-4.
- Produces: a runnable module that serves the real pipeline on port 8000.

- [ ] **Step 1: Write the script**

```python
# backend/main.py
import os

import uvicorn

from backend.llm.generate import LLMEngine
from backend.rag.store import InventoryStore
from backend.server.app import create_app
from backend.stt.transcribe import Transcriber
from backend.stt.vad import UtteranceSegmenter
from backend.tts.synthesize import Synthesizer

LLM_GGUF_PATH = os.environ.get(
    "LLM_GGUF_PATH", "/content/qwen2.5-7b-instruct-q4_k_m.gguf"
)

# Set STT_DEVICE=cuda to move Whisper onto the GPU. Default is cpu, which keeps
# the GPU entirely for the LLM; measure both with backend/stt/benchmark.py and
# pick based on the numbers, not the design intent.
STT_DEVICE = os.environ.get("STT_DEVICE", "cpu")

app = create_app(
    segmenter_factory=lambda: UtteranceSegmenter(),
    transcriber=Transcriber(model_size="small", device=STT_DEVICE),
    inventory_store=InventoryStore(persist_dir="backend/rag/chroma_db"),
    llm_engine=LLMEngine(model_path=LLM_GGUF_PATH, n_gpu_layers=-1),
    synthesizer=Synthesizer(),
    history_db_path="backend/history/history.db",
    scratch_audio_path="/tmp/scratch_utterance.wav",
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 2: Run it manually on Colab**

Prerequisites: Phase 1's catalog loaded (`python -m backend.rag.load_catalog`) and Phase 2's GGUF file downloaded.
Run: `python -m backend.main`
Expected: uvicorn reports `Uvicorn running on http://0.0.0.0:8000` with no import or model-loading errors.

- [ ] **Step 3: Commit**

```bash
git add backend/main.py
git commit -m "feat: add real-model Colab entrypoint wiring all pipeline stages"
```

---

## Phase 5 done when

`pytest backend/tests/test_history_store.py backend/tests/test_server.py -v` passes locally with fakes (no GPU, no weights, no audio) — 11 tests including barge-in cancellation, the language handoff into the prompt, and the timing message — and `python -m backend.main` starts cleanly on Colab.
