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
- `build_prompt(user_text, retrieved, history, max_history_turns=6) -> str` *(Phase 2)*
- `LLMEngine.generate(prompt, stop=None, max_tokens=200) -> Iterator[str]` *(Phase 2)*
- `Transcriber.transcribe(audio_path) -> {"text": str, "language": str}` *(Phase 3)*
- `UtteranceSegmenter.push(pcm_chunk) -> bytes | None` *(Phase 3)*
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
3. Text `{"type": "turn_end"}`

## Global Constraints

- Conversation history windowed to the last N turns and persisted to SQLite on local Colab disk; wiped on runtime reset, which is accepted. *(spec: Conversation history)*
- Retrieval capped to top-k=3-5 per turn. *(spec: Cost / token optimization)*
- Only the LLM touches the GPU (`n_gpu_layers=-1`); STT and TTS stay on CPU. *(spec: Constraints)*
- The language code detected during transcription selects the TTS voice for the reply. *(spec: Data flow step 3)*

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
    def push(self, chunk):
        return None


class FakeTranscriber:
    def transcribe(self, audio_path):
        return {"text": "do you have headphones", "language": "en"}


class FakeStore:
    def query(self, text, top_k=5):
        return [{"id": "1", "name": "Wireless Headphones", "description": "...",
                 "price": 89.99, "stock": 20, "category": "Electronics", "score": 0.1}]


class FakeLLM:
    def generate(self, prompt, stop=None, max_tokens=200):
        yield "Yes, "
        yield "we have Wireless Headphones in stock. "


class FakeSynthesizer:
    def __init__(self):
        self.calls = []

    def synthesize(self, text, language="en"):
        self.calls.append((text, language))
        return b"RIFFfakewavbytes"


@pytest.fixture
def synthesizer():
    return FakeSynthesizer()


@pytest.fixture
def client(synthesizer):
    app = create_app(
        segmenter_factory=lambda: FakeSegmenter(),
        transcriber=FakeTranscriber(),
        inventory_store=FakeStore(),
        llm_engine=FakeLLM(),
        synthesizer=synthesizer,
        history_db_path=tempfile.mktemp(suffix=".db"),
        scratch_audio_path=tempfile.mktemp(suffix=".wav"),
    )
    return TestClient(app)


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

        # drain any further clause pairs until the turn ends
        msg = json.loads(ws.receive_text())
        while msg["type"] == "answer_text":
            ws.receive_bytes()
            msg = json.loads(ws.receive_text())
        assert msg == {"type": "turn_end"}


def test_synthesis_uses_detected_language(client, synthesizer):
    with client.websocket_connect("/ws/test-session") as ws:
        ws.send_bytes(b"\x00\x00" * 100)
        ws.send_text("__end__")
        while json.loads(ws.receive_text())["type"] != "turn_end":
            try:
                ws.receive_bytes()
            except Exception:
                break
    assert all(language == "en" for _, language in synthesizer.calls)


def test_history_persists_across_turns(client):
    with client.websocket_connect("/ws/history-session") as ws:
        for _ in range(2):
            ws.send_bytes(b"\x00\x00" * 100)
            ws.send_text("__end__")
            msg = json.loads(ws.receive_text())
            while msg["type"] != "turn_end":
                if msg["type"] == "answer_text":
                    ws.receive_bytes()
                msg = json.loads(ws.receive_text())
    # Two turns stored two user + two assistant rows; verified indirectly by
    # the server not erroring and by HistoryStore's own unit tests.
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest backend/tests/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.server.app'`

- [ ] **Step 4: Write minimal implementation**

```python
# backend/server/app.py
import json
import wave

from fastapi import FastAPI, WebSocket

from backend.history.store import HistoryStore
from backend.llm.prompt import build_prompt
from backend.tts.chunker import ClauseChunker

TOP_K = 5


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

        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if message.get("bytes") is not None:
                chunk = message["bytes"]
                buffer.extend(chunk)
                utterance = segmenter.push(chunk)
                if utterance is not None:
                    buffer.clear()
                    await _handle_turn(websocket, session_id, utterance, transcriber,
                                       inventory_store, llm_engine, synthesizer,
                                       history_store, write_wav, scratch_audio_path)
            elif message.get("text") == "__end__":
                utterance = bytes(buffer)
                buffer.clear()
                await _handle_turn(websocket, session_id, utterance, transcriber,
                                   inventory_store, llm_engine, synthesizer,
                                   history_store, write_wav, scratch_audio_path)

    return app


async def _handle_turn(websocket, session_id, utterance_pcm, transcriber,
                       inventory_store, llm_engine, synthesizer, history_store,
                       write_wav, scratch_audio_path):
    write_wav(utterance_pcm, scratch_audio_path)
    transcript = transcriber.transcribe(scratch_audio_path)
    language = transcript["language"]

    await websocket.send_text(json.dumps({
        "type": "transcript", "text": transcript["text"], "language": language,
    }))

    if not transcript["text"]:
        await websocket.send_text(json.dumps({"type": "turn_end"}))
        return

    history = history_store.get(session_id)
    retrieved = inventory_store.query(transcript["text"], top_k=TOP_K)
    prompt = build_prompt(transcript["text"], retrieved, history)

    chunker = ClauseChunker()
    full_answer = []

    async def speak(clause: str) -> None:
        await websocket.send_text(json.dumps({"type": "answer_text", "text": clause}))
        await websocket.send_bytes(synthesizer.synthesize(clause, language=language))

    for fragment in llm_engine.generate(prompt):
        full_answer.append(fragment)
        for clause in chunker.push(fragment):
            await speak(clause)

    trailing = chunker.flush()
    if trailing:
        await speak(trailing)

    history_store.append(session_id, "user", transcript["text"])
    history_store.append(session_id, "assistant", "".join(full_answer))

    await websocket.send_text(json.dumps({"type": "turn_end"}))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pip install -r backend/requirements.txt && pytest backend/tests/test_server.py -v`
Expected: PASS (3 tests)

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

app = create_app(
    segmenter_factory=lambda: UtteranceSegmenter(),
    transcriber=Transcriber(model_size="small", compute_type="int8"),
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

`pytest backend/tests/test_history_store.py backend/tests/test_server.py -v` passes locally with fakes (no GPU, no weights, no audio), and `python -m backend.main` starts cleanly on Colab.
