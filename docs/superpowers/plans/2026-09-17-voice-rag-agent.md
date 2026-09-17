# Voice-to-Voice RAG Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a streaming voice-to-voice agent (web/mWeb UI to Colab-hosted backend via ngrok) that answers spoken questions about a synthetic 1000+ item inventory catalog using RAG, in English/Hindi/Marathi.

**Architecture:** Browser captures mic audio, streams over WebSocket through an ngrok tunnel to a FastAPI server running in a Colab notebook. The server pipelines VAD → STT (CPU) → RAG retrieval (Chroma) → LLM (GPU) → TTS (CPU), streaming audio back to the browser. Each pipeline stage is built and tested as an independent module before wiring into the server; the server is built and tested independently of the frontend before final integration.

**Tech Stack:** Python 3.10+, FastAPI, websockets, faster-whisper, intfloat/multilingual-e5-small (sentence-transformers), ChromaDB, llama-cpp-python (Qwen2.5-7B-Instruct GGUF Q4_K_M), Meta MMS-TTS (transformers), webrtcvad/Silero-VAD, SQLite, vanilla HTML/JS frontend, ngrok.

**Spec:** `docs/superpowers/specs/2026-09-17-voice-rag-agent-design.md`

## Global Constraints

- STT and TTS run on CPU only. Only the LLM runs on GPU. (spec: Constraints)
- Must support English, Hindi, Marathi input and output. (spec: Constraints)
- Chunking: one Chroma chunk per inventory item, no text splitting. (spec: RAG & inference details)
- Embedding model: `intfloat/multilingual-e5-small`. (spec: RAG & inference details)
- LLM served via llama.cpp / llama-cpp-python, GGUF Q4_K_M, Qwen2.5-7B-Instruct. (spec: RAG & inference details)
- RAG retrieval capped to top-k=3-5 listings per turn; prompt only carries essential fields. (spec: Cost / token optimization)
- Conversation history windowed to last N turns / token budget, persisted to SQLite on local Colab disk. (spec: Conversation history)
- Chroma persists to local Colab disk, re-embedded fresh each notebook boot — no cross-session durability required. (spec: Inventory data)
- ngrok URL is not stable across restarts; frontend must not hardcode it. (spec: Frontend)

---

## Phase 1: Inventory Data + RAG Retrieval Module

Fully independent of every other phase. Produces a Python module that, given a text query, returns top-k relevant inventory items. Testable standalone via pytest, no server, no GPU.

### Task 1.1: Synthetic catalog generator

**Files:**
- Create: `backend/data/generate_catalog.py`
- Create: `backend/data/catalog.json` (generated output, committed as a fixture)
- Test: `backend/tests/test_generate_catalog.py`

**Interfaces:**
- Produces: `generate_catalog(n: int, seed: int = 42) -> list[dict]`, each dict shaped `{"id": str, "name": str, "description": str, "price": float, "stock": int, "category": str}`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_generate_catalog.py
from backend.data.generate_catalog import generate_catalog

def test_generate_catalog_produces_requested_count():
    items = generate_catalog(n=50, seed=1)
    assert len(items) == 50

def test_generate_catalog_items_have_required_fields():
    items = generate_catalog(n=5, seed=1)
    for item in items:
        assert set(item.keys()) == {"id", "name", "description", "price", "stock", "category"}
        assert isinstance(item["price"], float)
        assert isinstance(item["stock"], int)

def test_generate_catalog_ids_unique():
    items = generate_catalog(n=200, seed=2)
    ids = [item["id"] for item in items]
    assert len(ids) == len(set(ids))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_generate_catalog.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.data.generate_catalog'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/data/generate_catalog.py
import random
import uuid

CATEGORIES = ["Electronics", "Home & Kitchen", "Apparel", "Books", "Sports", "Toys", "Grocery"]
ADJECTIVES = ["Compact", "Premium", "Portable", "Wireless", "Eco-Friendly", "Heavy-Duty", "Classic"]
NOUNS = ["Blender", "Backpack", "Headphones", "Lamp", "Notebook", "Sneakers", "Kettle", "Charger"]

def generate_catalog(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        adjective = rng.choice(ADJECTIVES)
        noun = rng.choice(NOUNS)
        category = rng.choice(CATEGORIES)
        name = f"{adjective} {noun}"
        items.append({
            "id": str(uuid.uuid4()),
            "name": name,
            "description": f"{name} in the {category} category. Durable build, "
                            f"well-reviewed, ships within 2 days.",
            "price": round(rng.uniform(5.0, 500.0), 2),
            "stock": rng.randint(0, 500),
            "category": category,
        })
    return items

if __name__ == "__main__":
    import json
    catalog = generate_catalog(n=1200, seed=42)
    with open("backend/data/catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_generate_catalog.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Generate the fixture catalog and commit**

```bash
python -m backend.data.generate_catalog
git add backend/data/generate_catalog.py backend/data/catalog.json backend/tests/test_generate_catalog.py
git commit -m "feat: add synthetic inventory catalog generator"
```

### Task 1.2: RAG retrieval module (Chroma + multilingual-e5-small)

**Files:**
- Create: `backend/rag/store.py`
- Create: `backend/rag/__init__.py`
- Test: `backend/tests/test_rag_store.py`
- Modify: `backend/requirements.txt` (create if absent)

**Interfaces:**
- Consumes: `generate_catalog` output shape from Task 1.1 (`list[dict]` with `id/name/description/price/stock/category`).
- Produces:
  - `class InventoryStore` with `__init__(self, persist_dir: str)`
  - `InventoryStore.load(self, items: list[dict]) -> None` — embeds and upserts all items
  - `InventoryStore.query(self, text: str, top_k: int = 5) -> list[dict]` — returns items (same shape as input, plus `"score": float`) ranked by relevance

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_rag_store.py
import shutil
import tempfile
import pytest
from backend.rag.store import InventoryStore

SAMPLE_ITEMS = [
    {"id": "1", "name": "Wireless Headphones", "description": "Wireless Headphones in the Electronics category. Noise cancelling.", "price": 89.99, "stock": 20, "category": "Electronics"},
    {"id": "2", "name": "Classic Sneakers", "description": "Classic Sneakers in the Apparel category. Comfortable everyday shoe.", "price": 45.0, "stock": 5, "category": "Apparel"},
    {"id": "3", "name": "Portable Blender", "description": "Portable Blender in the Home & Kitchen category. Great for smoothies.", "price": 30.5, "stock": 0, "category": "Home & Kitchen"},
]

@pytest.fixture
def store():
    tmp_dir = tempfile.mkdtemp()
    s = InventoryStore(persist_dir=tmp_dir)
    s.load(SAMPLE_ITEMS)
    yield s
    shutil.rmtree(tmp_dir, ignore_errors=True)

def test_query_returns_top_k(store):
    results = store.query("noise cancelling headphones", top_k=2)
    assert len(results) == 2

def test_query_returns_most_relevant_first(store):
    results = store.query("I want headphones", top_k=1)
    assert results[0]["id"] == "1"

def test_query_result_has_score(store):
    results = store.query("shoes", top_k=1)
    assert "score" in results[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_rag_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.rag.store'`

- [ ] **Step 3: Add dependencies**

```text
# backend/requirements.txt
chromadb==0.5.5
sentence-transformers==3.0.1
```

- [ ] **Step 4: Write minimal implementation**

```python
# backend/rag/store.py
import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"


class InventoryStore:
    def __init__(self, persist_dir: str):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection("inventory")
        self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def _embed(self, texts: list[str], is_query: bool) -> list[list[float]]:
        prefix = "query: " if is_query else "passage: "
        prefixed = [prefix + t for t in texts]
        return self._model.encode(prefixed, normalize_embeddings=True).tolist()

    def load(self, items: list[dict]) -> None:
        docs = [f"{item['name']}. {item['description']}" for item in items]
        embeddings = self._embed(docs, is_query=False)
        self._collection.upsert(
            ids=[item["id"] for item in items],
            documents=docs,
            embeddings=embeddings,
            metadatas=[
                {
                    "name": item["name"],
                    "price": item["price"],
                    "stock": item["stock"],
                    "category": item["category"],
                }
                for item in items
            ],
        )

    def query(self, text: str, top_k: int = 5) -> list[dict]:
        embedding = self._embed([text], is_query=True)[0]
        result = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        items = []
        for i, item_id in enumerate(result["ids"][0]):
            meta = result["metadatas"][0][i]
            items.append({
                "id": item_id,
                "name": meta["name"],
                "description": result["documents"][0][i],
                "price": meta["price"],
                "stock": meta["stock"],
                "category": meta["category"],
                "score": result["distances"][0][i],
            })
        return items
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pip install -r backend/requirements.txt && pytest backend/tests/test_rag_store.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add backend/rag backend/tests/test_rag_store.py backend/requirements.txt
git commit -m "feat: add Chroma-backed RAG retrieval module with multilingual embeddings"
```

### Task 1.3: Load full catalog into persistent Chroma store (CLI script)

**Files:**
- Create: `backend/rag/load_catalog.py`

**Interfaces:**
- Consumes: `InventoryStore` from Task 1.2, `backend/data/catalog.json` from Task 1.1.
- Produces: populated Chroma store at `backend/rag/chroma_db/` (gitignored, regenerated on each run).

- [ ] **Step 1: Write the script**

```python
# backend/rag/load_catalog.py
import json
from backend.rag.store import InventoryStore

CATALOG_PATH = "backend/data/catalog.json"
PERSIST_DIR = "backend/rag/chroma_db"

if __name__ == "__main__":
    with open(CATALOG_PATH) as f:
        items = json.load(f)
    store = InventoryStore(persist_dir=PERSIST_DIR)
    store.load(items)
    print(f"Loaded {len(items)} items into {PERSIST_DIR}")
```

- [ ] **Step 2: Run it and verify manually**

Run: `python -m backend.rag.load_catalog`
Expected: prints `Loaded 1200 items into backend/rag/chroma_db`

Run: `python -c "from backend.rag.store import InventoryStore; s = InventoryStore('backend/rag/chroma_db'); print(s.query('cheap sneakers', top_k=3))"`
Expected: prints 3 items, at least one sneaker-related, each with a `score`.

- [ ] **Step 3: Gitignore the generated store and commit the script**

```bash
echo "backend/rag/chroma_db/" >> .gitignore
git add backend/rag/load_catalog.py .gitignore
git commit -m "feat: add script to load full catalog into Chroma"
```

**Phase 1 done when:** `pytest backend/tests/test_generate_catalog.py backend/tests/test_rag_store.py -v` passes, and the manual query above returns sensible results. No server, no LLM, no audio involved.

---

## Phase 2: LLM Answer Generation Module

Independent of Phase 1's storage internals — only depends on the *shape* of retrieved items (a `list[dict]`), which is stubbed directly in this phase's tests. Testable standalone with a text prompt in, text out. Requires GPU (Colab) to run for real; unit tests use a stub so they run without the actual model loaded.

### Task 2.1: Prompt builder (pure function, no model needed)

**Files:**
- Create: `backend/llm/prompt.py`
- Create: `backend/llm/__init__.py`
- Test: `backend/tests/test_prompt.py`

**Interfaces:**
- Consumes: retrieved items shaped like Task 1.2's `query()` output (`id/name/description/price/stock/category/score`); history entries shaped `{"role": "user"|"assistant", "text": str}`.
- Produces: `build_prompt(user_text: str, retrieved: list[dict], history: list[dict], max_history_turns: int = 6) -> str`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_prompt.py
from backend.llm.prompt import build_prompt

RETRIEVED = [
    {"id": "1", "name": "Wireless Headphones", "description": "...", "price": 89.99, "stock": 20, "category": "Electronics", "score": 0.1},
]

def test_prompt_includes_user_text():
    prompt = build_prompt("do you have headphones", RETRIEVED, [])
    assert "do you have headphones" in prompt

def test_prompt_includes_retrieved_item_fields():
    prompt = build_prompt("headphones?", RETRIEVED, [])
    assert "Wireless Headphones" in prompt
    assert "89.99" in prompt
    assert "20" in prompt

def test_prompt_excludes_full_description_field_name_noise():
    prompt = build_prompt("headphones?", [], [])
    assert "No matching listings" in prompt or "no items" in prompt.lower()

def test_prompt_windows_history_to_max_turns():
    history = [{"role": "user", "text": f"turn {i}"} for i in range(10)]
    prompt = build_prompt("latest question", [], history, max_history_turns=2)
    assert "turn 9" in prompt
    assert "turn 0" not in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/llm/prompt.py

SYSTEM_PROMPT = (
    "You are a phone assistant for a store. Answer only using the listings "
    "given below. If no listings are given, say there are no matching "
    "items in stock. Be brief, spoken-language style, one or two sentences."
)


def _format_items(items: list[dict]) -> str:
    if not items:
        return "No matching listings found."
    lines = []
    for item in items:
        lines.append(
            f"- {item['name']}: ${item['price']}, stock {item['stock']}, "
            f"category {item['category']}"
        )
    return "\n".join(lines)


def _format_history(history: list[dict], max_turns: int) -> str:
    windowed = history[-max_turns:] if max_turns else history
    lines = [f"{turn['role']}: {turn['text']}" for turn in windowed]
    return "\n".join(lines)


def build_prompt(user_text: str, retrieved: list[dict], history: list[dict],
                  max_history_turns: int = 6) -> str:
    parts = [
        SYSTEM_PROMPT,
        "",
        "Relevant listings:",
        _format_items(retrieved),
    ]
    history_text = _format_history(history, max_history_turns)
    if history_text:
        parts += ["", "Conversation so far:", history_text]
    parts += ["", f"user: {user_text}", "assistant:"]
    return "\n".join(parts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_prompt.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/llm/prompt.py backend/llm/__init__.py backend/tests/test_prompt.py
git commit -m "feat: add token-budgeted LLM prompt builder"
```

### Task 2.2: LLM generation wrapper (llama.cpp)

**Files:**
- Create: `backend/llm/generate.py`
- Test: `backend/tests/test_generate_manual.py` (manual/integration test, skipped by default — needs the actual GGUF weights)
- Modify: `backend/requirements.txt`

**Interfaces:**
- Consumes: `build_prompt()` output string from Task 2.1.
- Produces:
  - `class LLMEngine` with `__init__(self, model_path: str, n_gpu_layers: int = -1)`
  - `LLMEngine.generate(self, prompt: str, stop: list[str] = None, max_tokens: int = 200) -> Iterator[str]` — yields text chunks as they're generated (streaming)

- [ ] **Step 1: Add dependency**

```text
# backend/requirements.txt (append)
llama-cpp-python==0.2.90
```

- [ ] **Step 2: Write the implementation**

```python
# backend/llm/generate.py
from llama_cpp import Llama


class LLMEngine:
    def __init__(self, model_path: str, n_gpu_layers: int = -1):
        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,  # -1 = offload all layers to GPU
            n_ctx=4096,
            verbose=False,
        )

    def generate(self, prompt: str, stop: list[str] = None, max_tokens: int = 200):
        stop = stop or ["user:", "\nuser"]
        stream = self._llm(
            prompt,
            max_tokens=max_tokens,
            stop=stop,
            stream=True,
            temperature=0.3,
        )
        for chunk in stream:
            yield chunk["choices"][0]["text"]
```

- [ ] **Step 3: Write a manual integration test (requires downloaded GGUF weights, so marked skip by default)**

```python
# backend/tests/test_generate_manual.py
import os
import pytest
from backend.llm.generate import LLMEngine

MODEL_PATH = os.environ.get("LLM_GGUF_PATH", "")

@pytest.mark.skipif(not MODEL_PATH, reason="Set LLM_GGUF_PATH to run this manual test")
def test_generate_produces_nonempty_text():
    engine = LLMEngine(model_path=MODEL_PATH)
    chunks = list(engine.generate("assistant:", max_tokens=20))
    assert "".join(chunks).strip() != ""
```

- [ ] **Step 4: Run test to verify it's skipped without weights, and run for real once weights are downloaded (Colab step, not local)**

Run: `pytest backend/tests/test_generate_manual.py -v`
Expected (no weights set): `SKIPPED`

On Colab, after downloading `Qwen2.5-7B-Instruct-Q4_K_M.gguf` (e.g. from a HuggingFace GGUF repo) to local disk:
Run: `LLM_GGUF_PATH=/content/qwen2.5-7b-instruct-q4_k_m.gguf pytest backend/tests/test_generate_manual.py -v`
Expected: PASS, prints generated text if run with `-s`

- [ ] **Step 5: Commit**

```bash
git add backend/llm/generate.py backend/tests/test_generate_manual.py backend/requirements.txt
git commit -m "feat: add streaming llama.cpp LLM generation wrapper"
```

**Phase 2 done when:** `test_prompt.py` passes locally (no GPU needed), and `test_generate_manual.py` passes on Colab with real weights. Phase 1 and Phase 2 have zero import dependency on each other — they only need to agree on the retrieved-item dict shape, which is fixed by the spec and tested independently on each side.

---

## Phase 3: STT Module (faster-whisper, CPU)

Independent of Phases 1-2. Input: a WAV file or raw PCM bytes. Output: transcript + detected language code. Runs entirely on CPU, testable locally without Colab.

### Task 3.1: Batch transcription wrapper

**Files:**
- Create: `backend/stt/transcribe.py`
- Create: `backend/stt/__init__.py`
- Create: `backend/tests/fixtures/hello_english.wav` (short recorded/generated sample, see Step 1)
- Test: `backend/tests/test_transcribe.py`
- Modify: `backend/requirements.txt`

**Interfaces:**
- Produces:
  - `class Transcriber` with `__init__(self, model_size: str = "small", compute_type: str = "int8")`
  - `Transcriber.transcribe(self, audio_path: str) -> dict` returning `{"text": str, "language": str}`

- [ ] **Step 1: Add dependency and generate a test fixture**

```text
# backend/requirements.txt (append)
faster-whisper==1.0.3
```

Generate a tiny test WAV using any TTS you have available, or record 2 seconds saying "hello, do you have headphones" and save as `backend/tests/fixtures/hello_english.wav` (16kHz mono). This fixture is a one-time manual asset, not generated by code.

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_transcribe.py
import os
import pytest
from backend.stt.transcribe import Transcriber

FIXTURE = "backend/tests/fixtures/hello_english.wav"

@pytest.mark.skipif(not os.path.exists(FIXTURE), reason="Add the fixture WAV first")
def test_transcribe_returns_text_and_language():
    t = Transcriber(model_size="small", compute_type="int8")
    result = t.transcribe(FIXTURE)
    assert "headphones" in result["text"].lower()
    assert result["language"] == "en"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest backend/tests/test_transcribe.py -v`
Expected: FAIL with `ModuleNotFoundError` (or SKIPPED if fixture missing — add fixture before continuing)

- [ ] **Step 4: Write minimal implementation**

```python
# backend/stt/transcribe.py
from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model_size: str = "small", compute_type: str = "int8"):
        self._model = WhisperModel(model_size, device="cpu", compute_type=compute_type)

    def transcribe(self, audio_path: str) -> dict:
        segments, info = self._model.transcribe(audio_path)
        text = " ".join(segment.text.strip() for segment in segments)
        return {"text": text.strip(), "language": info.language}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pip install -r backend/requirements.txt && pytest backend/tests/test_transcribe.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/stt backend/tests/test_transcribe.py backend/tests/fixtures/hello_english.wav backend/requirements.txt
git commit -m "feat: add CPU faster-whisper transcription wrapper"
```

### Task 3.2: Streaming VAD segmenter

**Files:**
- Create: `backend/stt/vad.py`
- Test: `backend/tests/test_vad.py`
- Modify: `backend/requirements.txt`

**Interfaces:**
- Produces:
  - `class UtteranceSegmenter` with `__init__(self, sample_rate: int = 16000, silence_ms: int = 600)`
  - `UtteranceSegmenter.push(self, pcm_chunk: bytes) -> bytes | None` — feed raw PCM chunks continuously; returns the accumulated utterance bytes when silence-triggered end-of-speech is detected, otherwise `None`.

- [ ] **Step 1: Add dependency**

```text
# backend/requirements.txt (append)
webrtcvad==2.0.10
```

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_vad.py
from backend.stt.vad import UtteranceSegmenter

def _silence_frame(sample_rate=16000, ms=30):
    n_samples = int(sample_rate * ms / 1000)
    return (b"\x00\x00" * n_samples)

def _tone_frame(sample_rate=16000, ms=30):
    import struct, math
    n_samples = int(sample_rate * ms / 1000)
    samples = [int(3000 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(n_samples)]
    return struct.pack("<%dh" % n_samples, *samples)

def test_segmenter_returns_none_during_speech():
    seg = UtteranceSegmenter(silence_ms=300)
    for _ in range(5):
        result = seg.push(_tone_frame())
    assert result is None

def test_segmenter_returns_bytes_after_trailing_silence():
    seg = UtteranceSegmenter(silence_ms=90)
    for _ in range(5):
        seg.push(_tone_frame())
    result = None
    for _ in range(6):
        result = seg.push(_silence_frame())
        if result is not None:
            break
    assert result is not None
    assert len(result) > 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest backend/tests/test_vad.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Write minimal implementation**

```python
# backend/stt/vad.py
import webrtcvad


class UtteranceSegmenter:
    def __init__(self, sample_rate: int = 16000, silence_ms: int = 600, frame_ms: int = 30):
        self._vad = webrtcvad.Vad(2)
        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._silence_frames_needed = silence_ms // frame_ms
        self._buffer = bytearray()
        self._silence_run = 0
        self._speech_started = False

    def push(self, pcm_chunk: bytes) -> bytes | None:
        is_speech = self._vad.is_speech(pcm_chunk, self._sample_rate)
        if is_speech:
            self._speech_started = True
            self._silence_run = 0
            self._buffer.extend(pcm_chunk)
            return None

        if not self._speech_started:
            return None

        self._buffer.extend(pcm_chunk)
        self._silence_run += 1
        if self._silence_run >= self._silence_frames_needed:
            utterance = bytes(self._buffer)
            self._buffer = bytearray()
            self._silence_run = 0
            self._speech_started = False
            return utterance
        return None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest backend/tests/test_vad.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add backend/stt/vad.py backend/tests/test_vad.py backend/requirements.txt
git commit -m "feat: add VAD-based utterance segmenter for streaming STT"
```

**Phase 3 done when:** both test files pass on CPU, no GPU, no dependency on Phases 1/2/4.

---

## Phase 4: TTS Module (Meta MMS-TTS, CPU)

Independent of every other phase. Input: text + language code. Output: audio bytes. Runs on CPU.

### Task 4.1: Multi-language synthesis wrapper

**Files:**
- Create: `backend/tts/synthesize.py`
- Create: `backend/tts/__init__.py`
- Test: `backend/tests/test_synthesize.py`
- Modify: `backend/requirements.txt`

**Interfaces:**
- Produces:
  - `class Synthesizer` with `__init__(self, language_model_map: dict[str, str] = None)` — maps language code (`"en"`, `"hi"`, `"mr"`) to MMS-TTS checkpoint name (defaults to `facebook/mms-tts-eng`, `facebook/mms-tts-hin`, `facebook/mms-tts-mar`)
  - `Synthesizer.synthesize(self, text: str, language: str = "en") -> bytes` — returns WAV audio bytes

- [ ] **Step 1: Add dependency**

```text
# backend/requirements.txt (append)
transformers==4.44.2
torch==2.4.0
scipy==1.14.0
```

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_synthesize.py
from backend.tts.synthesize import Synthesizer

def test_synthesize_returns_nonempty_wav_bytes_english():
    synth = Synthesizer()
    audio = synth.synthesize("hello, we have that in stock", language="en")
    assert len(audio) > 1000
    assert audio[:4] == b"RIFF"

def test_synthesize_returns_nonempty_wav_bytes_hindi():
    synth = Synthesizer()
    audio = synth.synthesize("नमस्ते, यह स्टॉक में है", language="hi")
    assert len(audio) > 1000
    assert audio[:4] == b"RIFF"

def test_synthesize_unknown_language_raises():
    synth = Synthesizer()
    try:
        synth.synthesize("test", language="xx")
        assert False, "expected ValueError"
    except ValueError:
        pass
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest backend/tests/test_synthesize.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Write minimal implementation**

```python
# backend/tts/synthesize.py
import io
import scipy.io.wavfile
from transformers import VitsModel, AutoTokenizer

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
            self._models[language] = VitsModel.from_pretrained(checkpoint)
            self._tokenizers[language] = AutoTokenizer.from_pretrained(checkpoint)
        return self._models[language], self._tokenizers[language]

    def synthesize(self, text: str, language: str = "en") -> bytes:
        model, tokenizer = self._load(language)
        inputs = tokenizer(text, return_tensors="pt")
        with_no_grad_output = model(**inputs).waveform
        waveform = with_no_grad_output.squeeze().detach().numpy()

        buffer = io.BytesIO()
        scipy.io.wavfile.write(buffer, rate=model.config.sampling_rate, data=waveform)
        return buffer.getvalue()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pip install -r backend/requirements.txt && pytest backend/tests/test_synthesize.py -v`
Expected: PASS (3 tests) — first run downloads MMS checkpoints, slower

- [ ] **Step 6: Commit**

```bash
git add backend/tts backend/tests/test_synthesize.py backend/requirements.txt
git commit -m "feat: add multi-language MMS-TTS CPU synthesis wrapper"
```

### Task 4.2: Incremental clause chunker (for streaming TTS from partial LLM output)

**Files:**
- Create: `backend/tts/chunker.py`
- Test: `backend/tests/test_chunker.py`

**Interfaces:**
- Produces:
  - `class ClauseChunker` with `__init__(self)`
  - `ClauseChunker.push(self, text_fragment: str) -> list[str]` — feed LLM token/text fragments as they arrive; returns zero or more complete clauses (split on `.`, `!`, `?`, `,` followed by whitespace) ready to send to TTS
  - `ClauseChunker.flush(self) -> str | None` — call at end of generation to get any trailing partial clause

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_chunker.py
from backend.tts.chunker import ClauseChunker

def test_chunker_emits_clause_on_sentence_end():
    chunker = ClauseChunker()
    result = chunker.push("We have that in stock. ")
    assert result == ["We have that in stock."]

def test_chunker_buffers_incomplete_clause():
    chunker = ClauseChunker()
    result = chunker.push("We have that")
    assert result == []

def test_chunker_flush_returns_remaining_text():
    chunker = ClauseChunker()
    chunker.push("We have that")
    remaining = chunker.flush()
    assert remaining == "We have that"

def test_chunker_flush_returns_none_when_empty():
    chunker = ClauseChunker()
    chunker.push("Done.")
    assert chunker.flush() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError`

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
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/tts/chunker.py backend/tests/test_chunker.py
git commit -m "feat: add clause chunker for incremental streaming TTS"
```

**Phase 4 done when:** all TTS tests pass on CPU, zero dependency on Phases 1-3.

---

## Phase 5: Backend Orchestration Server (FastAPI WebSocket)

Depends on Phases 1-4 as library imports (not on their internal implementation — only their tested public interfaces above). Independently testable via a Python WebSocket test client, without any frontend.

### Task 5.1: Conversation history store (SQLite)

**Files:**
- Create: `backend/history/store.py`
- Create: `backend/history/__init__.py`
- Test: `backend/tests/test_history_store.py`

**Interfaces:**
- Produces:
  - `class HistoryStore` with `__init__(self, db_path: str)`
  - `HistoryStore.append(self, session_id: str, role: str, text: str) -> None`
  - `HistoryStore.get(self, session_id: str, max_turns: int = 6) -> list[dict]` — returns `[{"role": str, "text": str}, ...]` oldest-to-newest, limited to last `max_turns`

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
    s = HistoryStore(db_path=path)
    yield s
    os.remove(path)

def test_append_and_get(store):
    store.append("session1", "user", "hello")
    store.append("session1", "assistant", "hi there")
    history = store.get("session1")
    assert history == [
        {"role": "user", "text": "hello"},
        {"role": "assistant", "text": "hi there"},
    ]

def test_sessions_are_isolated(store):
    store.append("session1", "user", "a")
    store.append("session2", "user", "b")
    assert store.get("session1") == [{"role": "user", "text": "a"}]

def test_get_windows_to_max_turns(store):
    for i in range(10):
        store.append("session1", "user", f"turn {i}")
    history = store.get("session1", max_turns=3)
    assert [h["text"] for h in history] == ["turn 7", "turn 8", "turn 9"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_history_store.py -v`
Expected: FAIL with `ModuleNotFoundError`

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
            "SELECT role, text FROM history WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, max_turns),
        )
        rows = cursor.fetchall()
        rows.reverse()
        return [{"role": role, "text": text} for role, text in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_history_store.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/history backend/tests/test_history_store.py
git commit -m "feat: add SQLite conversation history store"
```

### Task 5.2: WebSocket orchestration server

**Files:**
- Create: `backend/server/app.py`
- Create: `backend/server/__init__.py`
- Test: `backend/tests/test_server.py`
- Modify: `backend/requirements.txt`

**Interfaces:**
- Consumes: `InventoryStore` (1.2), `build_prompt` (2.1), `LLMEngine` (2.2), `UtteranceSegmenter` (3.2), `Transcriber` (3.1), `Synthesizer` (4.1), `ClauseChunker` (4.2), `HistoryStore` (5.1).
- Produces: FastAPI `app` object with a `/ws/{session_id}` WebSocket endpoint. Protocol:
  - Client sends binary frames: raw 16kHz mono PCM chunks.
  - Client sends text frame `"__end__"` to force end-of-turn (used by tests instead of real silence).
  - Server sends text frames: `{"type": "transcript", "text": str, "language": str}`, then `{"type": "answer_text", "text": str}` (one message per clause), then binary frames of WAV audio per clause, then `{"type": "turn_end"}`.

- [ ] **Step 1: Add dependency**

```text
# backend/requirements.txt (append)
fastapi==0.112.0
uvicorn==0.30.5
websockets==12.0
httpx==0.27.0
```

- [ ] **Step 2: Write the failing test (uses fakes for the heavy models, real HistoryStore/prompt builder)**

```python
# backend/tests/test_server.py
import json
import tempfile
import pytest
from fastapi.testclient import TestClient
from backend.server.app import create_app


class FakeSegmenter:
    def push(self, chunk):
        return None  # server test drives end-of-turn via "__end__" text frame


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
    def synthesize(self, text, language="en"):
        return b"RIFFfakewavbytes"


@pytest.fixture
def client():
    fd_db = tempfile.mktemp(suffix=".db")
    fd_audio = tempfile.mktemp(suffix=".wav")
    app = create_app(
        segmenter_factory=lambda: FakeSegmenter(),
        transcriber=FakeTranscriber(),
        inventory_store=FakeStore(),
        llm_engine=FakeLLM(),
        synthesizer=FakeSynthesizer(),
        history_db_path=fd_db,
        scratch_audio_path=fd_audio,
    )
    return TestClient(app)


def test_full_turn_over_websocket(client):
    with client.websocket_connect("/ws/test-session") as ws:
        ws.send_bytes(b"\x00\x00" * 100)
        ws.send_text("__end__")

        transcript_msg = json.loads(ws.receive_text())
        assert transcript_msg == {"type": "transcript", "text": "do you have headphones", "language": "en"}

        answer_msg = json.loads(ws.receive_text())
        assert answer_msg["type"] == "answer_text"

        audio_msg = ws.receive_bytes()
        assert audio_msg.startswith(b"RIFF")

        end_msg = json.loads(ws.receive_text())
        assert end_msg == {"type": "turn_end"}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest backend/tests/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Write minimal implementation**

```python
# backend/server/app.py
import json
import wave
from fastapi import FastAPI, WebSocket

from backend.llm.prompt import build_prompt
from backend.tts.chunker import ClauseChunker
from backend.history.store import HistoryStore


def create_app(segmenter_factory, transcriber, inventory_store, llm_engine,
               synthesizer, history_db_path: str, scratch_audio_path: str):
    app = FastAPI()
    history_store = HistoryStore(db_path=history_db_path)

    def _write_wav(pcm_bytes: bytes, path: str):
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
            if "bytes" in message and message["bytes"] is not None:
                chunk = message["bytes"]
                buffer.extend(chunk)
                utterance = segmenter.push(chunk)
                if utterance is not None:
                    await _handle_turn(websocket, session_id, utterance,
                                        transcriber, inventory_store, llm_engine,
                                        synthesizer, history_store, _write_wav,
                                        scratch_audio_path)
                    buffer.clear()
            elif message.get("text") == "__end__":
                utterance = bytes(buffer)
                buffer.clear()
                await _handle_turn(websocket, session_id, utterance,
                                    transcriber, inventory_store, llm_engine,
                                    synthesizer, history_store, _write_wav,
                                    scratch_audio_path)
            elif message.get("type") == "websocket.disconnect":
                break

    return app


async def _handle_turn(websocket, session_id, utterance_pcm, transcriber,
                        inventory_store, llm_engine, synthesizer, history_store,
                        write_wav, scratch_audio_path):
    write_wav(utterance_pcm, scratch_audio_path)
    transcript = transcriber.transcribe(scratch_audio_path)
    await websocket.send_text(json.dumps({
        "type": "transcript", "text": transcript["text"], "language": transcript["language"],
    }))

    history = history_store.get(session_id)
    retrieved = inventory_store.query(transcript["text"], top_k=5)
    prompt = build_prompt(transcript["text"], retrieved, history)

    chunker = ClauseChunker()
    full_answer = []
    for fragment in llm_engine.generate(prompt):
        full_answer.append(fragment)
        for clause in chunker.push(fragment):
            await websocket.send_text(json.dumps({"type": "answer_text", "text": clause}))
            audio = synthesizer.synthesize(clause, language=transcript["language"])
            await websocket.send_bytes(audio)
    trailing = chunker.flush()
    if trailing:
        await websocket.send_text(json.dumps({"type": "answer_text", "text": trailing}))
        audio = synthesizer.synthesize(trailing, language=transcript["language"])
        await websocket.send_bytes(audio)

    history_store.append(session_id, "user", transcript["text"])
    history_store.append(session_id, "assistant", "".join(full_answer))

    await websocket.send_text(json.dumps({"type": "turn_end"}))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pip install -r backend/requirements.txt && pytest backend/tests/test_server.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/server backend/tests/test_server.py backend/requirements.txt
git commit -m "feat: add WebSocket orchestration server wiring STT/RAG/LLM/TTS/history"
```

### Task 5.3: Real-model wiring script (Colab entrypoint)

**Files:**
- Create: `backend/main.py`

**Interfaces:**
- Consumes: `create_app` from Task 5.2, real implementations from Phases 1-4.
- Produces: a runnable script that starts the real server (not fakes) on port 8000.

- [ ] **Step 1: Write the script**

```python
# backend/main.py
import uvicorn
from backend.rag.store import InventoryStore
from backend.llm.generate import LLMEngine
from backend.stt.transcribe import Transcriber
from backend.stt.vad import UtteranceSegmenter
from backend.tts.synthesize import Synthesizer
from backend.server.app import create_app

LLM_GGUF_PATH = "/content/qwen2.5-7b-instruct-q4_k_m.gguf"  # set for your Colab session

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

- [ ] **Step 2: Run manually on Colab (after Phase 1 catalog is loaded, Phase 2 GGUF downloaded)**

Run: `python -m backend.main` in a Colab cell (backgrounded), then in another cell start ngrok pointing at port 8000.
Expected: server starts without error, ngrok prints a public `wss://...` URL.

- [ ] **Step 3: Commit**

```bash
git add backend/main.py
git commit -m "feat: add real-model Colab entrypoint wiring all pipeline stages"
```

**Phase 5 done when:** `pytest backend/tests/test_server.py -v` passes locally with fakes (no GPU/models needed), and `backend/main.py` runs successfully on Colab with real models, reachable through ngrok with a raw WebSocket client (e.g. `websocat` or a short Python script) sending PCM + `"__end__"`.

---

## Phase 6: Frontend Web/mWeb UI

Independent of Phase 5's real models — built and tested against a lightweight local mock WebSocket server that mimics the Phase 5.2 protocol. Wired to the real Colab backend only in Phase 7.

### Task 6.1: Mock WebSocket server for frontend development

**Files:**
- Create: `frontend/mock_server.py`

**Interfaces:**
- Produces: a standalone script, port 8001, implementing the same `/ws/{session_id}` protocol as Phase 5.2 (transcript → answer_text → audio bytes → turn_end), with canned responses.

- [ ] **Step 1: Write the mock server**

```python
# frontend/mock_server.py
import asyncio
import json
import wave
import io
import struct
from fastapi import FastAPI, WebSocket
import uvicorn

app = FastAPI()


def _fake_wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<100h", *([1000] * 100)))
    return buffer.getvalue()


@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    while True:
        message = await websocket.receive()
        if message.get("text") == "__end__" or "bytes" in message:
            if message.get("text") != "__end__":
                continue
            await websocket.send_text(json.dumps({"type": "transcript", "text": "do you have headphones", "language": "en"}))
            await asyncio.sleep(0.3)
            await websocket.send_text(json.dumps({"type": "answer_text", "text": "Yes, we have Wireless Headphones in stock."}))
            await websocket.send_bytes(_fake_wav_bytes())
            await websocket.send_text(json.dumps({"type": "turn_end"}))
        elif message.get("type") == "websocket.disconnect":
            break

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

- [ ] **Step 2: Run it manually**

Run: `python -m frontend.mock_server`
Expected: server starts on port 8001, no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/mock_server.py
git commit -m "feat: add mock WebSocket server for frontend-only development"
```

### Task 6.2: Web/mWeb UI page

**Files:**
- Create: `frontend/index.html`
- Create: `frontend/app.js`
- Create: `frontend/style.css`

**Interfaces:**
- Consumes: the WebSocket protocol from Task 6.1/Phase 5.2 (`transcript` / `answer_text` / binary audio / `turn_end` messages).
- Produces: a static page, mobile-responsive, with a backend-URL config field, mic toggle, live transcript, connection status.

- [ ] **Step 1: Write the HTML shell**

```html
<!-- frontend/index.html -->
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Voice RAG Agent</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <main>
    <h1>Voice Inventory Assistant</h1>

    <label for="backend-url">Backend WebSocket URL</label>
    <input id="backend-url" type="text" placeholder="wss://xxxx.ngrok-free.app/ws/session1" />
    <button id="connect-btn">Connect</button>
    <span id="status">disconnected</span>

    <button id="mic-btn" disabled>🎤 Hold to talk</button>

    <section id="transcript-log" aria-live="polite"></section>
  </main>
  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write the client logic**

```javascript
// frontend/app.js
let ws = null;
let audioContext = null;
let mediaStream = null;
let processor = null;

const statusEl = document.getElementById("status");
const logEl = document.getElementById("transcript-log");
const micBtn = document.getElementById("mic-btn");
const connectBtn = document.getElementById("connect-btn");
const urlInput = document.getElementById("backend-url");

function log(role, text) {
  const p = document.createElement("p");
  p.textContent = `${role}: ${text}`;
  logEl.appendChild(p);
}

connectBtn.addEventListener("click", () => {
  const url = urlInput.value.trim();
  if (!url) return;
  ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    statusEl.textContent = "connected";
    micBtn.disabled = false;
  };
  ws.onclose = () => {
    statusEl.textContent = "disconnected";
    micBtn.disabled = true;
  };
  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      const msg = JSON.parse(event.data);
      if (msg.type === "transcript") log("you", msg.text);
      if (msg.type === "answer_text") log("agent", msg.text);
      if (msg.type === "turn_end") statusEl.textContent = "connected (turn done)";
    } else {
      playAudio(event.data);
    }
  };
});

function playAudio(arrayBuffer) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  ctx.decodeAudioData(arrayBuffer, (buffer) => {
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.start();
  });
}

async function startMic() {
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: 16000 } });
  audioContext = new AudioContext({ sampleRate: 16000 });
  const source = audioContext.createMediaStreamSource(mediaStream);
  processor = audioContext.createScriptProcessor(4096, 1, 1);

  processor.onaudioprocess = (e) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const input = e.inputBuffer.getChannelData(0);
    const pcm16 = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
      pcm16[i] = Math.max(-32768, Math.min(32767, input[i] * 32768));
    }
    ws.send(pcm16.buffer);
  };

  source.connect(processor);
  processor.connect(audioContext.destination);
}

function stopMic() {
  if (processor) processor.disconnect();
  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
  if (ws && ws.readyState === WebSocket.OPEN) ws.send("__end__");
}

micBtn.addEventListener("mousedown", startMic);
micBtn.addEventListener("touchstart", startMic);
micBtn.addEventListener("mouseup", stopMic);
micBtn.addEventListener("touchend", stopMic);
```

- [ ] **Step 3: Write minimal responsive styling**

```css
/* frontend/style.css */
body {
  font-family: system-ui, sans-serif;
  max-width: 480px;
  margin: 0 auto;
  padding: 16px;
}
input, button {
  width: 100%;
  padding: 10px;
  margin: 6px 0;
  font-size: 16px;
}
#mic-btn {
  background: #2563eb;
  color: white;
  border: none;
  border-radius: 8px;
}
#mic-btn:disabled {
  background: #94a3b8;
}
#transcript-log {
  margin-top: 16px;
  border-top: 1px solid #e2e8f0;
  padding-top: 8px;
}
```

- [ ] **Step 4: Manual test against the mock server**

Run: `python -m frontend.mock_server` in one terminal, then serve the frontend statically: `python -m http.server 5500 --directory frontend` in another.
Open `http://localhost:5500/index.html` on desktop and on a phone on the same network (`http://<lan-ip>:5500`), enter `ws://<lan-ip>:8001/ws/session1`, connect, hold mic button, release.
Expected: transcript and canned answer text appear, a short tone plays.

- [ ] **Step 5: Commit**

```bash
git add frontend/index.html frontend/app.js frontend/style.css
git commit -m "feat: add mobile-responsive web voice UI"
```

**Phase 6 done when:** the manual test above works against the mock server on both desktop and mobile browser, with zero code dependency on the real backend.

---

## Phase 7: End-to-End Integration (Colab + ngrok + real frontend)

Wires Phase 5's real backend (Task 5.3) to Phase 6's frontend. This is glue and configuration, not new modules — each side was already independently verified.

### Task 7.1: Colab notebook cells for full startup

**Files:**
- Create: `backend/colab_notebook.md` (documented cell-by-cell instructions, since the actual `.ipynb` is created interactively in Colab)

**Interfaces:**
- Consumes: `backend/main.py` (5.3), `backend/rag/load_catalog.py` (1.3).

- [ ] **Step 1: Write the notebook instructions**

```markdown
# backend/colab_notebook.md

Cell 1 — clone repo and install deps:
    !git clone <repo-url> plivio-demo
    %cd plivio-demo
    !pip install -r backend/requirements.txt -q

Cell 2 — download the GGUF model:
    !wget -O /content/qwen2.5-7b-instruct-q4_k_m.gguf <gguf-download-url>

Cell 3 — load the catalog into Chroma:
    !python -m backend.rag.load_catalog

Cell 4 — install and configure ngrok:
    !pip install pyngrok -q
    from pyngrok import ngrok
    ngrok.set_auth_token("<your-ngrok-token>")

Cell 5 — start the server and tunnel:
    import subprocess, time
    from pyngrok import ngrok
    subprocess.Popen(["python", "-m", "backend.main"])
    time.sleep(5)
    public_url = ngrok.connect(8000, "http")
    ws_url = public_url.replace("https://", "wss://") + "/ws/session1"
    print("WebSocket URL for the frontend:", ws_url)
```

- [ ] **Step 2: Manually run through all 5 cells in a real Colab notebook**

Expected: Cell 5 prints a `wss://...ngrok-free.app/ws/session1` URL, server responds to a raw WebSocket ping.

- [ ] **Step 3: Commit**

```bash
git add backend/colab_notebook.md
git commit -m "docs: add Colab notebook startup instructions"
```

### Task 7.2: Full end-to-end manual test

**Files:** none (manual verification only)

- [ ] **Step 1: Start the Colab backend** per Task 7.1, copy the printed `wss://` URL.
- [ ] **Step 2: Serve the real frontend** (not the mock): `python -m http.server 5500 --directory frontend`.
- [ ] **Step 3: Open the page** on a phone browser, paste the `wss://` URL into the backend-URL field, connect.
- [ ] **Step 4: Hold the mic button, ask** "do you have wireless headphones", release.
- [ ] **Step 5: Verify:** transcript appears, an answer referencing an actual catalog item appears, spoken audio plays back.
- [ ] **Step 6: Repeat in Hindi and Marathi** to confirm multilingual STT/TTS round-trip.
- [ ] **Step 7: Ask a follow-up question** referencing the prior turn (e.g. "what about in blue?") to confirm conversation history is actually used.

**Phase 7 done when:** all 7 manual steps above succeed against the real Colab-hosted pipeline through ngrok on an actual mobile browser.

---

## Where things live (quick reference)

| Concern | Location | Persistence |
|---|---|---|
| Chroma vector DB | `backend/rag/chroma_db/` inside the Colab runtime's local disk | Ephemeral — wiped on Colab runtime reset, re-seeded from `catalog.json` every boot via Task 1.3's script |
| Conversation history | `backend/history/history.db` (SQLite) inside the Colab runtime's local disk | Ephemeral — wiped on Colab runtime reset |
| LLM weights | Downloaded GGUF file on Colab local disk (`/content/...`) | Re-downloaded each fresh Colab session unless Drive-mounted (out of scope) |
| Frontend | Static files (`frontend/`), served independently of Colab — can run on any static host or `python -m http.server` locally | N/A, stateless |
| Backend reachability | ngrok tunnel, URL regenerated every Colab restart | N/A — frontend takes the URL as user input each session |
