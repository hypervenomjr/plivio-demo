# Phase 2: LLM Answer Generation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a user question plus retrieved inventory listings plus conversation history into a short spoken-style answer, streamed token-by-token from a locally-hosted quantized LLM.

**Architecture:** A pure prompt-builder function (no model, no I/O, fully unit-testable) assembles a token-budgeted prompt. A thin `LLMEngine` wrapper around llama-cpp-python loads the GGUF Q4_K_M build of Qwen2.5-7B-Instruct with all layers offloaded to GPU and yields text fragments as they generate.

**Tech Stack:** Python 3.10+, llama-cpp-python, Qwen2.5-7B-Instruct GGUF Q4_K_M.

**Spec:** `docs/superpowers/specs/2026-09-17-voice-rag-agent-design.md`

**Index:** [README.md](README.md)

## Isolation

Depends on **no other phase's code**. It only needs the *shape* of a retrieved item and of a history turn, both of which are stubbed inline in this phase's tests. Task 2.1 runs on CPU with no model at all; only Task 2.2's manual test needs the GPU and the downloaded weights.

## Contracts

**Consumed** (shapes only, stubbed in tests here — no imports from other phases):
- Retrieved item dict: `{"id": str, "name": str, "description": str, "price": float, "stock": int, "category": str, "score": float}` *(defined in Phase 1)*
- History turn dict: `{"role": "user"|"assistant", "text": str}` *(defined in Phase 5)*

**Produced** (Phase 5 relies on these — do not change without updating it):
- `build_prompt(user_text: str, retrieved: list[dict], history: list[dict], max_history_turns: int = 6) -> str`
- `LLMEngine(model_path: str, n_gpu_layers: int = -1)`
- `LLMEngine.generate(prompt: str, stop: list[str] = None, max_tokens: int = 200) -> Iterator[str]`

## Global Constraints

- LLM served via llama-cpp-python, GGUF Q4_K_M, Qwen2.5-7B-Instruct — chosen over vLLM because there is a single concurrent caller. *(spec: RAG & inference details)*
- Only the LLM runs on GPU; STT and TTS stay on CPU, so all layers can be offloaded (`n_gpu_layers=-1`). *(spec: Constraints)*
- Retrieved listings truncated to essential fields in the prompt (name, price, stock, category) — never the full raw record. *(spec: Cost / token optimization)*
- Conversation history windowed to the last N turns, oldest dropped first. *(spec: Cost / token optimization)*
- Short, fixed system prompt — no verbose per-turn restatement. *(spec: Cost / token optimization)*
- Stop sequences set so generation halts at the answer instead of over-generating. *(spec: Cost / token optimization)*
- When retrieval returns nothing, the prompt must instruct the model to say so rather than hallucinate. *(spec: Error handling)*

---

### Task 2.1: Prompt builder (pure function, no model needed)

**Files:**
- Create: `backend/llm/prompt.py`
- Create: `backend/llm/__init__.py`
- Test: `backend/tests/test_prompt.py`

**Interfaces:**
- Consumes: retrieved item dicts and history turn dicts (shapes above, stubbed in the test).
- Produces: `build_prompt(user_text: str, retrieved: list[dict], history: list[dict], max_history_turns: int = 6) -> str`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_prompt.py
from backend.llm.prompt import build_prompt

RETRIEVED = [
    {"id": "1", "name": "Wireless Headphones", "description": "...", "price": 89.99,
     "stock": 20, "category": "Electronics", "score": 0.1},
]


def test_prompt_includes_user_text():
    prompt = build_prompt("do you have headphones", RETRIEVED, [])
    assert "do you have headphones" in prompt


def test_prompt_includes_retrieved_item_essential_fields():
    prompt = build_prompt("headphones?", RETRIEVED, [])
    assert "Wireless Headphones" in prompt
    assert "89.99" in prompt
    assert "20" in prompt


def test_prompt_states_no_listings_when_retrieval_empty():
    prompt = build_prompt("headphones?", [], [])
    assert "No matching listings found." in prompt


def test_prompt_windows_history_to_max_turns():
    history = [{"role": "user", "text": f"turn {i}"} for i in range(10)]
    prompt = build_prompt("latest question", [], history, max_history_turns=2)
    assert "turn 9" in prompt
    assert "turn 0" not in prompt


def test_prompt_omits_history_section_when_empty():
    prompt = build_prompt("hello", [], [])
    assert "Conversation so far:" not in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.llm.prompt'`

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
    return "\n".join(f"{turn['role']}: {turn['text']}" for turn in windowed)


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
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/llm/prompt.py backend/llm/__init__.py backend/tests/test_prompt.py
git commit -m "feat: add token-budgeted LLM prompt builder"
```

---

### Task 2.2: LLM generation wrapper (llama.cpp, streaming)

**Files:**
- Create: `backend/llm/generate.py`
- Test: `backend/tests/test_generate_manual.py` (integration test, skipped unless weights are present)
- Modify: `backend/requirements.txt`

**Interfaces:**
- Consumes: a prompt string (from `build_prompt` in Task 2.1, but the wrapper accepts any string).
- Produces:
  - `class LLMEngine` with `__init__(self, model_path: str, n_gpu_layers: int = -1)`
  - `LLMEngine.generate(self, prompt: str, stop: list[str] = None, max_tokens: int = 200)` — yields text fragments as they are generated

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
            n_gpu_layers=n_gpu_layers,  # -1 offloads every layer to the GPU
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

- [ ] **Step 3: Write the manual integration test**

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


@pytest.mark.skipif(not MODEL_PATH, reason="Set LLM_GGUF_PATH to run this manual test")
def test_generate_streams_more_than_one_chunk():
    engine = LLMEngine(model_path=MODEL_PATH)
    chunks = list(engine.generate("Say three short sentences.\nassistant:", max_tokens=40))
    assert len(chunks) > 1
```

- [ ] **Step 4: Verify it skips locally, then run it for real on Colab**

Run locally: `pytest backend/tests/test_generate_manual.py -v`
Expected: 2 SKIPPED (no weights set).

On Colab, download `Qwen2.5-7B-Instruct-Q4_K_M.gguf` from a HuggingFace GGUF repo to local disk, then run:
`LLM_GGUF_PATH=/content/qwen2.5-7b-instruct-q4_k_m.gguf pytest backend/tests/test_generate_manual.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/llm/generate.py backend/tests/test_generate_manual.py backend/requirements.txt
git commit -m "feat: add streaming llama.cpp LLM generation wrapper"
```

---

## Phase 2 done when

`pytest backend/tests/test_prompt.py -v` passes locally with no GPU, and `test_generate_manual.py` passes on Colab with the real weights.
