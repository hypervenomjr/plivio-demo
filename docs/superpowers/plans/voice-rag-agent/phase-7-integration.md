# Phase 7: End-to-End Integration (Colab + ngrok) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the real backend on Colab behind an ngrok tunnel and drive it from the real frontend on a phone, confirming a full voice-to-voice turn in English, Hindi and Marathi.

**Architecture:** Configuration and verification only — no new modules. Colab cells install dependencies, download the GGUF weights, seed Chroma, start `backend/main.py`, and open an ngrok tunnel that prints the `wss://` URL to paste into the frontend.

**Tech Stack:** Google Colab, pyngrok, the artifacts built in Phases 1-6.

**Spec:** `docs/superpowers/specs/2026-09-17-voice-rag-agent-design.md`

**Index:** [README.md](README.md)

## Isolation

This is the only phase that requires other phases to be finished. It writes **no library code** — just a documented notebook runbook and a manual test checklist. If a phase's internals change, nothing here changes; only a change to startup steps or to the wire protocol touches this document.

## Prerequisites

- Phase 1 complete: `backend/data/catalog.json` exists and `backend/rag/load_catalog.py` runs.
- Phase 2 complete: `backend/llm/generate.py` exists; a GGUF URL is known.
- Phases 3, 4, 5 complete: `backend/main.py` starts locally without import errors.
- Phase 6 complete: `frontend/` serves and works against the mock server.

## Global Constraints

- The ngrok URL regenerates on every Colab restart — it is pasted into the frontend by hand each session, never baked into the page. *(spec: Frontend)*
- Chroma and SQLite live on Colab local disk and are wiped on runtime reset; the catalog is re-seeded every boot. *(spec: Inventory data, Conversation history)*
- Only the LLM occupies the GPU; STT and TTS stay on CPU. *(spec: Constraints)*

---

### Task 7.1: Colab notebook runbook

**Files:**
- Create: `backend/colab_notebook.md`

**Interfaces:**
- Consumes: `backend/main.py` (Phase 5, Task 5.3) and `backend/rag/load_catalog.py` (Phase 1, Task 1.3).
- Produces: a cell-by-cell runbook, since the `.ipynb` itself is created interactively in Colab.

- [ ] **Step 1: Write the runbook**

````markdown
<!-- backend/colab_notebook.md -->
# Colab runbook

Run these cells in order in a **GPU runtime** (Runtime → Change runtime type → T4 GPU).

## Cell 1 — clone and install

```python
!git clone <repo-url> plivio-demo
%cd plivio-demo
!pip install -r backend/requirements.txt -q
```

## Cell 2 — download the LLM weights

```python
!wget -O /content/qwen2.5-7b-instruct-q4_k_m.gguf <gguf-download-url>
```

Use a Qwen2.5-7B-Instruct GGUF repo on HuggingFace and pick the `Q4_K_M` file.

## Cell 3 — seed Chroma with the catalog

```python
!python -m backend.rag.load_catalog
```

Expected output: `Loaded 1200 items into backend/rag/chroma_db`

## Cell 4 — configure ngrok

```python
!pip install pyngrok -q
from pyngrok import ngrok
ngrok.set_auth_token("<your-ngrok-token>")
```

## Cell 5 — start the server and open the tunnel

```python
import subprocess, time
from pyngrok import ngrok

subprocess.Popen(["python", "-m", "backend.main"])
time.sleep(60)  # model loading (Whisper + MMS + 7B GGUF) takes a while

public_url = ngrok.connect(8000, "http").public_url
ws_url = public_url.replace("https://", "wss://") + "/ws/session1"
print("Paste this into the frontend:", ws_url)
```

## Cell 6 — check GPU headroom (optional)

```python
!nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Only the LLM should be resident — expect roughly 5-6GB used, not 12GB+. If it is much higher, something moved STT or TTS onto the GPU.
````

- [ ] **Step 2: Run all cells in a real Colab notebook**

Expected: Cell 5 prints a `wss://....ngrok-free.app/ws/session1` URL and the server logs no errors. Cell 6 shows only the LLM's memory footprint.

- [ ] **Step 3: Commit**

```bash
git add backend/colab_notebook.md
git commit -m "docs: add Colab notebook startup runbook"
```

---

### Task 7.2: End-to-end manual verification

**Files:** none — verification only.

**Interfaces:**
- Consumes: the running Colab backend from Task 7.1 and the frontend from Phase 6.

- [ ] **Step 1: Start the backend** — run the Task 7.1 cells, copy the printed `wss://` URL.

- [ ] **Step 2: Serve the real frontend**

Run: `python -m http.server 5500 --directory frontend`
Open `http://<your-lan-ip>:5500/index.html` on a phone browser.

- [ ] **Step 3: Connect** — paste the `wss://` URL into the backend-URL field, press Connect.
Expected: status shows `connected`, mic button becomes enabled.

- [ ] **Step 4: Ask in English** — hold the mic button, say "do you have wireless headphones", release.
Expected: transcript line appears with roughly that text, language shows `en`, an answer naming a real catalog item appears, and the answer plays back as speech.

- [ ] **Step 5: Ask in Hindi** — hold, say "क्या आपके पास हेडफ़ोन हैं", release.
Expected: language shows `hi`, the reply is spoken in a Hindi voice.

- [ ] **Step 6: Ask in Marathi** — hold, say "तुमच्याकडे हेडफोन आहेत का", release.
Expected: language shows `mr`, the reply is spoken in a Marathi voice. Marathi transcription is the weakest link in the stack — if the transcript is wrong, note it as a known limitation rather than a blocker.

- [ ] **Step 7: Test conversation memory** — ask a follow-up that only makes sense with history, such as "and how many are in stock?"
Expected: the answer refers to the item from the previous turn, proving history reaches the prompt.

- [ ] **Step 8: Test the empty-retrieval path** — ask for something absent from the catalog, such as "do you sell live elephants".
Expected: the agent says there are no matching listings rather than inventing one.

- [ ] **Step 9: Test reconnection** — stop the Colab cell running the server, observe the UI, restart it, take the new URL and reconnect.
Expected: UI shows `disconnected — press Connect to retry`; reconnecting with the new URL works.

- [ ] **Step 10: Record the results**

Write what passed and what did not into `docs/superpowers/plans/voice-rag-agent/phase-7-results.md`, then commit:

```bash
git add docs/superpowers/plans/voice-rag-agent/phase-7-results.md
git commit -m "docs: record end-to-end integration test results"
```

---

## Phase 7 done when

Steps 1-10 have been run against the real Colab pipeline from an actual mobile browser, and the results file records the outcome of each, including any known-weak spots such as Marathi transcription accuracy.
