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
- The LLM occupies the GPU and TTS stays on CPU; STT placement is chosen from the Cell 7 benchmark. *(spec: Constraints)*
- The frontend must be served over HTTPS, or mobile browsers refuse mic access. *(spec: Frontend)*

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

`requirements.txt` deliberately excludes `torch` (Colab ships a CUDA-matched build) and `llama-cpp-python` (needs CUDA build flags — next cell).

## Cell 2 — install llama-cpp-python WITH CUDA

```python
!CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
    pip install llama-cpp-python==0.2.90 --no-cache-dir -q
```

**Do not replace this with a plain `pip install`.** The default wheel has no CUDA support: it installs fine, accepts `n_gpu_layers=-1` without complaint, then runs every layer on the CPU at 1-3 tokens/sec. This cell compiles from source and takes several minutes.

## Cell 3 — download the LLM weights

```python
!wget -O /content/qwen2.5-7b-instruct-q4_k_m.gguf <gguf-download-url>
```

Use a Qwen2.5-7B-Instruct GGUF repo on HuggingFace and pick the `Q4_K_M` file.

## Cell 4 — verify the GPU is actually being used

```python
from llama_cpp import Llama
llm = Llama(model_path="/content/qwen2.5-7b-instruct-q4_k_m.gguf",
            n_gpu_layers=-1, verbose=True)
del llm
```

Look for **`offloaded 29/29 layers to GPU`**. If it reports `offloaded 0/29`, Cell 2 did not take — fix it and restart the runtime before continuing, because everything downstream will be unusably slow.

## Cell 5 — seed Chroma with the catalog

```python
!python -m backend.rag.load_catalog
```

Expected: `Loaded 1200 items into backend/rag/chroma_db`

## Cell 6 — measure retrieval quality

```python
!python -m backend.rag.evaluate
```

Record the recall@5 figures, per language, in `docs/interview-defense.md` §12 (Measurement checklist).

## Cell 7 — benchmark STT placement

```python
!python -m backend.stt.benchmark
```

Compare the `cpu` and `cuda` lines and use the result to choose `STT_DEVICE` below. Record both numbers.

## Cell 8 — configure ngrok

```python
!pip install pyngrok -q
from pyngrok import ngrok
ngrok.set_auth_token("<your-ngrok-token>")
```

## Cell 9 — start the server and open the tunnel

```python
import os, subprocess, time
from pyngrok import ngrok

env = dict(os.environ, STT_DEVICE="cpu")  # or "cuda", per Cell 7
subprocess.Popen(["python", "-m", "backend.main"], env=env)
time.sleep(90)  # Whisper + three MMS voices + a 4.7GB GGUF is slow to load

public_url = ngrok.connect(8000, "http").public_url
ws_url = public_url.replace("https://", "wss://") + "/ws/session1"
print("Paste this into the frontend:", ws_url)
```

## Cell 10 — check GPU headroom

```python
!nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

With `STT_DEVICE=cpu` expect roughly 5-6GB used; with `cuda`, roughly 6-7GB. Anything near 12GB+ means something unexpected is resident.
````

- [ ] **Step 2: Run all cells in a real Colab notebook**

Expected: Cell 4 confirms full GPU offload, Cell 9 prints a `wss://....ngrok-free.app/ws/session1` URL with no server errors, and Cell 10 shows a footprint consistent with the chosen `STT_DEVICE`.

- [ ] **Step 3: Serve the frontend over HTTPS**

The page needs a secure context for mic access on mobile, so a plain local `http.server` opened by LAN IP **will not work**. Either deploy `frontend/` to a static host (Netlify, GitHub Pages, Vercel) or run a second ngrok tunnel against a local static server:

```bash
python -m http.server 5500 --directory frontend
ngrok http 5500
```

The page takes the backend URL as input, so it does not care where it is hosted.

- [ ] **Step 4: Commit**

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

- [ ] **Step 2: Serve the real frontend over HTTPS** — per Task 7.1 Step 3. Open the `https://` page URL on a phone browser. A plain `http://<lan-ip>` page will silently fail to get mic access.

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

- [ ] **Step 7a: Verify the reply language matches the question language.** In Steps 5 and 6, confirm the agent answered *in Hindi and Marathi*, not in English spoken by a Hindi voice. An English reply through an Indic TTS checkpoint is the specific failure this design guards against — if it happens, the language is not reaching `build_prompt`.

- [ ] **Step 7b: Test barge-in** — ask a question, then start talking again while the agent is still speaking.
Expected: playback stops immediately, an `interrupted` line appears in the log, and the new utterance is processed as a fresh turn.

- [ ] **Step 7c: Record the timing numbers** — after a few turns, read the timing row on the page.
Copy a representative set of per-stage figures into `docs/interview-defense.md` §12 (Measurement checklist). `time_to_first_audio` is the headline number.

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
