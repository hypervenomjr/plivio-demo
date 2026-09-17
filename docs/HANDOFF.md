# Handoff — Voice-to-Voice RAG Agent

Session snapshot, written mid-Phase-7 so work can resume cleanly in a new session. Read this first, then `docs/interview-defense.md` for the "why," then the plan docs under `docs/superpowers/plans/voice-rag-agent/` for task-level detail.

## Where the code lives

- **GitHub**: https://github.com/hypervenomjr/plivio-demo — **public** (made public deliberately, see "Decisions made this session" below; confirmed clean, no secrets).
- **Local**: `/home/vedant/Desktop/work/plivio-demo`, on branch `main`, fast-forwarded through all seven phase branches (`phase-1-rag` … `phase-7-integration` still exist locally and on `main`'s history — nothing to merge, they're all reachable from `main`).
- **Local Python env**: `.venv/` — Python 3.11 via `uv` (system Python is 3.14, which has no wheels for this stack; do not use system `python3`). Activate or invoke directly as `.venv/bin/python`.
- Last commit on `main`: `4844b53` "docs: add Colab notebook runbook grounded in the actual codebase"

## What's done: Phases 1–6, fully built, tested, committed

All seven phase plan docs exist under `docs/superpowers/plans/voice-rag-agent/`. Phases 1–6 are **code-complete, tested, and pushed**. Phase 7 (Colab integration) is **in progress** — see next section.

Run the full local suite: `.venv/bin/python -m pytest -q` → **91 passed, 6 skipped** (the 6 skips are `test_generate_manual.py`, which need real GGUF weights + CUDA and only run on Colab).

| Phase | What it built | Key measured numbers |
|---|---|---|
| 1 — RAG | Synthetic 1200-item catalog (all unique names), Chroma + `multilingual-e5-small` retrieval, recall@5 eval | **96.7% recall@5** overall (en 100%, hi 100%, mr 90%); catalog embeds in 22s |
| 2 — LLM | Token-budgeted prompt builder (names reply language explicitly), llama.cpp streaming wrapper | Real prompt measured ~238 tokens (vs ~400 est.) |
| 3 — STT | faster-whisper wrapper with hallucination guards, VAD segmenter (re-frames internally) | CPU RTF **0.49** (8-core i7); CUDA not measurable locally |
| 4 — TTS | MMS-TTS synthesizer (3 languages), clause chunker (handles Devanagari danda, decimals) | Warm RTF 0.20–0.26; **first clause audible at 0.28s** vs 2.45s for whole answer |
| 5 — Backend | FastAPI WebSocket server: barge-in via `asyncio.Task`, per-stage timing, SQLite history | Barge-in cancellation **~38ms**; real-model cold-start (minus LLM) **17.0s** |
| 6 — Frontend | Voice UI (queued playback, iOS downsample), mock server | Verified live in a real Chrome browser — full turn, barge-in, disconnect/reconnect, downsample math all checked via actual page state, not just visual |

**Full details, every measured number, and the "why" behind every architectural choice**: `docs/interview-defense.md`. It already has real numbers filled in for everything above under §2–§7. Sections 3 (latency) and 4 (cost) still show **estimated** figures pending real Colab measurements — that's what Phase 7 exists to fill in.

## Real bugs found and fixed during the build (not in the original plan)

Worth knowing before touching this code, and worth mentioning in an interview — these were found by actually running things, not by inspection:

1. **`llama-cpp-python` needs CUDA build flags** — plain `pip install` gives a silent CPU-only build (1-3 tok/s, no error). Must use `CMAKE_ARGS="-DGGML_CUDA=on"`. This is why it's deliberately excluded from `requirements.txt`.
2. **Whisper hallucinates on non-speech** — silence transcribed as "you"; noise produced a Korean news sign-off, detected as `ko`, which would have crashed the TTS stage (no Korean voice). Fixed with `vad_filter` + no-speech filtering + language clamping to `en`/`hi`/`mr`.
3. **`webrtcvad` needs `setuptools<81`** — it imports the now-removed `pkg_resources`.
4. **VAD frame-size mismatch** — webrtcvad wants exact 10/20/30ms frames; a browser sends arbitrary chunk sizes. `UtteranceSegmenter.push()` now re-frames internally so callers can push anything.
5. **The mock server had the exact bug the real server was built to avoid** — first draft processed turns inline in the receive loop, so a barge-in chunk sat unread until the turn finished on its own. Every unit test passed anyway. Only caught by testing the real frontend against it in an actual browser. Fixed with `asyncio.create_task`, matching the real server.
6. **Catalog price realism** — a flat `$5–$500` range across all categories produced things like a `$300` chocolate bar. Added per-category price bands.
7. **Retrieval eval metric design** — the catalog has ~14 brands per product+variant, so scoring generic queries against one exact UUID would cap recall@5 near 5/14 regardless of retrieval quality. Added product-type/category judgment types alongside exact-ID.

## Decisions made this session (with reasoning, in case they need revisiting)

- **Repo made public**, not private. Original plan was a private repo + SSH deploy key for Colab to clone. Generating the deploy key succeeded (read-only, single-repo, confirmed via GitHub API), but reading its private half to type into the Colab notebook was blocked by the Bash permission classifier — correctly, since private key material shouldn't casually pass through model context. Rather than work around that, asked the user, who chose public. Deploy key was deleted afterward (`gh api -X DELETE .../keys/163621514`). Repo was confirmed clean (grepped for secrets/tokens/credentials) before making it public.
- **GGUF weights URL**: `https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf` — a real, standard quantization repo. Not yet verified to actually download successfully on Colab (Cell 3 hadn't run yet when this session paused).
- **ngrok token**: left as an explicit placeholder in Colab Cell 8 (`"PASTE_YOUR_NGROK_TOKEN_HERE"`) — the user needs to fill this in themselves from https://dashboard.ngrok.com/get-started/your-authtoken.

## Phase 7 — exact state when this session paused

**Colab notebook**: https://colab.research.google.com/drive/1kU44SZcRFOhRsQhKUnDAwDk6JcVKCrVJ (`plivo_demo.ipynb`)

- Runtime: **T4 GPU, connected** (was showing live RAM/Disk gauges, "T4 (Python 3)" in the status bar).
- **10 cells typed into the notebook**, matching `backend/colab_notebook.md` exactly:
  1. `git clone` + `cd` + `pip install -r backend/requirements.txt -q` — **was running when session paused, at ~12m36s elapsed** (local install took ~10-12 min for comparison, so this was in the expected range, likely close to done or just finishing)
  2. `CMAKE_ARGS="-DGGML_CUDA=on" ... pip install llama-cpp-python==0.2.90` — typed, not yet run
  3. `wget` the GGUF weights — typed, not yet run
  4. `from llama_cpp import Llama; Llama(..., verbose=True); del llm` — GPU-offload verification — typed, not yet run (note: this cell had a typing bug fixed mid-session — an earlier `Escape`+`b` keyboard-shortcut attempt to insert a new cell instead typed a literal `b` character into this cell as `del llmb`; caught and fixed to `del llm`. Worth double-checking this cell's exact contents before running.)
  5. `python -m backend.rag.load_catalog` — typed, not yet run
  6. `python -m backend.rag.evaluate` — typed, not yet run (expect ~96.7% recall@5, matching local; if it differs materially, that's a finding — write it into `docs/interview-defense.md` §11/12)
  7. `python -m backend.stt.benchmark` — typed, not yet run (this is the one that gives the real CUDA RTF for STT, filling the biggest gap in the interview doc)
  8. `pip install pyngrok -q; from pyngrok import ngrok; ngrok.set_auth_token("PASTE_YOUR_NGROK_TOKEN_HERE")` — **needs the user's real ngrok token pasted in before running**
  9. Starts `backend.main` via `subprocess.Popen` with `STT_DEVICE="cpu"`, sleeps 90s, opens the ngrok tunnel, prints the `wss://` URL — typed, not yet run
  10. `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` — typed, not yet run

**Lesson learned mid-session**: Colab's cell editor sometimes has an autocomplete popup open after typing an identifier. Pressing `Escape` in that state closes the *popup*, not cell edit mode, so a following keyboard shortcut (`b` for "insert cell below") gets typed as a literal character instead. Safer to click elsewhere first to dismiss any popup, or just use the toolbar `+ Code` button instead of `Escape`+`b`.

### To resume Phase 7 in a new session

1. Re-open the Colab notebook URL above. Confirm the T4 runtime is still connected (Colab free-tier disconnects after idle periods — if disconnected, `backend/colab_notebook.md` has a "if the runtime disconnects mid-session" note on what survives vs. what needs redoing).
2. Check Cell 1's actual output — did the clone + install finish? If it errored or is stuck, re-run it.
3. Before running Cell 4, **open it and verify it reads `del llm`, not `del llmb`** (see the typing bug above).
4. Run Cells 2–7 in order, confirming each one's expected output against `backend/colab_notebook.md`'s inline commentary (especially Cell 4's `offloaded 29/29 layers to GPU` check — if it says `0/29`, Cell 2's CUDA build didn't take).
5. **Before Cell 8**, get an ngrok auth token (free) from https://dashboard.ngrok.com/get-started/your-authtoken and paste it in, replacing the placeholder.
6. Run Cells 8–10, copy the printed `wss://` URL.
7. Serve `frontend/` over HTTPS (a second ngrok tunnel against `python -m http.server 5500 --directory frontend`, or deploy to Netlify/Vercel/GitHub Pages — plain `http://<lan-ip>` will not work for mic access on a phone).
8. Do the actual live voice test from `docs/superpowers/plans/voice-rag-agent/phase-7-integration.md` Task 7.2 (English/Hindi/Marathi questions, barge-in, conversation memory, empty-retrieval path, reconnect) — **this step needs a real person with a real phone**, which is why it couldn't be finished in-session.
9. Record every real number (recall@5, STT benchmark, cold-start, time-to-first-audio measured live) into `docs/interview-defense.md` §11/12, replacing the "estimated" figures in §3 and §4.

## Quick reference — commands that matter

```bash
# Run the full local test suite (no GPU needed)
cd /home/vedant/Desktop/work/plivio-demo
.venv/bin/python -m pytest -q

# Regenerate the catalog / reload Chroma locally
.venv/bin/python -m backend.data.generate_catalog
.venv/bin/python -m backend.rag.load_catalog

# Check retrieval quality locally
.venv/bin/python -m backend.rag.evaluate

# Check STT CPU benchmark locally (no CUDA available on this machine)
.venv/bin/python -m backend.stt.benchmark
```

## Open items not yet started

- Task 7.2's live manual verification (see above — needs a physical phone/mic).
- Filling §3/§4's estimated latency and cost numbers in `docs/interview-defense.md` with real Colab measurements.
- Nothing else — Phases 1–6 are complete, tested, and committed. Phase 7's only remaining work is the Colab run-through and the live voice test.
