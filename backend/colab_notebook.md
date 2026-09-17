# Colab runbook

Run these cells in order, in a **GPU runtime** (Runtime → Change runtime type → T4 GPU).

Every command below matches what actually exists in this repo as of Phase 1-6 — file paths, module names, and env vars are copy-pasteable, not illustrative.

## Cell 1 — clone and install

```python
!git clone <repo-url> plivio-demo
%cd plivio-demo
!pip install -r backend/requirements.txt -q
```

`backend/requirements.txt` deliberately excludes `torch` (Colab ships a
CUDA-matched build already; pinning it either takes ~10 minutes or silently
breaks GPU support) and `llama-cpp-python` (needs CUDA build flags — next
cell). It does pin `setuptools<81`, because `webrtcvad` imports the now-removed
`pkg_resources` module.

## Cell 2 — install llama-cpp-python WITH CUDA

```python
!CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
    pip install llama-cpp-python==0.2.90 --no-cache-dir -q
```

**Do not replace this with a plain `pip install`.** The default wheel has no
CUDA support: it installs fine, accepts `n_gpu_layers=-1` without complaint,
and then runs every layer on the CPU at 1-3 tokens/sec with no error pointing
at the cause. This cell compiles from source and takes several minutes.

## Cell 3 — download the LLM weights

```python
!wget -O /content/qwen2.5-7b-instruct-q4_k_m.gguf <gguf-download-url>
```

Use a Qwen2.5-7B-Instruct GGUF repo on HuggingFace and pick the `Q4_K_M` file.
This is the path `backend/main.py` reads from `LLM_GGUF_PATH` by default.

## Cell 4 — verify the GPU is actually being used

```python
from llama_cpp import Llama
llm = Llama(model_path="/content/qwen2.5-7b-instruct-q4_k_m.gguf",
            n_gpu_layers=-1, verbose=True)
del llm
```

Look for **`offloaded 29/29 layers to GPU`** in the output. If it reports
`offloaded 0/29`, Cell 2 did not take — fix it and restart the runtime before
continuing, because everything downstream will be unusably slow. This exact
check is also `test_generation_is_fast_enough_to_be_on_gpu` in
`backend/tests/test_generate_manual.py`, runnable directly:

```python
!LLM_GGUF_PATH=/content/qwen2.5-7b-instruct-q4_k_m.gguf \
    python -m pytest backend/tests/test_generate_manual.py -v -s
```

## Cell 5 — seed Chroma with the catalog

```python
!python -m backend.rag.load_catalog
```

Expected: `Loaded 1200 items into backend/rag/chroma_db` — measured at 22s on
an 8-core dev machine; Colab's CPU may differ.

## Cell 6 — measure retrieval quality

```python
!python -m backend.rag.evaluate
```

Already measured on the dev machine at 96.7% recall@5 overall (en 100%,
hi 100%, mr 90%) — this cell confirms the same result reproduces on Colab's
CPU and catches any drift from a different package version. Record the
output in `docs/interview-defense.md` §11 if it differs from what's already
there.

## Cell 7 — benchmark STT placement

```python
!python -m backend.stt.benchmark
```

The dev machine measured CPU RTF 0.49 (8 fast cores); CUDA was unmeasurable
locally. Colab's free-tier CPU (2 vCPU, older Xeon) will likely show a much
worse CPU RTF than 0.49 — that's expected, and it's exactly why this cell
exists: to get the real number instead of assuming. Compare the `cpu` and
`cuda` lines and use the faster one for `STT_DEVICE` in Cell 9.

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

env = dict(os.environ, STT_DEVICE="cpu")  # or "cuda", per Cell 7's result
subprocess.Popen(["python", "-m", "backend.main"], env=env)
time.sleep(90)  # Whisper + three MMS voices + a 4.7GB GGUF is slow to load
                 # (the dev machine measured 17.0s for everything except the
                 # LLM; give the LLM itself real time to read + upload)

public_url = ngrok.connect(8000, "http").public_url
ws_url = public_url.replace("https://", "wss://") + "/ws/session1"
print("Paste this into the frontend:", ws_url)
```

If the server errors on startup, re-run this cell after checking the
subprocess's stdout/stderr — `subprocess.Popen` swallows it silently unless
you redirect. For debugging, run `!STT_DEVICE=cpu python -m backend.main` in
its own cell instead, so errors print directly.

## Cell 10 — check GPU headroom

```python
!nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

With `STT_DEVICE=cpu` expect roughly 5-6GB used (LLM only); with `cuda`,
roughly 6-7GB (LLM + Whisper). Anything near 12GB+ means something
unexpected is resident — check that STT and TTS didn't both end up on GPU.

## Serving the frontend

The page needs HTTPS for mic access on mobile — `getUserMedia` is blocked on
insecure origins, and `http://<lan-ip>` opened from a phone fails silently.
Either deploy `frontend/` to a static host (Netlify, GitHub Pages, Vercel) or
run a second ngrok tunnel against a local static server:

```bash
python -m http.server 5500 --directory frontend
ngrok http 5500
```

The page takes the backend URL as a runtime input (never hardcoded, since the
ngrok URL from Cell 9 regenerates on every restart), so it doesn't matter
where the frontend itself is hosted.

## If the runtime disconnects mid-session

Colab free-tier disconnects are the single most likely failure during a live
demo. On reconnect: re-run Cells 1-2 only if the runtime was fully reset
(check with `!nvidia-smi` — if it errors, the GPU allocation is gone and
everything needs reinstalling); otherwise re-run from Cell 5 onward, since
Chroma and the history SQLite file are wiped but the installed packages and
downloaded GGUF survive a soft disconnect. Keep the GGUF download (Cell 3)
and the `llama-cpp-python` CUDA build (Cell 2) as separate cells specifically
so a partial recovery doesn't have to redo the slowest steps unnecessarily.
