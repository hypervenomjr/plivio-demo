# Voice-to-Voice RAG Agent — Phased Plan Index

**Goal:** A streaming voice-to-voice agent (web/mWeb UI → Colab backend via ngrok) that answers spoken questions about a synthetic 1000+ item inventory catalog using RAG, in English/Hindi/Marathi.

**Spec:** `docs/superpowers/specs/2026-09-17-voice-rag-agent-design.md`

Each phase below is a **separate plan document**, executable and testable on its own. Phases talk to each other only through the frozen contracts listed in the table — change a phase's internals and no other document needs to move.

## Phases

| Phase | Document | Builds | Needs a GPU? | Depends on |
|---|---|---|---|---|
| 1 | [phase-1-rag.md](phase-1-rag.md) | Synthetic catalog + Chroma retrieval + recall@5 evaluation | No | nothing |
| 2 | [phase-2-llm.md](phase-2-llm.md) | Prompt builder + llama.cpp streaming generation | Only for the manual test | nothing |
| 3 | [phase-3-stt.md](phase-3-stt.md) | faster-whisper transcription + VAD segmenter + device benchmark | Only for the benchmark | nothing |
| 4 | [phase-4-tts.md](phase-4-tts.md) | MMS-TTS synthesis + clause chunker | No | nothing |
| 5 | [phase-5-backend.md](phase-5-backend.md) | SQLite history + FastAPI WebSocket orchestration + barge-in + timing | Only for the real wiring script | Contracts of 1-4 (fakes in tests) |
| 6 | [phase-6-frontend.md](phase-6-frontend.md) | Mock server + web/mWeb voice UI | No | WebSocket protocol contract only |
| 7 | [phase-7-integration.md](phase-7-integration.md) | Colab + ngrok startup, end-to-end manual test | Yes | Phases 5 and 6 running |

**Phases 1, 2, 3, 4 and 6 are fully parallel** — no shared code, no shared state, no ordering between them. Phase 5 needs the *contracts* of 1-4 (its own tests use fakes, so it can be built before they exist). Phase 7 is glue only.

## Frozen contracts (the only cross-phase coupling)

Changing anything in this table means updating the consuming phase's document too. Everything else inside a phase is private to it.

| Contract | Defined in | Consumed by |
|---|---|---|
| Catalog item dict: `{"id": str, "name": str, "description": str, "price": float, "stock": int, "category": str}` | Phase 1 | Phase 1 (store), Phase 2 (prompt) |
| Retrieved item dict: catalog item + `{"score": float}` | Phase 1 | Phase 2, Phase 5 |
| `InventoryStore.query(text: str, top_k: int = 5) -> list[dict]` | Phase 1 | Phase 5 |
| `build_prompt(user_text, retrieved, history, language="en", max_history_turns=6) -> str` | Phase 2 | Phase 5 |
| `LLMEngine.generate(prompt, stop=None, max_tokens=200) -> Iterator[str]` | Phase 2 | Phase 5 |
| History turn dict: `{"role": "user"\|"assistant", "text": str}` | Phase 5 | Phase 2 (prompt input) |
| `Transcriber(model_size="small", device="cpu", compute_type=None)` | Phase 3 | Phase 5 |
| `Transcriber.transcribe(audio_path: str) -> {"text": str, "language": str}` | Phase 3 | Phase 5 |
| `UtteranceSegmenter.push(pcm_chunk: bytes) -> bytes \| None` — accepts **any** chunk length, re-frames internally | Phase 3 | Phase 5 |
| `UtteranceSegmenter.speech_active -> bool` (rising edge drives barge-in) | Phase 3 | Phase 5 |
| `Transcriber.transcribe` returns a language **clamped to `en`/`hi`/`mr`** | Phase 3 | Phase 4, Phase 5 |
| `Synthesizer.synthesize(text: str, language: str = "en") -> bytes` (WAV) | Phase 4 | Phase 5 |
| `ClauseChunker.push(fragment) -> list[str]` / `.flush() -> str \| None` | Phase 4 | Phase 5 |
| **WebSocket wire protocol** (see below) | Phase 5 | Phase 6 |

### WebSocket wire protocol

The single contract between backend and frontend. Endpoint: `/ws/{session_id}`.

**Client → server:**
- Binary frames: raw 16kHz mono PCM chunks (downsampled client-side from the device's native rate).
- Text frame `"__end__"`: force end-of-turn (used by push-to-talk release and by tests, instead of waiting for VAD silence).

**Server → client, in order, per turn:**
1. Text: `{"type": "transcript", "text": str, "language": str}`
2. Then, repeating per clause: text `{"type": "answer_text", "text": str}` followed by a binary WAV frame for that clause.
3. Text: `{"type": "timing", "stages": {...}}` — per-stage milliseconds for the turn.
4. Text: `{"type": "turn_end"}`

**Out of band, at any point during a turn:**
- Text: `{"type": "interrupted"}` — the caller spoke over the agent and generation was cancelled. The client stops playback and discards queued audio. No `turn_end` follows.

### The language code is used twice

Whisper detects it once (Phase 3) and it feeds **both** `build_prompt` (so the model answers in that language) **and** `Synthesizer.synthesize` (so the matching voice speaks it). Wiring only one of the two produces an English answer spoken by an Indic voice — unintelligible audio, and the most likely way the multilingual feature fails.

## Traps — read before writing code

Five failure modes found in review that are silent, misleading, or both. Each is handled in its phase document; they are listed together here because the cost of hitting one blind is hours.

| Trap | Symptom | Phase |
|---|---|---|
| `pip install llama-cpp-python` ships a **CPU-only** build | Installs fine, accepts `n_gpu_layers=-1`, then runs at 1-3 tok/s with no error. Needs `CMAKE_ARGS="-DGGML_CUDA=on"` | 2, 7 |
| `getUserMedia` needs a **secure context** | Mic silently unavailable on `http://<lan-ip>` from a phone. The page must be served over HTTPS | 6, 7 |
| The prompt must **name the reply language** | Hindi question gets an English answer, spoken by a Hindi voice — unintelligible. Language must reach `build_prompt`, not just the TTS | 2, 5 |
| iOS **ignores** `AudioContext({sampleRate})` | Audio arrives at 44.1/48kHz; VAD and Whisper both produce nonsense rather than an error. Downsample in software | 6 |
| A low-variety catalog makes **retrieval look broken** | Top-5 comes back as near-identical rows no embedding can rank. Every generated item needs a distinct name and description | 1 |

Four more found while building:

| Trap | Symptom | Phase |
|---|---|---|
| **Whisper hallucinates on non-speech** | Silence transcribes as "you"; noise produced a Korean news sign-off and a Spanish word. A cough tripping VAD becomes a spurious LLM turn, and a bogus language code raises in TTS mid-turn. Needs `vad_filter`, no-speech filtering, and a language clamp | 3 |
| faster-whisper 1.0.3 **raises on fully-filtered audio** | `ValueError: max() arg is an empty sequence` instead of an empty result | 3 |
| **webrtcvad needs `setuptools<81`** | Imports `pkg_resources`, removed in setuptools 84: `ModuleNotFoundError` | 3 |
| **webrtcvad frames vs browser chunks** | webrtcvad accepts only 10/20/30ms frames; a browser sends whatever its buffer works out to. The segmenter re-frames internally so callers can push any length | 3, 5 |

## Global constraints (apply to every phase)

- STT and TTS run on CPU only. Only the LLM runs on GPU. *(spec: Constraints)*
- Must support English, Hindi, Marathi input and output. *(spec: Constraints)*
- Chunking: one Chroma chunk per inventory item, no text splitting. *(spec: RAG & inference details)*
- Embedding model: `intfloat/multilingual-e5-small`. *(spec: RAG & inference details)*
- LLM served via llama-cpp-python, GGUF Q4_K_M, Qwen2.5-7B-Instruct. *(spec: RAG & inference details)*
- RAG retrieval capped to top-k=3-5 per turn; prompt carries only essential fields. *(spec: Cost / token optimization)*
- Conversation history windowed to last N turns, persisted to SQLite on local Colab disk. *(spec: Conversation history)*
- Chroma persists to local Colab disk, re-embedded fresh each notebook boot. *(spec: Inventory data)*
- ngrok URL is not stable across restarts; the frontend must not hardcode it. *(spec: Frontend)*

## Where things live

| Concern | Location | Persistence |
|---|---|---|
| Chroma vector DB | `backend/rag/chroma_db/` on the Colab runtime's local disk | Ephemeral — wiped on runtime reset, re-seeded from `catalog.json` each boot (Phase 1, Task 1.3) |
| Conversation history | `backend/history/history.db` (SQLite) on Colab local disk | Ephemeral — wiped on runtime reset |
| LLM weights | GGUF file at `/content/...` on Colab local disk | Re-downloaded each fresh Colab session |
| Frontend | Static files in `frontend/`, served anywhere independent of Colab | Stateless |
| Backend reachability | ngrok tunnel, URL regenerated every Colab restart | Frontend takes the URL as user input each session |
