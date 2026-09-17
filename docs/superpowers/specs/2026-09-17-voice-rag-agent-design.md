# Voice-to-Voice RAG Agent — Design

## Purpose

Phone-call-style agent: user speaks, agent transcribes, retrieves relevant
inventory listings via RAG, generates a spoken answer. Prototype targets
web/browser voice first; real telephony (Plivo/Twilio) is a later, separate
integration.

## Constraints

- No GPU available locally. Build/run on Google Colab (free GPU tier),
  expose endpoints via ngrok for testing.
- GPU reserved solely for the LLM. STT and TTS both run on CPU, so they
  never compete with the LLM for VRAM.
- Must support multiple Indian languages (e.g. Hindi, Marathi) for both
  input and output, not just English.
- Streaming pipeline chosen over turn-based despite added complexity —
  user wants real-time feel, accepted the higher build/debug risk.

## Architecture

```
Browser (mic capture, WebSocket client, audio playback)
        │ WebSocket (audio chunks + control msgs)
        ▼
ngrok tunnel
        ▼
Colab notebook: FastAPI + WebSocket server
   ├─ STT (CPU): faster-whisper (small/medium, int8), streaming decode
   │   — multilingual, covers Hindi well, Marathi with more error margin
   ├─ RAG: Chroma (in-notebook, local) + sentence-transformers embeddings
   ├─ LLM (GPU): Qwen2.5-7B-Instruct, 4-bit quant (AWQ/GGUF via
   │   llama.cpp or vLLM) — only model on GPU
   └─ TTS (CPU): Meta MMS-TTS (VITS architecture, same lightweight
       family as Piper) — per-language checkpoints, covers Hindi (hin),
       Marathi (mar), and most other Indian languages individually.
       Chosen over Piper (no Marathi voice) and AI4Bharat Indic-TTS
       (heavier, more GPU-leaning) specifically for CPU-native fit +
       Indic language coverage.
```

STT and TTS both CPU-only, so the entire Colab GPU is free for the
quantized LLM (~5GB) — much larger safety margin than sharing VRAM
three ways.

## Data flow (streaming turn)

1. Browser opens WebSocket, streams mic audio (16kHz PCM chunks)
   continuously.
2. Server runs VAD (webrtcvad or Silero-VAD) on incoming chunks to detect
   utterance start/end — this is the turn boundary in a streaming setup.
3. On speech-end: buffered audio finalized to text via faster-whisper.
   Whisper auto-detects spoken language; detected language code carried
   forward to select the matching MMS-TTS checkpoint for the reply.
4. Text embedded, Chroma similarity search over the inventory catalog,
   top-k listings retrieved.
5. Retrieved listings + user text + conversation history → prompt → local
   LLM generates response, streamed token-by-token.
6. As LLM tokens arrive, sentence/clause chunks are pulled off and fed to
   MMS-TTS incrementally — audio starts playing before the full LLM
   response finishes generating.
7. Audio chunks streamed back over WebSocket; browser plays them in
   order.
8. Barge-in: if VAD detects the user speaking again while TTS is playing,
   server cancels current LLM/TTS generation and starts a new turn.

## Conversation history

- Persisted to SQLite, keyed by session id, so history survives within a
  running Colab session.
- Stored on local Colab disk (not Drive-mounted) — accepted tradeoff:
  history is wiped whenever the Colab runtime resets or disconnects.
- Included in the LLM prompt each turn, windowed to the last N turns /
  a token budget to avoid overflowing the model's context.

## RAG & inference details

- **Chunking:** one chunk per inventory item — no sliding-window text
  splitting. Each item's fields (name, description, price, stock)
  concatenated into a single chunk, item id kept as metadata. Structured
  short records don't benefit from further splitting.
- **Embedding model:** `intfloat/multilingual-e5-small` (CPU, ~118M
  params). Must be multilingual, not plain English MiniLM, since queries
  can arrive in Hindi/Marathi and still need to match English-described
  inventory. Computed once per item at catalog load (cached in Chroma)
  and once per live user query at runtime.
- **LLM inference:** llama.cpp (via llama-cpp-python) serving the GGUF
  Q4_K_M quant of Qwen2.5-7B-Instruct. Chosen over vLLM — single
  concurrent caller in this prototype, so vLLM's batching/throughput
  advantage doesn't apply; llama.cpp has lower setup overhead on Colab
  and native token-streaming.

## Cost / token optimization

1. 4-bit quant on the LLM — largest single lever, ~4x memory/compute cut
   vs fp16.
2. RAG retrieval capped to top-k=3-5 listings per turn, not the whole
   catalog.
3. Retrieved listings truncated to essential fields in the prompt (name,
   price, stock, short description) — not the full raw record.
4. Conversation history windowed to last N turns / a fixed token budget
   (see above), oldest turns dropped first.
5. Short, fixed system prompt — no verbose re-stated instructions per
   turn.
6. Stop sequences on LLM generation so it halts at the actual answer
   instead of over-generating.
7. Embeddings cached at catalog load; never recomputed except for the
   live user query.

## Inventory data

- Synthetic catalog, ~1000+ items (name, price, description, stock, etc),
  generated for this build — no real business data.
- Embedded once at startup into Chroma.
- Chroma runs in-notebook (embedded, not a separate server), persists to
  local Colab disk. Same tradeoff as conversation history: wiped on
  runtime reset. Acceptable — catalog is seeded/re-embedded fresh on
  every notebook boot, not meant to survive across sessions.

## Frontend (web/mWeb UI)

- Single-page web app (desktop + mobile browser), served separately from
  the Colab backend — static HTML/JS, connects to the Colab+ngrok
  WebSocket URL.
- Mic capture (getUserMedia), streams PCM chunks over WebSocket,
  receives + plays back streamed TTS audio.
- Minimal UI: mic on/off state, live transcript display, connection
  status, language indicator (once detected).
- ngrok URL changes each time the Colab notebook restarts (free tier) —
  UI needs a way to point at the current tunnel URL (config field or
  query param), not hardcoded.

## Error handling

- ASR fails / empty transcript: skip turn, wait for next VAD-triggered
  speech, no LLM call.
- LLM GPU OOM: catch, respond with a pre-baked (not generated) fallback
  audio clip, log the error, keep the WebSocket alive if possible.
- RAG returns nothing relevant: LLM prompt instructs it to say no
  matching listings were found rather than hallucinate.
- ngrok tunnel drops: browser detects WebSocket close, shows reconnect
  UI, client retries.
- LLM generation hangs: hard per-turn timeout (~15s), cancel and fall
  back to an error message.

## Testing

- Unit-level: RAG retrieval correctness (query → expected top-k items)
  against the seeded synthetic catalog.
- Manual end-to-end: browser test page, live voice test against the
  running Colab+ngrok endpoint — no automated audio-quality testing in
  scope.

## Explicitly out of scope for this build

- Real telephony (Plivo/Twilio) integration — future work.
- Cross-session history persistence beyond a single Colab runtime.
- Multi-tenant / multi-business inventory support.
