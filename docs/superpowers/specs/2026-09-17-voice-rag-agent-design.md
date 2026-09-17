# Voice-to-Voice RAG Agent — Design

## Purpose

Phone-call-style agent: user speaks, agent transcribes, retrieves relevant
inventory listings via RAG, generates a spoken answer. Prototype targets
web/browser voice first; real telephony (Plivo/Twilio) is a later, separate
integration.

## Constraints

- No GPU available locally. Build/run on Google Colab (free GPU tier),
  expose endpoints via ngrok for testing.
- All models must coexist in a single GPU session (~15GB VRAM on free
  Colab T4): STT, LLM, TTS all resident at once.
- Streaming pipeline chosen over turn-based despite added complexity —
  user wants real-time feel, accepted the higher build/debug risk.

## Architecture

```
Browser (mic capture, WebSocket client, audio playback)
        │ WebSocket (audio chunks + control msgs)
        ▼
ngrok tunnel
        ▼
Colab notebook: FastAPI + WebSocket server (single GPU process)
   ├─ STT: faster-whisper (small/medium, int8), streaming decode
   ├─ RAG: Chroma (in-notebook, local) + sentence-transformers embeddings
   ├─ LLM: Qwen2.5-7B-Instruct, 4-bit quant (AWQ/GGUF via llama.cpp or vLLM)
   └─ TTS: Piper (CPU-only, streaming-friendly) — chosen over XTTS
     specifically to save VRAM headroom for the LLM
```

Single process, single GPU. Piper runs on CPU so only STT + LLM compete
for VRAM: quantized LLM (~5GB) + Whisper int8 (~1-2GB) should fit a 15GB
T4.

## Data flow (streaming turn)

1. Browser opens WebSocket, streams mic audio (16kHz PCM chunks)
   continuously.
2. Server runs VAD (webrtcvad or Silero-VAD) on incoming chunks to detect
   utterance start/end — this is the turn boundary in a streaming setup.
3. On speech-end: buffered audio finalized to text via faster-whisper.
4. Text embedded, Chroma similarity search over the inventory catalog,
   top-k listings retrieved.
5. Retrieved listings + user text + conversation history → prompt → local
   LLM generates response, streamed token-by-token.
6. As LLM tokens arrive, sentence/clause chunks are pulled off and fed to
   Piper incrementally — audio starts playing before the full LLM
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

## Inventory data

- Synthetic catalog, ~1000+ items (name, price, description, stock, etc),
  generated for this build — no real business data.
- Embedded once at startup into Chroma.

## Error handling

- ASR fails / empty transcript: skip turn, wait for next VAD-triggered
  speech, no LLM call.
- GPU OOM: catch, respond with a pre-baked (not generated) fallback audio
  clip, log the error, keep the WebSocket alive if possible.
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
