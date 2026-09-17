"""WebSocket orchestration server.

Wires VAD -> STT -> RAG -> LLM -> TTS into one turn, streaming clause audio
back as it's synthesized. Every model is injected so the full turn flow -
including barge-in and the language handoff - is testable with fakes, no
GPU or weights required.

Two things worth understanding before touching this file:

- `llm_engine.generate` is a synchronous generator. Without the explicit
  `await asyncio.sleep(0)` inside the generation loop, the event loop never
  regains control during generation, and the receive loop can't notice a
  barge-in until the turn ends on its own. This is the cheapest correct fix;
  a fully concurrent version would run generation in a thread executor.
- `time_to_first_audio` is the number that defines the experience, and it
  cannot be reconstructed from the other stage timings: streaming means
  stages overlap, so their sum exceeds the real elapsed time.
"""
import asyncio
import json
import time
import wave
from contextlib import contextmanager

from fastapi import FastAPI, WebSocket

from backend.history.store import HistoryStore
from backend.llm.prompt import build_prompt
from backend.tts.chunker import ClauseChunker

TOP_K = 5


@contextmanager
def _timed(stages: dict, name: str):
    """Record a stage's wall-clock duration in milliseconds."""
    start = time.perf_counter()
    yield
    stages[name] = round((time.perf_counter() - start) * 1000, 1)


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
        turn_task = None

        def start_turn(utterance: bytes):
            return asyncio.create_task(_handle_turn(
                websocket, session_id, utterance, transcriber, inventory_store,
                llm_engine, synthesizer, history_store, write_wav, scratch_audio_path,
            ))

        async def cancel_turn_if_running():
            """Barge-in: abandon the in-flight answer.

            Cancel, then await the task so its cancellation actually completes
            before we touch the socket again - Starlette websockets are not
            safe for concurrent sends from two coroutines.
            """
            nonlocal turn_task
            if turn_task is not None and not turn_task.done():
                turn_task.cancel()
                try:
                    await turn_task
                except asyncio.CancelledError:
                    pass
                await websocket.send_text(json.dumps({"type": "interrupted"}))
            turn_task = None

        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                await cancel_turn_if_running()
                break

            if message.get("bytes") is not None:
                chunk = message["bytes"]
                was_speaking = segmenter.speech_active
                utterance = segmenter.push(chunk)

                # Rising edge of speech while the agent is answering = barge-in.
                if not was_speaking and segmenter.speech_active:
                    await cancel_turn_if_running()

                buffer.extend(chunk)
                if utterance is not None:
                    buffer.clear()
                    turn_task = start_turn(utterance)

            elif message.get("text") == "__end__":
                await cancel_turn_if_running()
                utterance = bytes(buffer)
                buffer.clear()
                turn_task = start_turn(utterance)

    return app


async def _handle_turn(websocket, session_id, utterance_pcm, transcriber,
                       inventory_store, llm_engine, synthesizer, history_store,
                       write_wav, scratch_audio_path):
    stages = {}
    turn_start = time.perf_counter()

    with _timed(stages, "stt"):
        write_wav(utterance_pcm, scratch_audio_path)
        transcript = transcriber.transcribe(scratch_audio_path)

    language = transcript["language"]
    await websocket.send_text(json.dumps({
        "type": "transcript", "text": transcript["text"], "language": language,
    }))

    if not transcript["text"]:
        await websocket.send_text(json.dumps({"type": "turn_end"}))
        return

    with _timed(stages, "retrieval"):
        history = history_store.get(session_id)
        retrieved = inventory_store.query(transcript["text"], top_k=TOP_K)

    # language is used twice: here so the model answers in it, and below so
    # the matching TTS voice speaks it.
    prompt = build_prompt(transcript["text"], retrieved, history, language=language)

    chunker = ClauseChunker()
    full_answer = []
    first_audio_at = None
    tts_ms = 0.0

    async def speak(clause: str) -> None:
        nonlocal first_audio_at, tts_ms
        await websocket.send_text(json.dumps({"type": "answer_text", "text": clause}))
        tts_start = time.perf_counter()
        audio = synthesizer.synthesize(clause, language=language)
        tts_ms += (time.perf_counter() - tts_start) * 1000
        await websocket.send_bytes(audio)
        if first_audio_at is None:
            first_audio_at = time.perf_counter()

    llm_start = time.perf_counter()
    first_token_at = None

    for fragment in llm_engine.generate(prompt):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        full_answer.append(fragment)
        for clause in chunker.push(fragment):
            await speak(clause)
        # Yield to the event loop so an incoming barge-in can be noticed
        # mid-generation rather than only between turns.
        await asyncio.sleep(0)

    trailing = chunker.flush()
    if trailing:
        await speak(trailing)

    if first_token_at is not None:
        stages["llm_first_token"] = round((first_token_at - llm_start) * 1000, 1)
    stages["llm_total"] = round((time.perf_counter() - llm_start) * 1000, 1)
    stages["tts_total"] = round(tts_ms, 1)
    if first_audio_at is not None:
        stages["time_to_first_audio"] = round((first_audio_at - turn_start) * 1000, 1)
    stages["turn_total"] = round((time.perf_counter() - turn_start) * 1000, 1)

    history_store.append(session_id, "user", transcript["text"])
    history_store.append(session_id, "assistant", "".join(full_answer))

    await websocket.send_text(json.dumps({"type": "timing", "stages": stages}))
    await websocket.send_text(json.dumps({"type": "turn_end"}))
