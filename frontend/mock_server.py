"""Mock backend for frontend development.

Implements the same /ws/{session_id} protocol as backend.server.app so the
UI can be built and tested without any real model, GPU, or Colab dependency.
Run: python -m frontend.mock_server
"""
import asyncio
import io
import json
import struct
import wave

import uvicorn
from fastapi import FastAPI, WebSocket

app = FastAPI()


def _fake_wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # ~0.25s square-wave tone so playback is audibly verifiable.
        samples = [int(6000 * ((i // 20) % 2 * 2 - 1)) for i in range(4000)]
        wf.writeframes(struct.pack("<%dh" % len(samples), *samples))
    return buffer.getvalue()


async def _run_turn(websocket: WebSocket):
    """Runs as a background task, mirroring backend.server.app._handle_turn.

    Must NOT run inline in the receive loop: if it did, a chunk sent while
    "speaking" would sit unread in the socket buffer until the turn finished,
    and the interrupted path could never actually fire. The real server has
    the identical requirement, for the identical reason.
    """
    await websocket.send_text(json.dumps({
        "type": "transcript", "text": "do you have headphones", "language": "en",
    }))
    await asyncio.sleep(0.3)

    for clause in ["Yes,", "we have Wireless Headphones in stock."]:
        await websocket.send_text(json.dumps({"type": "answer_text", "text": clause}))
        await websocket.send_bytes(_fake_wav_bytes())
        await asyncio.sleep(0.4)

    await websocket.send_text(json.dumps({"type": "timing", "stages": {
        "stt": 1800.0, "retrieval": 25.0, "llm_first_token": 900.0,
        "llm_total": 2100.0, "tts_total": 1200.0,
        "time_to_first_audio": 3100.0, "turn_total": 4300.0,
    }}))
    await websocket.send_text(json.dumps({"type": "turn_end"}))


@app.websocket("/ws/{session_id}")
async def ws_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    turn_task = None

    while True:
        message = await websocket.receive()

        if message.get("type") == "websocket.disconnect":
            if turn_task and not turn_task.done():
                turn_task.cancel()
            break

        if message.get("bytes") is not None:
            # A chunk arriving while a turn is in flight simulates barge-in,
            # so the frontend's interrupt handling is exercisable without a
            # real VAD.
            if turn_task is not None and not turn_task.done():
                turn_task.cancel()
                try:
                    await turn_task
                except asyncio.CancelledError:
                    pass
                await websocket.send_text(json.dumps({"type": "interrupted"}))
                turn_task = None
            continue

        if message.get("text") != "__end__":
            continue

        turn_task = asyncio.create_task(_run_turn(websocket))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
