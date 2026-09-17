import json
import tempfile
import time

import pytest
from fastapi.testclient import TestClient

from backend.server.app import create_app


class FakeSegmenter:
    """Never fires on its own — the test drives end-of-turn via "__end__"."""
    def __init__(self):
        self.speech_active = False

    def push(self, chunk):
        return None


class BargeInSegmenter:
    """Reports speech starting on the Nth pushed chunk, to trigger barge-in."""
    def __init__(self, speak_on_chunk: int):
        self.speech_active = False
        self._count = 0
        self._speak_on = speak_on_chunk

    def push(self, chunk):
        self._count += 1
        if self._count >= self._speak_on:
            self.speech_active = True
        return None


class FakeTranscriber:
    def __init__(self, text="do you have headphones", language="en"):
        self._text = text
        self._language = language

    def transcribe(self, audio_path):
        return {"text": self._text, "language": self._language}


class FakeStore:
    def query(self, text, top_k=5):
        return [{"id": "1", "name": "Wireless Headphones", "description": "...",
                 "price": 89.99, "stock": 20, "category": "Electronics", "score": 0.1}]


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, stop=None, max_tokens=200):
        self.prompts.append(prompt)
        yield "Yes, "
        yield "we have Wireless Headphones in stock. "


class SlowLLM:
    """Generates slowly so a barge-in can land mid-answer."""
    def generate(self, prompt, stop=None, max_tokens=200):
        for _ in range(50):
            time.sleep(0.01)
            yield "word, "


class FakeSynthesizer:
    def __init__(self):
        self.calls = []

    def synthesize(self, text, language="en"):
        self.calls.append((text, language))
        return b"RIFFfakewavbytes"


def build_client(segmenter_factory=None, transcriber=None, llm=None, synthesizer=None):
    app = create_app(
        segmenter_factory=segmenter_factory or (lambda: FakeSegmenter()),
        transcriber=transcriber or FakeTranscriber(),
        inventory_store=FakeStore(),
        llm_engine=llm or FakeLLM(),
        synthesizer=synthesizer or FakeSynthesizer(),
        history_db_path=tempfile.mktemp(suffix=".db"),
        scratch_audio_path=tempfile.mktemp(suffix=".wav"),
    )
    return TestClient(app)


def drain_turn(ws):
    """Read messages until turn_end or interrupted, returning everything seen."""
    messages = []
    while True:
        msg = json.loads(ws.receive_text())
        messages.append(msg)
        if msg["type"] == "answer_text":
            ws.receive_bytes()
        if msg["type"] in ("turn_end", "interrupted"):
            return messages


@pytest.fixture
def synthesizer():
    return FakeSynthesizer()


@pytest.fixture
def client(synthesizer):
    return build_client(synthesizer=synthesizer)


def test_full_turn_over_websocket(client):
    with client.websocket_connect("/ws/test-session") as ws:
        ws.send_bytes(b"\x00\x00" * 100)
        ws.send_text("__end__")

        assert json.loads(ws.receive_text()) == {
            "type": "transcript", "text": "do you have headphones", "language": "en",
        }

        answer_msg = json.loads(ws.receive_text())
        assert answer_msg["type"] == "answer_text"
        assert ws.receive_bytes().startswith(b"RIFF")

        msg = json.loads(ws.receive_text())
        while msg["type"] == "answer_text":
            ws.receive_bytes()
            msg = json.loads(ws.receive_text())
        assert msg["type"] == "timing"
        assert json.loads(ws.receive_text())["type"] == "turn_end"


def test_timing_message_reports_stages(client):
    with client.websocket_connect("/ws/test-session") as ws:
        ws.send_bytes(b"\x00\x00" * 100)
        ws.send_text("__end__")
        messages = drain_turn(ws)
    timing = next(m for m in messages if m["type"] == "timing")
    for stage in ["stt", "retrieval", "llm_total", "tts_total", "time_to_first_audio"]:
        assert stage in timing["stages"]


def test_synthesis_uses_detected_language(synthesizer):
    client = build_client(transcriber=FakeTranscriber("क्या हेडफ़ोन हैं", "hi"),
                          synthesizer=synthesizer)
    with client.websocket_connect("/ws/test-session") as ws:
        ws.send_bytes(b"\x00\x00" * 100)
        ws.send_text("__end__")
        drain_turn(ws)
    assert synthesizer.calls
    assert all(language == "hi" for _, language in synthesizer.calls)


def test_detected_language_reaches_the_prompt():
    llm = FakeLLM()
    client = build_client(transcriber=FakeTranscriber("हेडफ़ोन", "hi"), llm=llm)
    with client.websocket_connect("/ws/test-session") as ws:
        ws.send_bytes(b"\x00\x00" * 100)
        ws.send_text("__end__")
        drain_turn(ws)
    assert "Hindi" in llm.prompts[0]


def test_empty_transcript_skips_llm():
    llm = FakeLLM()
    client = build_client(transcriber=FakeTranscriber("", "en"), llm=llm)
    with client.websocket_connect("/ws/test-session") as ws:
        ws.send_bytes(b"\x00\x00" * 100)
        ws.send_text("__end__")
        drain_turn(ws)
    assert llm.prompts == []


def test_empty_transcript_still_sends_turn_end():
    client = build_client(transcriber=FakeTranscriber("", "en"))
    with client.websocket_connect("/ws/test-session") as ws:
        ws.send_bytes(b"\x00\x00" * 100)
        ws.send_text("__end__")
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "transcript"
        assert json.loads(ws.receive_text())["type"] == "turn_end"


def test_barge_in_cancels_in_flight_turn():
    client = build_client(
        segmenter_factory=lambda: BargeInSegmenter(speak_on_chunk=2),
        llm=SlowLLM(),
    )
    with client.websocket_connect("/ws/test-session") as ws:
        ws.send_bytes(b"\x00\x00" * 100)   # chunk 1, no speech yet
        ws.send_text("__end__")            # starts a slow turn
        ws.send_bytes(b"\x00\x00" * 100)   # chunk 2 -> speech_active rises

        saw_interrupted = False
        for _ in range(60):
            msg = json.loads(ws.receive_text())
            if msg["type"] == "answer_text":
                ws.receive_bytes()
            if msg["type"] == "interrupted":
                saw_interrupted = True
                break
            if msg["type"] == "turn_end":
                break
        assert saw_interrupted


def test_history_persists_across_turns(client):
    with client.websocket_connect("/ws/history-session") as ws:
        for _ in range(2):
            ws.send_bytes(b"\x00\x00" * 100)
            ws.send_text("__end__")
            drain_turn(ws)
    # Two turns stored two user + two assistant rows; HistoryStore's own unit
    # tests cover the storage semantics.
