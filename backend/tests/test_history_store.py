import os
import tempfile

import pytest

from backend.history.store import HistoryStore


@pytest.fixture
def store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield HistoryStore(db_path=path)
    os.remove(path)


def test_append_and_get(store):
    store.append("session1", "user", "hello")
    store.append("session1", "assistant", "hi there")
    assert store.get("session1") == [
        {"role": "user", "text": "hello"},
        {"role": "assistant", "text": "hi there"},
    ]


def test_sessions_are_isolated(store):
    store.append("session1", "user", "a")
    store.append("session2", "user", "b")
    assert store.get("session1") == [{"role": "user", "text": "a"}]


def test_get_windows_to_max_turns_keeping_newest(store):
    for i in range(10):
        store.append("session1", "user", f"turn {i}")
    history = store.get("session1", max_turns=3)
    assert [h["text"] for h in history] == ["turn 7", "turn 8", "turn 9"]


def test_get_unknown_session_returns_empty(store):
    assert store.get("nobody") == []


def test_persists_across_store_instances(store, tmp_path):
    """The point of SQLite over an in-memory dict: it must survive a
    reconnect, since the server can restart mid-session on Colab."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        s1 = HistoryStore(db_path=path)
        s1.append("session1", "user", "remember this")
        s2 = HistoryStore(db_path=path)
        assert s2.get("session1") == [{"role": "user", "text": "remember this"}]
    finally:
        os.remove(path)


def test_handles_unicode_text(store):
    """Hindi and Marathi transcripts must round-trip without mangling."""
    store.append("session1", "user", "क्या आपके पास हेडफ़ोन हैं")
    assert store.get("session1") == [
        {"role": "user", "text": "क्या आपके पास हेडफ़ोन हैं"}
    ]
