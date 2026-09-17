"""Per-session conversation history, persisted to SQLite.

Lives on local Colab disk and is wiped on runtime reset — an accepted
tradeoff, since the catalog itself re-seeds fresh on every boot. Persists
across a HistoryStore reconnect within one runtime (the server can restart
without losing history mid-session), just not across a Colab restart.
"""
import sqlite3


class HistoryStore:
    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_session ON history(session_id, id)"
        )
        self._conn.commit()

    def append(self, session_id: str, role: str, text: str) -> None:
        self._conn.execute(
            "INSERT INTO history (session_id, role, text) VALUES (?, ?, ?)",
            (session_id, role, text),
        )
        self._conn.commit()

    def get(self, session_id: str, max_turns: int = 6) -> list[dict]:
        cursor = self._conn.execute(
            "SELECT role, text FROM history WHERE session_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, max_turns),
        )
        rows = cursor.fetchall()
        rows.reverse()  # newest-first from SQL, oldest-first for the prompt
        return [{"role": role, "text": text} for role, text in rows]
