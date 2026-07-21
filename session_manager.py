"""
Session persistence for MAZU Agent — SQLite-backed conversation storage.

Usage:
    from session_manager import SessionManager

    sm = SessionManager()
    sid = sm.create_session()
    sm.save_messages(sid, st.session_state.messages, st.session_state.display)
    sessions = sm.list_sessions()
    data = sm.get_session(sid)
"""

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone

log = logging.getLogger("mazu.session")


class SessionManager:
    """SQLite-backed session store for MAZU Agent conversations.

    Each session holds a dual-list conversation:
    - ``messages`` — OpenAI-format list (system / user / assistant / tool)
    - ``display`` — UI render list (user / assistant with inlined tool_calls)

    Both are serialised to the database and restored together.
    The system prompt is NOT persisted; it is regenerated on each restore.
    """

    def __init__(self, db_path: str = "sessions.db"):
        project_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(project_dir, db_path)
        self._init_db()

    # ── schema ──

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id         TEXT PRIMARY KEY,
                    title      TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role            TEXT NOT NULL,
                    content         TEXT,
                    tool_calls_json TEXT,
                    tool_call_id    TEXT,
                    display_json    TEXT,
                    created_at      TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, id);
            """)

    # ── public API ──

    def create_session(self, title: str = "") -> str:
        """Create a new empty session. Returns the session UUID."""
        sid = uuid.uuid4().hex[:12]
        now = _now()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO sessions(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (sid, title, now, now),
            )
        log.info("Created session %s", sid)
        return sid

    def list_sessions(self) -> list[dict]:
        """Return all sessions ordered by most-recently-updated first."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT s.id, s.title, s.created_at, s.updated_at,
                       COUNT(m.id) AS msg_count
                FROM sessions s
                LEFT JOIN messages m ON m.session_id = s.id
                GROUP BY s.id
                ORDER BY s.updated_at DESC
            """).fetchall()
        return [
            {
                "id": r[0],
                "title": r[1],
                "created_at": r[2],
                "updated_at": r[3],
                "message_count": r[4],
            }
            for r in rows
        ]

    def get_session(self, session_id: str) -> dict | None:
        """Restore a session, returning {messages, display}.

        Returns None if the session does not exist.
        The system prompt is NOT returned — the caller prepends it fresh.
        """
        with sqlite3.connect(self.db_path) as conn:
            exists = conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if not exists:
                return None

            rows = conn.execute(
                "SELECT role, content, tool_calls_json, tool_call_id, display_json "
                "FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()

        messages = []
        display = []

        for role, content, tc_json, tc_id, disp_json in rows:
            # ── Rebuild LLM message ──
            msg = {"role": role}
            if content is not None:
                msg["content"] = content
            if tc_json:
                msg["tool_calls"] = json.loads(tc_json)
            if tc_id:
                msg["tool_call_id"] = tc_id
            messages.append(msg)

            # ── Rebuild display entry ──
            if disp_json:
                display.append(json.loads(disp_json))

        return {"messages": messages, "display": display}

    def save_messages(self, session_id: str, messages: list, display: list) -> None:
        """Persist the full conversation state for *session_id*.

        Skips ``messages[0]`` (the system prompt — regenerated each restore).
        If *display* is empty we still persist (the session exists but has no
        user messages yet — just update the timestamp).
        """
        now = _now()
        with sqlite3.connect(self.db_path) as conn:
            # Replace all messages for this session in a single transaction
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))

            if display:
                conn.executemany(
                    "INSERT INTO messages(session_id, role, content, tool_calls_json, "
                    "tool_call_id, display_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    _serialise_messages(session_id, messages, display, now),
                )

            # Update title from first user message if still empty
            row = conn.execute(
                "SELECT title FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row and not row[0] and display:
                first_user = next(
                    (d["content"] for d in display if d["role"] == "user"), ""
                )
                title = first_user[:30].replace("\n", " ").strip()
                if title:
                    conn.execute(
                        "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                        (title, now, session_id),
                    )
                    return

            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )

    def delete_session(self, session_id: str) -> None:
        """Delete a session and all its messages."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        log.info("Deleted session %s", session_id)


# ── helpers ──

def _normalise(msg) -> dict:
    """Convert a message (dict or Pydantic SDK object) to a plain dict."""
    if isinstance(msg, dict):
        return msg
    return {
        "role": getattr(msg, "role", ""),
        "content": getattr(msg, "content", None),
        "tool_calls": getattr(msg, "tool_calls", None),
        "tool_call_id": getattr(msg, "tool_call_id", None),
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialise_messages(
    session_id: str, messages: list, display: list, now: str
) -> list[tuple]:
    """Convert messages + display into rows for the messages table.

    Skips messages[0] (system prompt).
    Each display entry is attached to its corresponding assistant message.
    """
    rows = []
    di = 0  # display index (incremented for each display entry consumed)

    for mi, msg in enumerate(messages):
        if mi == 0:
            continue  # skip system prompt

        # Normalise: handle both dict and Pydantic SDK objects
        m = _normalise(msg)

        role = m.get("role", "")
        content = m.get("content")
        tc_json = None
        tc_id = m.get("tool_call_id")

        # Serialise tool_calls if present
        tcs = m.get("tool_calls")
        if tcs:
            # Normalise each tool_call too (may be SDK objects)
            norm_tcs = []
            for tc in tcs:
                if isinstance(tc, dict):
                    norm_tcs.append(tc)
                else:
                    norm_tcs.append({
                        "id": getattr(tc, "id", ""),
                        "type": getattr(tc, "type", "function"),
                        "function": {
                            "name": getattr(getattr(tc, "function", None), "name", ""),
                            "arguments": getattr(getattr(tc, "function", None), "arguments", ""),
                        },
                    })
            tc_json = json.dumps(norm_tcs, ensure_ascii=False)

        # Serialise display entry
        # display entries are ordered: [user, assistant, user, assistant, ...]
        # We attach the next unconsumed display entry to each user/assistant message
        disp_json = None
        if di < len(display):
            disp_entry = display[di]
            # User display: matches when current msg is user
            if role == "user" and disp_entry["role"] == "user":
                disp_json = json.dumps(disp_entry, ensure_ascii=False)
                di += 1
            # Assistant display: matches when current msg is assistant with text content
            elif role == "assistant" and disp_entry["role"] == "assistant" and content:
                disp_json = json.dumps(disp_entry, ensure_ascii=False)
                di += 1

        rows.append(
            (session_id, role, content, tc_json, tc_id, disp_json, now)
        )

    return rows
