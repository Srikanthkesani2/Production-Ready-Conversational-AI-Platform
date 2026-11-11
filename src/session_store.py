"""Persistent per-session chat history, backed by SQLite.

History is stored in a local SQLite file so conversations survive page
refreshes and app restarts (unlike Streamlit's in-memory session_state).
RunnableWithMessageHistory calls `get_session_history` to read/write messages.
"""
from __future__ import annotations

import os

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from sqlalchemy import create_engine, text

# Store the DB at the project root (one level up from src/).
_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chat_history.db")
DB_URL = f"sqlite:///{_DB_PATH}"
TABLE = "message_store"

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return a SQLite-backed history for the given session id."""
    return SQLChatMessageHistory(
        session_id=session_id, connection=DB_URL, table_name=TABLE
    )


def list_sessions() -> list[str]:
    """Return all session ids that already have stored messages."""
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT DISTINCT session_id FROM {TABLE}")
            ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        # Table doesn't exist yet (no messages stored).
        return []


def message_count(session_id: str) -> int:
    """Number of stored messages for a session (cheap COUNT query)."""
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT COUNT(*) FROM {TABLE} WHERE session_id = :sid"),
                {"sid": session_id},
            ).fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def clear_session(session_id: str) -> None:
    """Delete all stored messages for a session."""
    get_session_history(session_id).clear()
