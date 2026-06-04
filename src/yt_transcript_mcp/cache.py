"""Tiny SQLite cache keyed by (video_id, lang). Avoids re-fetching/re-transcribing the
same video and keeps request volume low (helps stay under YouTube's flagging threshold).

Thread-safe: FastMCP runs sync tools in a worker threadpool, so the cache can be hit
concurrently. We open the connection in WAL mode (concurrent readers + one writer) and
guard every write with a process-wide lock to avoid `database is locked` / interleaved
writes on the single shared connection.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

_DEFAULT_DIR = Path(os.environ.get("YT_MCP_CACHE_DIR", Path.home() / ".cache" / "yt-transcript-mcp"))
_DB_PATH = _DEFAULT_DIR / "cache.db"

_conn: Optional[sqlite3.Connection] = None
_lock = threading.Lock()  # guards _conn init and all writes


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        with _lock:
            if _conn is None:  # double-checked: another thread may have built it
                _DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False, timeout=30.0)
                # WAL: concurrent readers don't block the writer; survives crashes better.
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA busy_timeout=30000")
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS transcripts (
                           video_id TEXT, lang TEXT, source TEXT,
                           payload TEXT, created_at REAL,
                           PRIMARY KEY (video_id, lang))"""
                )
                conn.commit()
                _conn = conn
    return _conn


def get(video_id: str, lang: str) -> Optional[dict]:
    """Read a cached payload. Reads are safe without the lock under WAL."""
    row = _db().execute(
        "SELECT payload FROM transcripts WHERE video_id=? AND lang=?", (video_id, lang)
    ).fetchone()
    return json.loads(row[0]) if row else None


def put(video_id: str, lang: str, source: str, payload: dict) -> None:
    """Write a payload. Serialized through _lock so concurrent worker threads don't clash."""
    conn = _db()
    with _lock:
        conn.execute(
            "INSERT OR REPLACE INTO transcripts VALUES (?,?,?,?,?)",
            (video_id, lang, source, json.dumps(payload), time.time()),
        )
        conn.commit()


def local_file_key(path: str) -> str:
    """Stable cache key for a local file: identity = abspath + mtime + size.

    If the file is edited/replaced, mtime or size changes and the key changes, so a stale
    transcript is never returned for new content. Hashed to keep the key short.
    """
    st = os.stat(path)
    raw = f"{os.path.abspath(path)}|{int(st.st_mtime)}|{st.st_size}"
    return "local:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
