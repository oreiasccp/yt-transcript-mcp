"""Tiny SQLite cache keyed by (video_id, lang). Avoids re-fetching/re-transcribing the
same video and keeps request volume low (helps stay under YouTube's flagging threshold)."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

_DEFAULT_DIR = Path(os.environ.get("YT_MCP_CACHE_DIR", Path.home() / ".cache" / "yt-transcript-mcp"))
_DB_PATH = _DEFAULT_DIR / "cache.db"

_conn: Optional[sqlite3.Connection] = None


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.execute(
            """CREATE TABLE IF NOT EXISTS transcripts (
                   video_id TEXT, lang TEXT, source TEXT,
                   payload TEXT, created_at REAL,
                   PRIMARY KEY (video_id, lang))"""
        )
        _conn.commit()
    return _conn


def get(video_id: str, lang: str) -> Optional[dict]:
    row = _db().execute(
        "SELECT payload FROM transcripts WHERE video_id=? AND lang=?", (video_id, lang)
    ).fetchone()
    return json.loads(row[0]) if row else None


def put(video_id: str, lang: str, source: str, payload: dict) -> None:
    _db().execute(
        "INSERT OR REPLACE INTO transcripts VALUES (?,?,?,?,?)",
        (video_id, lang, source, json.dumps(payload), time.time()),
    )
    _db().commit()
