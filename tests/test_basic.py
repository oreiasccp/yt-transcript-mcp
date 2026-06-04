"""Fast, offline unit tests — no network, no GPU, no model load.

Covers the pure logic paths: caption parsing, video-ID extraction, segment rendering,
cache round-trip + thread-safety, the local-file cache key, and the local-transcribe
error path for a missing file. Whisper/yt-dlp paths need real media/network and are
exercised manually, not here.

Run:  uv run pytest -q   (or)  .venv/Scripts/python.exe -m pytest -q
"""

from __future__ import annotations

import asyncio
import concurrent.futures

from yt_transcript_mcp import cache, parse, server


# --------------------------------------------------------------------------- parse

def test_extract_video_id_variants():
    assert parse.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert parse.extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert parse.extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert parse.extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_parse_json3_basic():
    raw = (
        '{"events":[{"tStartMs":0,"dDurationMs":1500,"segs":[{"utf8":"hello "},{"utf8":"world"}]},'
        '{"tStartMs":1500,"dDurationMs":500,"segs":[{"utf8":"\\n"}]}]}'
    )
    segs = parse.parse_json3(raw)
    assert len(segs) == 1
    assert segs[0]["text"] == "hello world"
    assert segs[0]["start"] == 0.0
    assert segs[0]["duration"] == 1.5


def test_parse_vtt_dedupes_rolling_lines():
    raw = (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.000\nhello\n\n"
        "00:00:01.000 --> 00:00:02.000\nhello\n\n"  # duplicate -> dropped
        "00:00:02.000 --> 00:00:03.000\nworld\n"
    )
    segs = parse.parse_vtt(raw)
    assert [s["text"] for s in segs] == ["hello", "world"]


def test_segments_to_text_timestamps_toggle():
    segs = [{"text": "a", "start": 0.0, "duration": 1.0}, {"text": "b", "start": 1.0, "duration": 1.0}]
    assert parse.segments_to_text(segs, True) == "[0.0s] a\n[1.0s] b"
    assert parse.segments_to_text(segs, False) == "a b"


# --------------------------------------------------------------------------- cache

def test_cache_roundtrip(tmp_path, monkeypatch):
    # Point the cache at a temp dir and reset the module-level connection.
    monkeypatch.setenv("YT_MCP_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(cache, "_DEFAULT_DIR", tmp_path)
    monkeypatch.setattr(cache, "_DB_PATH", tmp_path / "cache.db")
    monkeypatch.setattr(cache, "_conn", None)

    assert cache.get("vid1", "en") is None
    cache.put("vid1", "en", "captions:manual", {"segments": [1, 2, 3]})
    got = cache.get("vid1", "en")
    assert got == {"segments": [1, 2, 3]}


def test_cache_concurrent_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_DEFAULT_DIR", tmp_path)
    monkeypatch.setattr(cache, "_DB_PATH", tmp_path / "cache.db")
    monkeypatch.setattr(cache, "_conn", None)

    def write(i: int) -> None:
        cache.put(f"vid{i}", "en", "src", {"i": i})

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(write, range(50)))  # must not raise "database is locked"

    assert cache.get("vid7", "en") == {"i": 7}
    assert cache.get("vid49", "en") == {"i": 49}


def test_local_file_key_changes_with_content(tmp_path):
    f = tmp_path / "a.wav"
    f.write_bytes(b"x" * 10)
    k1 = cache.local_file_key(str(f))
    f.write_bytes(b"y" * 20)  # size changes -> key changes
    k2 = cache.local_file_key(str(f))
    assert k1 != k2
    assert k1.startswith("local:")


# --------------------------------------------------------------------------- server tool

def test_transcribe_local_file_missing():
    out = asyncio.run(server.transcribe_local_file(r"C:\definitely\nope.wav"))
    assert "error" in out and "not found" in out["error"].lower()


def test_path_allowed_unrestricted(monkeypatch):
    monkeypatch.setattr(server, "_ALLOWED_DIRS", [])
    assert server._path_allowed("/anywhere/file.wav") is True


def test_path_allowed_enforced(tmp_path, monkeypatch):
    allowed = tmp_path / "ok"
    allowed.mkdir()
    monkeypatch.setattr(server, "_ALLOWED_DIRS", [str(allowed)])
    assert server._path_allowed(str(allowed / "a.wav")) is True
    assert server._path_allowed(str(tmp_path / "outside.wav")) is False


def test_transcribe_local_file_access_denied(tmp_path, monkeypatch):
    f = tmp_path / "real.wav"
    f.write_bytes(b"RIFF0000")  # exists, but outside the allowlist
    monkeypatch.setattr(server, "_ALLOWED_DIRS", [str(tmp_path / "elsewhere")])
    out = asyncio.run(server.transcribe_local_file(str(f)))
    assert "error" in out and "Access denied" in out["error"]


def test_listing_markdown_render():
    env = {
        "count": 1, "offset": 0, "has_more": True, "next_offset": 1,
        "items": [{"id": "x", "title": "T", "channel": "C", "duration": 5, "view_count": 9, "url": "u"}],
    }
    md = server._listing_markdown(env, "Title")
    assert "# Title" in md and "next_offset=1" in md and "**T** (x)" in md
