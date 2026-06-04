"""Parse YouTube caption payloads (json3 / srv / vtt) into segments + extract video IDs."""

from __future__ import annotations

import json
import re
from typing import Optional

# Segment shape used everywhere: {"text": str, "start": float, "duration": float}
Segment = dict


_VIDEO_ID_RE = re.compile(r"[0-9A-Za-z_-]{11}")


def extract_video_id(url_or_id: str) -> Optional[str]:
    """Pull an 11-char YouTube video ID from a URL, short URL, or bare ID."""
    s = url_or_id.strip()
    # Bare ID
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", s):
        return s
    # Common URL params/paths: watch?v=, youtu.be/, /shorts/, /embed/, /live/
    patterns = [
        r"(?:v=|/shorts/|/embed/|/live/|youtu\.be/)([0-9A-Za-z_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, s)
        if m:
            return m.group(1)
    # Last resort: first 11-char token that looks like an ID
    m = _VIDEO_ID_RE.search(s)
    return m.group(0) if m else None


def parse_json3(raw: bytes | str) -> list[Segment]:
    """Parse the YouTube json3 caption format into segments.

    json3 shape: {"events": [{"tStartMs": int, "dDurationMs": int,
                              "segs": [{"utf8": str}, ...]}, ...]}
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    data = json.loads(raw)
    out: list[Segment] = []
    for ev in data.get("events", []):
        segs = ev.get("segs")
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs).strip()
        if not text or text == "\n":
            continue
        start = ev.get("tStartMs", 0) / 1000.0
        dur = ev.get("dDurationMs", 0) / 1000.0
        out.append({"text": text, "start": round(start, 3), "duration": round(dur, 3)})
    return out


_VTT_TS = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)
_VTT_TAG = re.compile(r"<[^>]+>")


def _ts_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(raw: bytes | str) -> list[Segment]:
    """Parse WebVTT / SRT-ish caption text into segments. Dedupes the rolling
    duplicate lines YouTube auto-captions emit."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    lines = raw.splitlines()
    out: list[Segment] = []
    i = 0
    last_text = None
    while i < len(lines):
        m = _VTT_TS.search(lines[i])
        if not m:
            i += 1
            continue
        start = _ts_to_seconds(m.group(1), m.group(2), m.group(3), m.group(4))
        end = _ts_to_seconds(m.group(5), m.group(6), m.group(7), m.group(8))
        i += 1
        text_lines = []
        while i < len(lines) and lines[i].strip() and not _VTT_TS.search(lines[i]):
            text_lines.append(_VTT_TAG.sub("", lines[i]).strip())
            i += 1
        text = " ".join(t for t in text_lines if t).strip()
        if not text or text == last_text:
            continue
        last_text = text
        out.append(
            {"text": text, "start": round(start, 3), "duration": round(end - start, 3)}
        )
    return out


def segments_to_text(segments: list[Segment], include_timestamps: bool = True) -> str:
    """Render segments as markdown-ish plain text."""
    if include_timestamps:
        return "\n".join(f"[{s['start']:.1f}s] {s['text']}" for s in segments)
    return " ".join(s["text"] for s in segments)
