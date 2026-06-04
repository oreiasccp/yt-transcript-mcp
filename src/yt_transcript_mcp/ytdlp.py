"""yt-dlp layer: caption fetch with cookieless->cookies escalation, audio download,
search, channel, playlist, and RSS latest-uploads. Uses the yt-dlp Python API so the
HTTP session/headers/cookies stay consistent across info extraction and caption fetch.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Optional
from urllib.parse import quote

# defusedxml hardens against XXE / billion-laughs in the RSS feed parse.
from defusedxml import ElementTree as ET

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from . import parse

# Browser yt-dlp reads cookies from on the escalation retry. Override with env.
COOKIES_BROWSER = os.environ.get("YT_COOKIES_BROWSER", "chrome").strip()

# Substrings that mean "YouTube flagged this request, retry with cookies".
_BOT_MARKERS = (
    "sign in to confirm",
    "confirm you're not a bot",
    "confirm you’re not a bot",
    "login_required",
    "this video is only available to",
    "http error 403",
)

# Caption format preference: json3 is cleanest to parse, then srv3, then vtt.
_SUB_FORMAT_PREF = ("json3", "srv3", "srv1", "vtt")


def _is_bot_block(err: Exception) -> bool:
    msg = str(err).lower()
    return any(m in msg for m in _BOT_MARKERS)


def _base_opts(use_cookies: bool, **extra: Any) -> dict:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "retries": 2,
        "extractor_retries": 2,
        # android_vr + web_safari avoid PO-token requirements for player/GVS in most cases.
        "extractor_args": {"youtube": {"player_client": ["android_vr", "web_safari"]}},
    }
    if use_cookies and COOKIES_BROWSER:
        opts["cookiesfrombrowser"] = (COOKIES_BROWSER,)
    opts.update(extra)
    return opts


def with_escalation(make_opts: Callable[[bool], dict], run: Callable[[YoutubeDL], Any]) -> Any:
    """Run `run` with cookieless opts; on a bot/login block, retry once with cookies.

    `make_opts(use_cookies)` builds the YoutubeDL options for each attempt.
    """
    try:
        with YoutubeDL(make_opts(False)) as ydl:
            return run(ydl)
    except DownloadError as e:
        if not _is_bot_block(e):
            raise
        # Escalate: reuse the logged-in browser session (e.g. your Premium account).
        with YoutubeDL(make_opts(True)) as ydl:
            return run(ydl)


def _canonical_url(url_or_id: str) -> str:
    vid = parse.extract_video_id(url_or_id)
    if vid and ("/" not in url_or_id or "watch?v=" in url_or_id or "youtu.be" in url_or_id
                or "/shorts/" in url_or_id):
        return f"https://www.youtube.com/watch?v={vid}"
    return url_or_id


# --------------------------------------------------------------------------- captions

def _pick_track(tracks_by_lang: dict, lang: Optional[str]) -> Optional[list[dict]]:
    if not tracks_by_lang:
        return None
    if lang:
        # exact, then prefix match (e.g. "en" matches "en-US")
        if lang in tracks_by_lang:
            return tracks_by_lang[lang]
        for k, v in tracks_by_lang.items():
            if k.split("-")[0] == lang.split("-")[0]:
                return v
    # default: English-ish first, else whatever exists
    for k in tracks_by_lang:
        if k.startswith("en"):
            return tracks_by_lang[k]
    return next(iter(tracks_by_lang.values()))


def _ordered_formats(track: list[dict]) -> list[dict]:
    def rank(fmt: dict) -> int:
        ext = fmt.get("ext", "")
        return _SUB_FORMAT_PREF.index(ext) if ext in _SUB_FORMAT_PREF else len(_SUB_FORMAT_PREF)
    return sorted(track, key=rank)


def _meta(info: dict) -> dict:
    """Lean metadata block pulled from an already-fetched info dict (zero extra requests)."""
    return {
        "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"),
        "channel_id": info.get("channel_id"),
        "channel_url": info.get("channel_url") or info.get("uploader_url"),
        "duration": info.get("duration"),
        "duration_string": info.get("duration_string"),
        "view_count": info.get("view_count"),
        "upload_date": info.get("upload_date"),
        "thumbnail": info.get("thumbnail"),
        "webpage_url": info.get("webpage_url"),
    }


def fetch_captions(url_or_id: str, lang: Optional[str] = None) -> Optional[dict]:
    """Return {"segments", "lang", "kind", "used_cookies"} or None if no captions exist.

    Tries manual subtitles first, then automatic captions. Iterates caption formats and
    tracks defensively to dodge yt-dlp's empty/zero-byte-track bug (#13443).
    """
    url = _canonical_url(url_or_id)
    used_cookies = {"v": False}

    def make_opts(use_cookies: bool) -> dict:
        used_cookies["v"] = use_cookies
        return _base_opts(
            use_cookies,
            writesubtitles=False,
            writeautomaticsub=False,  # we read URLs from info, not files
        )

    def run(ydl: YoutubeDL) -> Optional[dict]:
        info = ydl.extract_info(url, download=False)
        manual = info.get("subtitles") or {}
        auto = info.get("automatic_captions") or {}
        for kind, store in (("manual", manual), ("auto", auto)):
            track = _pick_track(store, lang)
            if not track:
                continue
            track_lang = next(
                (k for k, v in store.items() if v is track), lang or "unknown"
            )
            for fmt in _ordered_formats(track):
                sub_url = fmt.get("url")
                if not sub_url:
                    continue
                try:
                    raw = ydl.urlopen(sub_url).read()
                except Exception:
                    continue
                if not raw or len(raw.strip()) < 3:
                    continue  # empty/zero-byte track -> try next format/kind
                try:
                    segs = (
                        parse.parse_json3(raw)
                        if fmt.get("ext") == "json3"
                        else parse.parse_vtt(raw)
                    )
                except Exception:
                    continue
                if segs:
                    return {
                        "segments": segs,
                        "lang": track_lang,
                        "kind": kind,
                        "used_cookies": used_cookies["v"],
                        "metadata": _meta(info),
                    }
        return None

    return with_escalation(make_opts, run)


# --------------------------------------------------------------------------- audio

def download_audio(url_or_id: str, dest_dir: str) -> dict:
    """Download audio to dest_dir for Whisper fallback. Returns {"path", "info", "used_cookies"}.

    Uses format 18 (360p progressive, audio adequate for ASR) as a SABR-resistant fallback,
    else bestaudio. Extracts to m4a/opus via ffmpeg.
    """
    url = _canonical_url(url_or_id)
    used_cookies = {"v": False}
    outtmpl = os.path.join(dest_dir, "%(id)s.%(ext)s")

    def make_opts(use_cookies: bool) -> dict:
        used_cookies["v"] = use_cookies
        return _base_opts(
            use_cookies,
            skip_download=False,
            format="bestaudio/18",
            outtmpl=outtmpl,
            postprocessors=[
                {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "0"}
            ],
        )

    def run(ydl: YoutubeDL) -> dict:
        info = ydl.extract_info(url, download=True)
        vid = info["id"]
        path = os.path.join(dest_dir, f"{vid}.mp3")
        return {
            "path": path,
            "info": info,
            "used_cookies": used_cookies["v"],
            "metadata": _meta(info),
        }

    return with_escalation(make_opts, run)


# --------------------------------------------------------------------------- metadata

def video_info(url_or_id: str) -> dict:
    url = _canonical_url(url_or_id)

    def make_opts(use_cookies: bool) -> dict:
        return _base_opts(use_cookies)

    def run(ydl: YoutubeDL) -> dict:
        info = ydl.extract_info(url, download=False)
        return {
            "id": info.get("id"),
            "title": info.get("title"),
            "channel": info.get("channel") or info.get("uploader"),
            "channel_id": info.get("channel_id"),
            "channel_url": info.get("channel_url") or info.get("uploader_url"),
            "duration": info.get("duration"),
            "duration_string": info.get("duration_string"),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "upload_date": info.get("upload_date"),
            "description": (info.get("description") or "")[:2000],
            "thumbnail": info.get("thumbnail"),
            "has_subtitles": bool(info.get("subtitles")),
            "has_auto_captions": bool(info.get("automatic_captions")),
            "webpage_url": info.get("webpage_url"),
        }

    return with_escalation(make_opts, run)


# --------------------------------------------------------------------------- flat listings

def _flat_entries(target: str, limit: int, offset: int = 0) -> dict:
    """Flat-extract a listing and return a paginated envelope.

    Fetches one extra entry (offset+limit+1) so we can report `has_more` without a second
    request, then slices the requested window. YouTube doesn't expose a reliable total for
    search/channel listings, so `total` is omitted; `has_more`/`next_offset` drive paging.
    """
    fetch_end = offset + limit + 1  # +1 sentinel to detect a further page

    def make_opts(use_cookies: bool) -> dict:
        return _base_opts(
            use_cookies,
            extract_flat="in_playlist",
            playliststart=1,
            playlistend=fetch_end,
            noplaylist=False,
        )

    def run(ydl: YoutubeDL) -> dict:
        info = ydl.extract_info(target, download=False)
        entries = [e for e in (info.get("entries") or []) if e]
        has_more = len(entries) > offset + limit
        window = entries[offset : offset + limit]
        items = [
            {
                "id": e.get("id"),
                "title": e.get("title"),
                "url": e.get("url") or (f"https://www.youtube.com/watch?v={e.get('id')}"),
                "duration": e.get("duration"),
                "view_count": e.get("view_count"),
                "channel": e.get("channel") or e.get("uploader"),
                "channel_id": e.get("channel_id"),
            }
            for e in window
        ]
        return {
            "count": len(items),
            "offset": offset,
            "items": items,
            "has_more": has_more,
            "next_offset": offset + len(items) if has_more else None,
        }

    return with_escalation(make_opts, run)


def search(query: str, search_type: str = "video", limit: int = 20, offset: int = 0) -> dict:
    if search_type == "channel":
        # YouTube channel-search results page; query must be URL-encoded.
        target = f"https://www.youtube.com/results?search_query={quote(query)}&sp=EgIQAg%253D%253D"
        return _flat_entries(target, limit, offset)
    # ytsearchN:<query> — N is the raw count yt-dlp fetches from the top; query is literal
    # search syntax (NOT a URL), so it must NOT be percent-encoded.
    target = f"ytsearch{offset + limit + 1}:{query}"
    return _flat_entries(target, limit, offset)


def _channel_base(channel: str) -> str:
    c = channel.strip()
    if c.startswith("UC") and len(c) == 24:
        return f"https://www.youtube.com/channel/{c}"
    if c.startswith("@"):
        return f"https://www.youtube.com/{c}"
    if c.startswith("http"):
        return c.rstrip("/")
    return f"https://www.youtube.com/@{c}"


def list_channel_videos(channel: str, limit: int = 50, offset: int = 0) -> dict:
    return _flat_entries(_channel_base(channel) + "/videos", limit, offset)


def search_channel_videos(channel: str, query: str, limit: int = 30, offset: int = 0) -> dict:
    target = f"{_channel_base(channel)}/search?query={quote(query)}"
    return _flat_entries(target, limit, offset)


def list_playlist_videos(playlist: str, limit: int = 100, offset: int = 0) -> dict:
    p = playlist.strip()
    if not p.startswith("http"):
        p = f"https://www.youtube.com/playlist?list={p}"
    return _flat_entries(p, limit, offset)


# --------------------------------------------------------------------------- RSS (free)

def _resolve_channel_id(channel: str) -> Optional[str]:
    c = channel.strip()
    if c.startswith("UC") and len(c) == 24:
        return c
    m = re.search(r"/channel/(UC[0-9A-Za-z_-]{22})", c)
    if m:
        return m.group(1)

    def make_opts(use_cookies: bool) -> dict:
        return _base_opts(use_cookies, extract_flat=True, playlistend=1)

    def run(ydl: YoutubeDL) -> Optional[str]:
        info = ydl.extract_info(_channel_base(channel), download=False)
        return info.get("channel_id") or info.get("uploader_id")

    return with_escalation(make_opts, run)


def latest_videos(channel: str) -> list[dict]:
    """~15 most recent uploads via the public RSS feed. No credits, no auth, no PO token."""
    cid = _resolve_channel_id(channel)
    if not cid:
        raise ValueError(f"Could not resolve channel id for {channel!r}")
    feed_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={cid}"
    with YoutubeDL(_base_opts(False)) as ydl:
        raw = ydl.urlopen(feed_url).read()
    ns = {
        "a": "http://www.w3.org/2005/Atom",
        "yt": "http://www.youtube.com/xml/schemas/2015",
        "media": "http://search.yahoo.com/mrss/",
    }
    root = ET.fromstring(raw)
    out = []
    for entry in root.findall("a:entry", ns):
        vid_el = entry.find("yt:videoId", ns)
        title_el = entry.find("a:title", ns)
        pub_el = entry.find("a:published", ns)
        vid = vid_el.text if vid_el is not None else None
        out.append(
            {
                "id": vid,
                "title": title_el.text if title_el is not None else None,
                "published": pub_el.text if pub_el is not None else None,
                "url": f"https://www.youtube.com/watch?v={vid}" if vid else None,
            }
        )
    return out
