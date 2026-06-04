"""Local YouTube transcript + search MCP server (FastMCP, stdio).

Escalation strategy per research (2026):
  captions cookieless  ->  captions with browser cookies (your Premium session)
                       ->  audio download + faster-whisper (only if no captions exist)
All results cached in SQLite so repeated calls never re-hit YouTube.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Optional

from mcp.server.fastmcp import FastMCP

from . import cache, parse, transcribe, ytdlp

mcp = FastMCP("yt-transcript-mcp")


# --------------------------------------------------------------------------- transcript

@mcp.tool()
def get_transcript(
    video_url: str,
    lang: Optional[str] = None,
    output: str = "text",
    include_timestamps: bool = True,
    allow_whisper: bool = True,
    send_metadata: bool = True,
) -> dict:
    """Get a YouTube video's transcript.

    Tries the native caption track first (fast, no GPU). If the video has no captions and
    allow_whisper is True, downloads the audio and transcribes locally with faster-whisper.

    Args:
        video_url: YouTube URL (full/short/shorts) or 11-char video ID.
        lang: Preferred caption language code (e.g. "en", "pt"). None = auto-pick.
        output: "text" (markdown with [t] timestamps) or "json" (segments + metadata).
        include_timestamps: Include per-segment timestamps in text output.
        allow_whisper: Allow audio+Whisper fallback when no captions exist.
        send_metadata: Include video metadata (title, channel, duration, thumbnail…).

    Returns:
        {"video_id", "source", "lang", "used_cookies", "metadata",
         "transcript"/"segments", ...}
    """
    vid = parse.extract_video_id(video_url)
    if not vid:
        return {"error": f"Could not extract a video ID from {video_url!r}"}
    cache_key_lang = lang or "auto"

    cached = cache.get(vid, cache_key_lang)
    if cached:
        return _format_transcript(cached, output, include_timestamps, from_cache=True)

    # 1) native captions (cookieless -> cookies escalation handled inside)
    result = None
    try:
        cap = ytdlp.fetch_captions(video_url, lang)
    except Exception as e:
        cap = None
        cap_err = str(e)
    else:
        cap_err = None
    if cap:
        result = {
            "video_id": vid,
            "source": f"captions:{cap['kind']}",
            "lang": cap["lang"],
            "used_cookies": cap["used_cookies"],
            "segments": cap["segments"],
            "metadata": cap.get("metadata"),
        }

    # 2) whisper fallback (no captions)
    if result is None and allow_whisper:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                dl = ytdlp.download_audio(video_url, tmp)
                tr = transcribe.transcribe(dl["path"], language=lang)
            result = {
                "video_id": vid,
                "source": "whisper:faster-whisper",
                "lang": tr["lang"],
                "used_cookies": dl["used_cookies"],
                "segments": tr["segments"],
                "metadata": dl.get("metadata"),
            }
        except Exception as e:
            return {
                "video_id": vid,
                "error": f"No captions and Whisper fallback failed: {e}",
                "captions_error": cap_err,
            }

    if result is None:
        return {
            "video_id": vid,
            "error": "No captions available and allow_whisper is False.",
            "captions_error": cap_err,
        }

    cache.put(vid, cache_key_lang, result["source"], result)
    return _format_transcript(result, output, include_timestamps, send_metadata=send_metadata)


def _format_transcript(
    result: dict,
    output: str,
    include_timestamps: bool,
    from_cache: bool = False,
    send_metadata: bool = True,
) -> dict:
    base = {
        "video_id": result["video_id"],
        "source": result["source"],
        "lang": result["lang"],
        "used_cookies": result.get("used_cookies", False),
        "from_cache": from_cache,
        "segment_count": len(result["segments"]),
    }
    if send_metadata and result.get("metadata"):
        base["metadata"] = result["metadata"]
    if output == "json":
        base["segments"] = result["segments"]
    else:
        base["transcript"] = parse.segments_to_text(result["segments"], include_timestamps)
    return base


# --------------------------------------------------------------------------- local files

# Containers PyAV/faster-whisper decodes directly; others get an ffmpeg pre-extract.
_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}
_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".wmv", ".m4v", ".mpeg", ".mpg", ".3gp"}


def _ffmpeg_extract_audio(src: str, dst_dir: str) -> str:
    """Pull a 16kHz mono wav out of any container ffmpeg can read. Raises if ffmpeg missing."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not on PATH; cannot pre-extract audio from this container")
    out = os.path.join(dst_dir, "audio.wav")
    subprocess.run(
        [ffmpeg, "-nostdin", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", out],
        check=True,
        capture_output=True,
    )
    return out


@mcp.tool()
def transcribe_local_file(
    path: str,
    lang: Optional[str] = None,
    output: str = "text",
    include_timestamps: bool = True,
) -> dict:
    """Transcribe a LOCAL audio OR video file with faster-whisper. No YouTube, no network.

    faster-whisper decodes the audio stream directly via PyAV, so video files (mp4, mkv,
    mov, webm…) are transcribed without a separate audio-extraction step. If PyAV can't
    open the container, falls back to an ffmpeg pre-extract (16kHz mono wav) when ffmpeg
    is on PATH.

    Args:
        path: Absolute or relative path to a local audio/video file.
        lang: Language code hint (e.g. "en", "pt"). None = auto-detect.
        output: "text" (markdown with [t] timestamps) or "json" (segments + duration).
        include_timestamps: Include per-segment timestamps in text output.

    Returns:
        {"path", "source", "lang", "duration", "segment_count",
         "transcript"/"segments"}  — or {"error", ...} on failure.
    """
    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(p):
        return {"error": f"File not found: {path!r}"}

    ext = os.path.splitext(p)[1].lower()
    if ext and ext not in _AUDIO_EXTS and ext not in _VIDEO_EXTS:
        # Not fatal — PyAV may still decode it — but warn the caller in the result.
        unknown_ext = ext
    else:
        unknown_ext = None

    # 1) Try faster-whisper directly (PyAV handles most audio + video containers).
    try:
        tr = transcribe.transcribe(p, language=lang)
        source = "whisper:faster-whisper"
    except Exception as direct_err:
        # 2) ffmpeg pre-extract fallback for containers PyAV can't open.
        try:
            with tempfile.TemporaryDirectory() as tmp:
                wav = _ffmpeg_extract_audio(p, tmp)
                tr = transcribe.transcribe(wav, language=lang)
            source = "whisper:faster-whisper+ffmpeg"
        except Exception as ff_err:
            return {
                "path": p,
                "error": f"Transcription failed (direct: {direct_err}; ffmpeg fallback: {ff_err})",
            }

    base = {
        "path": p,
        "source": source,
        "lang": tr["lang"],
        "duration": tr["duration"],
        "segment_count": len(tr["segments"]),
    }
    if unknown_ext:
        base["warning"] = f"Unrecognized extension {unknown_ext!r}; decoded anyway."
    if output == "json":
        base["segments"] = tr["segments"]
    else:
        base["transcript"] = parse.segments_to_text(tr["segments"], include_timestamps)
    return base


# --------------------------------------------------------------------------- metadata

@mcp.tool()
def get_video_info(video_url: str) -> dict:
    """Get metadata for a YouTube video (title, channel, duration, views, caption availability)."""
    try:
        return ytdlp.video_info(video_url)
    except Exception as e:
        return {"error": str(e)}


# --------------------------------------------------------------------------- search

@mcp.tool()
def search_youtube(query: str, search_type: str = "video", limit: int = 20) -> dict:
    """Search YouTube for videos or channels.

    Args:
        query: Search query.
        search_type: "video" or "channel".
        limit: Max results (default 20).
    """
    try:
        return {"results": ytdlp.search(query, search_type, limit)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def list_channel_videos(channel: str, limit: int = 50) -> dict:
    """List recent videos on a channel. channel = @handle, channel URL, or UC… ID."""
    try:
        return {"results": ytdlp.list_channel_videos(channel, limit)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def search_channel_videos(channel: str, query: str, limit: int = 30) -> dict:
    """Search within a single channel for videos matching a query."""
    try:
        return {"results": ytdlp.search_channel_videos(channel, query, limit)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def list_playlist_videos(playlist: str, limit: int = 100) -> dict:
    """List every video in a YouTube playlist. playlist = URL or playlist ID."""
    try:
        return {"results": ytdlp.list_playlist_videos(playlist, limit)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_channel_latest_videos(channel: str) -> dict:
    """Get the ~15 most recent uploads from a channel via its public RSS feed.
    No auth, no PO token, no rate-limit risk. channel = @handle, URL, or UC… ID."""
    try:
        return {"results": ytdlp.latest_videos(channel)}
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
