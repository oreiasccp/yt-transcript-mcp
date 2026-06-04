"""Local YouTube transcript + search MCP server (FastMCP, stdio).

Escalation strategy per research (2026):
  captions cookieless  ->  captions with browser cookies (your Premium session)
                       ->  audio download + faster-whisper (only if no captions exist)
All results cached in SQLite so repeated calls never re-hit YouTube.

Design notes:
- Tools are async and offload blocking work (yt-dlp, faster-whisper) to a worker thread via
  anyio, so the stdio event loop stays responsive.
- Long transcriptions report incremental progress through the MCP Context.
- Inputs are constrained with Annotated/Field; enums replace free-form mode strings.
- Errors are returned in-result (never raised across the protocol) and sanitized.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import tempfile
from enum import Enum
from typing import Annotated, Any, Callable, Optional

import anyio
from anyio.from_thread import run as _run_on_loop
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from . import cache, parse, transcribe, ytdlp

mcp = FastMCP("yt_transcript_mcp")


# =========================================================================== enums

class ResponseFormat(str, Enum):
    """Output format: human-readable markdown text or machine-readable JSON."""
    TEXT = "text"
    JSON = "json"


class SearchType(str, Enum):
    """What kind of entity to search for."""
    VIDEO = "video"
    CHANNEL = "channel"


# Read-only, idempotent tools that DO reach out to YouTube.
_RO_REMOTE = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True
)
# Read-only, idempotent tools that stay fully local (no network).
_RO_LOCAL = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
)


# =========================================================================== helpers

async def _to_thread(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a blocking callable in a worker thread without blocking the event loop."""
    return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))


def _err(prefix: str, e: Exception) -> dict:
    """Sanitized, actionable error payload. Avoids leaking stack/internals to the client."""
    return {"error": f"{prefix}: {type(e).__name__}: {e}"}


def _make_progress(ctx: Optional[Context]) -> Optional[Callable[[float, float], None]]:
    """Build a throttled on_progress callback that reports back through the MCP Context.

    Called from the transcription worker thread, so it hops onto the event loop via
    anyio.from_thread.run. Throttled to ~every 5% of media to avoid notification spam.
    Fully guarded: any failure (no progress token, loop hiccup) is swallowed.
    """
    if ctx is None:
        return None
    state = {"last": -1.0}

    def cb(done: float, total: float) -> None:
        if total <= 0:
            return
        pct = done / total
        if state["last"] >= 0 and (pct - state["last"]) < 0.05:
            return
        state["last"] = pct
        try:
            _run_on_loop(
                ctx.report_progress,
                min(done, total),
                total,
                f"transcribing {done:.0f}/{total:.0f}s ({pct * 100:.0f}%)",
            )
        except Exception:
            pass

    return cb


def _format_transcript(
    result: dict,
    output: ResponseFormat,
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
    if output == ResponseFormat.JSON:
        base["segments"] = result["segments"]
    else:
        base["transcript"] = parse.segments_to_text(result["segments"], include_timestamps)
    return base


def _listing_markdown(env: dict, title: str) -> str:
    """Render a paginated listing envelope as compact markdown."""
    lines = [f"# {title}", "", f"Showing {env['count']} (offset {env['offset']})"]
    if env.get("has_more"):
        lines.append(f"More available — next_offset={env['next_offset']}")
    lines.append("")
    for it in env["items"]:
        head = it.get("title") or it.get("id") or "?"
        lines.append(f"- **{head}** ({it.get('id')})")
        meta = []
        if it.get("channel"):
            meta.append(it["channel"])
        if it.get("duration"):
            meta.append(f"{it['duration']}s")
        if it.get("view_count") is not None:
            meta.append(f"{it['view_count']} views")
        if meta:
            lines.append(f"  - {' · '.join(str(m) for m in meta)}")
        if it.get("url"):
            lines.append(f"  - {it['url']}")
    return "\n".join(lines)


def _wrap_listing(env: dict, output: ResponseFormat, title: str) -> dict:
    """Attach a markdown rendering when requested; always keep structured fields."""
    if output == ResponseFormat.TEXT:
        return {**env, "rendered": _listing_markdown(env, title)}
    return env


# =========================================================================== transcript

def _get_transcript_sync(
    video_url: str,
    lang: Optional[str],
    allow_whisper: bool,
    on_progress: Optional[Callable[[float, float], None]],
) -> dict:
    """Blocking transcript pipeline (captions -> whisper). Runs in a worker thread.

    Returns the raw result dict (pre-formatting) or {"error": ...}.
    """
    vid = parse.extract_video_id(video_url)
    if not vid:
        return {"error": f"Could not extract a YouTube video ID from {video_url!r}"}
    cache_key_lang = lang or "auto"

    cached = cache.get(vid, cache_key_lang)
    if cached:
        cached["_from_cache"] = True
        return cached

    # 1) native captions (cookieless -> cookies escalation handled inside)
    try:
        cap = ytdlp.fetch_captions(video_url, lang)
        cap_err = None
    except Exception as e:
        cap = None
        cap_err = str(e)

    result = None
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
                tr = transcribe.transcribe(dl["path"], language=lang, on_progress=on_progress)
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
                "error": f"No captions and Whisper fallback failed: {type(e).__name__}: {e}",
                "captions_error": cap_err,
            }

    if result is None:
        return {
            "video_id": vid,
            "error": "No captions available and allow_whisper is False.",
            "captions_error": cap_err,
        }

    cache.put(vid, cache_key_lang, result["source"], result)
    return result


@mcp.tool(name="get_transcript", annotations=_RO_REMOTE)
async def get_transcript(
    video_url: Annotated[str, Field(description="YouTube URL (full/short/shorts/live) or 11-char video ID", min_length=1)],
    lang: Annotated[Optional[str], Field(description="Preferred caption language code, e.g. 'en' or 'pt'. None = auto-pick.")] = None,
    output: Annotated[ResponseFormat, Field(description="'text' for markdown with [t] timestamps, 'json' for segments + metadata")] = ResponseFormat.TEXT,
    include_timestamps: Annotated[bool, Field(description="Include per-segment timestamps in text output")] = True,
    allow_whisper: Annotated[bool, Field(description="Allow audio+Whisper fallback when the video has no captions")] = True,
    send_metadata: Annotated[bool, Field(description="Include video metadata (title, channel, duration, thumbnail…)")] = True,
    ctx: Context = None,  # type: ignore[assignment]
) -> dict:
    """Get a YouTube video's transcript.

    Tries the native caption track first (fast, no GPU). If the video has no captions and
    allow_whisper is True, downloads the audio and transcribes locally with faster-whisper,
    reporting progress for long videos. Results are cached by (video_id, lang).

    Returns (JSON): {"video_id", "source", "lang", "used_cookies", "from_cache",
    "segment_count", "metadata"?, and either "transcript" (text) or "segments" (json)}.
    On failure: {"error", ...} with an actionable message.
    """
    result = await _get_transcript_sync(video_url, lang, allow_whisper, _make_progress(ctx))
    if "error" in result and "segments" not in result:
        return result
    from_cache = result.pop("_from_cache", False)
    return _format_transcript(result, output, include_timestamps, from_cache, send_metadata)


# =========================================================================== local files

_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}
_VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".wmv", ".m4v", ".mpeg", ".mpg", ".3gp"}

# Optional access restriction for local-file transcription. When YT_MCP_ALLOWED_DIRS is set
# (os-path-separator list of directories), transcribe_local_file may only read files inside
# those roots. Unset = unrestricted (the default for a local single-user server), mirroring
# postgres-mcp's --access-mode model.
_ALLOWED_DIRS = [
    os.path.abspath(os.path.expanduser(p))
    for p in os.environ.get("YT_MCP_ALLOWED_DIRS", "").split(os.pathsep)
    if p.strip()
]


def _path_allowed(abs_path: str) -> bool:
    """True if abs_path is inside an allowed root (or no allowlist is configured)."""
    if not _ALLOWED_DIRS:
        return True
    for root in _ALLOWED_DIRS:
        try:
            if os.path.commonpath([abs_path, root]) == root:
                return True
        except ValueError:
            continue  # different drive (Windows) -> not under this root
    return False


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


def _transcribe_local_sync(
    path: str,
    lang: Optional[str],
    on_progress: Optional[Callable[[float, float], None]],
) -> dict:
    """Blocking local-file transcription with cache + ffmpeg fallback. Worker-thread side."""
    # Cache by content identity (path+mtime+size) so re-runs are instant.
    try:
        ckey = cache.local_file_key(path)
        cached = cache.get(ckey, lang or "auto")
        if cached:
            cached["_from_cache"] = True
            return cached
    except OSError:
        ckey = None  # stat failed; skip cache, surface the real error below

    # 1) faster-whisper directly (PyAV handles most audio + video containers).
    try:
        tr = transcribe.transcribe(path, language=lang, on_progress=on_progress)
        source = "whisper:faster-whisper"
    except Exception as direct_err:
        # 2) ffmpeg pre-extract fallback for containers PyAV can't open.
        try:
            with tempfile.TemporaryDirectory() as tmp:
                wav = _ffmpeg_extract_audio(path, tmp)
                tr = transcribe.transcribe(wav, language=lang, on_progress=on_progress)
            source = "whisper:faster-whisper+ffmpeg"
        except Exception as ff_err:
            return {
                "path": path,
                "error": f"Transcription failed (direct: {type(direct_err).__name__}: "
                f"{direct_err}; ffmpeg fallback: {type(ff_err).__name__}: {ff_err})",
            }

    result = {
        "path": path,
        "source": source,
        "lang": tr["lang"],
        "duration": tr["duration"],
        "segments": tr["segments"],
    }
    if ckey:
        cache.put(ckey, lang or "auto", source, result)
    return result


@mcp.tool(name="transcribe_local_file", annotations=_RO_LOCAL)
async def transcribe_local_file(
    path: Annotated[str, Field(description="Absolute or relative path to a local audio or video file", min_length=1)],
    lang: Annotated[Optional[str], Field(description="Language code hint, e.g. 'en' or 'pt'. None = auto-detect.")] = None,
    output: Annotated[ResponseFormat, Field(description="'text' for markdown with [t] timestamps, 'json' for segments + duration")] = ResponseFormat.TEXT,
    include_timestamps: Annotated[bool, Field(description="Include per-segment timestamps in text output")] = True,
    ctx: Context = None,  # type: ignore[assignment]
) -> dict:
    """Transcribe a LOCAL audio OR video file with faster-whisper. No YouTube, no network.

    faster-whisper decodes the audio stream directly via PyAV, so video files (mp4, mkv,
    mov, webm…) are transcribed without a separate audio-extraction step. If PyAV can't
    open the container, falls back to an ffmpeg pre-extract (16kHz mono wav) when ffmpeg is
    on PATH. Results are cached by file identity (path + mtime + size); progress is reported
    for long media. If YT_MCP_ALLOWED_DIRS is set, the path must resolve inside one of those
    directories, otherwise access is denied.

    Returns (JSON): {"path", "source", "lang", "duration", "from_cache", "segment_count",
    and either "transcript" (text) or "segments" (json)}. On failure: {"error", ...}.
    """
    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(p):
        return {"error": f"File not found: {path!r}"}
    if not _path_allowed(p):
        return {"error": f"Access denied: {path!r} is outside YT_MCP_ALLOWED_DIRS"}

    ext = os.path.splitext(p)[1].lower()
    unknown_ext = ext if (ext and ext not in _AUDIO_EXTS and ext not in _VIDEO_EXTS) else None

    result = await _to_thread(_transcribe_local_sync, p, lang, _make_progress(ctx))
    if "error" in result:
        return result

    from_cache = result.pop("_from_cache", False)
    base = {
        "path": result["path"],
        "source": result["source"],
        "lang": result["lang"],
        "duration": result["duration"],
        "from_cache": from_cache,
        "segment_count": len(result["segments"]),
    }
    if unknown_ext:
        base["warning"] = f"Unrecognized extension {unknown_ext!r}; decoded anyway."
    if output == ResponseFormat.JSON:
        base["segments"] = result["segments"]
    else:
        base["transcript"] = parse.segments_to_text(result["segments"], include_timestamps)
    return base


# =========================================================================== metadata

@mcp.tool(name="get_video_info", annotations=_RO_REMOTE)
async def get_video_info(
    video_url: Annotated[str, Field(description="YouTube URL or 11-char video ID", min_length=1)],
) -> dict:
    """Get metadata for a YouTube video (title, channel, duration, views, caption availability).

    Returns (JSON): {"id", "title", "channel", "channel_id", "channel_url", "duration",
    "duration_string", "view_count", "like_count", "upload_date", "description", "thumbnail",
    "has_subtitles", "has_auto_captions", "webpage_url"}. On failure: {"error", ...}.
    """
    try:
        return await _to_thread(ytdlp.video_info, video_url)
    except Exception as e:
        return _err("get_video_info failed", e)


# =========================================================================== search

@mcp.tool(name="search_youtube", annotations=_RO_REMOTE)
async def search_youtube(
    query: Annotated[str, Field(description="Search query", min_length=1, max_length=500)],
    search_type: Annotated[SearchType, Field(description="Search for 'video' or 'channel'")] = SearchType.VIDEO,
    limit: Annotated[int, Field(description="Max results to return", ge=1, le=50)] = 20,
    offset: Annotated[int, Field(description="Results to skip for pagination", ge=0, le=1000)] = 0,
    output: Annotated[ResponseFormat, Field(description="'json' (default) or 'text' for markdown")] = ResponseFormat.JSON,
) -> dict:
    """Search YouTube for videos or channels.

    Returns a paginated envelope (JSON): {"count", "offset", "items":[{id,title,url,duration,
    view_count,channel,channel_id}], "has_more", "next_offset"}. With output='text' a
    "rendered" markdown field is added. On failure: {"error", ...}.
    """
    try:
        env = await _to_thread(ytdlp.search, query, search_type.value, limit, offset)
        return _wrap_listing(env, output, f"Search: {query}")
    except Exception as e:
        return _err("search_youtube failed", e)


@mcp.tool(name="list_channel_videos", annotations=_RO_REMOTE)
async def list_channel_videos(
    channel: Annotated[str, Field(description="@handle, channel URL, or UC… channel ID", min_length=1)],
    limit: Annotated[int, Field(description="Max results to return", ge=1, le=100)] = 50,
    offset: Annotated[int, Field(description="Results to skip for pagination", ge=0, le=5000)] = 0,
    output: Annotated[ResponseFormat, Field(description="'json' (default) or 'text' for markdown")] = ResponseFormat.JSON,
) -> dict:
    """List recent videos on a channel.

    Returns a paginated envelope (see search_youtube for the schema). On failure: {"error", ...}.
    """
    try:
        env = await _to_thread(ytdlp.list_channel_videos, channel, limit, offset)
        return _wrap_listing(env, output, f"Channel videos: {channel}")
    except Exception as e:
        return _err("list_channel_videos failed", e)


@mcp.tool(name="search_channel_videos", annotations=_RO_REMOTE)
async def search_channel_videos(
    channel: Annotated[str, Field(description="@handle, channel URL, or UC… channel ID", min_length=1)],
    query: Annotated[str, Field(description="Search query within the channel", min_length=1, max_length=500)],
    limit: Annotated[int, Field(description="Max results to return", ge=1, le=100)] = 30,
    offset: Annotated[int, Field(description="Results to skip for pagination", ge=0, le=5000)] = 0,
    output: Annotated[ResponseFormat, Field(description="'json' (default) or 'text' for markdown")] = ResponseFormat.JSON,
) -> dict:
    """Search within a single channel for videos matching a query.

    Returns a paginated envelope (see search_youtube for the schema). On failure: {"error", ...}.
    """
    try:
        env = await _to_thread(ytdlp.search_channel_videos, channel, query, limit, offset)
        return _wrap_listing(env, output, f"{channel} / {query}")
    except Exception as e:
        return _err("search_channel_videos failed", e)


@mcp.tool(name="list_playlist_videos", annotations=_RO_REMOTE)
async def list_playlist_videos(
    playlist: Annotated[str, Field(description="Playlist URL or playlist ID", min_length=1)],
    limit: Annotated[int, Field(description="Max results to return", ge=1, le=200)] = 100,
    offset: Annotated[int, Field(description="Results to skip for pagination", ge=0, le=10000)] = 0,
    output: Annotated[ResponseFormat, Field(description="'json' (default) or 'text' for markdown")] = ResponseFormat.JSON,
) -> dict:
    """List videos in a YouTube playlist.

    Returns a paginated envelope (see search_youtube for the schema). On failure: {"error", ...}.
    """
    try:
        env = await _to_thread(ytdlp.list_playlist_videos, playlist, limit, offset)
        return _wrap_listing(env, output, f"Playlist: {playlist}")
    except Exception as e:
        return _err("list_playlist_videos failed", e)


@mcp.tool(name="get_channel_latest_videos", annotations=_RO_REMOTE)
async def get_channel_latest_videos(
    channel: Annotated[str, Field(description="@handle, channel URL, or UC… channel ID", min_length=1)],
) -> dict:
    """Get the ~15 most recent uploads from a channel via its public RSS feed.

    No auth, no PO token, no rate-limit risk. Returns (JSON): {"count", "items":[{id,title,
    published,url}]}. On failure: {"error", ...}.
    """
    try:
        items = await _to_thread(ytdlp.latest_videos, channel)
        return {"count": len(items), "items": items}
    except Exception as e:
        return _err("get_channel_latest_videos failed", e)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
