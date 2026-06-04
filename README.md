# yt-transcript-mcp

Local, self-hosted MCP server for YouTube transcripts + search. Runs from your own
(residential) IP alongside Claude Code — no third-party API, no credits, no account.

[![test](https://github.com/oreiasccp/yt-transcript-mcp/actions/workflows/test.yml/badge.svg)](https://github.com/oreiasccp/yt-transcript-mcp/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/yt-transcript-mcp.svg)](https://pypi.org/project/yt-transcript-mcp/)
[![Python](https://img.shields.io/pypi/pyversions/yt-transcript-mcp.svg)](https://pypi.org/project/yt-transcript-mcp/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Why local wins (2026):** YouTube blocks most cloud-provider IPs (AWS/GCP/Azure). Running
from your home IP sidesteps the bot-detection that breaks server-hosted scrapers. See the
research that drove this design below.

## Strategy

Per-request escalation, cheapest path first:

```
1. native captions, cookieless            ← works for most videos on a residential IP
2. native captions, with browser cookies  ← auto-retry if YouTube says "confirm you're not a bot"
3. audio download + faster-whisper (GPU)   ← only when the video has NO captions at all
+ SQLite cache by video_id                 ← never fetch/transcribe the same video twice
```

Captions are preferred because they need only Python + yt-dlp (no GPU, no model, no ffmpeg)
and are instant. Whisper is a fallback for caption-less videos.

## Tools

| Tool | What |
|------|------|
| `get_transcript` | Transcript via captions → Whisper fallback. Returns video metadata (title, channel, duration, views, thumbnail) alongside the text. Args: `output=text\|json`, `lang`, `include_timestamps`, `allow_whisper`, `send_metadata`. |
| `transcribe_local_file` | Transcribe a **local** audio **or** video file (no YouTube, no network) with faster-whisper. Video containers (mp4/mkv/mov/webm…) are decoded directly via PyAV — no separate audio extraction — with an `ffmpeg` pre-extract fallback for containers PyAV can't open. Args: `path`, `lang`, `output=text\|json`, `include_timestamps`. |
| `get_video_info` | Title, channel, duration, views, caption availability. |
| `search_youtube` | Search videos or channels (`search_type=video\|channel`). |
| `list_channel_videos` | Recent videos on a channel (`@handle` / URL / `UC…`). |
| `search_channel_videos` | Search within one channel. |
| `list_playlist_videos` | Every video in a playlist. |
| `get_channel_latest_videos` | ~15 newest uploads via public RSS — no auth, no PO token, zero block risk. |

## Requirements

| Need | Why | Required? |
|------|-----|-----------|
| Python 3.10+ and [uv](https://docs.astral.sh/uv/) | runtime + deps | **Yes** |
| `ffmpeg` on PATH | audio extraction for the Whisper fallback (caption-less videos) and the `transcribe_local_file` container fallback | only when PyAV can't decode a container |
| NVIDIA GPU + CUDA driver | GPU is the priority device for Whisper (CUDA libs are bundled by default). Without a GPU it falls back to CPU (int8) automatically. | optional |
| A browser logged into YouTube (Chrome) | cookie escalation when YouTube flags a request | optional (helps reliability) |

The native-caption path (most videos) needs only Python + uv. YouTube Premium helps:
Premium accounts are exempt from the GVS PO-token requirement.

## Install & register (copy-paste)

### Option A — from PyPI with `uvx` (recommended, no clone)

`uvx` fetches the published package into an isolated cache and runs it — the same launch
idiom as the reference servers (`uvx mcp-server-git`). GPU acceleration is bundled by default
(NVIDIA, non-macOS) and the server falls back to CPU automatically when no GPU is present.

```sh
claude mcp add -s user yt-transcript -- uvx yt-transcript-mcp
```

### Option B — from source (for development or local edits)

```sh
# clone + install dependencies into a managed .venv
gh repo clone oreiasccp/yt-transcript-mcp
cd yt-transcript-mcp
uv sync
```

Then register the server in Claude Code. Pick the launch command that matches how you
installed it:

```sh
# Recommended (stable): the console-script entry point in the project venv.
# One process, no per-launch re-sync — the launch idiom used by the reference servers.
claude mcp add -s user yt-transcript -- "$(pwd)/.venv/bin/yt-transcript-mcp"
# Windows PowerShell:
#   claude mcp add -s user yt-transcript -- "$((Get-Location).Path)\.venv\Scripts\yt-transcript-mcp.exe"

# Equivalent, SDK-canonical: run the package as a module.
claude mcp add -s user yt-transcript -- "$(pwd)/.venv/bin/python" -m yt_transcript_mcp
```

> **Why not `uv --directory … run`?** `uv run` re-syncs the *editable* project on every
> launch, which rewrites the console-script binary. On Windows the binary is often still
> open from the previous launch, so the rewrite fails (`os error 32`), the server can't
> start (`-32000 Connection closed`), and orphaned processes accumulate. Launching the
> installed entry point (or `python -m`) skips the re-sync entirely. Reserve `uv run` for
> local development (`uv run mcp dev` / `uv run pytest`), not as a registered server command.

Verify, then use it:

```sh
claude mcp list      # → yt-transcript: ... ✓ Connected
```

Then ask Claude Code: *"get the transcript of <youtube-url> and summarize it."*

### Run standalone (without the registry)

```sh
uv run yt-transcript-mcp        # dev
python -m yt_transcript_mcp     # after `uv sync` / pip install (stdio server)
```

## Configuration (env vars)

| Var | Default | Purpose |
|-----|---------|---------|
| `YT_COOKIES_BROWSER` | `chrome` | Browser yt-dlp reads cookies from on the escalation retry. `firefox`, `edge`, `brave`… or empty to disable. |
| `YT_WHISPER_MODEL` | `large-v3` | faster-whisper model. `turbo`, `medium`, `small`, `tiny`. |
| `YT_WHISPER_DEVICE` | `cuda` | `cuda` or `cpu`. |
| `YT_WHISPER_COMPUTE` | `int8_float16` | Quantization. `float16`, `int8`. |
| `YT_WHISPER_VAD` | `0` | Voice-activity-detection filter. **Off by default** — the bundled VAD over-filters and can drop all segments in the current faster-whisper. Set `1` to try it. |
| `YT_MCP_CACHE_DIR` | `~/.cache/yt-transcript-mcp` | SQLite cache location. |
| `YT_MCP_ALLOWED_DIRS` | _(unset)_ | Restrict `transcribe_local_file` to these directories (OS-path-separator list, e.g. `/home/me/media:/tmp` or `D:\media;C:\Users\me\Downloads`). Unset = unrestricted, appropriate for a local single-user server. |

## Notes & caveats

- **Account risk:** the cookie escalation reuses your logged-in browser session. For low-volume
  manual use the risk is small, but using your main Google account for automated downloads
  violates YouTube ToS. Consider a secondary account if you scale up. Never run in a tight loop.
- **Moving target:** YouTube's anti-bot posture (PO tokens, SABR, client-format 403s) shifts
  between yt-dlp releases. Keep yt-dlp updated (`uv sync -U`).
- **Empty-track bug:** yt-dlp can return zero-byte caption tracks (#13443). The caption fetch
  defensively iterates formats/tracks to route around it.
- **Captions can be POT-gated** on specific videos (#13075) — intermittent, not universal. When
  the cookieless caption fetch fails, the cookie retry usually recovers it.

## Releasing (maintainers)

Publishing is automated via PyPI **Trusted Publishing** (OIDC — no API token in the repo):

```sh
# 1. bump `version` in pyproject.toml, commit
# 2. tag and push — the publish workflow builds, checks, and uploads to PyPI
git tag v0.2.0 && git push origin v0.2.0
```

One-time PyPI setup: add a Trusted Publisher for project `yt-transcript-mcp` pointing at
this repo's `publish.yml` workflow and the `pypi` environment. Local dry run:

```sh
uv build && uvx twine check dist/*
```

The MCP registry manifest lives in [`server.json`](server.json); submit/update it with the
`mcp-publisher` CLI after the PyPI release is live.

## Research basis

Design verified against a 2026 deep-research pass (24 sources, adversarial verification):
- Native-caption path is what every reference MCP server actually ships (yttranscript_mcp,
  youtube-mcp-server-enhanced, youtube_transcribe_mcp_server, youtube-transcript-api).
- Cloud IPs are blocked; residential IPs are the meaningful advantage of local hosting.
- faster-whisper (CTranslate2) ≈4× openai-whisper at equal accuracy; whisper.cpp is the CPU choice.
- PO tokens "may help" but don't guarantee bypass; SABR is forced even on Premium.
