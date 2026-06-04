# yt-transcript-mcp

Local, self-hosted MCP server for YouTube transcripts + search. Runs from your own
(residential) IP alongside Claude Code — no third-party API, no credits, no account.

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
| `ffmpeg` on PATH | audio extraction for the Whisper fallback | only for caption-less videos |
| NVIDIA GPU + CUDA driver | fast local Whisper (falls back to CPU int8) | optional |
| A browser logged into YouTube (Chrome) | cookie escalation when YouTube flags a request | optional (helps reliability) |

The native-caption path (most videos) needs only Python + uv. YouTube Premium helps:
Premium accounts are exempt from the GVS PO-token requirement.

## Install & register (copy-paste)

```sh
# clone
gh repo clone oreiasccp/yt-transcript-mcp
cd yt-transcript-mcp

# install dependencies
uv sync

# register globally in Claude Code (works in every project + the VS Code extension)
claude mcp add -s user yt-transcript -- uv --directory "$(pwd)" run yt-transcript-mcp
```

`$(pwd)` resolves to wherever you cloned it — no hardcoded path. On Windows PowerShell use
`"$((Get-Location).Path)"` instead of `"$(pwd)"`.

Verify:

```sh
claude mcp list      # → yt-transcript: ... ✓ Connected
```

Then ask Claude Code: *"get the transcript of <youtube-url> and summarize it."*

## Configuration (env vars)

| Var | Default | Purpose |
|-----|---------|---------|
| `YT_COOKIES_BROWSER` | `chrome` | Browser yt-dlp reads cookies from on the escalation retry. `firefox`, `edge`, `brave`… or empty to disable. |
| `YT_WHISPER_MODEL` | `large-v3` | faster-whisper model. `turbo`, `medium`, `small`, `tiny`. |
| `YT_WHISPER_DEVICE` | `cuda` | `cuda` or `cpu`. |
| `YT_WHISPER_COMPUTE` | `int8_float16` | Quantization. `float16`, `int8`. |
| `YT_WHISPER_VAD` | `0` | Voice-activity-detection filter. **Off by default** — the bundled VAD over-filters and can drop all segments in the current faster-whisper. Set `1` to try it. |
| `YT_MCP_CACHE_DIR` | `~/.cache/yt-transcript-mcp` | SQLite cache location. |

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

## Research basis

Design verified against a 2026 deep-research pass (24 sources, adversarial verification):
- Native-caption path is what every reference MCP server actually ships (yttranscript_mcp,
  youtube-mcp-server-enhanced, youtube_transcribe_mcp_server, youtube-transcript-api).
- Cloud IPs are blocked; residential IPs are the meaningful advantage of local hosting.
- faster-whisper (CTranslate2) ≈4× openai-whisper at equal accuracy; whisper.cpp is the CPU choice.
- PO tokens "may help" but don't guarantee bypass; SABR is forced even on Premium.
