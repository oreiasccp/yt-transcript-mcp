"""Module entry point so the server can be launched the canonical way:

    python -m yt_transcript_mcp

This mirrors how the reference MCP servers (mcp-server-git, mcp-server-fetch) are run
after a pip/uv install, and avoids the editable-project re-sync that `uv run --directory`
performs on every launch.
"""

from .server import main

if __name__ == "__main__":
    main()
