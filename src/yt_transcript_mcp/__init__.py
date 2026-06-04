"""Local YouTube transcript + search MCP server.

Version is single-sourced from the installed package metadata (pyproject `version`),
so there is no second place to keep in sync.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("yt-transcript-mcp")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+dev"

__all__ = ["__version__"]
