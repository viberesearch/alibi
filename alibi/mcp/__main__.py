"""Entry point for running alibi MCP server as a module.

Usage:
    uv run python -m alibi.mcp
"""

from alibi.mcp.server import run_server

run_server()
