"""MCP server for exposing alibi data to Claude.

Provides tools and resources for querying transactions, spending,
budgets, and line items through the Model Context Protocol.
"""

from alibi.mcp.server import mcp, run_server

__all__ = [
    "mcp",
    "run_server",
]
