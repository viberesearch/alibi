"""MCP server for alibi life tracker.

Exposes transaction, spending, budget, and item data through the
Model Context Protocol for Claude Code integration.

Usage:
    uv run python -m alibi.mcp.server
"""

from mcp.server.fastmcp import FastMCP

from alibi.mcp.resources import register_resources
from alibi.mcp.tools import register_tools

mcp = FastMCP(
    name="alibi",
    instructions=(
        "Alibi life tracker - ingest documents, query facts and items, "
        "correct vendors, annotate entities, and analyze spending patterns. "
        "Full lifecycle: ingestion, query, correction, annotation, analytics."
    ),
)

# Register tools and resources on the server instance
register_tools(mcp)
register_resources(mcp)


def run_server() -> None:
    """Run the MCP server with stdio transport for Claude Code."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_server()
