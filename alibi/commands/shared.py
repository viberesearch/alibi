"""Shared CLI utilities used across all command modules."""

from __future__ import annotations

from typing import Any

from rich.console import Console

console = Console()

# Status indicators for consistent output
STATUS_OK = "[green]OK[/green]"
STATUS_WARN = "[yellow]Warning[/yellow]"
STATUS_ERROR = "[red]Error[/red]"

# Verbosity level (0=quiet, 1=normal, 2=verbose)
_verbosity = 1


def set_verbosity(level: int) -> None:
    """Set the global verbosity level."""
    global _verbosity
    _verbosity = level


def is_quiet() -> bool:
    """Check if quiet mode is enabled."""
    return _verbosity == 0


def is_verbose() -> bool:
    """Check if verbose mode is enabled."""
    return _verbosity >= 2


def format_amount(amount: float | None, currency: str = "") -> str:
    """Format amount with color coding."""
    if amount is None:
        return ""
    if amount >= 0:
        return f"[green]{amount:,.2f}[/green] {currency}".strip()
    else:
        return f"[red]{amount:,.2f}[/red] {currency}".strip()


def resolve_prefix(db: Any, table: str, id_column: str, prefix: str) -> str | None:
    """Resolve a short ID prefix to a full ID.

    Returns the full ID if exactly one match, None otherwise.
    """
    rows = db.fetchall(
        f"SELECT {id_column} FROM {table} WHERE {id_column} LIKE ?",  # noqa: S608
        (f"{prefix}%",),
    )
    if len(rows) == 1:
        return str(rows[0][0])
    return None
