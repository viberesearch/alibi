"""Currency / FX commands: normalise receipt currencies to EUR for analytics."""

from __future__ import annotations

import click
from rich.table import Table

from alibi.commands.shared import console
from alibi.db.connection import get_db


@click.group()
def fx() -> None:
    """Currency conversion: historical receipt-currency -> EUR for analytics."""


@fx.command("backfill")
@click.option(
    "--no-fetch",
    is_flag=True,
    default=False,
    help="Use only cached rates (offline); do not call the FX API",
)
def fx_backfill(no_fetch: bool) -> None:
    """Resolve each fact's EUR rate (historical, at its date) and rebuild stars.

    Fetches the EUR reference rate at every non-EUR fact's event_date from
    Frankfurter (cached in ``exchange_rates``), stamps ``facts.eur_rate``, then
    rebuilds ``item_stars`` so the EUR-normalised analytics (spend, basket,
    comparable price) reflect the conversion. Re-runnable; EUR facts convert 1:1.
    """
    from alibi.services import rebuild_item_stars
    from alibi.services.fx import backfill_fact_rates

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    console.print("[dim]Resolving historical EUR rates...[/dim]")
    stats = backfill_fact_rates(db, fetch=not no_fetch)

    console.print(
        f"[green]Resolved {stats['pairs_resolved']} currency/date rate(s)[/green]"
        + (
            f"; [yellow]{stats['pairs_unresolved']} unresolved[/yellow]"
            if stats["pairs_unresolved"]
            else ""
        )
    )
    if stats["currencies"]:
        console.print(
            f"[dim]Non-EUR currencies: {', '.join(stats['currencies'])}[/dim]"
        )
    if stats["facts_unconverted"]:
        console.print(
            f"[yellow]{stats['facts_unconverted']} fact(s) remain unconverted "
            "(no rate / no date) and are excluded from EUR analytics.[/yellow]"
        )

    count = rebuild_item_stars(db)
    console.print(f"[green]Rebuilt item_stars: {count} rows (EUR-normalised).[/green]")


@fx.command("rates")
@click.option("-l", "--limit", default=50, show_default=True, help="Max rows")
def fx_rates(limit: int) -> None:
    """Show the cached historical exchange rates (EUR per 1 unit)."""
    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    rows = db.fetchall(
        "SELECT base, rate_date, eur_per_unit FROM exchange_rates "
        "ORDER BY base, rate_date DESC LIMIT ?",
        (limit,),
    )
    if not rows:
        console.print("[dim]No cached rates. Run [bold]lt fx backfill[/bold].[/dim]")
        return

    table = Table(title="Cached exchange rates (EUR per 1 unit)")
    table.add_column("Currency")
    table.add_column("Date")
    table.add_column("EUR/unit", justify="right")
    for r in rows:
        table.add_row(r["base"], str(r["rate_date"]), f"{r['eur_per_unit']:.4f}")
    console.print(table)
