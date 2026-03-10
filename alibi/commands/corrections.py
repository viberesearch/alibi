"""Corrections group: correction event log and adaptive learning."""

from __future__ import annotations

import click
from rich.table import Table

from alibi.commands.shared import console
from alibi.db.connection import get_db


@click.group()
def corrections() -> None:
    """Correction event log -- adaptive learning foundation."""
    pass


@corrections.command("list")
@click.option("--entity-type", "-t", default=None, help="Filter by entity type")
@click.option("--entity-id", "-e", default=None, help="Filter by entity ID")
@click.option("--field", "-f", default=None, help="Filter by field name")
@click.option("--limit", "-l", default=30, show_default=True, help="Max results")
def corrections_list_cmd(
    entity_type: str | None,
    entity_id: str | None,
    field: str | None,
    limit: int,
) -> None:
    """List correction events with optional filters."""
    from alibi.services import correction_log

    db = get_db()
    results = correction_log.list_corrections(
        db, entity_type=entity_type, entity_id=entity_id, field=field, limit=limit
    )
    if not results:
        console.print("[yellow]No correction events found.[/yellow]")
        return

    table = Table(title=f"Correction Events ({len(results)})")
    table.add_column("Date", style="dim")
    table.add_column("Entity")
    table.add_column("Field")
    table.add_column("Old Value")
    table.add_column("New Value")
    table.add_column("Source")

    for c in results:
        created = str(c.get("created_at", ""))[:19]
        entity = f"{c['entity_type']}:{c['entity_id'][:8]}"
        table.add_row(
            created,
            entity,
            c["field"],
            str(c.get("old_value") or "")[:30],
            str(c.get("new_value") or "")[:30],
            c.get("source", ""),
        )
    console.print(table)


@corrections.command("rate")
@click.argument("vendor_key")
@click.option("--window", "-w", default=90, show_default=True, help="Window in days")
def corrections_rate_cmd(vendor_key: str, window: int) -> None:
    """Show correction rate for a vendor."""
    from alibi.services import correction_log

    db = get_db()
    rate = correction_log.get_vendor_correction_rate(db, vendor_key, window)
    console.print(
        f"Vendor [bold]{vendor_key}[/bold] (last {window} days): "
        f"{rate['corrected_facts']}/{rate['total_facts']} facts corrected "
        f"({rate['rate']:.1%})"
    )


@corrections.command("snapshot")
@click.option(
    "--clear", is_flag=True, help="Delete existing snapshot instead of creating one"
)
def corrections_snapshot_cmd(clear: bool) -> None:
    """Take (or clear) a fact_items snapshot for out-of-band edit detection."""
    from alibi.services.snapshot import delete_snapshot, take_snapshot

    if clear:
        if delete_snapshot():
            console.print("[green]Snapshot deleted.[/green]")
        else:
            console.print("[yellow]No snapshot to delete.[/yellow]")
        return

    db = get_db()
    try:
        count = take_snapshot(db)
        console.print(f"[green]Snapshot saved:[/green] {count} fact_items captured.")
    except FileExistsError as e:
        console.print(f"[red]{e}[/red]")


@corrections.command("detect")
@click.option(
    "--apply", "do_apply", is_flag=True, help="Record correction events and apply"
)
@click.option(
    "--source", default="tableplus", show_default=True, help="Correction source tag"
)
def corrections_detect_cmd(do_apply: bool, source: str) -> None:
    """Detect fact_item changes since last snapshot (dry-run by default)."""
    from alibi.services.snapshot import (
        apply_changes,
        delete_snapshot,
        detect_changes,
        load_snapshot,
    )

    snapshot = load_snapshot()
    if snapshot is None:
        console.print(
            "[red]No snapshot found. Run 'lt corrections snapshot' first.[/red]"
        )
        return

    db = get_db()
    changes = detect_changes(db, snapshot)

    if not changes:
        console.print("[green]No changes detected.[/green]")
        if do_apply:
            delete_snapshot()
            console.print("[dim]Snapshot cleaned up.[/dim]")
        return

    table = Table(title=f"Detected Changes ({len(changes)})")
    table.add_column("Item ID", style="dim", max_width=8)
    table.add_column("Item Name")
    table.add_column("Field")
    table.add_column("Old", style="red")
    table.add_column("New", style="green")

    for c in changes:
        table.add_row(
            c["item_id"][:8],
            str(c.get("item_name", ""))[:40],
            c["field"],
            str(c.get("old_value", "")),
            str(c.get("new_value", "")),
        )
    console.print(table)

    if do_apply:
        recorded = apply_changes(db, changes, source=source)
        delete_snapshot()
        console.print(
            f"\n[green]Applied:[/green] {recorded} correction events recorded. "
            "Snapshot cleaned up."
        )
    else:
        console.print("\n[dim]Dry run. Add --apply to record correction events.[/dim]")
