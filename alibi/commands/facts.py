"""Facts group: inspect and correct v2 facts, clouds, and bundles."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import click
from rich.table import Table

from alibi.commands.shared import console
from alibi.db.connection import DatabaseManager, get_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_fact_id(db_manager: DatabaseManager, prefix: str) -> str | None:
    """Resolve a fact ID prefix to a full UUID."""
    if len(prefix) >= 36:
        return prefix

    rows = db_manager.fetchall("SELECT id FROM facts WHERE id LIKE ?", (f"{prefix}%",))
    if len(rows) == 1:
        result: str = rows[0]["id"]
        return result
    if len(rows) == 0:
        console.print(f"[red]No fact matching '{prefix}'[/red]")
        return None
    console.print(
        f"[red]Ambiguous prefix '{prefix}' matches {len(rows)} facts. Be more specific.[/red]"
    )
    return None


def _resolve_bundle_id(db_manager: DatabaseManager, prefix: str) -> str | None:
    """Resolve a bundle ID prefix to a full UUID."""
    if len(prefix) >= 36:
        return prefix

    rows = db_manager.fetchall(
        "SELECT id FROM bundles WHERE id LIKE ?", (f"{prefix}%",)
    )
    if len(rows) == 1:
        result: str = rows[0]["id"]
        return result
    if len(rows) == 0:
        console.print(f"[red]No bundle matching '{prefix}'[/red]")
        return None
    console.print(
        f"[red]Ambiguous prefix '{prefix}' matches {len(rows)} bundles. Be more specific.[/red]"
    )
    return None


def _resolve_cloud_id(db_manager: DatabaseManager, prefix: str) -> str | None:
    """Resolve a cloud ID prefix to a full UUID."""
    if len(prefix) >= 36:
        return prefix

    rows = db_manager.fetchall("SELECT id FROM clouds WHERE id LIKE ?", (f"{prefix}%",))
    if len(rows) == 1:
        result: str = rows[0]["id"]
        return result
    if len(rows) == 0:
        console.print(f"[red]No cloud matching '{prefix}'[/red]")
        return None
    console.print(
        f"[red]Ambiguous prefix '{prefix}' matches {len(rows)} clouds. Be more specific.[/red]"
    )
    return None


def _atom_summary(atom_type: str, data: dict[str, Any]) -> str:
    """Format a one-line summary of an atom's data."""
    if atom_type == "vendor":
        return str(data.get("name", ""))
    if atom_type == "amount":
        return f"{data.get('value', '')} {data.get('currency', '')} ({data.get('semantic_type', '')})"
    if atom_type == "datetime":
        return str(data.get("value", ""))
    if atom_type == "item":
        qty = data.get("quantity", 1)
        name = data.get("name", "")
        price = data.get("total_price")
        return f"{qty}x {name}" + (f" = {price}" if price else "")
    if atom_type == "payment":
        method = data.get("method", "")
        last4 = data.get("card_last4", "")
        return f"{method}" + (f" *{last4}" if last4 else "")
    if atom_type == "tax":
        return f"{data.get('rate', '')}% ({data.get('type', '')})"
    return str(data)[:60]


# ---------------------------------------------------------------------------
# Group
# ---------------------------------------------------------------------------


@click.group()
def facts() -> None:
    """Inspect and correct v2 facts, clouds, and bundles."""
    pass


@facts.command("list")
@click.option("--vendor", "-v", help="Filter by vendor name")
@click.option(
    "--type",
    "-t",
    "fact_type",
    help="Filter by type (purchase, refund, subscription_payment)",
)
@click.option(
    "--since", "-s", type=click.DateTime(formats=["%Y-%m-%d"]), help="Since date"
)
@click.option(
    "--until", "-u", type=click.DateTime(formats=["%Y-%m-%d"]), help="Until date"
)
@click.option("--limit", "-l", default=50, help="Maximum results")
def facts_list(
    vendor: str | None,
    fact_type: str | None,
    since: datetime | None,
    until: datetime | None,
    limit: int,
) -> None:
    """List facts with optional filters."""
    from alibi.db import v2_store

    db_manager = get_db()
    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    rows = v2_store.list_facts(
        db_manager,
        date_from=since.date() if since else None,
        date_to=until.date() if until else None,
        vendor=vendor,
        fact_type=fact_type,
        limit=limit,
    )

    if not rows:
        console.print("[yellow]No facts found.[/yellow]")
        return

    table = Table(title=f"Facts ({len(rows)})")
    table.add_column("ID", style="dim", max_width=8)
    table.add_column("Type", style="cyan")
    table.add_column("Vendor", style="green")
    table.add_column("Amount", justify="right")
    table.add_column("Date")
    table.add_column("Status")

    for row in rows:
        fact_id = row["id"][:8]
        amount_str = ""
        if row.get("total_amount") is not None:
            amount_str = f"{float(row['total_amount']):,.2f} {row.get('currency', '')}"
        status = row.get("status", "")
        status_styled = {
            "confirmed": f"[green]{status}[/green]",
            "partial": f"[yellow]{status}[/yellow]",
            "needs_review": f"[red]{status}[/red]",
        }.get(status, status)

        table.add_row(
            fact_id,
            row.get("fact_type", ""),
            row.get("vendor", ""),
            amount_str,
            str(row.get("event_date", "")),
            status_styled,
        )

    console.print(table)


@facts.command("inspect")
@click.argument("fact_id")
def facts_inspect(fact_id: str) -> None:
    """Inspect a fact: show cloud, bundles, atoms, source documents.

    FACT_ID can be a full UUID or a prefix (min 4 chars).
    """
    from alibi.db import v2_store

    db_manager = get_db()
    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    # Resolve prefix
    resolved_id = _resolve_fact_id(db_manager, fact_id)
    if not resolved_id:
        return

    result = v2_store.inspect_fact(db_manager, resolved_id)
    if not result:
        console.print(f"[red]Fact not found: {fact_id}[/red]")
        return

    fact = result["fact"]
    cloud = result["cloud"]

    # Fact header
    console.print(f"\n[bold blue]Fact:[/bold blue] {fact['id']}")
    console.print(f"  Type: [cyan]{fact.get('fact_type', '')}[/cyan]")
    console.print(f"  Vendor: [green]{fact.get('vendor', '')}[/green]")
    amount = fact.get("total_amount")
    if amount is not None:
        console.print(f"  Amount: {float(amount):,.2f} {fact.get('currency', '')}")
    console.print(f"  Date: {fact.get('event_date', '')}")
    console.print(f"  Status: {fact.get('status', '')}")

    # Cloud info
    console.print(f"\n[bold blue]Cloud:[/bold blue] {cloud.get('id', '')}")
    console.print(f"  Status: {cloud.get('status', '')}")
    console.print(f"  Confidence: {cloud.get('confidence', '')}")

    # Bundles
    console.print(f"\n[bold blue]Bundles ({len(result['bundles'])}):[/bold blue]")
    for b in result["bundles"]:
        doc = b.get("document", {})
        console.print(
            f"\n  [cyan]{b['id'][:8]}[/cyan] "
            f"({b['bundle_type']}) "
            f"match={b.get('match_type', '')} "
            f"conf={b.get('match_confidence', '')}"
        )
        console.print(f"    Document: {doc.get('file_path', '')}")

        # Atoms in this bundle
        for a in b.get("atoms", []):
            atype = a["atom_type"]
            data = a["data"]
            summary = _atom_summary(atype, data)
            console.print(f"    [{atype}] {summary}")

    # Items
    if result["items"]:
        console.print(f"\n[bold blue]Items ({len(result['items'])}):[/bold blue]")
        item_table = Table(show_header=True)
        item_table.add_column("Name")
        item_table.add_column("Qty", justify="right")
        item_table.add_column("Unit")
        item_table.add_column("Price", justify="right")
        item_table.add_column("Atom", style="dim", max_width=8)

        for item in result["items"]:
            qty = str(item.get("quantity", ""))
            price = ""
            if item.get("total_price") is not None:
                price = f"{float(item['total_price']):,.2f}"
            item_table.add_row(
                item.get("name", ""),
                qty,
                item.get("unit", ""),
                price,
                item.get("atom_id", "")[:8] if item.get("atom_id") else "",
            )
        console.print(item_table)


@facts.command("clouds")
@click.option("--status", "-s", help="Filter by status (forming, collapsed, disputed)")
@click.option("--limit", "-l", default=50, help="Maximum results")
def facts_clouds(status: str | None, limit: int) -> None:
    """List clouds with bundle count and fact summary."""
    from alibi.db import v2_store

    db_manager = get_db()
    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    rows = v2_store.list_clouds(db_manager, status=status, limit=limit)
    if not rows:
        console.print("[yellow]No clouds found.[/yellow]")
        return

    table = Table(title=f"Clouds ({len(rows)})")
    table.add_column("Cloud ID", style="dim", max_width=8)
    table.add_column("Status", style="cyan")
    table.add_column("Bundles", justify="right")
    table.add_column("Fact Vendor", style="green")
    table.add_column("Amount", justify="right")
    table.add_column("Date")

    for row in rows:
        status_val = row.get("status", "")
        status_styled = {
            "collapsed": f"[green]{status_val}[/green]",
            "forming": f"[yellow]{status_val}[/yellow]",
            "disputed": f"[red]{status_val}[/red]",
        }.get(status_val, status_val)

        amount_str = ""
        if row.get("total_amount") is not None:
            amount_str = f"{float(row['total_amount']):,.2f}"

        table.add_row(
            row["id"][:8],
            status_styled,
            str(row.get("bundle_count", 0)),
            row.get("fact_vendor") or "",
            amount_str,
            str(row.get("event_date", "") or ""),
        )

    console.print(table)


@facts.command("unassigned")
def facts_unassigned() -> None:
    """List bundles with no cloud assignment (detached)."""
    from alibi.db import v2_store

    db_manager = get_db()
    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    rows = v2_store.get_unassigned_bundles(db_manager)
    if not rows:
        console.print("[green]No unassigned bundles.[/green]")
        return

    table = Table(title=f"Unassigned Bundles ({len(rows)})")
    table.add_column("Bundle ID", style="dim", max_width=8)
    table.add_column("Type", style="cyan")
    table.add_column("Document")

    for row in rows:
        table.add_row(
            row["id"][:8],
            row.get("bundle_type", ""),
            row.get("file_path", ""),
        )

    console.print(table)


@facts.command("move")
@click.argument("bundle_id")
@click.option("--to", "target_cloud", help="Target cloud ID (omit to create new cloud)")
def facts_move(bundle_id: str, target_cloud: str | None) -> None:
    """Move a bundle to a different cloud (or a new one).

    BUNDLE_ID can be a full UUID or a prefix (min 4 chars).
    """
    from alibi.clouds.correction import move_bundle

    db_manager = get_db()
    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    # Resolve bundle prefix
    resolved = _resolve_bundle_id(db_manager, bundle_id)
    if not resolved:
        return

    # Resolve target cloud prefix if provided
    resolved_target = None
    if target_cloud:
        resolved_target = _resolve_cloud_id(db_manager, target_cloud)
        if not resolved_target:
            return

    result = move_bundle(db_manager, resolved, target_cloud_id=resolved_target)

    if not result.success:
        console.print(f"[red]Move failed:[/red] {result.error}")
        return

    console.print("[green]Bundle moved successfully.[/green]")
    if result.target_cloud_id:
        console.print(f"  Target cloud: {result.target_cloud_id[:8]}")
    if result.source_fact_id:
        console.print(f"  Source re-collapsed: fact {result.source_fact_id[:8]}")
    if result.target_fact_id:
        console.print(f"  Target re-collapsed: fact {result.target_fact_id[:8]}")
    if result.deleted_clouds:
        console.print(f"  Cleaned up {result.deleted_clouds} empty cloud(s)")


@facts.command("set-cloud")
@click.argument("bundle_id")
@click.argument("cloud_id", required=False)
@click.option("--detach", is_flag=True, help="Detach bundle (set cloud_id to NULL)")
def facts_set_cloud(bundle_id: str, cloud_id: str | None, detach: bool) -> None:
    """Set the cloud_id on a bundle directly.

    BUNDLE_ID and CLOUD_ID can be full UUIDs or prefixes.
    Use --detach to clear the cloud assignment.
    """
    from alibi.db import v2_store

    db_manager = get_db()
    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    resolved_bundle = _resolve_bundle_id(db_manager, bundle_id)
    if not resolved_bundle:
        return

    target: str | None = None
    if detach:
        target = None
    elif cloud_id:
        target = _resolve_cloud_id(db_manager, cloud_id)
        if not target:
            return
    else:
        console.print("[red]Provide a CLOUD_ID or use --detach.[/red]")
        return

    ok = v2_store.set_bundle_cloud(db_manager, resolved_bundle, target)
    if ok:
        if target:
            console.print(
                f"[green]Bundle {resolved_bundle[:8]} -> cloud {target[:8]}[/green]"
            )
        else:
            console.print(
                f"[green]Bundle {resolved_bundle[:8]} detached (cloud_id = NULL)[/green]"
            )
    else:
        console.print("[red]Failed to update bundle.[/red]")


@facts.command("recollapse")
@click.argument("cloud_id")
def facts_recollapse(cloud_id: str) -> None:
    """Force re-collapse a cloud into a fact.

    CLOUD_ID can be a full UUID or a prefix.
    """
    from alibi.clouds.correction import recollapse_cloud

    db_manager = get_db()
    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    resolved = _resolve_cloud_id(db_manager, cloud_id)
    if not resolved:
        return

    fact_id = recollapse_cloud(db_manager, resolved)
    if fact_id:
        console.print(f"[green]Cloud re-collapsed -> fact {fact_id[:8]}[/green]")
    else:
        console.print("[yellow]Cloud could not collapse (stays forming).[/yellow]")


@facts.command("dispute")
@click.argument("cloud_id")
def facts_dispute(cloud_id: str) -> None:
    """Mark a cloud as disputed (needs human review).

    CLOUD_ID can be a full UUID or a prefix.
    """
    from alibi.clouds.correction import mark_disputed

    db_manager = get_db()
    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    resolved = _resolve_cloud_id(db_manager, cloud_id)
    if not resolved:
        return

    mark_disputed(db_manager, resolved)
    console.print(f"[yellow]Cloud {resolved[:8]} marked as disputed.[/yellow]")
