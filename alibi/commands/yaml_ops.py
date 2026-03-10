"""YAML sidecar and clouds commands."""

from __future__ import annotations

from typing import Any

import click
from rich.table import Table

from alibi.commands.shared import console, format_amount
from alibi.config import get_config
from alibi.db.connection import get_db
from alibi.errors import NO_INBOX_CONFIGURED


# ---------------------------------------------------------------------------
# yaml group
# ---------------------------------------------------------------------------


@click.group()
def yaml() -> None:
    """Batch YAML sidecar editing and inspection commands."""
    pass


@yaml.command("list")
@click.option(
    "--inbox",
    "-i",
    type=click.Path(exists=True),
    default=None,
    help="Inbox directory to scan (default: configured inbox)",
)
@click.option(
    "--type",
    "-t",
    "doc_type",
    default=None,
    help="Filter by document type (receipt, invoice, etc.)",
)
@click.option(
    "--vendor",
    "-v",
    default=None,
    help="Filter by vendor name (case-insensitive substring match)",
)
def yaml_list(inbox: str | None, doc_type: str | None, vendor: str | None) -> None:
    """List all .alibi.yaml files in the inbox with a summary.

    Shows filename, document type, vendor, total/amount, confidence, and
    needs_review flag for each YAML sidecar found.

    Examples:

        lt yaml list

        lt yaml list --type invoice

        lt yaml list --vendor acme
    """
    import yaml as _yaml
    from pathlib import Path as P

    from alibi.extraction.yaml_cache import resolve_source_from_yaml, scan_yaml_store

    config = get_config()

    inbox_path: P | None = P(inbox) if inbox else config.get_inbox_path()

    if inbox_path is None:
        NO_INBOX_CONFIGURED.display(console)
        raise click.Abort()

    if not inbox_path.exists():
        console.print(f"[red]Inbox not found:[/red] {inbox_path}")
        raise click.Abort()

    entries: list[tuple[P, dict[str, Any]]] = []

    for yaml_path in scan_yaml_store():
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = _yaml.safe_load(f)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        # Apply type filter
        yaml_doc_type = data.get("document_type", "")
        if doc_type and yaml_doc_type != doc_type:
            continue

        # Resolve vendor field (field name varies by type)
        yaml_vendor = data.get("vendor") or data.get("issuer") or data.get("bank") or ""

        # Apply vendor filter
        if vendor and vendor.lower() not in (yaml_vendor or "").lower():
            continue

        resolved = resolve_source_from_yaml(yaml_path)
        source_name = resolved[0].name if resolved else yaml_path.name

        meta = data.get("_meta", {}) if isinstance(data.get("_meta"), dict) else {}
        entries.append(
            (P(source_name), {"data": data, "meta": meta, "yaml_path": yaml_path})
        )

    if not entries:
        console.print("[yellow]No .alibi.yaml files found.[/yellow]")
        return

    table = Table(title=f"YAML Sidecars ({len(entries)} found)")
    table.add_column("File", style="cyan", no_wrap=False, max_width=35)
    table.add_column("Type", style="dim", max_width=20)
    table.add_column("Vendor", style="green", max_width=25)
    table.add_column("Total", justify="right")
    table.add_column("Conf", justify="right")
    table.add_column("Review", justify="center")

    for entry_path, info in entries:
        d = info["data"]
        meta = info["meta"]

        yaml_doc_type = d.get("document_type", "")
        yaml_vendor = d.get("vendor") or d.get("issuer") or d.get("bank") or ""
        # Amount field varies: total (receipt), amount (invoice/payment), closing_balance (statement)
        amount = d.get("total") or d.get("amount") or d.get("closing_balance")
        currency = d.get("currency", "")
        amount_str = (
            format_amount(float(amount), currency)
            if amount is not None
            else "[dim]n/a[/dim]"
        )

        confidence = meta.get("confidence")
        conf_str = f"{confidence:.2f}" if confidence is not None else "[dim]n/a[/dim]"
        needs_review = meta.get("needs_review", False)
        review_str = "[red]yes[/red]" if needs_review else ""

        table.add_row(
            str(entry_path)[:35],
            yaml_doc_type[:20],
            (yaml_vendor or "")[:25],
            amount_str,
            conf_str,
            review_str,
        )

    console.print(table)


@yaml.command("set-field")
@click.argument("path", type=click.Path())
@click.argument("field")
@click.argument("value")
def yaml_set_field(path: str, field: str, value: str) -> None:
    """Set a field in a specific .alibi.yaml sidecar.

    PATH is the path to the source document (e.g. receipt.jpg), not the
    YAML file itself.  The YAML sidecar must already exist alongside it.

    VALUE is always parsed as a string; numeric values will be stored as
    strings unless the field already contained a number.

    Examples:

        lt yaml set-field /inbox/receipt.jpg vendor "ACME Corp"

        lt yaml set-field /inbox/invoice.pdf total 149.99
    """
    from pathlib import Path as P

    from alibi.extraction.yaml_cache import get_yaml_path
    from alibi.services.ingestion import patch_yaml_field

    source_path = P(path)

    # Validate YAML exists (checks yaml_store then sidecar)
    is_group = source_path.is_dir()
    yaml_path = get_yaml_path(source_path, is_group=is_group)
    if not yaml_path.exists():
        console.print(f"[red]No YAML sidecar found for:[/red] {source_path}")
        raise click.Abort()

    # Try to coerce value to a number if it looks like one
    coerced_value: str | float | int = value
    try:
        if "." in value:
            coerced_value = float(value)
        else:
            coerced_value = int(value)
    except ValueError:
        pass  # Keep as string

    modified = patch_yaml_field(source_path, field, coerced_value)

    if modified:
        console.print(
            f"[green]Updated[/green] [cyan]{field}[/cyan] = [bold]{coerced_value!r}[/bold]"
        )
        console.print(f"  in {yaml_path}")
    else:
        console.print(
            f"[yellow]No change:[/yellow] field [cyan]{field}[/cyan] already has value {coerced_value!r}"
        )
        console.print(f"  YAML: {yaml_path}")


@yaml.command("rename-vendor")
@click.argument("inbox", type=click.Path(exists=True))
@click.argument("old_name")
@click.argument("new_name")
@click.option(
    "--dry-run", "-n", is_flag=True, help="Show matches without applying changes"
)
def yaml_rename_vendor(inbox: str, old_name: str, new_name: str, dry_run: bool) -> None:
    """Batch rename a vendor across all .alibi.yaml files in the inbox.

    Walks all .alibi.yaml files under INBOX, finds those where the vendor
    field (case-insensitive substring match) contains OLD_NAME, and renames
    them to NEW_NAME.

    The vendor field name searched depends on the document type:
    receipt/payment/contract/warranty -> vendor
    invoice -> issuer
    statement -> bank

    Examples:

        lt yaml rename-vendor /inbox "old name" "New Name Ltd"

        lt yaml rename-vendor /inbox acme "ACME Corp" --dry-run
    """
    import yaml as _yaml
    from pathlib import Path as P

    from alibi.extraction.yaml_cache import resolve_source_from_yaml, scan_yaml_store
    from alibi.services.ingestion import patch_yaml_field

    inbox_path = P(inbox)

    # Collect all matches
    matches: list[tuple[P, str, str]] = []  # (source_path, vendor_field, current_value)

    for yaml_path in scan_yaml_store():
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = _yaml.safe_load(f)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        doc_type = data.get("document_type", "")
        # Determine which field holds the vendor name
        if doc_type == "invoice":
            vendor_field = "issuer"
        elif doc_type == "statement":
            vendor_field = "bank"
        else:
            vendor_field = "vendor"

        current = data.get(vendor_field, "") or ""
        if old_name.lower() not in current.lower():
            continue

        resolved = resolve_source_from_yaml(yaml_path)
        if resolved is None:
            continue
        source_path, _is_group = resolved
        matches.append((source_path, vendor_field, current))

    if not matches:
        console.print(
            f"[yellow]No YAML files found with vendor containing '{old_name}'.[/yellow]"
        )
        return

    console.print(
        f"Found [bold]{len(matches)}[/bold] match(es) for '[cyan]{old_name}[/cyan]':\n"
    )

    table = Table(show_header=True, header_style="bold")
    table.add_column("File", style="cyan", max_width=40)
    table.add_column("Field", style="dim")
    table.add_column("Current Vendor", style="yellow", max_width=30)

    for source_path, vendor_field, current in matches:
        table.add_row(source_path.name[:40], vendor_field, current[:30])

    console.print(table)

    if dry_run:
        console.print(f"\n[dim]Dry run -- no changes applied.[/dim]")
        console.print(f"  Would rename to: [bold]{new_name}[/bold]")
        return

    # Apply renames
    updated = 0
    failed = 0
    for source_path, vendor_field, _current in matches:
        ok = patch_yaml_field(source_path, vendor_field, new_name)
        if ok:
            updated += 1
        else:
            failed += 1
            console.print(
                f"[yellow]Warning:[/yellow] could not update {source_path.name}"
            )

    console.print(
        f"\n[green]Renamed {updated} file(s)[/green] to '[bold]{new_name}[/bold]'."
    )
    if failed:
        console.print(f"[yellow]{failed} file(s) could not be updated.[/yellow]")


# ---------------------------------------------------------------------------
# clouds group
# ---------------------------------------------------------------------------


@click.group()
def clouds() -> None:
    """Cloud management and maintenance."""
    pass


@clouds.command("reconcile")
def clouds_reconcile() -> None:
    """Re-try collapse on all FORMING clouds.

    Useful after batch processing or when documents arrive out of order.
    Scans all clouds with FORMING status and attempts to collapse them
    into facts.
    """
    from alibi.services.correction import reconcile_forming_clouds

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    count = reconcile_forming_clouds(db)
    if count > 0:
        console.print(f"[green]Collapsed {count} forming cloud(s) into facts.[/green]")
    else:
        console.print("[dim]No forming clouds could be collapsed.[/dim]")
