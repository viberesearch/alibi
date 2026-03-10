"""Database, maintenance, and schedule commands."""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import click
from rich.table import Table

from alibi.commands.shared import console, is_quiet
from alibi.config import get_config
from alibi.db.connection import DatabaseManager, get_db


# ---------------------------------------------------------------------------
# db group
# ---------------------------------------------------------------------------


@click.group()
def db() -> None:
    """Database management commands."""
    pass


@db.command("info")
def db_info() -> None:
    """Show database information."""
    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        console.print("Run 'lt init' to initialize.")
        return

    console.print(f"[bold]Database:[/bold] {db_manager.db_path}")
    console.print(f"[bold]Schema version:[/bold] {db_manager.get_schema_version()}")

    # Show stats
    stats = db_manager.get_stats()
    console.print("\n[bold]Table counts:[/bold]")
    for table_name, count in stats.items():
        console.print(f"  {table_name}: {count}")


@db.command("reset")
@click.confirmation_option(
    prompt="Are you sure you want to reset the database? All data will be lost!"
)
def db_reset() -> None:
    """Reset the database (delete all data)."""
    db_manager = get_db()

    if db_manager.db_path.exists():
        db_manager.close()
        db_manager.db_path.unlink()
        console.print("[yellow]Database deleted.[/yellow]")

    # Reinitialize
    db_manager.initialize()
    console.print("[green]Database reinitialized.[/green]")


@db.command("backup")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option(
    "--include-vectors/--no-vectors",
    default=True,
    help="Include LanceDB vector index",
)
def db_backup(output: str | None, include_vectors: bool) -> None:
    """Create a backup of database and vector index."""
    from pathlib import Path

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from alibi.backup import create_backup

    config = get_config()
    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        console.print("Run 'lt init' to initialize.")
        return

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"alibi_backup_{timestamp}.tar.gz")

    # Get paths
    db_path = db_manager.db_path
    lance_path = config.get_lance_path() if include_vectors else None

    console.print("[bold blue]Creating backup...[/bold blue]")
    console.print(f"  Database: {db_path}")
    if lance_path and lance_path.exists():
        console.print(f"  Vectors: {lance_path}")
    elif include_vectors:
        console.print("  Vectors: [dim]not found[/dim]")
        lance_path = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Creating backup archive...", total=None)
        try:
            result = create_backup(
                output_path=output_path,
                db_path=db_path,
                lance_path=lance_path,
            )
        except Exception as e:
            console.print(f"[red]Backup failed:[/red] {e}")
            raise click.Abort() from e

    console.print()
    console.print("[green]Backup created successfully![/green]")
    console.print(f"  File: {result.path}")
    console.print(f"  Size: {result.size_bytes:,} bytes")
    console.print(f"  Files: {result.file_count}")
    console.print(f"  Created: {result.manifest.created_at}")


@db.command("restore")
@click.argument("backup_file", type=click.Path(exists=True))
@click.option("--verify/--no-verify", default=True, help="Verify checksums")
@click.option("--force", is_flag=True, help="Overwrite existing data")
def db_restore(backup_file: str, verify: bool, force: bool) -> None:
    """Restore from a backup file."""
    from pathlib import Path

    from rich.progress import Progress, SpinnerColumn, TextColumn

    from alibi.backup import get_backup_info, restore_backup

    config = get_config()
    db_manager = get_db()

    backup_path = Path(backup_file)

    # Show backup info first
    try:
        manifest = get_backup_info(backup_path)
    except Exception as e:
        console.print(f"[red]Invalid backup file:[/red] {e}")
        raise click.Abort() from e

    console.print("[bold blue]Backup Information[/bold blue]")
    console.print(f"  Created: {manifest.created_at}")
    console.print(f"  Version: {manifest.version}")
    console.print(f"  Files: {len(manifest.files)}")

    # Confirm restoration
    if not force:
        db_path = db_manager.db_path
        lance_path = config.get_lance_path()

        if db_path.exists():
            console.print(f"\n[yellow]Warning:[/yellow] Database exists at {db_path}")
        if lance_path.exists() and any(lance_path.iterdir()):
            console.print(
                f"[yellow]Warning:[/yellow] LanceDB directory exists at {lance_path}"
            )

        if not click.confirm("\nProceed with restore?"):
            console.print("[yellow]Restore cancelled.[/yellow]")
            return

    # Close existing database connection
    db_manager.close()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Restoring backup...", total=None)
        try:
            result = restore_backup(
                backup_path=backup_path,
                db_path=db_manager.db_path,
                lance_path=config.get_lance_path(),
                verify_checksums=verify,
                overwrite=force,
            )
        except Exception as e:
            console.print(f"[red]Restore failed:[/red] {e}")
            raise click.Abort() from e

    console.print()
    console.print("[green]Restore completed successfully![/green]")
    console.print(f"  Files restored: {result.files_restored}")
    console.print(f"  Files verified: {result.files_verified}")
    if result.checksum_failures:
        console.print(
            f"  [yellow]Checksum failures:[/yellow] {len(result.checksum_failures)}"
        )


@db.command("backup-info")
@click.argument("backup_file", type=click.Path(exists=True))
def db_backup_info(backup_file: str) -> None:
    """Show information about a backup file."""
    from pathlib import Path

    from alibi.backup import get_backup_info

    backup_path = Path(backup_file)

    try:
        manifest = get_backup_info(backup_path)
    except Exception as e:
        console.print(f"[red]Invalid backup file:[/red] {e}")
        raise click.Abort() from e

    console.print("[bold]Backup Information[/bold]\n")
    console.print(f"  File: {backup_path}")
    console.print(f"  Size: {backup_path.stat().st_size:,} bytes")
    console.print(f"  Created: {manifest.created_at}")
    console.print(f"  Version: {manifest.version}")
    console.print(f"  Total files: {len(manifest.files)}")

    if manifest.metadata:
        console.print("\n[bold]Metadata:[/bold]")
        for key, value in manifest.metadata.items():
            console.print(f"  {key}: {value}")

    # Group files by type
    db_files = [f for f in manifest.files if f.startswith("database/")]
    lance_files = [f for f in manifest.files if f.startswith("lancedb/")]

    console.print("\n[bold]Contents:[/bold]")
    console.print(f"  Database files: {len(db_files)}")
    console.print(f"  LanceDB files: {len(lance_files)}")


@db.command("cleanup")
@click.option("--dry-run/--execute", default=True, help="Show what would be deleted")
@click.option(
    "--max-duplicate-age",
    default=90,
    help="Max age in days for duplicate artifact logs",
)
@click.option("--max-error-age", default=30, help="Max age in days for error artifacts")
def db_cleanup(dry_run: bool, max_duplicate_age: int, max_error_age: int) -> None:
    """Clean up old data based on retention policy."""
    from alibi.retention import RetentionPolicy, cleanup_old_data

    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        console.print("Run 'lt init' to initialize.")
        return

    policy = RetentionPolicy(
        max_duplicate_age_days=max_duplicate_age,
        max_error_documents_days=max_error_age,
    )

    console.print("[bold blue]Cleanup Policy[/bold blue]")
    console.print(f"  Max duplicate age: {max_duplicate_age} days")
    console.print(f"  Max error age: {max_error_age} days")
    console.print(f"  Mode: {'Dry run' if dry_run else 'Execute'}")
    console.print()

    result = cleanup_old_data(db_manager, policy, dry_run=dry_run)

    if not result.candidates:
        console.print("[green]No records to clean up.[/green]")
        return

    # Show candidates
    table = Table(title=f"Cleanup Candidates ({len(result.candidates)} records)")
    table.add_column("Table", style="cyan")
    table.add_column("ID", max_width=8, style="dim")
    table.add_column("Reason")
    table.add_column("Age", justify="right")

    now = datetime.now()
    for candidate in result.candidates[:20]:  # Show first 20
        age_days = (now - candidate.created_at).days
        table.add_row(
            candidate.table,
            candidate.id[:8],
            candidate.reason,
            f"{age_days} days",
        )

    console.print(table)

    if len(result.candidates) > 20:
        console.print(f"\n  [dim]... and {len(result.candidates) - 20} more[/dim]")

    console.print()
    if dry_run:
        console.print(
            f"[yellow]Dry run:[/yellow] Would delete {len(result.candidates)} records."
        )
        console.print("Use --execute to actually delete.")
    else:
        console.print(f"[green]Deleted {result.deleted_count} records.[/green]")
        if result.errors:
            console.print(f"[yellow]Errors:[/yellow] {len(result.errors)}")
            for error in result.errors[:5]:
                console.print(f"  {error}")


@db.command("cleanup-duplicates")
@click.option("--dry-run/--execute", default=True, help="Show what would change")
def db_cleanup_duplicates(dry_run: bool) -> None:
    """Fix duplicate fact items, BARCODE vendors, and non-product lines.

    Finds and fixes three data quality issues:

    1. Duplicate items within the same fact (same name + total_price)

    2. Facts with "BARCODE: ..." as vendor name

    3. Non-product line items (discounts, price-change annotations)
    """
    import re

    from alibi.db import v2_store

    db_manager = get_db()
    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    mode = "DRY RUN" if dry_run else "EXECUTE"
    console.print(f"[bold blue]Cleanup Duplicates ({mode})[/bold blue]\n")

    items_to_delete: list[str] = []
    vendor_fixes: list[tuple[str, str, str]] = []  # (fact_id, old, new)

    # --- 1. Duplicate items within same fact ---
    dupes = db_manager.fetchall(
        "SELECT a.id AS keep_id, b.id AS drop_id, "
        "a.fact_id, a.name, a.total_price "
        "FROM fact_items a "
        "JOIN fact_items b ON a.fact_id = b.fact_id "
        "  AND LOWER(TRIM(a.name)) = LOWER(TRIM(b.name)) "
        "  AND a.total_price = b.total_price "
        "  AND a.id < b.id"
    )

    if dupes:
        console.print(f"[yellow]Duplicate items:[/yellow] {len(dupes)} pairs")
        seen_drop: set[str] = set()
        for row in dupes:
            drop_id = row[1]
            if drop_id in seen_drop:
                continue
            seen_drop.add(drop_id)
            items_to_delete.append(drop_id)
            if not is_quiet():
                console.print(
                    f"  fact {row[2][:8]}.. | {row[3]} @ {row[4]} | drop {drop_id[:8]}.."
                )
    else:
        console.print("[green]No duplicate items found.[/green]")

    # --- 2. BARCODE vendor facts ---
    barcode_facts = db_manager.fetchall(
        "SELECT id, vendor FROM facts WHERE vendor LIKE 'BARCODE%'"
    )

    if barcode_facts:
        console.print(f"\n[yellow]BARCODE vendor facts:[/yellow] {len(barcode_facts)}")
        for row in barcode_facts:
            fact_id, old_vendor = row[0], row[1]
            # Try to find the real vendor from the cloud's other bundles
            new_vendor = _guess_vendor_for_barcode_fact(db_manager, fact_id)
            vendor_fixes.append((fact_id, old_vendor, new_vendor))
            console.print(f"  {fact_id[:8]}.. | {old_vendor} -> {new_vendor}")
    else:
        console.print("[green]No BARCODE vendor facts.[/green]")

    # --- 3. Non-product line items ---
    non_product_patterns = [
        re.compile(r"^FROM\s+[\d.,]+\s+TO\s+[\d.,]+", re.IGNORECASE),
        re.compile(r"\d+%\s*OFF\b", re.IGNORECASE),
        re.compile(r"^\d+\s+ea\s+[\d.,]+$", re.IGNORECASE),
        re.compile(r"^VAT\d?\s+\d+", re.IGNORECASE),
        re.compile(r"^Subtotal$", re.IGNORECASE),
    ]
    non_product_keywords = {"discount", "coupon"}
    total_line_names = {"total", "σynoao", "σynοδο", "συνολο", "σύνολο"}

    all_items = db_manager.fetchall("SELECT id, name, total_price FROM fact_items")

    non_product_ids: list[str] = []
    for row in all_items:
        item_id, name, price = row[0], row[1] or "", row[2]
        name_stripped = name.strip()
        if not name_stripped:
            continue
        is_bad = False
        for pat in non_product_patterns:
            if pat.search(name_stripped):
                is_bad = True
                break
        if not is_bad:
            name_lower = name_stripped.lower()
            if name_lower in total_line_names:
                is_bad = True
            else:
                for kw in non_product_keywords:
                    if kw in name_lower and (price is None or price <= 0):
                        is_bad = True
                        break
        if is_bad:
            non_product_ids.append(item_id)
            if not is_quiet():
                console.print(f"  non-product: {name_stripped} ({price})")

    if non_product_ids:
        console.print(f"\n[yellow]Non-product items:[/yellow] {len(non_product_ids)}")
        items_to_delete.extend(non_product_ids)
    else:
        console.print("[green]No non-product items found.[/green]")

    # --- Summary & execute ---
    console.print(
        f"\n[bold]Total: {len(items_to_delete)} items to delete, "
        f"{len(vendor_fixes)} vendor fixes[/bold]"
    )

    if dry_run:
        console.print(
            "[yellow]Dry run -- no changes made. Use --execute to apply.[/yellow]"
        )
        return

    # Apply deletions
    if items_to_delete:
        deleted = v2_store.delete_fact_items(db_manager, items_to_delete)
        console.print(f"[green]Deleted {deleted} fact items.[/green]")

    # Apply vendor fixes
    for fact_id, old_vendor, new_vendor in vendor_fixes:
        from alibi.services.correction import update_fact

        update_fact(db_manager, fact_id, {"vendor": new_vendor})
        console.print(f"[green]Fixed vendor: {old_vendor} -> {new_vendor}[/green]")

    console.print("[bold green]Cleanup complete.[/bold green]")


def _guess_vendor_for_barcode_fact(db_manager: DatabaseManager, fact_id: str) -> str:
    """Try to find the real vendor for a fact with BARCODE as vendor.

    Looks at header atoms in the cloud's bundles for a real vendor name.
    Falls back to "UNKNOWN" if nothing found.
    """
    row = db_manager.fetchone(
        "SELECT c.id FROM facts f "
        "JOIN clouds c ON f.cloud_id = c.id "
        "WHERE f.id = ?",
        (fact_id,),
    )
    if not row:
        return "UNKNOWN"

    cloud_id = row[0]
    atoms = db_manager.fetchall(
        "SELECT a.data FROM atoms a "
        "JOIN bundle_atoms ba ON a.id = ba.atom_id "
        "JOIN bundles b ON ba.bundle_id = b.id "
        "WHERE b.cloud_id = ? AND a.atom_type = 'header'",
        (cloud_id,),
    )

    for atom_row in atoms:
        try:
            data = (
                json.loads(atom_row[0]) if isinstance(atom_row[0], str) else atom_row[0]
            )
        except (json.JSONDecodeError, TypeError):
            continue
        vendor: str = data.get("vendor", "")
        if vendor and not vendor.upper().startswith("BARCODE"):
            return vendor

    # Known barcode-as-vendor mappings (from manual investigation)
    _KNOWN_BARCODE_VENDORS = {
        "BARCODE: 7400/2612354": "ALPHAMEGA",
    }
    fact_row = db_manager.fetchone("SELECT vendor FROM facts WHERE id = ?", (fact_id,))
    if fact_row:
        return _KNOWN_BARCODE_VENDORS.get(fact_row[0], "UNKNOWN")

    return "UNKNOWN"


@db.command("stats")
def db_stats() -> None:
    """Show database statistics including age distribution."""
    from alibi.retention import get_retention_stats

    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        console.print("Run 'lt init' to initialize.")
        return

    stats = get_retention_stats(db_manager)

    console.print("[bold]Database Statistics[/bold]\n")

    # Totals
    console.print(f"  Total documents: {stats['total_documents']}")
    console.print(f"  Total facts: {stats['total_facts']}")

    # Status distribution
    if stats.get("documents_by_status"):
        console.print("\n[bold]Documents by Status:[/bold]")
        for status, count in stats["documents_by_status"].items():
            console.print(f"  {status}: {count}")

    # Age distribution
    if stats.get("age_distribution"):
        console.print("\n[bold]Age Distribution:[/bold]")
        table = Table()
        table.add_column("Age Range", style="cyan")
        table.add_column("Count", justify="right", style="green")

        for age_range, count in stats["age_distribution"].items():
            table.add_row(age_range, str(count))

        console.print(table)

    # Oldest/Newest
    if stats.get("oldest_document"):
        oldest = stats["oldest_document"]
        console.print("\n[bold]Oldest document:[/bold]")
        console.print(f"  ID: {oldest['id'][:8]}")
        console.print(f"  Created: {oldest['created_at']}")
        console.print(f"  Age: {oldest['age_days']} days")

    if stats.get("newest_document"):
        newest = stats["newest_document"]
        console.print("\n[bold]Newest document:[/bold]")
        console.print(f"  ID: {newest['id'][:8]}")
        console.print(f"  Created: {newest['created_at']}")
        console.print(f"  Age: {newest['age_days']} days")


# ---------------------------------------------------------------------------
# maintain group
# ---------------------------------------------------------------------------


@click.group()
def maintain() -> None:
    """Maintenance and learning aggregation commands."""
    pass


@maintain.command("run")
@click.option(
    "--stale-days", default=90, show_default=True, help="Days before template is stale"
)
@click.option(
    "--max-history",
    default=20,
    show_default=True,
    help="Max confidence history entries",
)
def maintain_run(stale_days: int, max_history: int) -> None:
    """Run full maintenance cycle (templates + identities)."""
    from alibi.services.maintenance import run_maintenance

    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    console.print("[bold blue]Running maintenance...[/bold blue]")
    report_result = run_maintenance(
        db_manager, max_history=max_history, stale_days=stale_days
    )

    console.print(f"  Templates recalculated: {report_result.templates_recalculated}")
    console.print(f"  Templates marked stale: {report_result.templates_marked_stale}")
    console.print(f"  Confidence history pruned: {report_result.templates_pruned}")
    console.print(f"  Members deduplicated: {report_result.members_deduplicated}")
    console.print(
        f"  Orphaned members removed: {report_result.orphaned_members_removed}"
    )
    total = (
        report_result.templates_recalculated
        + report_result.templates_marked_stale
        + report_result.templates_pruned
        + report_result.members_deduplicated
        + report_result.orphaned_members_removed
    )
    if total:
        console.print(f"[green]Maintenance complete: {total} operations.[/green]")
    else:
        console.print("[dim]No maintenance needed.[/dim]")


@maintain.command("recalculate")
def maintain_recalculate() -> None:
    """Recalculate template reliability from correction history."""
    from alibi.services.maintenance import recalculate_templates

    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    count = recalculate_templates(db_manager)
    console.print(f"Recalculated {count} templates")


@maintain.command("deduplicate")
def maintain_deduplicate() -> None:
    """Deduplicate identity members and remove orphans."""
    from alibi.services.maintenance import cleanup_identities

    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    count = cleanup_identities(db_manager)
    if count:
        console.print(f"[green]Cleaned up {count} identity members[/green]")
    else:
        console.print("[dim]No duplicates or orphans found.[/dim]")


@maintain.command("rebuild-fts")
def maintain_rebuild_fts() -> None:
    """Rebuild FTS5 product name search index."""
    from alibi.enrichment.product_resolver import rebuild_product_fts

    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    count = rebuild_product_fts(db_manager)
    console.print(f"[green]Rebuilt FTS5 index: {count} items indexed[/green]")


@maintain.command("fix-quality")
@click.option("-n", "--dry-run", is_flag=True, help="Preview fixes without applying")
def maintain_fix_quality(dry_run: bool) -> None:
    """Fix known data quality issues (weighed item units, missing unit_quantity)."""
    from alibi.services.maintenance import fix_data_quality

    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    if dry_run:
        console.print("[bold]Dry run -- previewing fixes...[/bold]\n")

    report_result = fix_data_quality(db_manager)

    if report_result.total:
        console.print(f"  Units fixed (pcs->kg): {report_result.units_fixed}")
        console.print(
            f"  Unit quantities backfilled: {report_result.unit_quantities_backfilled}"
        )
        if report_result.details:
            console.print("\n[bold]Details:[/bold]")
            for detail in report_result.details:
                console.print(f"  {detail}")
        console.print(
            f"\n[green]Data quality: {report_result.total} fixes applied.[/green]"
        )
    else:
        console.print("[dim]No data quality issues found.[/dim]")


@maintain.command("delete-garbage")
def maintain_delete_garbage() -> None:
    """Delete non-product garbage items (OCR artifacts, totals, VAT lines)."""
    from alibi.services.maintenance import delete_garbage

    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    report_result = delete_garbage(db_manager)

    if report_result.units_fixed:
        console.print(f"  Deleted: {report_result.units_fixed} garbage items")
        if report_result.details:
            console.print("\n[bold]Details:[/bold]")
            for detail in report_result.details:
                console.print(f"  {detail}")
        console.print(
            f"\n[green]Cleanup: {report_result.units_fixed} garbage items deleted.[/green]"
        )
    else:
        console.print("[dim]No garbage items found.[/dim]")


@maintain.command("stamp-extraction")
def maintain_stamp_extraction() -> None:
    """Stamp items with brand+category but missing enrichment_source."""
    from alibi.services.maintenance import stamp_extraction

    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    report_result = stamp_extraction(db_manager)

    if report_result.units_fixed:
        console.print(f"  Stamped: {report_result.units_fixed} items")
        if report_result.details:
            console.print("\n[bold]Details:[/bold]")
            for detail in report_result.details:
                console.print(f"  {detail}")
        console.print(
            f"\n[green]Stamped {report_result.units_fixed} items"
            f" with enrichment_source='extraction'.[/green]"
        )
    else:
        console.print("[dim]No items need stamping.[/dim]")


@maintain.command("fill-categories")
def maintain_fill_categories() -> None:
    """Fill missing categories from known brand mappings."""
    from alibi.services.maintenance import fill_category_gaps

    db_manager = get_db()

    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    report_result = fill_category_gaps(db_manager)

    if report_result.units_fixed:
        console.print(f"  Filled: {report_result.units_fixed} items")
        if report_result.details:
            console.print("\n[bold]Details:[/bold]")
            for detail in report_result.details:
                console.print(f"  {detail}")
        console.print(
            f"\n[green]Filled {report_result.units_fixed} category gaps"
            f" from brand mappings.[/green]"
        )
    else:
        console.print("[dim]No category gaps to fill.[/dim]")


# ---------------------------------------------------------------------------
# schedule group
# ---------------------------------------------------------------------------


@click.group()
def schedule() -> None:
    """Scheduled enrichment management."""


@schedule.command("status")
def schedule_status() -> None:
    """Show enrichment scheduler status and last run results."""
    from alibi.daemon.scheduler import SchedulerState, _load_state

    config = get_config()
    state = _load_state()

    console.print("\n[bold]Enrichment Schedule Configuration[/bold]")
    console.print(f"  Enabled:              {config.enrichment_schedule_enabled}")
    console.print(
        f"  Cycle interval:       {config.enrichment_schedule_interval // 3600}h"
        f" ({config.enrichment_schedule_interval}s)"
    )
    console.print(
        f"  Gemini interval:      {config.enrichment_schedule_gemini_interval // 86400}d"
        f" ({config.enrichment_schedule_gemini_interval}s)"
    )
    console.print(
        f"  Maintenance interval: {config.enrichment_schedule_maintenance_interval // 86400}d"
        f" ({config.enrichment_schedule_maintenance_interval}s)"
    )
    console.print(f"  Item limit per phase: {config.enrichment_schedule_limit}")

    console.print("\n[bold]Scheduler State[/bold]")
    console.print(f"  Total cycles run:     {state.cycle_count}")

    if state.last_cycle > 0:
        import datetime as dt

        last = dt.datetime.fromtimestamp(state.last_cycle)
        ago = int(time.time() - state.last_cycle)
        console.print(f"  Last cycle:           {last:%Y-%m-%d %H:%M} ({ago}s ago)")
    else:
        console.print("  Last cycle:           [dim]never[/dim]")

    if state.last_gemini > 0:
        import datetime as dt

        last = dt.datetime.fromtimestamp(state.last_gemini)
        ago = int(time.time() - state.last_gemini)
        console.print(f"  Last Gemini batch:    {last:%Y-%m-%d %H:%M} ({ago}s ago)")
    else:
        console.print("  Last Gemini batch:    [dim]never[/dim]")

    if state.last_maintenance > 0:
        import datetime as dt

        last = dt.datetime.fromtimestamp(state.last_maintenance)
        ago = int(time.time() - state.last_maintenance)
        console.print(f"  Last maintenance:     {last:%Y-%m-%d %H:%M} ({ago}s ago)")
    else:
        console.print("  Last maintenance:     [dim]never[/dim]")

    console.print()


@schedule.command("run-now")
@click.option("-l", "--limit", default=500, help="Max items per phase.")
def schedule_run_now(limit: int) -> None:
    """Run enrichment cycle immediately (all phases)."""
    from alibi.daemon.scheduler import EnrichmentScheduler

    db_manager = get_db()
    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    config = get_config()
    config.enrichment_schedule_limit = limit

    scheduler = EnrichmentScheduler(
        db_factory=lambda: db_manager,
        config=config,
    )

    console.print("[bold]Running enrichment cycle...[/bold]\n")
    result = scheduler.run_now()

    table = Table(title="Enrichment Cycle Results")
    table.add_column("Phase", style="cyan")
    table.add_column("Items", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Status", style="green")

    for phase in result.phases:
        if phase.skipped:
            status = "[dim]skipped[/dim]"
        elif phase.error:
            status = f"[red]error: {phase.error[:40]}[/red]"
        else:
            status = "[green]ok[/green]"

        table.add_row(
            phase.name,
            str(phase.items_processed),
            f"{phase.duration_seconds:.1f}s",
            status,
        )

    console.print(table)
    console.print(
        f"\n[green]Total: {result.total_enriched} items "
        f"in {result.duration_seconds:.1f}s[/green]"
    )
