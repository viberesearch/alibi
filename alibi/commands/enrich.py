"""Enrichment commands: product enrichment from external data sources."""

from __future__ import annotations

import click
from rich.table import Table

from alibi.commands.shared import console
from alibi.config import get_config
from alibi.db.connection import DatabaseManager, get_db


@click.group()
def enrich() -> None:
    """Product enrichment from external data sources."""
    pass


@enrich.command("pending")
@click.option(
    "--limit",
    "-l",
    default=100,
    show_default=True,
    help="Max items to enrich",
)
def enrich_pending(limit: int) -> None:
    """Enrich fact items that have a barcode but no brand/category.

    Looks up barcodes in Open Food Facts (cached locally) and updates
    brand and category fields on matching fact items.
    """
    from alibi.enrichment.service import enrich_pending_items

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    console.print(f"[dim]Enriching up to {limit} items...[/dim]")
    results = enrich_pending_items(db, limit=limit)

    success = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)

    if not results:
        console.print("[dim]No items with barcodes need enrichment.[/dim]")
    else:
        if success:
            console.print(f"[green]Enriched {success} item(s).[/green]")
        if failed:
            console.print(
                f"[yellow]{failed} barcode(s) not found in Open Food Facts.[/yellow]"
            )


@enrich.command("barcode")
@click.argument("barcode")
def enrich_barcode(barcode: str) -> None:
    """Enrich all fact items matching a specific barcode.

    Looks up the barcode in Open Food Facts and updates all matching
    fact items with brand and category data.
    """
    from alibi.enrichment.service import enrich_by_barcode

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    results = enrich_by_barcode(db, barcode)

    if not results:
        console.print(f"[yellow]No fact items found with barcode {barcode}.[/yellow]")
        return

    success = sum(1 for r in results if r.success)
    for r in results:
        if r.success:
            console.print(
                f"[green]Item {r.item_id[:8]}: "
                f"brand={r.brand or '?'}, "
                f"category={r.category or '?'}[/green]"
            )
        else:
            console.print(
                f"[yellow]Item {r.item_id[:8]}: "
                f"barcode {r.barcode} not found in OFF[/yellow]"
            )

    console.print(f"\n{success}/{len(results)} item(s) enriched.")


@enrich.command("cascade")
@click.option("-l", "--limit", default=100, help="Max items to process")
def enrich_cascade(limit: int) -> None:
    """Enrich items using the full multi-source barcode cascade (OFF -> UPCitemdb -> GS1)."""
    from alibi.enrichment.service import enrich_item_cascade

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    rows = db.fetchall(
        "SELECT id, barcode FROM fact_items "
        "WHERE barcode IS NOT NULL AND barcode != '' "
        "AND (brand IS NULL OR brand = '' "
        "     OR category IS NULL OR category = '') "
        "LIMIT ?",
        (limit,),
    )
    if not rows:
        console.print("[dim]No items with barcodes need enrichment.[/dim]")
        return

    enriched = 0
    for row in rows:
        result = enrich_item_cascade(db, row["id"], row["barcode"])
        if result.success:
            enriched += 1
            console.print(
                f"  {row['barcode']}: "
                f"{result.brand or '?'} / {result.category or '?'} "
                f"({result.source})"
            )

    console.print(f"\nEnriched {enriched}/{len(rows)} items via cascade")


@enrich.command("resolve")
@click.option(
    "--limit",
    "-l",
    default=100,
    show_default=True,
    help="Max items to resolve",
)
def enrich_resolve(limit: int) -> None:
    """Resolve brand/category for items without barcodes via name matching.

    Fuzzy-matches item names against previously enriched products to
    propagate brand and category data.
    """
    from alibi.enrichment.service import enrich_pending_by_name

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    console.print(f"[dim]Resolving up to {limit} items by name...[/dim]")
    results = enrich_pending_by_name(db, limit=limit)

    success = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)

    if not results:
        console.print("[dim]No items without barcodes need enrichment.[/dim]")
    else:
        for r in results:
            if r.success:
                console.print(
                    f"[green]  {r.item_id[:8]}: "
                    f"brand={r.brand or '?'}, "
                    f"category={r.category or '?'} "
                    f"(matched: {r.product_name})[/green]"
                )
        if success:
            console.print(f"[green]Resolved {success} item(s).[/green]")
        if failed:
            console.print(f"[dim]{failed} item(s) had no matching products.[/dim]")


@enrich.command("infer")
@click.option(
    "--limit",
    "-l",
    default=100,
    show_default=True,
    help="Max items to process",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Ollama model override (default: config)",
)
def enrich_infer(limit: int, model: str | None) -> None:
    """Infer brand/category using local LLM for items without matches.

    Uses qwen3:8b (or configured model) to infer brand and product
    category from item names, grouped by vendor for context.
    Requires Ollama to be running.
    """
    from alibi.enrichment.llm_enrichment import enrich_pending_by_llm

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    console.print(
        f"[dim]Inferring brand/category for up to {limit} items via LLM...[/dim]"
    )
    results = enrich_pending_by_llm(db, limit=limit, model=model)

    success = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)

    if not results:
        console.print("[dim]No items need LLM enrichment.[/dim]")
    else:
        for r in results:
            if r.success:
                console.print(
                    f"[green]  {r.item_id[:8]}: "
                    f"brand={r.brand or '?'}, "
                    f"category={r.category or '?'}[/green]"
                )
        if success:
            console.print(f"[green]Inferred {success} item(s).[/green]")
        if failed:
            console.print(f"[dim]{failed} item(s) could not be inferred.[/dim]")


@enrich.command("cloud")
@click.option(
    "--limit",
    "-l",
    default=100,
    show_default=True,
    help="Max items to process",
)
def enrich_cloud(limit: int) -> None:
    """Infer brand/category using Anthropic Claude API (cloud, privacy-safe).

    Sends ONLY product names and barcodes -- no financial data, dates,
    or vendor information. Requires ALIBI_CLOUD_ENRICHMENT_ENABLED=1
    and ANTHROPIC_API_KEY to be set.

    This is Tier 3 enrichment: a last resort after Open Food Facts,
    product resolver, and local LLM have all been attempted.
    """
    from alibi.enrichment.cloud_enrichment import enrich_pending_by_cloud

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    console.print(
        f"[dim]Inferring brand/category for up to {limit} items via cloud API...[/dim]"
    )
    results = enrich_pending_by_cloud(db, limit=limit)

    success = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)

    if not results:
        console.print(
            "[dim]No items need cloud enrichment "
            "(or ALIBI_CLOUD_ENRICHMENT_ENABLED is not set).[/dim]"
        )
    else:
        for r in results:
            if r.success:
                console.print(
                    f"[green]  {r.item_id[:8]}: "
                    f"brand={r.brand or '?'}, "
                    f"category={r.category or '?'}[/green]"
                )
        if success:
            console.print(f"[green]Inferred {success} item(s) via cloud.[/green]")
        if failed:
            console.print(f"[dim]{failed} item(s) could not be inferred.[/dim]")


@enrich.command("refine")
@click.option(
    "--limit",
    "-l",
    default=100,
    show_default=True,
    help="Max items to process",
)
def enrich_refine(limit: int) -> None:
    """Refine LLM-inferred categories using cloud API (Sonnet).

    Sends items with LLM-assigned categories to Claude Sonnet for
    verification. Only corrects categories where the cloud model
    disagrees with the local LLM assignment.

    Uses the configured cloud_enrichment_model (default: claude-sonnet-4-6).
    Requires ANTHROPIC_API_KEY to be set.
    """
    from alibi.enrichment.cloud_enrichment import refine_categories_by_cloud

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    console.print(
        f"[dim]Refining LLM-inferred categories for up to {limit} items via cloud API...[/dim]"
    )
    results = refine_categories_by_cloud(db, limit=limit)

    if not results:
        console.print(
            "[dim]No category corrections made "
            "(all LLM categories verified, or ANTHROPIC_API_KEY is not set).[/dim]"
        )
    else:
        for r in results:
            if r.success:
                console.print(
                    f"[green]  {r.item_id[:8]}: "
                    f"category -> {r.category or '?'}[/green]"
                )
        console.print(
            f"[green]Corrected {len(results)} category/ies via cloud.[/green]"
        )


@enrich.command("gemini")
@click.option(
    "-l",
    "--limit",
    default=500,
    help="Max items to enrich",
)
def enrich_gemini(limit: int) -> None:
    """Enrich items using Gemini mega-batch (brand + category + unit).

    Sends all pending items in a single Gemini API call. Historical
    unit_quantity data is applied first where available.
    """
    from alibi.enrichment.gemini_enrichment import enrich_pending_by_gemini

    config = get_config()
    db = DatabaseManager(config)
    if not db.is_initialized():
        db.initialize()

    results = enrich_pending_by_gemini(db, limit=limit)

    if not results:
        console.print(
            "[dim]No items need Gemini enrichment or feature is disabled.[/dim]"
        )
        return

    success = sum(1 for r in results if r.success)
    with_uq = sum(1 for r in results if r.unit_quantity is not None)

    table = Table(title=f"Gemini Enrichment: {success}/{len(results)} items")
    table.add_column("Item", style="cyan")
    table.add_column("Brand")
    table.add_column("Category")
    table.add_column("Unit Qty")
    table.add_column("Unit")
    table.add_column("Status")

    for r in results[:20]:  # Show first 20
        status = "[green]OK[/green]" if r.success else "[red]SKIP[/red]"
        table.add_row(
            r.item_id[:8],
            r.brand or "",
            r.category or "",
            str(r.unit_quantity) if r.unit_quantity else "",
            r.unit or "",
            status,
        )

    console.print(table)
    console.print(f"\n{success}/{len(results)} enriched, {with_uq} with unit_quantity")


@enrich.command("normalize-names")
@click.option("-l", "--limit", default=500, help="Max items to process")
def enrich_normalize_names(limit: int) -> None:
    """Normalize item names via Gemini (generate English comparable_name).

    Finds items where name_normalized equals the raw name (no translation)
    and sends them to Gemini for comparable_name generation.
    """
    from alibi.enrichment.gemini_enrichment import normalize_names_by_gemini

    config = get_config()
    db = DatabaseManager(config)
    if not db.is_initialized():
        db.initialize()

    results = normalize_names_by_gemini(db, limit=limit)

    if not results:
        console.print(
            "[dim]No items need name normalization or feature is disabled.[/dim]"
        )
        return

    success = sum(1 for r in results if r.success)
    with_cn = sum(1 for r in results if r.comparable_name is not None)

    table = Table(title=f"Name Normalization: {success}/{len(results)} items")
    table.add_column("Item", style="cyan")
    table.add_column("Brand")
    table.add_column("Category")
    table.add_column("Comparable Name")
    table.add_column("Status")

    for r in results[:20]:
        status = "[green]OK[/green]" if r.success else "[red]SKIP[/red]"
        table.add_row(
            r.item_id[:8],
            r.brand or "",
            r.category or "",
            r.comparable_name or "",
            status,
        )

    console.print(table)
    console.print(
        f"\n{success}/{len(results)} enriched, {with_cn} with comparable_name"
    )


@enrich.command("lookup")
@click.argument("barcode")
def enrich_lookup(barcode: str) -> None:
    """Look up a barcode in Open Food Facts (no DB update).

    Shows product information for the given barcode without modifying
    any fact items. Useful for testing barcode lookups.
    """
    from alibi.enrichment.off_client import cached_lookup

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    product = cached_lookup(db, barcode)
    if not product:
        console.print(
            f"[yellow]Barcode {barcode} not found in Open Food Facts.[/yellow]"
        )
        return

    table = Table(title=f"Product: {barcode}")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Name", product.get("product_name", "?"))
    table.add_row("Brands", product.get("brands", "?"))
    table.add_row("Categories", product.get("categories", "?"))
    table.add_row("Quantity", product.get("quantity", "?"))
    table.add_row("Nutriscore", product.get("nutriscore_grade", "?"))

    nutriments = product.get("nutriments", {})
    if nutriments:
        kcal = nutriments.get("energy-kcal_100g")
        if kcal is not None:
            table.add_row("Energy (kcal/100g)", str(kcal))
        sugar = nutriments.get("sugars_100g")
        if sugar is not None:
            table.add_row("Sugars (g/100g)", str(sugar))
        protein = nutriments.get("proteins_100g")
        if protein is not None:
            table.add_row("Protein (g/100g)", str(protein))

    console.print(table)


@enrich.command("review")
@click.option(
    "--threshold",
    "-t",
    default=0.8,
    show_default=True,
    help="Confidence threshold -- items below this need review",
)
@click.option(
    "--limit",
    "-l",
    default=50,
    show_default=True,
    help="Max items to show",
)
def enrich_review(threshold: float, limit: int) -> None:
    """Show fact items with low-confidence enrichment that need review.

    Items are shown worst-first (lowest confidence at the top).
    """
    from alibi.services.enrichment_review import get_review_queue

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    items = get_review_queue(db, threshold=threshold, limit=limit)

    if not items:
        console.print(
            f"[green]No items with confidence below {threshold} need review.[/green]"
        )
        return

    table = Table(title=f"Enrichment Review Queue (threshold={threshold})")
    table.add_column("ID", style="dim", width=9)
    table.add_column("Name")
    table.add_column("Brand")
    table.add_column("Category")
    table.add_column("Source")
    table.add_column("Conf", justify="right")
    table.add_column("Vendor", style="dim")

    for item in items:
        conf = item.get("enrichment_confidence")
        conf_str = f"{conf:.2f}" if conf is not None else "?"
        table.add_row(
            (item["id"] or "")[:8],
            item.get("name") or "",
            item.get("brand") or "",
            item.get("category") or "",
            item.get("enrichment_source") or "",
            conf_str,
            item.get("vendor") or "",
        )

    console.print(table)
    console.print(f"\n[dim]{len(items)} item(s) need review.[/dim]")


@enrich.command("confirm")
@click.argument("item_id")
@click.option("--brand", "-b", default=None, help="Override brand value")
@click.option("--cat", "-c", default=None, help="Override category value")
def enrich_confirm(item_id: str, brand: str | None, cat: str | None) -> None:
    """Confirm enrichment for a fact item, optionally correcting brand/category.

    Sets enrichment_source='user_confirmed' and enrichment_confidence=1.0.
    """
    from alibi.services.enrichment_review import confirm_enrichment

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    ok = confirm_enrichment(db, item_id, brand=brand, category=cat)
    if ok:
        parts = ["[green]Confirmed enrichment for item"]
        if brand:
            parts.append(f" brand={brand!r}")
        if cat:
            parts.append(f" category={cat!r}")
        console.print("".join(parts) + "[/green]")
    else:
        console.print(f"[yellow]Item {item_id} not found.[/yellow]")


@enrich.command("reject")
@click.argument("item_id")
def enrich_reject(item_id: str) -> None:
    """Reject enrichment for a fact item, clearing brand/category back to NULL.

    Use this when the enrichment is wrong and you want the item to be
    re-enriched or left unenriched.
    """
    from alibi.services.enrichment_review import reject_enrichment

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    ok = reject_enrichment(db, item_id)
    if ok:
        console.print(f"[green]Enrichment rejected for item {item_id[:8]}.[/green]")
    else:
        console.print(f"[yellow]Item {item_id} not found.[/yellow]")


@enrich.command("stats")
def enrich_stats() -> None:
    """Show enrichment statistics: counts by source and average confidence."""
    from alibi.services.enrichment_review import get_review_stats

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    stats = get_review_stats(db)

    total = stats.get("total_enriched", 0)
    avg = stats.get("avg_confidence")
    pending = stats.get("pending_review", 0)

    console.print("\n[bold]Enrichment Statistics[/bold]")
    console.print(f"Total enriched items: {total}")
    if avg is not None:
        console.print(f"Average confidence:   {avg:.3f}")
    console.print(f"Pending review (<0.8): {pending}")

    by_source = stats.get("by_source", [])
    if by_source:
        table = Table(title="By Source")
        table.add_column("Source")
        table.add_column("Count", justify="right")
        table.add_column("Avg Confidence", justify="right")
        for row in by_source:
            avg_conf = row.get("avg_confidence")
            avg_str = f"{avg_conf:.3f}" if avg_conf is not None else "?"
            table.add_row(
                row.get("enrichment_source") or "",
                str(row.get("count", 0)),
                avg_str,
            )
        console.print(table)
    else:
        console.print("[dim]No enriched items found.[/dim]")


@enrich.command("match")
@click.option("-c", "--category", default=None, help="Filter by category")
@click.option("-l", "--limit", default=200, help="Max products to analyze")
def enrich_match(category: str | None, limit: int) -> None:
    """Find cross-vendor product matches via Gemini."""
    db = get_db()
    from alibi.services import find_product_matches

    groups = find_product_matches(db, category=category, limit=limit)

    if not groups:
        console.print("[yellow]No cross-vendor matches found.[/yellow]")
        return

    console.print(f"Found [bold]{len(groups)}[/bold] product match groups:\n")
    for g in groups:
        console.print(
            f"[bold]{g.canonical_name}[/bold] (confidence: {g.confidence:.0%})"
        )
        console.print(f"  Reasoning: {g.reasoning}")
        for p in g.products:
            console.print(f'  - "{p.name}" at {p.vendor_name}')
        console.print()


@enrich.command("barcode-match")
@click.option("-l", "--limit", default=500, help="Max barcodes to process")
def enrich_barcode_match(limit: int) -> None:
    """Propagate enrichment across vendors sharing the same barcode."""
    from alibi.enrichment.barcode_matcher import (
        get_barcode_coverage,
        match_all_barcodes,
    )

    db = get_db()
    if not db.is_initialized():
        console.print("[yellow]Database not initialized.[/yellow]")
        return

    # Show coverage stats first
    stats = get_barcode_coverage(db)
    console.print(
        f"Barcode items: {stats['total_with_barcode']} total, "
        f"{stats['enriched']} enriched, {stats['unenriched']} unenriched"
    )
    console.print(
        f"Cross-vendor barcodes: {stats['cross_vendor_barcodes']}, "
        f"matchable: {stats['matchable']}"
    )

    if stats["matchable"] == 0:
        console.print("[dim]No items to match via barcode.[/dim]")
        return

    results = match_all_barcodes(db, limit=limit)
    if results:
        for r in results:
            console.print(
                f"  {r.barcode}: {r.brand or '?'} / {r.category or '?'} "
                f"-> item {r.item_id[:8]}"
            )
        console.print(f"\n[green]Matched {len(results)} items via barcode.[/green]")
    else:
        console.print("[dim]No new matches found.[/dim]")
