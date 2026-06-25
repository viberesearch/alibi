"""Items group: the item-as-star analytics surface over ``item_stars``.

``item_stars`` is a materialised mirror of fact_items + parent fact axes
(migration 039), kept in sync on the collapse/store path and the per-fact
enrichment hook. ``lt items rebuild`` is the drift safety net -- run it after
batch enrichment passes (``lt enrich categorize``, comparable backfills) that
write fact_items directly without going through the incremental hooks.
"""

from __future__ import annotations

from typing import Any

import click
from rich.table import Table

from alibi.commands.shared import console
from alibi.db.connection import get_db


@click.group()
def items() -> None:
    """Item-level analytics (the 'item sky')."""


# Base A-axis filter options shared by every analytics command. comparable_name
# is added separately because `trend` takes it as a positional argument.
def _base_filter_options(fn: Any) -> Any:
    decorators = [
        click.option("--name", help="Substring match on item name"),
        click.option(
            "--category-path", help="Hierarchical prefix, e.g. 'food > dairy'"
        ),
        click.option("--vendor", help="Substring match on vendor"),
        click.option("--country", help="Exact jurisdiction (ISO alpha-2)"),
        click.option("--currency", help="Exact ISO 4217 currency"),
        click.option("--date-from", help="Event date lower bound (ISO)"),
        click.option("--date-to", help="Event date upper bound (ISO)"),
        click.option(
            "--state",
            help="Product state facet: fresh, frozen, canned, dried, cured, "
            "pickled, roasted, cooked",
        ),
    ]
    for dec in reversed(decorators):
        fn = dec(fn)
    return fn


def _filters(opts: dict[str, Any]) -> dict[str, Any]:
    """Collect recognised non-empty A-axis filter options into a dict.

    ``--state`` is folded into the flexible ``attributes`` facet filter (the same
    surface ``attr:state`` groups on), so a state-discriminated price view —
    fresh vs canned vs frozen — needs no bespoke plumbing.
    """
    keys = (
        "name",
        "comparable_name",
        "category_path",
        "vendor",
        "country",
        "currency",
        "date_from",
        "date_to",
    )
    out = {k: opts[k] for k in keys if opts.get(k)}
    if opts.get("state"):
        out["attributes"] = {"state": str(opts["state"]).strip().lower()}
    return out


@items.command("rebuild")
def items_rebuild() -> None:
    """Fully rebuild the item_stars table from fact_items + facts.

    Reconciles the materialised mirror with the canonical tables so it can
    never silently drift. Safe to run any time.
    """
    from alibi.services import rebuild_item_stars

    db = get_db()
    count = rebuild_item_stars(db)
    console.print(f"[green]Rebuilt item_stars: {count} rows[/green]")


@items.command("avg-price")
@click.option("--comparable-name", help="Substring match on normalised product name")
@_base_filter_options
@click.option(
    "--group-by",
    default="comparable_name",
    help="Comma-separated dims: comparable_name, vendor, country, "
    "currency, brand, category, category_path, year, month, quarter, "
    "or any facet as attr:<key> (e.g. attr:state, attr:size)",
)
def items_avg_price(group_by: str, **opts: Any) -> None:
    """Average comparable unit price grouped along the requested axes.

    Example -- avg EUR/L of milk in CY in Q1 2026 across vendors:

      lt items avg-price --comparable-name milk --country CY \\
          --date-from 2026-01-01 --date-to 2026-03-31

    Example -- fresh vs canned vs frozen price of a product (the #58 state
    facet), each within its own comparable unit:

      lt items avg-price --comparable-name salmon --group-by comparable_name,attr:state
    """
    from alibi.services import avg_comparable_price

    db = get_db()
    dims = [d.strip() for d in group_by.split(",") if d.strip()]
    # A facet dim "attr:state" is aliased to its bare key ("state") in the
    # result rows, so look up / label by that alias, not the raw dim string.
    aliases = [d[len("attr:") :] if d.startswith("attr:") else d for d in dims]
    try:
        rows = avg_comparable_price(db, _filters(opts), dims)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return

    table = Table(title="Average comparable unit price")
    for alias in aliases:
        table.add_column(alias)
    table.add_column("avg", justify="right")
    table.add_column("unit")
    table.add_column("min", justify="right")
    table.add_column("max", justify="right")
    table.add_column("items", justify="right")
    table.add_column("vendors", justify="right")
    for r in rows:
        table.add_row(
            *[str(r.get(a) or "-") for a in aliases],
            f"{r['avg_comparable_unit_price']:.3f}",
            str(r.get("comparable_unit") or "-"),
            f"{r['min_comparable_unit_price']:.3f}",
            f"{r['max_comparable_unit_price']:.3f}",
            str(r["item_count"]),
            str(r["vendor_count"]),
        )
    console.print(table)


@items.command("price-by-state")
@click.option("--comparable-name", help="Substring match on normalised product name")
@_base_filter_options
@click.option(
    "--min-states",
    default=2,
    show_default=True,
    help="Only products seen in at least this many distinct states",
)
def items_price_by_state(min_states: int, **opts: Any) -> None:
    """Compare comparable unit price across product STATE, within a product.

    The #58 state facet as a price view: for each product sold in more than one
    state, its normalised price per state (fresh vs canned vs frozen; raw vs
    roasted), each within its own comparable_unit.

      lt items price-by-state --comparable-name salmon
    """
    from alibi.services import price_by_state

    db = get_db()
    rows = price_by_state(db, _filters(opts), min_states=max(2, min_states))

    if not rows:
        console.print(
            "[yellow]No multi-state products found. Run `lt enrich states` "
            "first, or widen the filters.[/yellow]"
        )
        return

    table = Table(title="Comparable unit price by product state")
    table.add_column("product")
    table.add_column("unit")
    table.add_column("state")
    table.add_column("avg", justify="right")
    table.add_column("min", justify="right")
    table.add_column("max", justify="right")
    table.add_column("items", justify="right")
    last_key = None
    for r in rows:
        key = (r.get("comparable_name"), r.get("comparable_unit"), r.get("currency"))
        # Blank the repeated product/unit cells so each product's states group
        # visually, cheapest-first.
        same = key == last_key
        table.add_row(
            "" if same else str(r.get("comparable_name") or "-"),
            "" if same else str(r.get("comparable_unit") or "-"),
            str(r.get("state") or "-"),
            f"{r['avg_comparable_unit_price']:.3f}",
            f"{r['min_comparable_unit_price']:.3f}",
            f"{r['max_comparable_unit_price']:.3f}",
            str(r["item_count"]),
        )
        last_key = key
    console.print(table)


@items.command("trend")
@click.argument("comparable_name")
@_base_filter_options
@click.option("--period", default="month", help="Bucket: year, month, or quarter")
@click.option("--no-vendor-split", is_flag=True, help="Aggregate all vendors together")
def items_trend(
    comparable_name: str,
    period: str,
    no_vendor_split: bool,
    **opts: Any,
) -> None:
    """Price trend for a product over time across vendors."""
    from alibi.services import price_trend

    db = get_db()
    try:
        rows = price_trend(
            db,
            comparable_name,
            _filters(opts),
            period=period,
            by_vendor=not no_vendor_split,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return

    table = Table(title=f"Price trend: {comparable_name}")
    table.add_column("period")
    if not no_vendor_split:
        table.add_column("vendor")
    table.add_column("avg", justify="right")
    table.add_column("unit")
    table.add_column("items", justify="right")
    for r in rows:
        cells = [str(r.get("period") or "-")]
        if not no_vendor_split:
            cells.append(str(r.get("vendor") or "-"))
        cells += [
            f"{r['avg_comparable_unit_price']:.3f}",
            str(r.get("comparable_unit") or "-"),
            str(r["item_count"]),
        ]
        table.add_row(*cells)
    console.print(table)


@items.command("basket")
@click.option("--comparable-name", help="Substring match on normalised product name")
@_base_filter_options
@click.option("--by", default="category", help="Group dimension (default category)")
def items_basket(by: str, **opts: Any) -> None:
    """Basket composition: spend grouped by a categorical axis."""
    from alibi.services import basket_composition

    db = get_db()
    try:
        rows = basket_composition(db, _filters(opts), by=by)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return

    table = Table(title="Basket composition")
    table.add_column(by)
    table.add_column("items", justify="right")
    table.add_column("spent", justify="right")
    for r in rows:
        table.add_row(
            str(r.get(by) or "-"),
            str(r["item_count"]),
            f"{r['total_spent']:.2f}" if r["total_spent"] is not None else "-",
        )
    console.print(table)
