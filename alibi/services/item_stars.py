"""Item-analytics service over the materialised ``item_stars`` table.

``item_stars`` (migration 039) denormalises each ``fact_item`` together with
its parent fact's vendor / vendor_key / currency / country / event_date /
event_time, so the "item as star" analytics queries -- average comparable unit
price by product across vendors/countries/periods, price trends, basket
composition -- run without per-call joins or GROUP-BY-over-joins.

This module owns:

* **sync** -- :func:`rebuild_item_stars` (full table rebuild, the drift
  safety net behind ``lt items rebuild``) and
  :func:`refresh_item_stars_for_fact` / :func:`refresh_item_stars_for_document`
  (incremental refresh after enrichment writes to ``fact_items``);
* **read** -- :func:`list_item_stars` plus the aggregation endpoints
  :func:`avg_comparable_price`, :func:`price_trend`, :func:`basket_composition`.

All reads accept the same A-axis filter dict used by
:func:`alibi.services.query.list_fact_items_with_fact`, so filters compose
across the item surface.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any

from alibi.db import item_stars as _item_stars_sql
from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)

# Attribute facet keys are interpolated into JSON paths, so they must be a
# strict safe identifier (validated, never parameterised) to avoid injection.
_ATTR_KEY_RE = re.compile(r"^[a-z0-9_]{1,40}$")


def _attr_path_expr(key: str) -> str:
    """json_extract expression for a validated attribute facet key."""
    k = str(key).strip().lower()
    if not _ATTR_KEY_RE.match(k):
        raise ValueError(f"Invalid attribute key: {key!r}")
    return f"json_extract(attributes, '$.{k}')"


# ---------------------------------------------------------------------------
# Sync (write) operations
# ---------------------------------------------------------------------------


def rebuild_item_stars(db: DatabaseManager) -> int:
    """Fully rebuild item_stars from fact_items + facts.

    Truncates the table and re-materialises every row. This is the
    authoritative reconciliation used by ``lt items rebuild`` so the mirror can
    never silently drift from the canonical fact_items/facts tables (e.g. after
    batch enrichment passes that bypass the incremental hooks).

    Returns the number of rows materialised.
    """
    with db.transaction() as cursor:
        count = _item_stars_sql.rebuild_all(cursor)
    logger.info("Rebuilt item_stars: %d rows", count)
    return count


def refresh_item_stars_for_fact(db: DatabaseManager, fact_id: str) -> None:
    """Re-materialise item_stars rows for one fact (delete + reinsert).

    Use after a write that touched a single fact's items outside the collapse
    path -- e.g. a line-item correction or per-fact enrichment.
    """
    with db.transaction() as cursor:
        _item_stars_sql.refresh_fact(cursor, fact_id)


def refresh_item_stars_for_items(db: DatabaseManager, item_ids: list[str]) -> int:
    """Refresh item_stars for the facts owning the given fact_item ids.

    Resolves each item's parent fact and refreshes those facts (deduplicated).
    Used after batch enrichment (e.g. the Gemini Phase 3 flush) writes
    brand/category to a set of items spanning several facts. Returns the number
    of distinct facts refreshed.
    """
    if not item_ids:
        return 0
    placeholders = ",".join("?" for _ in item_ids)
    rows = db.fetchall(
        "SELECT DISTINCT fact_id FROM fact_items "  # noqa: S608
        f"WHERE id IN ({placeholders})",
        tuple(item_ids),
    )
    fact_ids = [row["fact_id"] for row in rows]
    if not fact_ids:
        return 0
    with db.transaction() as cursor:
        for fact_id in fact_ids:
            _item_stars_sql.refresh_fact(cursor, fact_id)
    return len(fact_ids)


def refresh_item_stars_for_document(db: DatabaseManager, document_id: str) -> int:
    """Refresh item_stars for every fact derived from a document.

    Resolves the document's facts (via its cloud) and refreshes each. Used by
    the enrichment subscriber after it writes brand/category/comparable fields
    directly to fact_items. Returns the number of facts refreshed.
    """
    rows = db.fetchall(
        "SELECT DISTINCT f.id AS fact_id "
        "FROM facts f "
        "JOIN clouds c ON f.cloud_id = c.id "
        "JOIN cloud_bundles cb ON c.id = cb.cloud_id "
        "JOIN bundles b ON cb.bundle_id = b.id "
        "JOIN bundle_atoms ba ON b.id = ba.bundle_id "
        "JOIN atoms a ON ba.atom_id = a.id "
        "WHERE a.document_id = ?",
        (document_id,),
    )
    fact_ids = [row["fact_id"] for row in rows]
    if not fact_ids:
        return 0
    with db.transaction() as cursor:
        for fact_id in fact_ids:
            _item_stars_sql.refresh_fact(cursor, fact_id)
    return len(fact_ids)


# ---------------------------------------------------------------------------
# Read: shared filter builder (A axes over item_stars)
# ---------------------------------------------------------------------------

# Group-by dimensions exposed to aggregation callers, mapped to SQL columns.
_GROUP_DIMENSIONS = {
    "comparable_name": "comparable_name",
    "comparable_unit": "comparable_unit",
    "product_variant": "product_variant",
    "vendor": "vendor",
    "vendor_key": "vendor_key",
    "brand": "brand",
    "category": "category",
    "category_path": "category_path",
    "currency": "currency",
    "country": "country",
}

# Period bucket dimensions over event_date (SQLite strftime).
_PERIOD_EXPR = {
    "year": "strftime('%Y', event_date)",
    "month": "strftime('%Y-%m', event_date)",
    "quarter": (
        "strftime('%Y', event_date) || '-Q' || "
        "CAST((CAST(strftime('%m', event_date) AS INTEGER) + 2) / 3 AS TEXT)"
    ),
}


# Keep-condition for "real product" rows: excludes non-product receipt lines
# (tax, tip, fee, discount, deposit, totals, "non_item") that the categorize pass
# files under the taxonomy's ``adjustment`` branch and the broad-category pass
# marks ``Non_Item``. They are receipt adjustments, not basket spend, so spend
# composition leaves them out.
_PRODUCT_ONLY_PREDICATE = (
    "(category_path IS NULL OR category_path NOT LIKE 'adjustment%') "
    "AND COALESCE(category, '') != 'Non_Item'"
)


def _build_filters(filters: dict[str, Any] | None) -> tuple[list[str], list[Any]]:
    """Translate the A-axis filter dict into WHERE conditions over item_stars.

    Mirrors the recognised keys of
    :func:`alibi.services.query.list_fact_items_with_fact` but resolves them
    against the flat item_stars columns (no JOIN). Recognised keys:
    name, comparable_name, brand, vendor (substring); category, category (exact),
    vendor_key, currency, country, comparable_unit (exact); category_path
    (hierarchical prefix); date_from / date_to, datetime_from / datetime_to;
    price_min / price_max (on total_price).
    """
    filters = filters or {}
    conditions: list[str] = ["1=1"]
    params: list[Any] = []

    # Substring (LIKE) filters
    for key, column in (
        ("name", "name"),
        ("comparable_name", "comparable_name"),
        ("brand", "brand"),
        ("vendor", "vendor"),
    ):
        if filters.get(key):
            conditions.append(f"LOWER({column}) LIKE ?")
            params.append(f"%{str(filters[key]).lower()}%")

    # Exact-match filters
    for key, column in (
        ("category", "category"),
        ("comparable_unit", "comparable_unit"),
        ("vendor_key", "vendor_key"),
        ("currency", "currency"),
        ("country", "country"),
        ("product_variant", "product_variant"),
    ):
        if filters.get(key):
            conditions.append(f"{column} = ?")
            params.append(filters[key])

    date_from = filters.get("date_from")
    if date_from is not None:
        if isinstance(date_from, date):
            date_from = date_from.isoformat()
        conditions.append("event_date >= ?")
        params.append(str(date_from))

    date_to = filters.get("date_to")
    if date_to is not None:
        if isinstance(date_to, date):
            date_to = date_to.isoformat()
        conditions.append("event_date <= ?")
        params.append(str(date_to))

    if filters.get("datetime_from") is not None:
        conditions.append(
            "(event_date || ' ' || COALESCE(event_time, '00:00:00')) >= ?"
        )
        params.append(str(filters["datetime_from"]))
    if filters.get("datetime_to") is not None:
        conditions.append(
            "(event_date || ' ' || COALESCE(event_time, '23:59:59')) <= ?"
        )
        params.append(str(filters["datetime_to"]))

    if filters.get("price_min") is not None:
        conditions.append("CAST(total_price AS REAL) >= ?")
        params.append(float(filters["price_min"]))
    if filters.get("price_max") is not None:
        conditions.append("CAST(total_price AS REAL) <= ?")
        params.append(float(filters["price_max"]))

    category_path = filters.get("category_path")
    if category_path:
        prefix = str(category_path).strip().lower()
        conditions.append("(LOWER(category_path) = ? OR LOWER(category_path) LIKE ?)")
        params.append(prefix)
        params.append(f"{prefix} > %")

    # Flexible attribute facet filters: {"organic": True, "size": "L"}. A value
    # of None / "*" matches "facet present". Keys are validated; values are
    # parameterised. JSON true/false surface as integer 1/0 via json_extract.
    attrs = filters.get("attributes")
    if isinstance(attrs, dict):
        for key, value in attrs.items():
            col = _attr_path_expr(key)
            if value is None or value == "*":
                conditions.append(f"{col} IS NOT NULL")
            elif isinstance(value, bool):
                conditions.append(f"{col} = ?")
                params.append(1 if value else 0)
            elif isinstance(value, (int, float)):
                conditions.append(f"CAST({col} AS REAL) = ?")
                params.append(float(value))
            else:
                conditions.append(f"LOWER(CAST({col} AS TEXT)) = ?")
                params.append(str(value).strip().lower())

    return conditions, params


# ---------------------------------------------------------------------------
# Read: list + aggregations
# ---------------------------------------------------------------------------


def list_item_stars(
    db: DatabaseManager,
    filters: dict[str, Any] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """List item stars (one row per fact_item) matching the A-axis filters.

    Backs the "item sky" scatter/grid view. Each row is self-describing along
    every axis (item fields + parent fact fields), already denormalised.
    """
    conditions, params = _build_filters(filters)
    where = " AND ".join(conditions)
    sql = (
        "SELECT item_id, fact_id, name, comparable_name, quantity, unit, "
        "unit_price, total_price, brand, category, category_path, "
        "comparable_unit_price, comparable_unit, product_variant, "
        "vendor, vendor_key, currency, country, event_date, event_time "
        f"FROM item_stars WHERE {where} "  # noqa: S608
        "ORDER BY event_date DESC, event_time DESC"
    )
    if limit is not None:
        sql += " LIMIT ?"
        params = [*params, int(limit)]
    rows = db.fetchall(sql, tuple(params))
    return [dict(row) for row in rows]


def _resolve_group_columns(group_by: list[str] | None) -> list[tuple[str, str]]:
    """Resolve requested group dimensions to (alias, sql_expr) pairs.

    Accepts plain dimension names (vendor, country, ...) and period buckets
    (year, month, quarter). Unknown names raise ValueError.
    """
    resolved: list[tuple[str, str]] = []
    for dim in group_by or ["comparable_name"]:
        if dim in _GROUP_DIMENSIONS:
            resolved.append((dim, _GROUP_DIMENSIONS[dim]))
        elif dim in _PERIOD_EXPR:
            resolved.append((dim, _PERIOD_EXPR[dim]))
        elif dim.startswith("attr:"):
            # Group by a flexible facet, e.g. "attr:size" -> column alias "size".
            key = dim[len("attr:") :]
            resolved.append((key, _attr_path_expr(key)))
        else:
            raise ValueError(f"Unknown group dimension: {dim}")
    return resolved


def avg_comparable_price(
    db: DatabaseManager,
    filters: dict[str, Any] | None = None,
    group_by: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Average comparable_unit_price grouped along the requested dimensions.

    Answers e.g. "average EUR/L of milk in CY in Q1 2026 across vendors" in one
    call:

        avg_comparable_price(
            db,
            filters={"comparable_name": "milk", "country": "CY",
                     "date_from": "2026-01-01", "date_to": "2026-03-31"},
            group_by=["comparable_name"],
        )

    Filters compose with the A axes. Only items with a positive
    comparable_unit_price contribute (NULL and 0 are excluded -- a 0 is a
    missing/garbled price, not a real datum).

    ``comparable_unit`` is ALWAYS part of the grouping key, even if not in
    ``group_by``: averaging EUR/L together with EUR/piece is meaningless, so
    each returned row is a single unit. Group by ``product_variant`` too when a
    price-distinguishing attribute (egg size, milk/cheese fat %) must not be
    blended. Returns one row per (group keys, comparable_unit) with avg/min/max
    comparable_unit_price, item_count and vendor_count (distinct vendor_key).
    """
    group_cols = _resolve_group_columns(group_by)
    # Never blend distinct comparable_units into one average.
    if "comparable_unit" not in {alias for alias, _ in group_cols}:
        group_cols = [*group_cols, ("comparable_unit", "comparable_unit")]

    conditions, params = _build_filters(filters)
    # Average over the EUR-normalised price so cross-currency comparison is valid;
    # a non-EUR row not yet converted (NULL *_eur) drops out rather than blending.
    conditions.append("comparable_unit_price_eur IS NOT NULL")
    conditions.append("comparable_unit_price_eur > 0")

    select_parts = [f"{expr} AS {alias}" for alias, expr in group_cols]
    group_exprs = [expr for _, expr in group_cols]
    where = " AND ".join(conditions)
    group_clause = ", ".join(group_exprs)

    sql = (
        f"SELECT {', '.join(select_parts)}, "  # noqa: S608
        "AVG(comparable_unit_price_eur) AS avg_comparable_unit_price, "
        "MIN(comparable_unit_price_eur) AS min_comparable_unit_price, "
        "MAX(comparable_unit_price_eur) AS max_comparable_unit_price, "
        "COUNT(*) AS item_count, "
        "COUNT(DISTINCT vendor_key) AS vendor_count "
        f"FROM item_stars WHERE {where} "
        f"GROUP BY {group_clause} "
        "ORDER BY item_count DESC"
    )
    rows = db.fetchall(sql, tuple(params))
    return [dict(row) for row in rows]


def price_by_state(
    db: DatabaseManager,
    filters: dict[str, Any] | None = None,
    min_states: int = 2,
) -> list[dict[str, Any]]:
    """Compare comparable_unit_price across product STATE, within a product.

    The analytics answer the #58 state facet makes possible: for a single
    comparable product, how its normalised price differs by preservation /
    preparation form -- fresh vs canned vs frozen artichokes, raw vs roasted
    nuts, fresh vs smoked salmon. ``state`` is read from the ``attributes`` JSON
    (``$.state``), the same facet ``attr:state`` groups on.

    Unlike a plain ``avg_comparable_price(group_by=["attr:state"])`` this is
    scoped to a real *comparison*: only products that actually appear in at least
    ``min_states`` distinct states (within one ``comparable_unit`` + currency) are
    returned, so every row sits beside a sibling state to compare against. As
    everywhere, EUR/kg is never blended with EUR/L or EUR/pcs (the grouping key
    always carries ``comparable_unit``), and NULL / non-positive prices are
    excluded. Filters compose with the standard A axes.

    Returns one row per (comparable_name, comparable_unit, state, currency) with
    avg / min / max comparable_unit_price and item_count, ordered so a product's
    states sit together cheapest-first.
    """
    conditions, params = _build_filters(filters)
    conditions.append("comparable_name IS NOT NULL")
    conditions.append("comparable_name != ''")
    conditions.append("comparable_unit_price_eur IS NOT NULL")
    conditions.append("comparable_unit_price_eur > 0")
    conditions.append("json_extract(attributes, '$.state') IS NOT NULL")
    where = " AND ".join(conditions)

    # CTEs: project the stated rows once, find the products with >= min_states
    # distinct states, then aggregate per state but only for those products. The
    # price is the EUR-normalised one so fresh-vs-canned compares across currency.
    sql = (
        "WITH stated AS ("  # noqa: S608 - conditions are trusted internal literals
        "  SELECT comparable_name, comparable_unit, currency, "
        "         json_extract(attributes, '$.state') AS state, "
        "         comparable_unit_price_eur AS cup "
        "  FROM item_stars "
        f"  WHERE {where}"
        "), multi AS ("
        "  SELECT comparable_name, comparable_unit, currency "
        "  FROM stated "
        "  GROUP BY comparable_name, comparable_unit, currency "
        "  HAVING COUNT(DISTINCT state) >= ?"
        ") "
        "SELECT s.comparable_name, s.comparable_unit, s.state, s.currency, "
        "       COUNT(*) AS item_count, "
        "       AVG(s.cup) AS avg_comparable_unit_price, "
        "       MIN(s.cup) AS min_comparable_unit_price, "
        "       MAX(s.cup) AS max_comparable_unit_price "
        "FROM stated s "
        "JOIN multi m ON s.comparable_name = m.comparable_name "
        "  AND s.comparable_unit IS m.comparable_unit "
        "  AND s.currency IS m.currency "
        "GROUP BY s.comparable_name, s.comparable_unit, s.state, s.currency "
        "ORDER BY s.comparable_name, s.comparable_unit, avg_comparable_unit_price"
    )
    rows = db.fetchall(sql, (*params, int(min_states)))
    return [dict(row) for row in rows]


def price_trend(
    db: DatabaseManager,
    comparable_name: str,
    filters: dict[str, Any] | None = None,
    period: str = "month",
    by_vendor: bool = True,
) -> list[dict[str, Any]]:
    """Comparable unit-price trend for one product over time across vendors.

    Returns a series bucketed by ``period`` (year | month | quarter), optionally
    split by vendor, so callers can plot how a product's normalised price moves
    across vendors over time. Only items with a positive comparable_unit_price
    contribute. ``comparable_unit`` is part of the grouping key so a product
    measured in different units (e.g. some EUR/L, some EUR/piece rows) yields
    separate, non-blended series.
    """
    if period not in _PERIOD_EXPR:
        raise ValueError(f"Unknown period: {period}")

    merged = {**(filters or {}), "comparable_name": comparable_name}
    conditions, params = _build_filters(merged)
    conditions.append("comparable_unit_price_eur IS NOT NULL")
    conditions.append("comparable_unit_price_eur > 0")
    where = " AND ".join(conditions)

    period_expr = _PERIOD_EXPR[period]
    select_parts = [f"{period_expr} AS period", "comparable_unit"]
    group_exprs = [period_expr, "comparable_unit"]
    if by_vendor:
        select_parts.append("vendor")
        select_parts.append("vendor_key")
        group_exprs.append("vendor_key")

    sql = (
        f"SELECT {', '.join(select_parts)}, "  # noqa: S608
        "AVG(comparable_unit_price_eur) AS avg_comparable_unit_price, "
        "COUNT(*) AS item_count "
        f"FROM item_stars WHERE {where} "
        f"GROUP BY {', '.join(group_exprs)} "
        "ORDER BY period ASC"
    )
    rows = db.fetchall(sql, tuple(params))
    return [dict(row) for row in rows]


def basket_composition(
    db: DatabaseManager,
    filters: dict[str, Any] | None = None,
    by: str = "category",
) -> list[dict[str, Any]]:
    """Spend composition grouped by a categorical axis.

    ``by`` is one of the group dimensions (default ``category``; ``category_path``
    groups by the full hierarchical path, ``brand``/``vendor`` etc. also work).
    Returns one row per group with item_count and total_spent (sum of
    total_price_eur), ordered by spend. Rows with a NULL group key are folded into
    an "(uncategorised)" bucket so totals reconcile.

    Non-product receipt lines (tax / tip / fee / discount / deposit / "non_item")
    are EXCLUDED: a basket's spend is what you bought, not the receipt's
    adjustments. See :data:`_PRODUCT_ONLY_PREDICATE`.
    """
    if by in _GROUP_DIMENSIONS:
        expr = _GROUP_DIMENSIONS[by]
    elif by in _PERIOD_EXPR:
        expr = _PERIOD_EXPR[by]
    else:
        raise ValueError(f"Unknown composition dimension: {by}")

    conditions, params = _build_filters(filters)
    conditions.append(_PRODUCT_ONLY_PREDICATE)
    where = " AND ".join(conditions)
    # Spend is summed in EUR (total_price_eur) so categories never blend foreign
    # amounts as if 1 CAD == 1 EUR. A not-yet-converted foreign row is NULL and
    # simply doesn't contribute to its category's EUR spend.
    sql = (
        f"SELECT COALESCE({expr}, '(uncategorised)') AS {by}, "  # noqa: S608
        "COUNT(*) AS item_count, "
        "SUM(total_price_eur) AS total_spent "
        f"FROM item_stars WHERE {where} "
        f"GROUP BY COALESCE({expr}, '(uncategorised)') "
        "ORDER BY total_spent DESC"
    )
    rows = db.fetchall(sql, tuple(params))
    return [dict(row) for row in rows]


def list_attribute_facets(
    db: DatabaseManager,
    filters: dict[str, Any] | None = None,
    min_count: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    """Enumerate the distinct attribute facets present, with value counts.

    Expands every item's ``attributes`` JSON map (json_each) and groups by
    key+value, so callers (the Item Sky facet chips) can discover what facets
    exist and offer them as filters. Honours the A-axis filters so the facet
    list reflects the current view.

    Only (key, value) pairs covering at least ``min_count`` items are returned,
    and structurally-noisy keys are dropped, so one-off model artefacts don't
    clutter the facet chips. Returns ``{key: [{"value", "item_count"}, ...]}``.
    """
    from alibi.enrichment.attributes import _DENY_KEYS

    conditions, params = _build_filters(filters)
    conditions.append("attributes IS NOT NULL")
    conditions.append("attributes != '{}'")
    where = " AND ".join(conditions)
    rows = db.fetchall(
        "SELECT je.key AS facet_key, je.value AS facet_value, "  # noqa: S608
        "COUNT(*) AS item_count "
        "FROM item_stars, json_each(item_stars.attributes) je "
        f"WHERE {where} "
        "GROUP BY je.key, je.value "
        "HAVING COUNT(*) >= ? "
        "ORDER BY je.key ASC, item_count DESC",
        (*params, int(min_count)),
    )
    facets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = row["facet_key"]
        if key in _DENY_KEYS:
            continue
        facets.setdefault(key, []).append(
            {"value": row["facet_value"], "item_count": row["item_count"]}
        )
    return facets
