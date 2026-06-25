"""Low-level SQL to keep the materialised ``item_stars`` table in sync.

``item_stars`` is a denormalised mirror of ``fact_items`` joined to ``facts``
(see migration 039). This module owns the canonical column projection and the
refresh/rebuild SQL so every writer -- the store path
(:func:`alibi.db.v2_store.store_fact`), the enrichment hooks, and the
``lt items rebuild`` CLI -- materialises rows identically.

It is intentionally db-layer (operates on a raw ``sqlite3`` cursor) so it can
run inside the same transaction as the fact write. The service-layer wrappers
in :mod:`alibi.services.item_stars` provide transaction management for callers
outside an open transaction.
"""

from __future__ import annotations

from sqlite3 import Cursor

# Columns written to item_stars, in order. The first 15 mirror fact_items (fi);
# the rest are the parent fact axes (f). Keep in lockstep with _SELECT_EXPR.
_INSERT_COLUMNS = (
    "item_id",
    "fact_id",
    "name",
    "name_normalized",
    "comparable_name",
    "quantity",
    "unit",
    "unit_price",
    "total_price",
    "brand",
    "category",
    "category_path",
    "comparable_unit_price",
    "comparable_unit",
    "product_variant",
    "attributes",
    "fact_type",
    "vendor",
    "vendor_key",
    "currency",
    "country",
    "event_date",
    "event_time",
    "total_price_eur",
    "comparable_unit_price_eur",
)

# The fact's EUR conversion factor: its resolved ``eur_rate`` (set by
# ``lt fx backfill``), or 1.0 for an EUR / currency-less fact even before any
# backfill, or NULL for a not-yet-converted foreign fact (so its *_eur stays NULL
# and it drops out of EUR-only aggregations rather than blending). Multiplying the
# item's amounts by it here materialises the EUR figures the analytics sum.
_EUR_FACTOR = (
    "COALESCE(f.eur_rate, CASE WHEN COALESCE(f.currency, 'EUR') = 'EUR' "
    "THEN 1.0 END)"
)

_SELECT_EXPR = (
    "fi.id, fi.fact_id, fi.name, fi.name_normalized, fi.comparable_name, "
    "fi.quantity, fi.unit, fi.unit_price, fi.total_price, fi.brand, "
    "fi.category, fi.category_path, fi.comparable_unit_price, "
    "fi.comparable_unit, fi.product_variant, fi.attributes, "
    "f.fact_type, f.vendor, f.vendor_key, f.currency, f.country, "
    "f.event_date, f.event_time, "
    f"CAST(fi.total_price AS REAL) * {_EUR_FACTOR}, "
    f"CAST(fi.comparable_unit_price AS REAL) * {_EUR_FACTOR}"
)

# Built from module constants only (no user input) -- safe despite S608.
_INSERT_SQL = (  # noqa: S608
    f"INSERT OR REPLACE INTO item_stars ({', '.join(_INSERT_COLUMNS)}) "
    f"SELECT {_SELECT_EXPR} "
    "FROM fact_items fi JOIN facts f ON fi.fact_id = f.id"
)


def refresh_fact(cursor: Cursor, fact_id: str) -> None:
    """Re-materialise item_stars rows for a single fact.

    Deletes the fact's existing stars and reinserts them from the current
    ``fact_items`` + ``facts`` state, so the mirror reflects any item edits or
    enrichment writes made since the last refresh.
    """
    cursor.execute("DELETE FROM item_stars WHERE fact_id = ?", (fact_id,))
    cursor.execute(_INSERT_SQL + " WHERE fi.fact_id = ?", (fact_id,))


def rebuild_all(cursor: Cursor) -> int:
    """Truncate and fully rebuild item_stars from fact_items + facts.

    Returns the number of rows materialised.
    """
    cursor.execute("DELETE FROM item_stars")
    cursor.execute(_INSERT_SQL)
    row = cursor.execute("SELECT COUNT(*) FROM item_stars").fetchone()
    return int(row[0]) if row else 0
