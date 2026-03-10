"""Enrichment review service — feedback loop for low-confidence enrichments.

Provides functions to surface fact items that were enriched with low
confidence, and to record user decisions (confirm or reject).
"""

from __future__ import annotations

import logging
from typing import Any

from alibi.db.connection import DatabaseManager
from alibi.services.correction import update_fact_item

logger = logging.getLogger(__name__)


def get_review_queue(
    db: DatabaseManager,
    threshold: float = 0.8,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Get fact items needing review: have enrichment but confidence < threshold.

    Only items that have been enriched (enrichment_source IS NOT NULL) and
    whose confidence is below the threshold are returned.  Items are ordered
    worst-first so the most uncertain results appear at the top.

    Args:
        db: Database manager instance.
        threshold: Confidence below which an item needs review (exclusive).
            Defaults to 0.8.
        limit: Maximum number of items to return.

    Returns:
        List of dicts with keys: id, name, brand, category, enrichment_source,
        enrichment_confidence, fact_id, vendor (from parent fact).
    """
    rows = db.fetchall(
        """
        SELECT fi.id,
               fi.name,
               fi.brand,
               fi.category,
               fi.enrichment_source,
               fi.enrichment_confidence,
               fi.fact_id,
               f.vendor
        FROM fact_items fi
        JOIN facts f ON fi.fact_id = f.id
        WHERE fi.enrichment_source IS NOT NULL
          AND fi.enrichment_confidence IS NOT NULL
          AND fi.enrichment_confidence < ?
        ORDER BY fi.enrichment_confidence ASC
        LIMIT ?
        """,
        (threshold, limit),
    )
    return [dict(row) for row in rows]


def confirm_enrichment(
    db: DatabaseManager,
    fact_item_id: str,
    brand: str | None = None,
    category: str | None = None,
) -> bool:
    """User confirms (and optionally corrects) the enrichment for an item.

    Sets enrichment_source to 'user_confirmed' and enrichment_confidence to
    1.0.  If brand or category are provided they overwrite the current values.

    Args:
        db: Database manager instance.
        fact_item_id: ID of the fact item to confirm.
        brand: Optional brand correction; if None the existing value is kept.
        category: Optional category correction; if None the existing value is kept.

    Returns:
        True if the item was found and updated, False if not found.
    """
    fields: dict[str, object] = {
        "enrichment_source": "user_confirmed",
        "enrichment_confidence": 1.0,
    }
    if brand is not None:
        fields["brand"] = brand
    if category is not None:
        fields["category"] = category

    ok = update_fact_item(db, fact_item_id, fields)
    if ok:
        logger.info(
            "confirm_enrichment: item %s confirmed (brand=%r, category=%r)",
            fact_item_id[:8],
            brand,
            category,
        )
    return ok


def reject_enrichment(
    db: DatabaseManager,
    fact_item_id: str,
) -> bool:
    """User rejects the enrichment for an item.

    Clears brand, category, enrichment_source and enrichment_confidence back
    to NULL.  A direct SQL statement is used because update_fact_item() does
    not support setting columns to NULL via its allowlist mechanism.

    Args:
        db: Database manager instance.
        fact_item_id: ID of the fact item whose enrichment to discard.

    Returns:
        True if the item was found and updated, False if not found.
    """
    row = db.fetchone("SELECT id FROM fact_items WHERE id = ?", (fact_item_id,))
    if not row:
        return False

    with db.transaction() as cursor:
        cursor.execute(
            """
            UPDATE fact_items
            SET brand = NULL,
                category = NULL,
                enrichment_source = NULL,
                enrichment_confidence = NULL
            WHERE id = ?
            """,
            (fact_item_id,),
        )

    logger.info("reject_enrichment: item %s enrichment cleared", fact_item_id[:8])
    return True


def get_review_stats(
    db: DatabaseManager,
) -> dict[str, Any]:
    """Return enrichment statistics: counts by source and average confidence.

    Args:
        db: Database manager instance.

    Returns:
        Dict with keys:
            - by_source: list of {enrichment_source, count, avg_confidence}
            - total_enriched: total items with any enrichment_source
            - avg_confidence: overall average confidence across enriched items
            - pending_review: count of items with confidence < 0.8 (default threshold)
    """
    source_rows = db.fetchall(
        """
        SELECT enrichment_source,
               COUNT(*) AS count,
               AVG(enrichment_confidence) AS avg_confidence
        FROM fact_items
        WHERE enrichment_source IS NOT NULL
        GROUP BY enrichment_source
        ORDER BY count DESC
        """,
        (),
    )

    totals_row = db.fetchone(
        """
        SELECT COUNT(*) AS total_enriched,
               AVG(enrichment_confidence) AS avg_confidence
        FROM fact_items
        WHERE enrichment_source IS NOT NULL
        """,
        (),
    )

    pending_row = db.fetchone(
        """
        SELECT COUNT(*) AS pending_review
        FROM fact_items
        WHERE enrichment_source IS NOT NULL
          AND enrichment_confidence IS NOT NULL
          AND enrichment_confidence < 0.8
        """,
        (),
    )

    by_source = [
        {
            "enrichment_source": row["enrichment_source"],
            "count": row["count"],
            "avg_confidence": row["avg_confidence"],
        }
        for row in source_rows
    ]

    total_enriched: int = totals_row["total_enriched"] if totals_row else 0
    avg_confidence: float | None = totals_row["avg_confidence"] if totals_row else None
    pending_review: int = pending_row["pending_review"] if pending_row else 0

    return {
        "by_source": by_source,
        "total_enriched": total_enriched,
        "avg_confidence": avg_confidence,
        "pending_review": pending_review,
    }


def get_enrichment_trends(
    db: DatabaseManager,
    start_date: str | None = None,
    end_date: str | None = None,
    period: str = "month",
) -> dict[str, Any]:
    """Return enrichment activity over time grouped by source and period.

    Args:
        db: Database manager instance.
        start_date: Optional ISO date string (YYYY-MM-DD) for range start.
        end_date: Optional ISO date string (YYYY-MM-DD) for range end.
        period: Grouping granularity — 'day', 'week', or 'month'.

    Returns:
        Dict with key 'periods': list of {period, total, by_source} dicts.
    """
    fmt_map = {
        "day": "%Y-%m-%d",
        "week": "%Y-W%W",
        "month": "%Y-%m",
    }
    strftime_fmt = fmt_map.get(period, "%Y-%m")

    conditions = ["fi.enrichment_source IS NOT NULL"]
    params: list[object] = []

    if start_date:
        conditions.append("f.event_date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("f.event_date <= ?")
        params.append(end_date)

    where = " AND ".join(conditions)

    rows = db.fetchall(
        f"""
        SELECT strftime('{strftime_fmt}', f.event_date) AS period,
               fi.enrichment_source,
               COUNT(*) AS count
        FROM fact_items fi
        JOIN facts f ON fi.fact_id = f.id
        WHERE {where}
        GROUP BY period, fi.enrichment_source
        ORDER BY period ASC, fi.enrichment_source ASC
        """,
        tuple(params),
    )

    # Aggregate into period buckets
    period_map: dict[str, dict[str, Any]] = {}
    for row in rows:
        p = row["period"]
        src = row["enrichment_source"]
        cnt = row["count"]
        if p not in period_map:
            period_map[p] = {"period": p, "total": 0, "by_source": {}}
        period_map[p]["total"] += cnt
        period_map[p]["by_source"][src] = cnt

    return {"periods": list(period_map.values())}


def get_vendor_coverage(
    db: DatabaseManager,
    limit: int = 50,
) -> dict[str, Any]:
    """Return enrichment coverage percentage per vendor.

    Vendors are ordered by total item count descending so the busiest vendors
    appear first.  For each vendor the per-source breakdown is included.

    Args:
        db: Database manager instance.
        limit: Maximum number of vendors to return.

    Returns:
        Dict with key 'vendors': list of {vendor, vendor_key, total_items,
        enriched_items, coverage_pct, sources} dicts.
    """
    # Main coverage query: totals per vendor
    coverage_rows = db.fetchall(
        """
        SELECT f.vendor,
               f.vendor_key,
               COUNT(fi.id) AS total_items,
               SUM(CASE WHEN fi.enrichment_source IS NOT NULL THEN 1 ELSE 0 END)
                   AS enriched_items
        FROM facts f
        JOIN fact_items fi ON fi.fact_id = f.id
        GROUP BY f.vendor, f.vendor_key
        ORDER BY total_items DESC
        LIMIT ?
        """,
        (limit,),
    )

    if not coverage_rows:
        return {"vendors": []}

    # Collect vendor keys for per-source breakdown (use vendor+vendor_key pairs)
    vendor_keys = list({(row["vendor"], row["vendor_key"]) for row in coverage_rows})

    # Per-source breakdown — fetch for all vendors at once
    source_rows = db.fetchall(
        """
        SELECT f.vendor,
               f.vendor_key,
               fi.enrichment_source,
               COUNT(*) AS count
        FROM fact_items fi
        JOIN facts f ON fi.fact_id = f.id
        WHERE fi.enrichment_source IS NOT NULL
        GROUP BY f.vendor, f.vendor_key, fi.enrichment_source
        """,
        (),
    )

    # Build source lookup keyed by (vendor, vendor_key)
    sources_map: dict[tuple[str, str | None], dict[str, int]] = {}
    for row in source_rows:
        key = (row["vendor"], row["vendor_key"])
        if key not in sources_map:
            sources_map[key] = {}
        sources_map[key][row["enrichment_source"]] = row["count"]

    vendors = []
    for row in coverage_rows:
        total: int = row["total_items"]
        enriched: int = row["enriched_items"] or 0
        coverage_pct = round(enriched / total * 100, 2) if total else 0.0
        key = (row["vendor"], row["vendor_key"])
        vendors.append(
            {
                "vendor": row["vendor"],
                "vendor_key": row["vendor_key"],
                "total_items": total,
                "enriched_items": enriched,
                "coverage_pct": coverage_pct,
                "sources": sources_map.get(key, {}),
            }
        )

    return {"vendors": vendors}


def find_product_matches(
    db: DatabaseManager,
    category: str | None = None,
    limit: int = 200,
    api_key: str | None = None,
) -> list[Any]:
    """Find cross-vendor product matches via Gemini.

    Args:
        db: DatabaseManager.
        category: Filter to category, or None for all.
        limit: Max products to analyze.
        api_key: Optional API key override.

    Returns:
        List of MatchedProductGroup.
    """
    from alibi.enrichment.product_matcher import find_cross_vendor_matches

    return find_cross_vendor_matches(
        db, category=category, limit=limit, api_key=api_key
    )
