"""Barcode-based cross-vendor product matching.

When items at different vendors share the same EAN/UPC barcode, they are
the same product. This module propagates brand, category, and comparable_name
from enriched items to unenriched ones sharing the same barcode.

Higher confidence than fuzzy name matching (0.95) because barcode identity
is deterministic — no fuzzy logic involved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)

_BARCODE_MATCH_CONFIDENCE = 0.95
_ENRICHMENT_SOURCE = "barcode_cross_vendor"


@dataclass
class BarcodeMatchResult:
    """Result of cross-vendor barcode matching for a single item."""

    item_id: str
    barcode: str
    source_item_id: str
    source_vendor_key: str | None
    brand: str | None = None
    category: str | None = None
    comparable_name: str | None = None


def match_by_barcode(
    db: DatabaseManager,
    barcode: str | None,
) -> list[BarcodeMatchResult]:
    """Propagate enrichment across vendors for items sharing a barcode.

    Finds the best-enriched item with this barcode (most fields populated)
    and propagates its brand/category/comparable_name to all unenriched
    items with the same barcode.

    Args:
        db: Database connection.
        barcode: EAN/UPC barcode to match.

    Returns:
        List of BarcodeMatchResult for each item that was enriched.
    """
    if not barcode or not barcode.strip():
        return []

    conn = db.get_connection()

    # Find all items with this barcode, grouped by enrichment status
    rows = conn.execute(
        "SELECT fi.id, fi.name, fi.brand, fi.category, fi.comparable_name, "
        "fi.enrichment_source, fi.enrichment_confidence, f.vendor_key "
        "FROM fact_items fi "
        "JOIN facts f ON fi.fact_id = f.id "
        "WHERE fi.barcode = ? "
        "ORDER BY fi.enrichment_confidence DESC NULLS LAST",
        (barcode,),
    ).fetchall()

    if len(rows) < 2:
        return []

    # Find the best source: item with most enrichment data
    source = _pick_best_source(rows)
    if not source:
        return []

    source_brand = source["brand"]
    source_category = source["category"]
    source_comparable = source["comparable_name"]

    if not source_brand and not source_category:
        return []

    # Propagate to unenriched items
    results: list[BarcodeMatchResult] = []
    for row in rows:
        if row["id"] == source["id"]:
            continue

        # Skip items that are already well-enriched
        if _is_already_enriched(row, source_brand, source_category):
            continue

        fields: dict[str, Any] = {}
        propagated_brand: str | None = None
        propagated_category: str | None = None
        propagated_comparable: str | None = None

        if source_brand and not row["brand"]:
            fields["brand"] = source_brand
            propagated_brand = source_brand
        if source_category and not row["category"]:
            fields["category"] = source_category
            propagated_category = source_category
        if source_comparable and not row["comparable_name"]:
            fields["comparable_name"] = source_comparable
            propagated_comparable = source_comparable

        if not fields:
            continue

        fields["enrichment_source"] = _ENRICHMENT_SOURCE
        fields["enrichment_confidence"] = _BARCODE_MATCH_CONFIDENCE

        from alibi.services.correction import update_fact_item

        update_fact_item(db, row["id"], fields)

        results.append(
            BarcodeMatchResult(
                item_id=row["id"],
                barcode=barcode,
                source_item_id=source["id"],
                source_vendor_key=source["vendor_key"],
                brand=propagated_brand,
                category=propagated_category,
                comparable_name=propagated_comparable,
            )
        )

    if results:
        logger.info(
            "Barcode %s: propagated from %s to %d items across vendors",
            barcode,
            source["id"][:8],
            len(results),
        )

    return results


def match_all_barcodes(
    db: DatabaseManager,
    limit: int = 500,
) -> list[BarcodeMatchResult]:
    """Find all cross-vendor barcode matches and propagate enrichment.

    Identifies barcodes that appear at multiple vendors where at least one
    item is enriched and at least one is not, then propagates.

    Args:
        db: Database connection.
        limit: Max barcodes to process.

    Returns:
        List of all BarcodeMatchResult across all barcodes.
    """
    conn = db.get_connection()

    # Find barcodes with unenriched items that also have enriched siblings
    rows = conn.execute(
        "SELECT DISTINCT fi.barcode "
        "FROM fact_items fi "
        "WHERE fi.barcode IS NOT NULL AND fi.barcode != '' "
        "AND (fi.brand IS NULL OR fi.brand = '' "
        "     OR fi.category IS NULL OR fi.category = '') "
        "AND fi.barcode IN ("
        "  SELECT fi2.barcode FROM fact_items fi2 "
        "  WHERE fi2.barcode = fi.barcode "
        "  AND (fi2.brand IS NOT NULL AND fi2.brand != '')"
        ") "
        "LIMIT ?",
        (limit,),
    ).fetchall()

    all_results: list[BarcodeMatchResult] = []
    for row in rows:
        results = match_by_barcode(db, row["barcode"])
        all_results.extend(results)

    if all_results:
        logger.info(
            "Cross-vendor barcode matching: %d items enriched across %d barcodes",
            len(all_results),
            len(rows),
        )

    return all_results


def get_barcode_coverage(db: DatabaseManager) -> dict[str, int]:
    """Get statistics on barcode-based enrichment potential.

    Returns:
        Dict with counts: total_with_barcode, enriched, unenriched,
        cross_vendor_barcodes, matchable.
    """
    conn = db.get_connection()

    total = conn.execute(
        "SELECT COUNT(*) as cnt FROM fact_items "
        "WHERE barcode IS NOT NULL AND barcode != ''"
    ).fetchone()["cnt"]

    enriched = conn.execute(
        "SELECT COUNT(*) as cnt FROM fact_items "
        "WHERE barcode IS NOT NULL AND barcode != '' "
        "AND brand IS NOT NULL AND brand != ''"
    ).fetchone()["cnt"]

    # Barcodes appearing at 2+ different vendors
    cross_vendor = conn.execute(
        "SELECT COUNT(*) as cnt FROM ("
        "  SELECT fi.barcode FROM fact_items fi "
        "  JOIN facts f ON fi.fact_id = f.id "
        "  WHERE fi.barcode IS NOT NULL AND fi.barcode != '' "
        "  GROUP BY fi.barcode "
        "  HAVING COUNT(DISTINCT f.vendor_key) >= 2"
        ")"
    ).fetchone()["cnt"]

    # Items that could be enriched via cross-vendor matching
    matchable = conn.execute(
        "SELECT COUNT(*) as cnt FROM fact_items fi "
        "WHERE fi.barcode IS NOT NULL AND fi.barcode != '' "
        "AND (fi.brand IS NULL OR fi.brand = '') "
        "AND fi.barcode IN ("
        "  SELECT fi2.barcode FROM fact_items fi2 "
        "  WHERE fi2.barcode = fi.barcode "
        "  AND fi2.brand IS NOT NULL AND fi2.brand != ''"
        ")"
    ).fetchone()["cnt"]

    return {
        "total_with_barcode": total,
        "enriched": enriched,
        "unenriched": total - enriched,
        "cross_vendor_barcodes": cross_vendor,
        "matchable": matchable,
    }


def _pick_best_source(rows: list[Any]) -> Any:
    """Pick the best-enriched item from a set of barcode matches.

    Prefers items with both brand and category, then brand-only,
    then highest confidence.
    """
    best = None
    best_score = -1

    for row in rows:
        score = 0
        if row["brand"]:
            score += 2
        if row["category"]:
            score += 2
        if row["comparable_name"]:
            score += 1
        conf = row["enrichment_confidence"] or 0
        score += conf

        if score > best_score:
            best_score = score
            best = row

    if best and (best["brand"] or best["category"]):
        return best
    return None


def _is_already_enriched(
    row: Any, source_brand: str | None, source_category: str | None
) -> bool:
    """Check if an item already has the same enrichment as the source."""
    has_brand = bool(row["brand"])
    has_category = bool(row["category"])

    # If source has brand and target has it, and source has category and target has it
    if source_brand and has_brand and source_category and has_category:
        return True

    # If source only has brand, and target has brand
    if source_brand and not source_category and has_brand:
        return True

    # If source only has category, and target has category
    if not source_brand and source_category and has_category:
        return True

    return False
