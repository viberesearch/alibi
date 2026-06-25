"""Barcode service layer.

Wraps barcode detection, product lookup and barcode-driven enrichment so the
API (and through it the thin Telegram bot) never touches the extraction or
enrichment internals directly. Detection requires the optional ``pyzbar``
native dependency, which lives on the host alongside the pipeline -- the thin
bot forwards the photo bytes and lets the host do the work.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class ScannedBarcode:
    """A barcode detected in an image, with any cached product match."""

    data: str
    type: str
    valid_ean: bool
    product: dict[str, str] | None = None


def has_support() -> bool:
    """Return True if barcode detection is available on this host."""
    from alibi.extraction.barcode_detector import has_barcode_support

    return has_barcode_support()


def scan_image(db: DatabaseManager, image_data: bytes) -> list[ScannedBarcode]:
    """Detect barcodes in an image and attach any locally cached product info."""
    from alibi.extraction.barcode_detector import detect_barcodes

    results = detect_barcodes(image_data)
    scanned: list[ScannedBarcode] = []
    for r in results:
        product = lookup_cached_product(db, r.data) if r.valid_ean else None
        scanned.append(
            ScannedBarcode(
                data=r.data,
                type=r.type,
                valid_ean=r.valid_ean,
                product=product,
            )
        )
    return scanned


def lookup_cached_product(db: DatabaseManager, barcode: str) -> dict[str, str] | None:
    """Look up a barcode in the local ``product_cache`` table.

    Returns a small ``{product_name, brand, category}`` dict, or None when the
    barcode is absent or was cached as a known miss.
    """
    row = db.fetchone(
        "SELECT data FROM product_cache WHERE barcode = ?",
        (barcode,),
    )
    if not row:
        return None
    try:
        data: dict[str, object] = json.loads(row["data"])
    except (json.JSONDecodeError, TypeError):
        return None
    if data.get("_not_found"):
        return None
    tags = data.get("categories_tags")
    category = (
        ", ".join(str(t) for t in tags[:3])  # type: ignore[index]
        if isinstance(tags, list) and tags
        else ""
    )
    return {
        "product_name": str(data.get("product_name", "")),
        "brand": str(data.get("brands", "")),
        "category": category,
    }


def lookup_off_product(barcode: str) -> dict[str, Any] | None:
    """Look up a barcode in Open Food Facts (network call, no DB write)."""
    from alibi.enrichment.off_client import lookup_barcode

    return lookup_barcode(barcode)


def enrich_items_by_barcode(db: DatabaseManager, barcode: str) -> dict[str, int]:
    """Enrich all unenriched fact_items carrying ``barcode``.

    Returns ``{"matched": N, "enriched": M}`` where ``matched`` is the number
    of items with this barcode lacking a brand and ``enriched`` is how many were
    successfully enriched from a product lookup.
    """
    from alibi.enrichment.service import enrich_item

    rows = db.fetchall(
        "SELECT id FROM fact_items "
        "WHERE barcode = ? AND (brand IS NULL OR brand = '')",
        (barcode,),
    )
    enriched = 0
    for row in rows:
        result = enrich_item(db, row["id"], barcode)
        if result.success:
            enriched += 1
    return {"matched": len(rows), "enriched": enriched}
