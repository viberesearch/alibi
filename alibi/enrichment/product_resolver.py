"""Product resolution service — fuzzy match item names against known products.

Matches extracted item names from receipts against previously enriched
products (those with brand/category from OFF or user corrections) to
propagate brand and category to new items without barcodes.

Sources (checked in order):
1. FTS5 prefix search for candidate retrieval (sub-linear)
2. Exact normalized name match against candidates
3. Fuzzy match via SequenceMatcher against candidates (>= threshold)

Falls back to full-scan when FTS5 table is not available (pre-migration 033).

Vendor scoping: when vendor_key is provided, same-vendor matches get a
bonus, making them preferred over cross-vendor matches at similar scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)

# Minimum similarity for fuzzy matching (0.0 - 1.0)
_FUZZY_THRESHOLD = 0.80

# Bonus added to same-vendor matches (shifts them above cross-vendor ties)
_VENDOR_BONUS = 0.05

# Batch size threshold: below this, use per-item FTS5; above, use full scan
_FTS5_BATCH_THRESHOLD = 50


@dataclass
class ProductMatch:
    """Result of resolving a product name against the catalog."""

    brand: str | None
    category: str | None
    matched_name: str
    similarity: float
    source: str  # "exact_match" or "fuzzy_match"
    same_vendor: bool = False
    unit_quantity: float | None = None
    unit: str | None = None


def resolve_product(
    db: DatabaseManager,
    item_name: str,
    vendor_key: str | None = None,
    threshold: float = _FUZZY_THRESHOLD,
) -> ProductMatch | None:
    """Resolve an item name against known enriched products.

    Uses FTS5 for fast candidate retrieval when available, falling back
    to full table scan for older databases without migration 033.

    Args:
        db: Database connection.
        item_name: Raw item name from extraction.
        vendor_key: Optional vendor_key for same-vendor preference.
        threshold: Minimum similarity score (default 0.80).

    Returns:
        ProductMatch if a match is found above threshold, else None.
    """
    if not item_name or not item_name.strip():
        return None

    normalized = _normalize(item_name)
    if not normalized:
        return None

    # Try FTS5-accelerated path first
    if _fts5_available(db):
        return _resolve_via_fts5(db, normalized, vendor_key, threshold)

    # Fallback: full scan (for DBs without FTS5 migration)
    candidates = _get_enriched_candidates(db)
    if not candidates:
        return None

    exact = _find_exact_match(normalized, candidates, vendor_key)
    if exact:
        return exact
    return _find_fuzzy_match(normalized, candidates, vendor_key, threshold)


def resolve_products_batch(
    db: DatabaseManager,
    items: list[dict[str, str]],
    vendor_key: str | None = None,
    threshold: float = _FUZZY_THRESHOLD,
) -> dict[str, ProductMatch]:
    """Resolve multiple items in one batch (shares candidate loading).

    For small batches with FTS5 available, uses per-item FTS5 lookups.
    For large batches or without FTS5, loads all candidates once.

    Args:
        db: Database connection.
        items: List of dicts with at least 'id' and 'name' keys.
        vendor_key: Optional vendor_key for same-vendor preference.
        threshold: Minimum similarity score.

    Returns:
        Dict mapping item_id -> ProductMatch for resolved items.
    """
    use_fts5 = _fts5_available(db) and len(items) <= _FTS5_BATCH_THRESHOLD

    if use_fts5:
        return _batch_via_fts5(db, items, vendor_key, threshold)

    # Full-scan path: load all candidates once
    candidates = _get_enriched_candidates(db)
    if not candidates:
        return {}

    results: dict[str, ProductMatch] = {}
    for item in items:
        item_name = item.get("name", "")
        item_id = item.get("id", "")
        if not item_name or not item_name.strip():
            continue

        normalized = _normalize(item_name)
        if not normalized:
            continue

        match = _find_exact_match(normalized, candidates, vendor_key)
        if not match:
            match = _find_fuzzy_match(normalized, candidates, vendor_key, threshold)
        if match:
            results[item_id] = match

    return results


def rebuild_product_fts(db: DatabaseManager) -> int:
    """Rebuild FTS5 index from fact_items.

    Clears and repopulates the product_name_fts table.
    Useful after bulk operations or data cleanup.

    Returns:
        Count of items indexed.
    """
    conn = db.get_connection()
    conn.execute("DELETE FROM product_name_fts")
    cursor = conn.execute(
        "INSERT INTO product_name_fts "
        "(item_id, name, name_normalized, brand, category, "
        "vendor_key, unit_quantity, unit) "
        "SELECT fi.id, fi.name, fi.name_normalized, fi.brand, fi.category, "
        "f.vendor_key, fi.unit_quantity, fi.unit "
        "FROM fact_items fi "
        "JOIN facts f ON fi.fact_id = f.id "
        "WHERE (fi.brand IS NOT NULL AND fi.brand != '') "
        "   OR (fi.category IS NOT NULL AND fi.category != '')"
    )
    conn.commit()
    count = cursor.rowcount
    logger.info("Rebuilt FTS5 product index: %d items", count)
    return count


# ---------------------------------------------------------------------------
# FTS5-accelerated resolution
# ---------------------------------------------------------------------------


def _fts5_available(db: DatabaseManager) -> bool:
    """Check if FTS5 virtual table exists."""
    try:
        row = db.fetchone(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='product_name_fts'"
        )
        return row is not None
    except Exception:
        return False


def _resolve_via_fts5(
    db: DatabaseManager,
    normalized: str,
    vendor_key: str | None,
    threshold: float,
) -> ProductMatch | None:
    """Resolve using FTS5 for candidate retrieval."""
    # Phase 1: AND prefix query (narrow)
    candidates = _get_fts_candidates(db, normalized)
    if candidates:
        exact = _find_exact_match(normalized, candidates, vendor_key)
        if exact:
            return exact
        match = _find_fuzzy_match(normalized, candidates, vendor_key, threshold)
        if match:
            return match

    # Phase 2: OR prefix query (broad fallback)
    broad = _get_fts_candidates_broad(db, normalized)
    if broad:
        exact = _find_exact_match(normalized, broad, vendor_key)
        if exact:
            return exact
        return _find_fuzzy_match(normalized, broad, vendor_key, threshold)

    return None


def _get_fts_candidates(db: DatabaseManager, normalized: str) -> list[_Candidate]:
    """Get candidates via FTS5 AND prefix search.

    Tokenizes the query, adds * suffix to each token for prefix matching.
    Returns a focused candidate set much smaller than full scan.
    """
    tokens = normalized.split()
    if not tokens:
        return []

    # Each word as prefix query joined with AND
    fts_query = " ".join(f'"{t}"*' for t in tokens[:5])

    try:
        rows = db.fetchall(
            "SELECT item_id, name, name_normalized, brand, category, "
            "vendor_key, unit_quantity, unit "
            "FROM product_name_fts WHERE product_name_fts MATCH ? "
            "ORDER BY rank "
            "LIMIT 100",
            (fts_query,),
        )
    except Exception as e:
        logger.debug("FTS5 AND query failed: %s", e)
        return []

    return _rows_to_candidates(rows)


def _get_fts_candidates_broad(db: DatabaseManager, normalized: str) -> list[_Candidate]:
    """Broader FTS5 search using individual tokens with OR.

    Used when AND prefix query returns too few results.
    """
    tokens = normalized.split()
    if not tokens:
        return []

    fts_query = " OR ".join(f'"{t}"*' for t in tokens[:5])

    try:
        rows = db.fetchall(
            "SELECT item_id, name, name_normalized, brand, category, "
            "vendor_key, unit_quantity, unit "
            "FROM product_name_fts WHERE product_name_fts MATCH ? "
            "ORDER BY rank "
            "LIMIT 200",
            (fts_query,),
        )
    except Exception as e:
        logger.debug("FTS5 OR query failed: %s", e)
        return []

    return _rows_to_candidates(rows)


def _rows_to_candidates(rows: list[Any]) -> list[_Candidate]:
    """Convert FTS5 result rows to deduplicated _Candidate list."""
    seen: set[tuple[str, str | None]] = set()
    candidates: list[_Candidate] = []
    for row in rows:
        raw_name = row["name"] or ""
        norm = _normalize(raw_name)
        key = (norm, row["vendor_key"])
        if not norm or key in seen:
            continue
        seen.add(key)
        uq = row["unit_quantity"]
        candidates.append(
            _Candidate(
                name=raw_name,
                normalized=norm,
                brand=row["brand"],
                category=row["category"],
                vendor_key=row["vendor_key"],
                unit_quantity=float(uq) if uq is not None else None,
                unit=row["unit"],
            )
        )
    return candidates


def _batch_via_fts5(
    db: DatabaseManager,
    items: list[dict[str, str]],
    vendor_key: str | None,
    threshold: float,
) -> dict[str, ProductMatch]:
    """Batch resolution using per-item FTS5 lookups."""
    results: dict[str, ProductMatch] = {}
    for item in items:
        item_name = item.get("name", "")
        item_id = item.get("id", "")
        if not item_name or not item_name.strip():
            continue

        normalized = _normalize(item_name)
        if not normalized:
            continue

        match = _resolve_via_fts5(db, normalized, vendor_key, threshold)
        if match:
            results[item_id] = match

    return results


# ---------------------------------------------------------------------------
# Internal helpers (full-scan fallback)
# ---------------------------------------------------------------------------


@dataclass
class _Candidate:
    """An enriched fact_item used as a matching target."""

    name: str
    normalized: str
    brand: str | None
    category: str | None
    vendor_key: str | None
    unit_quantity: float | None = None
    unit: str | None = None


def _normalize(name: str) -> str:
    """Normalize an item name for matching.

    Lowercases, collapses whitespace, strips trailing unit quantities
    like '500g', '1l', '0.5kg'.
    """
    import re

    s = name.lower().strip()
    # Strip trailing unit quantities (e.g. "milk 1l", "butter 500g")
    s = re.sub(r"\s+\d+(\.\d+)?\s*(g|kg|ml|l|cl|oz|lb)\s*$", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _get_enriched_candidates(db: DatabaseManager) -> list[_Candidate]:
    """Load fact_items that have brand or category populated.

    Groups by normalized name to deduplicate (same product from
    multiple receipts). Takes the most recent entry per name.

    Used as fallback when FTS5 is not available.
    """
    rows = db.fetchall(
        "SELECT fi.name, fi.brand, fi.category, f.vendor_key, fi.unit_quantity, fi.unit "
        "FROM fact_items fi "
        "JOIN facts f ON fi.fact_id = f.id "
        "WHERE (fi.brand IS NOT NULL AND fi.brand != '') "
        "   OR (fi.category IS NOT NULL AND fi.category != '') "
        "ORDER BY fi.rowid DESC",
    )

    # Deduplicate by (normalized name, vendor_key) — keep first (most recent)
    # Separate vendor_key allows same product from different vendors
    seen: set[tuple[str, str | None]] = set()
    candidates: list[_Candidate] = []
    for row in rows:
        raw_name = row["name"] or ""
        norm = _normalize(raw_name)
        key = (norm, row["vendor_key"])
        if not norm or key in seen:
            continue
        seen.add(key)
        uq = row["unit_quantity"]
        candidates.append(
            _Candidate(
                name=raw_name,
                normalized=norm,
                brand=row["brand"],
                category=row["category"],
                vendor_key=row["vendor_key"],
                unit_quantity=float(uq) if uq is not None else None,
                unit=row["unit"],
            )
        )

    return candidates


def _find_exact_match(
    normalized: str,
    candidates: list[_Candidate],
    vendor_key: str | None,
) -> ProductMatch | None:
    """Find an exact normalized name match."""
    # Prefer same-vendor exact match
    if vendor_key:
        for c in candidates:
            if c.normalized == normalized and c.vendor_key == vendor_key:
                return ProductMatch(
                    brand=c.brand,
                    category=c.category,
                    matched_name=c.name,
                    similarity=1.0,
                    source="exact_match",
                    same_vendor=True,
                    unit_quantity=c.unit_quantity,
                    unit=c.unit,
                )

    # Any-vendor exact match
    for c in candidates:
        if c.normalized == normalized:
            return ProductMatch(
                brand=c.brand,
                category=c.category,
                matched_name=c.name,
                similarity=1.0,
                source="exact_match",
                same_vendor=(vendor_key is not None and c.vendor_key == vendor_key),
                unit_quantity=c.unit_quantity,
                unit=c.unit,
            )

    return None


def _find_fuzzy_match(
    normalized: str,
    candidates: list[_Candidate],
    vendor_key: str | None,
    threshold: float,
) -> ProductMatch | None:
    """Find the best fuzzy match above threshold."""
    best_score = 0.0
    best_candidate: _Candidate | None = None

    for c in candidates:
        raw_score = SequenceMatcher(None, normalized, c.normalized).ratio()

        # Apply vendor bonus for same-vendor matches
        effective_score = raw_score
        if vendor_key and c.vendor_key == vendor_key:
            effective_score = min(raw_score + _VENDOR_BONUS, 1.0)

        if effective_score > best_score:
            best_score = effective_score
            best_candidate = c

    if best_candidate is None or best_score < threshold:
        return None

    same_vendor = vendor_key is not None and best_candidate.vendor_key == vendor_key

    return ProductMatch(
        brand=best_candidate.brand,
        category=best_candidate.category,
        matched_name=best_candidate.name,
        similarity=best_score,
        source="fuzzy_match",
        same_vendor=same_vendor,
        unit_quantity=best_candidate.unit_quantity,
        unit=best_candidate.unit,
    )
