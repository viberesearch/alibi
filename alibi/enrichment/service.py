"""Product enrichment service.

Orchestrates barcode lookup via Open Food Facts and fuzzy name matching
against known products to update fact_items with brand, category, and
other product data.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from alibi.db.connection import DatabaseManager
from alibi.enrichment.off_client import cached_lookup

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentResult:
    """Result of enriching a single fact_item."""

    item_id: str
    barcode: str
    success: bool
    brand: str | None = None
    category: str | None = None
    product_name: str | None = None
    source: str = "openfoodfacts"


def enrich_item(
    db: DatabaseManager,
    item_id: str,
    barcode: str,
) -> EnrichmentResult:
    """Enrich a single fact_item by barcode lookup.

    Looks up the barcode in OFF (with cache), then updates the
    fact_item's brand and category if found.

    Returns:
        EnrichmentResult with success status and extracted fields.
    """
    product = cached_lookup(db, barcode)
    if not product:
        return EnrichmentResult(
            item_id=item_id,
            barcode=barcode,
            success=False,
        )

    brand = product.get("brands")
    category = _extract_primary_category(product)
    product_name = product.get("product_name")

    # Build update fields — only set non-empty values
    fields: dict[str, object] = {}
    if brand:
        fields["brand"] = brand
    if category:
        fields["category"] = category

    if fields:
        fields["enrichment_source"] = "openfoodfacts"
        fields["enrichment_confidence"] = 0.95

        from alibi.services.correction import update_fact_item

        update_fact_item(db, item_id, fields)
        logger.info(
            "Enriched item %s (%s): brand=%s, category=%s",
            item_id[:8],
            barcode,
            brand,
            category,
        )

    return EnrichmentResult(
        item_id=item_id,
        barcode=barcode,
        success=True,
        brand=brand,
        category=category,
        product_name=product_name,
    )


def _extract_primary_category(product: dict[str, Any]) -> str | None:
    """Extract the most specific category from OFF data.

    OFF returns categories as a comma-separated list from general to
    specific (e.g., "Breakfasts, Spreads, Sweet spreads, Hazelnut spreads").
    We take the last (most specific) category.

    Also tries categories_tags for a cleaner tag form.
    """
    # Prefer categories_tags (structured)
    tags = product.get("categories_tags")
    if tags and isinstance(tags, list):
        # Tags look like "en:hazelnut-spreads" — take last, clean up
        last_tag = str(tags[-1])
        # Strip language prefix
        if ":" in last_tag:
            last_tag = last_tag.split(":", 1)[1]
        # Convert hyphens to spaces, title case
        return last_tag.replace("-", " ").title()

    # Fall back to comma-separated categories string
    cats = product.get("categories")
    if cats and isinstance(cats, str):
        parts = [c.strip() for c in cats.split(",")]
        return parts[-1] if parts else None

    return None


def _extract_product_variant(product: dict[str, Any]) -> str | None:
    """Extract primary product variant from OFF data.

    Returns the single most price-relevant distinguishing attribute:
    fat percentage for dairy, etc.  Secondary attributes (organic,
    free-range) are handled separately via extract_product_attributes().
    """
    # Fat content (dairy, meat)
    nutriments = product.get("nutriments", {})
    if isinstance(nutriments, dict):
        fat = nutriments.get("fat_100g")
        if fat is not None:
            try:
                fat_val = float(fat)
                cats = str(product.get("categories_tags", "")).lower()
                if any(
                    kw in cats
                    for kw in ("milk", "dairy", "yogurt", "cheese", "cream", "butter")
                ):
                    return f"{fat_val:g}%"
            except (ValueError, TypeError):
                pass

    return None


def extract_product_attributes(
    product: dict[str, Any] | None = None,
    item_name: str = "",
    category: str = "",
) -> list[str]:
    """Extract secondary product attributes from OFF data and/or item name.

    Returns a list of normalized attribute strings like "organic",
    "free-range", "lactose-free", "wholegrain".  These are stored as
    annotations (key=product_attribute) rather than in the single-value
    product_variant field.
    """
    attrs: list[str] = []

    # From OFF labels
    if product:
        labels = product.get("labels_tags", [])
        if isinstance(labels, list):
            for label in labels:
                tag = str(label).lower()
                if ("organic" in tag or "bio" in tag) and "organic" not in attrs:
                    attrs.append("organic")
                if (
                    "wholegrain" in tag or "whole-grain" in tag
                ) and "wholegrain" not in attrs:
                    attrs.append("wholegrain")
                if "free-range" in tag and "free-range" not in attrs:
                    attrs.append("free-range")
                if (
                    "gluten-free" in tag or "sans gluten" in tag
                ) and "gluten-free" not in attrs:
                    attrs.append("gluten-free")
                if (
                    "lactose-free" in tag or "sans lactose" in tag
                ) and "lactose-free" not in attrs:
                    attrs.append("lactose-free")

    # From item name (heuristic)
    name_upper = item_name.upper()
    if re.search(r"\bORGANIC\b", name_upper) and "organic" not in attrs:
        attrs.append("organic")
    if re.search(r"\bFREE[\s-]*RANGE\b", name_upper) and "free-range" not in attrs:
        attrs.append("free-range")
    if re.search(r"\bDELACT\b", name_upper) and "lactose-free" not in attrs:
        attrs.append("lactose-free")
    if re.search(r"\bWHOLEGRAIN\b", name_upper) and "wholegrain" not in attrs:
        attrs.append("wholegrain")
    if re.search(r"\bGLUTEN[\s-]*FREE\b", name_upper) and "gluten-free" not in attrs:
        attrs.append("gluten-free")
    if re.search(r"\bSUGAR[\s-]*FREE\b", name_upper) and "sugar-free" not in attrs:
        attrs.append("sugar-free")

    return attrs


# Heuristic patterns for extracting primary variant from item names.
# The primary variant is the one that most affects price comparison
# (fat %, egg size, etc.).
_NAME_VARIANT_PATTERNS: list[tuple[re.Pattern[str], str | None]] = [
    # Fat percentage (dairy/deli): "3%", "3.6%", "15%", "20%", "5%", "9%"
    (re.compile(r"(\d+(?:\.\d+)?)\s*%"), "dairy_pct"),
    # Egg size indicators
    (re.compile(r"\b(LARGE|MEDIUM|SMALL|XL|XXL)\b", re.I), "egg_size_word"),
    # Egg size from "L X12", "M X12" patterns
    (re.compile(r"\b([LMS])\s*X\d+\b", re.I), "egg_size_letter"),
    # Light/low-fat
    (re.compile(r"\bLIGHT\b", re.I), "light"),
    # Unsalted (butter)
    (re.compile(r"\bUNSALTED\b", re.I), "unsalted"),
    # Strained/straggato yoghurt
    (re.compile(r"\b(?:STRAINED|STRAGGATO|ST\.)\b", re.I), "strained"),
    # Dried fruit size: 20/30, 30/40
    (re.compile(r"\b(\d+/\d+)\b"), "size_fraction"),
]

# Categories where fat % is the primary variant
_FAT_PCT_CATEGORIES = frozenset({"Dairy", "Deli"})
# Categories where egg size is relevant
_EGG_CATEGORIES = frozenset({"Eggs"})
# Categories where light is meaningful
_LIGHT_CATEGORIES = frozenset({"Dairy", "Beverages", "Canned Food"})
# Categories where size fractions matter
_SIZE_FRACTION_CATEGORIES = frozenset({"Fruit", "Snacks"})

_EGG_SIZE_MAP = {"LARGE": "L", "MEDIUM": "M", "SMALL": "S", "XL": "XL", "XXL": "XXL"}


def extract_variant_from_name(name: str, category: str) -> str | None:
    """Extract primary product variant from item name using heuristics.

    Returns a single normalized value like "3%", "L", "light", "strained",
    or None if no variant pattern matches the category context.
    """
    for pattern, kind in _NAME_VARIANT_PATTERNS:
        m = pattern.search(name)
        if not m:
            continue

        if kind == "dairy_pct" and category in _FAT_PCT_CATEGORIES:
            return f"{m.group(1)}%"
        if kind == "egg_size_word" and category in _EGG_CATEGORIES:
            return _EGG_SIZE_MAP.get(m.group(1).upper(), m.group(1).upper())
        if kind == "egg_size_letter" and category in _EGG_CATEGORIES:
            return m.group(1).upper()
        if kind == "light" and category in _LIGHT_CATEGORIES:
            return "light"
        if kind == "unsalted" and category == "Dairy":
            return "unsalted"
        if kind == "strained" and category == "Dairy":
            return "strained"
        if kind == "size_fraction" and category in _SIZE_FRACTION_CATEGORIES:
            return m.group(1)

    return None


def _store_product_attributes(
    db: DatabaseManager,
    item_id: str,
    attributes: list[str],
) -> None:
    """Store secondary product attributes as annotations on a fact_item.

    Each attribute (e.g. "organic", "free-range") becomes an annotation
    with annotation_type="product_attribute", key=attribute, value="true".
    Skips attributes that already exist to avoid duplicates.
    """
    from alibi.annotations.store import add_annotation, get_annotations

    existing = get_annotations(db, target_type="fact_item", target_id=item_id)
    existing_keys = {
        a["key"] for a in existing if a.get("annotation_type") == "product_attribute"
    }

    for attr in attributes:
        if attr not in existing_keys:
            add_annotation(
                db,
                annotation_type="product_attribute",
                target_type="fact_item",
                target_id=item_id,
                key=attr,
                value="true",
                source="enrichment",
            )
            logger.debug("Added product attribute '%s' to item %s", attr, item_id[:8])


def enrich_item_cascade(
    db: DatabaseManager,
    item_id: str,
    barcode: str,
) -> EnrichmentResult:
    """Enrich a single fact_item using the full barcode cascade.

    Cascade order:
    1. Open Food Facts (cached_lookup)
    2. UPCitemdb (if OFF misses)
    3. GS1 prefix brand propagation (brand-only, if both miss)

    Returns EnrichmentResult from the first source that provides data.
    """
    # Stage 1: OFF
    from alibi.enrichment.off_client import cached_lookup as off_lookup

    product = off_lookup(db, barcode)
    if product and not product.get("_not_found"):
        return _apply_product_data(db, item_id, barcode, product, "openfoodfacts", 0.95)

    # Stage 2: UPCitemdb
    from alibi.enrichment.upcitemdb_client import cached_lookup as upc_lookup

    product = upc_lookup(db, barcode)
    if product and not product.get("_not_found"):
        return _apply_product_data(db, item_id, barcode, product, "upcitemdb", 0.90)

    # Stage 3: GS1 prefix brand propagation (brand-only)
    from alibi.enrichment.gs1_client import lookup_brand_by_prefix

    gs1_result = lookup_brand_by_prefix(db, barcode)
    if gs1_result and gs1_result.get("brands"):
        from alibi.services.correction import update_fact_item

        fields: dict[str, object] = {
            "brand": gs1_result["brands"],
            "enrichment_source": "gs1",
            "enrichment_confidence": 0.80,
        }
        update_fact_item(db, item_id, fields)
        logger.info(
            "GS1 enriched item %s (%s): brand=%s",
            item_id[:8],
            barcode,
            gs1_result["brands"],
        )
        return EnrichmentResult(
            item_id=item_id,
            barcode=barcode,
            success=True,
            brand=gs1_result["brands"],
            source="gs1",
        )

    return EnrichmentResult(item_id=item_id, barcode=barcode, success=False)


def _apply_product_data(
    db: DatabaseManager,
    item_id: str,
    barcode: str,
    product: dict[str, Any],
    source: str,
    confidence: float,
) -> EnrichmentResult:
    """Apply product data from a barcode lookup source to a fact_item."""
    brand = product.get("brands")
    category = (
        _extract_primary_category(product)
        if source == "openfoodfacts"
        else product.get("categories")
    )
    product_name = product.get("product_name")

    fields: dict[str, object] = {}
    if brand:
        fields["brand"] = brand
    if category:
        fields["category"] = category

    # Extract unit/unit_quantity from product data (backup for items where
    # the parser couldn't determine the unit from the receipt text)
    _apply_unit_from_product(db, item_id, product, fields)

    # Extract product variant (e.g., fat percentage for dairy)
    variant = _extract_product_variant(product)
    if variant:
        fields["product_variant"] = variant

    if fields:
        fields["enrichment_source"] = source
        fields["enrichment_confidence"] = confidence

        from alibi.services.correction import update_fact_item

        update_fact_item(db, item_id, fields)
        logger.info(
            "Enriched item %s (%s) via %s: brand=%s, category=%s",
            item_id[:8],
            barcode,
            source,
            brand,
            category,
        )

    # Store secondary product attributes as annotations
    item_name = product.get("product_name", "")
    attrs = extract_product_attributes(product, item_name, category or "")
    if attrs:
        _store_product_attributes(db, item_id, attrs)

    return EnrichmentResult(
        item_id=item_id,
        barcode=barcode,
        success=bool(fields),
        brand=brand,
        category=category,
        product_name=product_name,
        source=source,
    )


_UNIT_PATTERN: re.Pattern[str] | None = None


def _get_unit_pattern() -> re.Pattern[str]:
    """Lazily compile the unit extraction pattern."""
    global _UNIT_PATTERN
    if _UNIT_PATTERN is None:
        _UNIT_PATTERN = re.compile(
            r"(\d+(?:[.,]\d+)?)\s*(kg|g|ml|l|cl|oz|lb)\b", re.IGNORECASE
        )
    return _UNIT_PATTERN


_UNIT_NORMALIZATION = {
    "g": ("g", 1.0),
    "kg": ("kg", 1.0),
    "ml": ("ml", 1.0),
    "cl": ("ml", 10.0),
    "l": ("l", 1.0),
    "oz": ("oz", 1.0),
    "lb": ("lb", 1.0),
}


def _apply_unit_from_product(
    db: DatabaseManager,
    item_id: str,
    product: dict[str, Any],
    fields: dict[str, object],
) -> None:
    """Extract unit/unit_quantity from product data and apply if item lacks them.

    This is the backup plan when the document doesn't specify the unit.
    Sources checked: OFF quantity field, historical data.
    Only updates items that currently have unit='pcs' and no unit_quantity.
    """
    # Check if item already has a proper unit (not pcs) or unit_quantity
    row = db.fetchone(
        "SELECT unit, unit_quantity FROM fact_items WHERE id = ?",
        (item_id,),
    )
    if not row:
        return
    if row["unit_quantity"] is not None:
        return  # Already has unit_quantity from extraction
    if row["unit"] and row["unit"] not in ("pcs", "other"):
        return  # Already has a measurement unit

    # Try to extract from OFF "quantity" field (e.g., "1l", "500ml", "250g")
    quantity_str = product.get("quantity", "")
    if quantity_str:
        match = _get_unit_pattern().search(str(quantity_str))
        if match:
            qty = float(match.group(1).replace(",", "."))
            raw_unit = match.group(2).lower()
            norm = _UNIT_NORMALIZATION.get(raw_unit)
            if norm:
                base_unit, multiplier = norm
                # Convert to base unit (e.g., 500ml -> 500ml, 1.5l -> 1.5l)
                # Map ml->l and g->kg for comparable_unit_price
                if base_unit == "ml":
                    fields["unit"] = "l"
                    fields["unit_quantity"] = qty * multiplier / 1000.0
                elif base_unit == "g":
                    fields["unit"] = "kg"
                    fields["unit_quantity"] = qty / 1000.0
                else:
                    fields["unit"] = base_unit
                    fields["unit_quantity"] = qty * multiplier
                return

    # Fallback: historical unit_quantity lookup
    from alibi.db.v2_store import get_canonical_unit_quantity

    item_row = db.fetchone(
        "SELECT name, barcode, brand, f.vendor_key "
        "FROM fact_items fi JOIN facts f ON fi.fact_id = f.id "
        "WHERE fi.id = ?",
        (item_id,),
    )
    if item_row:
        historical = get_canonical_unit_quantity(
            db,
            item_name=item_row["name"],
            barcode=item_row["barcode"],
            vendor_key=item_row["vendor_key"],
            brand=item_row["brand"],
        )
        if historical:
            fields["unit"] = historical["unit"]
            fields["unit_quantity"] = historical["unit_quantity"]


def trigger_off_contribution(
    db: DatabaseManager,
    item_id: str,
) -> bool:
    """After Gemini enriches an item with barcode, contribute to OFF."""
    row = db.fetchone(
        "SELECT barcode, brand, category, name FROM fact_items WHERE id = ?",
        (item_id,),
    )
    if not row or not row["barcode"] or not row["brand"]:
        return False

    from alibi.enrichment.off_client import contribute_if_enabled

    return contribute_if_enabled(
        db,
        row["barcode"],
        row["brand"],
        row["category"] or "",
        row["name"],
    )


def enrich_pending_items(
    db: DatabaseManager,
    limit: int = 100,
) -> list[EnrichmentResult]:
    """Find and enrich fact_items that have a barcode but no brand/category.

    Returns:
        List of EnrichmentResult for each processed item.
    """
    rows = db.fetchall(
        "SELECT id, barcode FROM fact_items "
        "WHERE barcode IS NOT NULL AND barcode != '' "
        "AND (brand IS NULL OR brand = '' "
        "     OR category IS NULL OR category = '') "
        "LIMIT ?",
        (limit,),
    )

    results = []
    for row in rows:
        result = enrich_item(db, row["id"], row["barcode"])
        results.append(result)

    return results


def enrich_by_barcode(
    db: DatabaseManager,
    barcode: str,
) -> list[EnrichmentResult]:
    """Enrich all fact_items matching a specific barcode.

    Returns:
        List of EnrichmentResult for each matching item.
    """
    rows = db.fetchall(
        "SELECT id, barcode FROM fact_items WHERE barcode = ?",
        (barcode,),
    )

    results = []
    for row in rows:
        result = enrich_item(db, row["id"], row["barcode"])
        results.append(result)

    return results


def enrich_item_by_name(
    db: DatabaseManager,
    item_id: str,
    item_name: str,
    vendor_key: str | None = None,
) -> EnrichmentResult:
    """Enrich a single fact_item by fuzzy name matching.

    Uses the product resolver to find previously enriched items with
    similar names and propagate their brand/category.

    Returns:
        EnrichmentResult with success status and matched fields.
    """
    from alibi.enrichment.product_resolver import resolve_product

    match = resolve_product(db, item_name, vendor_key=vendor_key)
    if not match:
        return EnrichmentResult(
            item_id=item_id,
            barcode="",
            success=False,
            source="product_resolver",
        )

    fields: dict[str, object] = {}
    if match.brand:
        fields["brand"] = match.brand
    if match.category:
        fields["category"] = match.category

    # Also try to fill unit/unit_quantity from the matched item's data
    if match.unit_quantity and match.unit:
        row = db.fetchone(
            "SELECT unit, unit_quantity FROM fact_items WHERE id = ?",
            (item_id,),
        )
        if (
            row
            and row["unit_quantity"] is None
            and row["unit"] in ("pcs", "other", None)
        ):
            fields["unit"] = match.unit
            fields["unit_quantity"] = match.unit_quantity

    if fields:
        fields["enrichment_source"] = "product_resolver"
        fields["enrichment_confidence"] = match.similarity

        from alibi.services.correction import update_fact_item

        update_fact_item(db, item_id, fields)
        logger.info(
            "Resolved item %s via %s (%.0f%%): brand=%s, category=%s, " "matched=%r",
            item_id[:8],
            match.source,
            match.similarity * 100,
            match.brand,
            match.category,
            match.matched_name,
        )

    return EnrichmentResult(
        item_id=item_id,
        barcode="",
        success=True,
        brand=match.brand,
        category=match.category,
        product_name=match.matched_name,
        source="product_resolver",
    )


def enrich_pending_by_name(
    db: DatabaseManager,
    limit: int = 100,
) -> list[EnrichmentResult]:
    """Find and enrich fact_items without barcode or brand/category.

    Uses fuzzy name matching against previously enriched products.

    Returns:
        List of EnrichmentResult for each processed item.
    """
    rows = db.fetchall(
        "SELECT fi.id, fi.name, f.vendor_key "
        "FROM fact_items fi "
        "JOIN facts f ON fi.fact_id = f.id "
        "WHERE (fi.barcode IS NULL OR fi.barcode = '') "
        "AND (fi.brand IS NULL OR fi.brand = '' "
        "     OR fi.category IS NULL OR fi.category = '') "
        "LIMIT ?",
        (limit,),
    )

    results = []
    for row in rows:
        result = enrich_item_by_name(
            db, row["id"], row["name"], vendor_key=row["vendor_key"]
        )
        results.append(result)

    return results
