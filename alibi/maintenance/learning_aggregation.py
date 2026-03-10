"""Cross-session learning aggregation and maintenance.

Provides periodic batch operations to:
1. Recalculate template reliability scores from correction events
2. Mark stale templates for re-learning
3. Deduplicate identity members
4. Prune old confidence history entries
5. Remove orphaned identity members

All functions are fail-safe: exceptions per-identity are caught and logged.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any

from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Summary of data quality fixes applied."""

    units_fixed: int = 0
    unit_quantities_backfilled: int = 0
    details: list[str] | None = None

    def __post_init__(self) -> None:
        if self.details is None:
            self.details = []

    @property
    def total(self) -> int:
        return self.units_fixed + self.unit_quantities_backfilled


@dataclass
class MaintenanceReport:
    """Summary of maintenance operations performed."""

    templates_recalculated: int = 0
    templates_marked_stale: int = 0
    templates_pruned: int = 0
    members_deduplicated: int = 0
    orphaned_members_removed: int = 0


def recalculate_template_reliability(db: DatabaseManager) -> int:
    """Recalculate template common_fixes from correction_events.

    For each vendor identity with a template, queries correction_events
    for that vendor's facts and updates common_fixes accordingly.
    Templates with high correction rates (>30%) are marked stale.

    Returns:
        Number of templates recalculated.
    """
    conn = db.get_connection()
    rows = conn.execute(
        "SELECT id, canonical_name, metadata FROM identities "
        "WHERE entity_type = 'vendor' AND active = 1 "
        "AND metadata IS NOT NULL AND metadata != '{}'"
    ).fetchall()

    count = 0
    for row in rows:
        try:
            identity_id = row["id"]
            metadata = (
                json.loads(row["metadata"])
                if isinstance(row["metadata"], str)
                else row["metadata"]
            )
            template_data = metadata.get("template")
            if not template_data:
                continue

            # Get vendor_key from identity members
            vk_row = conn.execute(
                "SELECT value FROM identity_members "
                "WHERE identity_id = ? AND member_type = 'vendor_key' "
                "LIMIT 1",
                (identity_id,),
            ).fetchone()
            if not vk_row:
                continue
            vendor_key = vk_row["value"]

            # Query correction events for this vendor's facts
            fix_rows = conn.execute(
                "SELECT field, COUNT(*) as cnt "
                "FROM correction_events "
                "WHERE entity_type = 'fact' "
                "AND entity_id IN ("
                "  SELECT id FROM facts WHERE vendor_key = ?"
                ") "
                "GROUP BY field",
                (vendor_key,),
            ).fetchall()

            if not fix_rows:
                continue

            # Build common_fixes from correction data
            common_fixes: dict[str, int] = {}
            total_corrections = 0
            for fr in fix_rows:
                common_fixes[fr["field"]] = fr["cnt"]
                total_corrections += fr["cnt"]

            # Update template
            from alibi.extraction.templates import VendorTemplate

            template = VendorTemplate.from_dict(template_data)

            # Merge: existing counters updated, new ones added
            merged_fixes = dict(template.common_fixes)
            for field, cnt in common_fixes.items():
                merged_fixes[field] = cnt  # replace with actual count

            template.common_fixes = merged_fixes

            # High correction rate -> mark stale
            fact_count_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM facts WHERE vendor_key = ?",
                (vendor_key,),
            ).fetchone()
            fact_count = fact_count_row["cnt"] if fact_count_row else 0

            if fact_count > 0 and total_corrections / fact_count > 0.3:
                template.stale = True

            # Save back
            metadata["template"] = template.to_dict()
            conn.execute(
                "UPDATE identities SET metadata = ? WHERE id = ?",
                (json.dumps(metadata), identity_id),
            )
            count += 1

        except Exception as e:
            logger.warning(
                "Failed to recalculate template for %s: %s",
                row["canonical_name"],
                e,
            )
            continue

    if count:
        conn.commit()
        logger.info("Recalculated %d vendor templates", count)
    return count


def mark_stale_templates(db: DatabaseManager, max_age_days: int = 90) -> int:
    """Mark templates as stale if not updated recently.

    Templates with last_updated older than max_age_days ago
    (or with no last_updated) and success_count > 0 are marked stale.

    Returns:
        Number of templates marked stale.
    """
    conn = db.get_connection()
    rows = conn.execute(
        "SELECT id, canonical_name, metadata FROM identities "
        "WHERE entity_type = 'vendor' AND active = 1 "
        "AND metadata IS NOT NULL AND metadata != '{}'"
    ).fetchall()

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    count = 0

    for row in rows:
        try:
            metadata = (
                json.loads(row["metadata"])
                if isinstance(row["metadata"], str)
                else row["metadata"]
            )
            template_data = metadata.get("template")
            if not template_data:
                continue

            from alibi.extraction.templates import VendorTemplate

            template = VendorTemplate.from_dict(template_data)

            # Skip already-stale or zero-count templates
            if template.stale or template.success_count == 0:
                continue

            # Check last_updated
            is_old = False
            if not template.last_updated:
                is_old = True
            else:
                try:
                    last = datetime.fromisoformat(template.last_updated)
                    if last.tzinfo is None:
                        last = last.replace(tzinfo=timezone.utc)
                    is_old = last < cutoff
                except (ValueError, TypeError):
                    is_old = True

            if is_old:
                template.stale = True
                metadata["template"] = template.to_dict()
                conn.execute(
                    "UPDATE identities SET metadata = ? WHERE id = ?",
                    (json.dumps(metadata), row["id"]),
                )
                count += 1

        except Exception as e:
            logger.warning(
                "Failed to check staleness for %s: %s",
                row["canonical_name"],
                e,
            )
            continue

    if count:
        conn.commit()
        logger.info("Marked %d templates as stale", count)
    return count


def prune_confidence_history(db: DatabaseManager, max_entries: int = 20) -> int:
    """Prune confidence_history to max_entries per template.

    Templates accumulate confidence_history entries over time.
    Keeps only the most recent max_entries.

    Returns:
        Number of templates pruned.
    """
    conn = db.get_connection()
    rows = conn.execute(
        "SELECT id, canonical_name, metadata FROM identities "
        "WHERE entity_type IN ('vendor', 'pos_provider') AND active = 1 "
        "AND metadata IS NOT NULL AND metadata != '{}'"
    ).fetchall()

    count = 0
    for row in rows:
        try:
            metadata = (
                json.loads(row["metadata"])
                if isinstance(row["metadata"], str)
                else row["metadata"]
            )
            template_data = metadata.get("template")
            if not template_data:
                continue

            history = template_data.get("confidence_history", [])
            if len(history) <= max_entries:
                continue

            template_data["confidence_history"] = history[-max_entries:]
            metadata["template"] = template_data
            conn.execute(
                "UPDATE identities SET metadata = ? WHERE id = ?",
                (json.dumps(metadata), row["id"]),
            )
            count += 1

        except Exception as e:
            logger.warning(
                "Failed to prune history for %s: %s",
                row["canonical_name"],
                e,
            )
            continue

    if count:
        conn.commit()
        logger.info("Pruned confidence history for %d templates", count)
    return count


def deduplicate_identity_members(db: DatabaseManager) -> int:
    """Remove duplicate identity members.

    Within each identity, if there are multiple members with the
    same (member_type, value) pair, keeps the oldest and removes
    duplicates.

    Returns:
        Number of duplicate members removed.
    """
    conn = db.get_connection()

    # Find duplicate groups
    dupes = conn.execute(
        "SELECT identity_id, member_type, value, "
        "COUNT(*) as cnt, MIN(rowid) as keep_rowid "
        "FROM identity_members "
        "GROUP BY identity_id, member_type, value "
        "HAVING cnt > 1"
    ).fetchall()

    count = 0
    for dupe in dupes:
        try:
            deleted = conn.execute(
                "DELETE FROM identity_members "
                "WHERE identity_id = ? AND member_type = ? AND value = ? "
                "AND rowid != ?",
                (
                    dupe["identity_id"],
                    dupe["member_type"],
                    dupe["value"],
                    dupe["keep_rowid"],
                ),
            ).rowcount
            count += deleted
        except Exception as e:
            logger.warning(
                "Failed to deduplicate member %s/%s: %s",
                dupe["member_type"],
                dupe["value"],
                e,
            )

    if count:
        conn.commit()
        logger.info("Removed %d duplicate identity members", count)
    return count


def remove_orphaned_members(db: DatabaseManager) -> int:
    """Remove identity members whose identity no longer exists.

    Returns:
        Number of orphaned members removed.
    """
    conn = db.get_connection()
    cursor = conn.execute(
        "DELETE FROM identity_members "
        "WHERE identity_id NOT IN (SELECT id FROM identities)"
    )
    count = cursor.rowcount
    if count:
        conn.commit()
        logger.info("Removed %d orphaned identity members", count)
    return count


def fix_weighed_item_units(db: DatabaseManager) -> DataQualityReport:
    """Fix weighed items with incorrect unit or missing unit_quantity.

    Detects and fixes two patterns:
    1. Items with unit='pcs'/'other' that are clearly weighed: have
       fractional unit_quantity in (0, 10) and price math proves per-kg
       pricing (unit_quantity * unit_price ≈ total_price). Sets unit='kg'.
    2. Items with unit='kg' and NULL unit_quantity where total_price
       and unit_price allow backfill: unit_quantity = total_price / unit_price.

    All changes are recorded as correction events via update_fact_item().

    Returns:
        DataQualityReport with counts and details.
    """
    from alibi.services.correction import update_fact_item

    report = DataQualityReport()
    conn = db.get_connection()

    # Pattern 1: unit='pcs'/'other' but clearly weighed
    # Criteria: fractional unit_quantity in (0, 10), and
    # unit_quantity * unit_price is within 2% of total_price
    pcs_weighed = conn.execute(
        """
        SELECT id, name, unit, unit_quantity, unit_price, total_price
        FROM fact_items
        WHERE unit IN ('pcs', 'other')
          AND unit_quantity IS NOT NULL
          AND unit_quantity > 0 AND unit_quantity < 10.0
          AND CAST(unit_quantity AS INTEGER) != unit_quantity
          AND unit_price IS NOT NULL AND unit_price > 0
          AND total_price IS NOT NULL AND total_price > 0
          AND ABS(total_price - unit_quantity * unit_price) / total_price < 0.02
        """
    ).fetchall()

    for row in pcs_weighed:
        item_id = row["id"]
        try:
            ok = update_fact_item(db, item_id, {"unit": "kg"})
            if ok:
                report.units_fixed += 1
                assert report.details is not None
                report.details.append(
                    f"unit pcs->kg: {row['name']!r} "
                    f"(uq={row['unit_quantity']}, "
                    f"up={row['unit_price']}, "
                    f"tp={row['total_price']})"
                )
        except Exception as e:
            logger.warning("Failed to fix unit for %s: %s", item_id[:8], e)

    # Pattern 2: unit='kg' with NULL unit_quantity but known prices
    # Backfill: unit_quantity = total_price / unit_price
    # Only when unit_price != total_price (otherwise ambiguous)
    kg_no_uq = conn.execute(
        """
        SELECT id, name, unit_price, total_price
        FROM fact_items
        WHERE unit = 'kg'
          AND unit_quantity IS NULL
          AND unit_price IS NOT NULL AND unit_price > 0
          AND total_price IS NOT NULL AND total_price > 0
          AND ABS(unit_price - total_price) > 0.01
        """
    ).fetchall()

    for row in kg_no_uq:
        item_id = row["id"]
        computed_uq = round(row["total_price"] / row["unit_price"], 3)
        if computed_uq <= 0 or computed_uq > 50:  # sanity bounds
            continue
        try:
            ok = update_fact_item(db, item_id, {"unit_quantity": computed_uq})
            if ok:
                report.unit_quantities_backfilled += 1
                assert report.details is not None
                report.details.append(
                    f"backfill uq={computed_uq}: {row['name']!r} "
                    f"(up={row['unit_price']}, tp={row['total_price']})"
                )
        except Exception as e:
            logger.warning(
                "Failed to backfill unit_quantity for %s: %s", item_id[:8], e
            )

    if report.total:
        logger.info(
            "Data quality: fixed %d units, backfilled %d unit_quantities",
            report.units_fixed,
            report.unit_quantities_backfilled,
        )
    return report


def fix_packaged_item_pricing(db: DatabaseManager) -> DataQualityReport:
    """Convert packaged-by-weight items from per-package to per-kg pricing.

    Detects items with unit='g' that have per-package pricing
    (unit_price ≈ total_price / quantity) and converts them to per-kg:
    - unit: g -> kg
    - unit_quantity: grams -> kg (divide by 1000)
    - unit_price: per-package -> per-kg

    Also fixes items already converted to unit='kg' (e.g. via TablePlus)
    where unit_price still reflects per-package pricing.

    Returns:
        DataQualityReport with counts and details.
    """
    from alibi.services.correction import update_fact_item

    report = DataQualityReport()
    conn = db.get_connection()

    # Pattern 1: unit='g' items with per-package pricing
    # unit_price * quantity ≈ total_price means unit_price is per-package
    g_items = conn.execute(
        """
        SELECT id, name, quantity, unit_quantity, unit_price, total_price
        FROM fact_items
        WHERE unit = 'g'
          AND unit_quantity IS NOT NULL AND unit_quantity > 0
          AND unit_price IS NOT NULL AND unit_price > 0
          AND total_price IS NOT NULL AND total_price > 0
          AND quantity IS NOT NULL AND quantity > 0
          AND ABS(total_price - unit_price * quantity) / total_price < 0.05
        """
    ).fetchall()

    for row in g_items:
        item_id = row["id"]
        uq_grams = float(row["unit_quantity"])
        uq_kg = round(uq_grams / 1000, 3)
        if uq_kg <= 0 or uq_kg > 50:
            continue
        per_kg_price = round(float(row["unit_price"]) / uq_kg, 2)
        try:
            ok = update_fact_item(
                db,
                item_id,
                {
                    "unit": "kg",
                    "unit_quantity": uq_kg,
                    "unit_price": per_kg_price,
                },
            )
            if ok:
                report.units_fixed += 1
                assert report.details is not None
                report.details.append(
                    f"g->kg: {row['name']!r} "
                    f"({uq_grams}g -> {uq_kg}kg, "
                    f"up={row['unit_price']} -> {per_kg_price}/kg)"
                )
        except Exception as e:
            logger.warning("Failed to convert %s: %s", item_id[:8], e)

    # Pattern 2: unit='kg' items where unit_price still equals per-package
    # (already converted unit/unit_quantity but unit_price not updated)
    # Detect: unit=kg, unit_quantity < 1, unit_price ≈ total_price / quantity
    kg_per_package = conn.execute(
        """
        SELECT id, name, quantity, unit_quantity, unit_price, total_price
        FROM fact_items
        WHERE unit = 'kg'
          AND unit_quantity IS NOT NULL AND unit_quantity > 0 AND unit_quantity < 1
          AND unit_price IS NOT NULL AND unit_price > 0
          AND total_price IS NOT NULL AND total_price > 0
          AND quantity IS NOT NULL AND quantity > 0
          AND ABS(total_price - unit_price * quantity) / total_price < 0.05
        """
    ).fetchall()

    for row in kg_per_package:
        item_id = row["id"]
        uq_kg = float(row["unit_quantity"])
        per_kg_price = round(float(row["unit_price"]) / uq_kg, 2)
        # Skip if unit_price is already per-kg (uq * up ≈ total / qty)
        current_check = uq_kg * float(row["unit_price"])
        expected = float(row["total_price"]) / float(row["quantity"])
        if abs(current_check - expected) / expected < 0.05:
            continue
        try:
            ok = update_fact_item(db, item_id, {"unit_price": per_kg_price})
            if ok:
                report.unit_quantities_backfilled += 1
                assert report.details is not None
                report.details.append(
                    f"per-kg price: {row['name']!r} "
                    f"(up={row['unit_price']} -> {per_kg_price}/kg, "
                    f"uq={uq_kg}kg)"
                )
        except Exception as e:
            logger.warning("Failed to fix pricing %s: %s", item_id[:8], e)

    if report.total:
        logger.info(
            "Packaged pricing: converted %d g->kg, fixed %d per-kg prices",
            report.units_fixed,
            report.unit_quantities_backfilled,
        )
    return report


def delete_garbage_items(db: DatabaseManager) -> DataQualityReport:
    """Delete non-product items from fact_items.

    Identifies and deletes items matching known garbage patterns:
    - Total/subtotal lines (ΣΥΝΟΛΟ, ΣYNDAO, ΣYNOAO)
    - Card payment lines (ΚΑΡΤΑ, KAPTA)
    - VAT summary headers (ΦΠΑ% ΦΠΑ + ΚΑΘΑΡΟ = ΜΕΙΚΤΟ variants)
    - Quantity multipliers (N ×)
    - Transaction metadata (PURCHASE, TIP)
    - OCR artifacts and non-product entries
    - Geographic region descriptions parsed as items
    - VAT detail lines (tax rate + amounts)

    Uses v2_store.delete_fact_items() directly since these aren't
    corrections to valid data — they're removing erroneous entries.

    Returns:
        DataQualityReport with counts and details.
    """
    from alibi.db.v2_store import delete_fact_items

    report = DataQualityReport()
    conn = db.get_connection()

    # Known garbage name patterns (exact matches)
    exact_garbage = {
        "PURCHASE",
        "TIP",
        "KAPTA",
        "ΣYNDAO",
        "ΣYNOAO",
    }

    # Regex-like patterns via SQL LIKE
    like_patterns = [
        "ΦΠA% ΦΠA%",  # VAT summary header variants
        "% × %",  # Quantity multiplier lines (e.g., "2 ×")
        "Βορείουνατολικός%",
        "Bορείουνατολικός%",
        "Αντολικός Μεσάγειος%",
        "DYNAMIC CONNECT DX",
        "JUNIOR TD",
        "E 0 %% 0%",  # VAT zero-rate detail lines
        "D 19 %% 0%",  # VAT 19% detail lines
        "B 5 %% %",  # VAT 5% detail lines
    ]

    # Collect garbage item IDs
    garbage_ids: list[str] = []
    garbage_details: list[str] = []

    # Exact name matches
    if exact_garbage:
        placeholders = ",".join("?" for _ in exact_garbage)
        rows = conn.execute(
            f"SELECT fi.id, fi.name, fi.total_price, f.vendor_key "
            f"FROM fact_items fi "
            f"JOIN facts f ON fi.fact_id = f.id "
            f"WHERE fi.name IN ({placeholders})",
            list(exact_garbage),
        ).fetchall()
        for row in rows:
            garbage_ids.append(row["id"])
            garbage_details.append(
                f"exact: {row['name']!r} "
                f"(vendor={row['vendor_key']}, "
                f"total={row['total_price']})"
            )

    # LIKE pattern matches
    for pattern in like_patterns:
        rows = conn.execute(
            "SELECT fi.id, fi.name, fi.total_price, f.vendor_key "
            "FROM fact_items fi "
            "JOIN facts f ON fi.fact_id = f.id "
            "WHERE fi.name LIKE ?",
            (pattern,),
        ).fetchall()
        for row in rows:
            if row["id"] not in garbage_ids:
                garbage_ids.append(row["id"])
                garbage_details.append(
                    f"pattern: {row['name']!r} "
                    f"(vendor={row['vendor_key']}, "
                    f"total={row['total_price']})"
                )

    # Items that are clearly footer/receipt concatenation blobs (>200 chars)
    long_garbage = conn.execute(
        "SELECT fi.id, fi.name, fi.total_price, f.vendor_key "
        "FROM fact_items fi "
        "JOIN facts f ON fi.fact_id = f.id "
        "WHERE LENGTH(fi.name) > 200"
    ).fetchall()
    for row in long_garbage:
        if row["id"] not in garbage_ids:
            garbage_ids.append(row["id"])
            garbage_details.append(
                f"long-name ({len(row['name'])} chars): "
                f"{row['name'][:60]!r}... "
                f"(vendor={row['vendor_key']})"
            )

    # Single-char or very short OCR artifacts that have no brand/category
    # and name matches patterns like "Q." or similar
    short_garbage = conn.execute(
        "SELECT fi.id, fi.name, fi.total_price, f.vendor_key "
        "FROM fact_items fi "
        "JOIN facts f ON fi.fact_id = f.id "
        "WHERE LENGTH(fi.name) <= 2 "
        "  AND fi.brand IS NULL AND fi.category IS NULL"
    ).fetchall()
    for row in short_garbage:
        if row["id"] not in garbage_ids:
            garbage_ids.append(row["id"])
            garbage_details.append(
                f"short-name: {row['name']!r} "
                f"(vendor={row['vendor_key']}, "
                f"total={row['total_price']})"
            )

    if garbage_ids:
        deleted = delete_fact_items(db, garbage_ids)
        report.units_fixed = deleted
        report.details = garbage_details
        logger.info("Deleted %d garbage items from fact_items", deleted)
    else:
        logger.info("No garbage items found")

    return report


def stamp_extraction_source(db: DatabaseManager) -> DataQualityReport:
    """Stamp items that have brand+category but no enrichment_source.

    Items that were enriched during extraction (by the heuristic parser
    or LLM) but never received an enrichment_source marker. Stamps them
    as 'extraction' with confidence 0.70.

    Returns:
        DataQualityReport with counts and details.
    """
    from alibi.services.correction import update_fact_item

    report = DataQualityReport()
    conn = db.get_connection()

    rows = conn.execute(
        "SELECT fi.id, fi.name, fi.brand, fi.category, f.vendor_key "
        "FROM fact_items fi "
        "JOIN facts f ON fi.fact_id = f.id "
        "WHERE fi.enrichment_source IS NULL "
        "  AND fi.brand IS NOT NULL AND fi.brand != '' "
        "  AND fi.category IS NOT NULL AND fi.category != ''"
    ).fetchall()

    for row in rows:
        item_id = row["id"]
        try:
            ok = update_fact_item(
                db,
                item_id,
                {
                    "enrichment_source": "extraction",
                    "enrichment_confidence": 0.70,
                },
            )
            if ok:
                report.units_fixed += 1
                assert report.details is not None
                report.details.append(
                    f"stamped: {row['name']!r} "
                    f"(brand={row['brand']}, cat={row['category']}, "
                    f"vendor={row['vendor_key']})"
                )
        except Exception as e:
            logger.warning("Failed to stamp enrichment for %s: %s", item_id[:8], e)

    if report.total:
        logger.info(
            "Stamped %d items with enrichment_source='extraction'",
            report.units_fixed,
        )
    return report


# Brand → category mapping for items that have a brand but missing category.
# Covers known brands from Cyprus supermarkets (Alphamega, LIDL, etc.)
_BRAND_CATEGORY_MAP: dict[str, str] = {
    # Dairy & eggs
    "ACTIVA": "Dairy",
    "ACTIVIA": "Dairy",
    "AGIA SKEPI": "Dairy",
    "ALPRO": "Dairy",
    "ARLA": "Dairy",
    "CD CHRISTIS": "Dairy",
    "CHRISTIS": "Dairy",
    "Charalambides Christis": "Dairy",
    "CHRYSOS": "Dairy",
    "FUNNY MILKMAN": "Dairy",
    "KERRYGOLI": "Dairy",
    "Kolios": "Dairy",
    "LAKXNANN": "Dairy",
    "MONOLITH": "Dairy",
    "NIKIFOROU": "Dairy",
    "Nikiforou": "Dairy",
    "SKAZK": "Dairy",
    "Toukaides": "Dairy",
    "Tzionis": "Dairy",
    "Tzionis Farm": "Dairy",
    "ZITA": "Dairy",
    "ΦAPMA TZIΩNH": "Dairy",
    "ΦΑРMA TZIΩNH": "Dairy",
    # Beverages
    "AHMAD": "Beverages",
    "RED BULL": "Beverages",
    "Red Bull": "Beverages",
    "REDBULL": "Beverages",
    "WH.EARTH": "Beverages",
    # Meat & poultry
    "ANNA": "Meat",
    "Anna": "Meat",
    "NEFELI": "Meat",
    "Palaichori": "Meat",
    # Seafood
    "AMAZING": "Seafood",
    # Fruits
    "ABOKATO-AVKATO": "Fruits",
    "BAJ": "Fruits",
    "MILA": "Fruits",
    "PIERIA": "Fruits",
    # Vegetables
    "EUROFRESH": "Vegetables",
    "GARDEN CYPRUS": "Vegetables",
    "GardenFresh": "Vegetables",
    # Frozen
    "7Seas": "Frozen",
    "Blue Green Wave": "Frozen",
    "Emafoods": "Frozen",
    "Lukoshko": "Frozen",
    # Deli
    "GRIGRIDOUJ": "Deli",
    "Grigoriou": "Deli",
    "XRYSODALIA": "Deli",
    "ΧΡΥΣΟΔΑΛ": "Deli",
    # Snacks & nuts
    "AMALIA": "Snacks",
    "Amalia": "Snacks",
    "KARPOS GIS": "Snacks",
    "Oriental Express": "Snacks",
    "RITTER": "Snacks",
    # Condiments & pantry
    "CAMPAGNA": "Condiments",
    "GAEA": "Condiments",
    "Gaea": "Condiments",
    "Heinz": "Condiments",
    "Ifantis": "Condiments",
    "Kaouris": "Condiments",
    "KIKKOMAN": "Condiments",
    "Morphakis": "Condiments",
    "OLYMPOS": "Condiments",
    # Oils
    "Fricook": "Oils",
    "Mas Best Brand": "Oils",
    # Grains & pasta
    "Barilla": "Grains",
    "MELISSA": "Grains",
    # Canned
    "Kyknos": "Canned",
    "Ponthier": "Canned",
    # Household
    "DOMESTOS": "Household",
    "EARTH RATED": "Household",
    "FINISH": "Household",
    "Fioro": "Household",
    "IROBOT": "Household",
    "KLEENEX": "Household",
    "TOPPITS": "Household",
    # Pet food
    "ACANA": "Pet Food",
    # Personal care
    "Fagron": "Personal Care",
    # Store brands — item-specific (matched by name substring)
    # ALPHAMEGA items are handled separately in the function.
}

# ALPHAMEGA store brand: product name → category
_ALPHAMEGA_ITEM_CATEGORIES: dict[str, str] = {
    "AGLA SKEPT": "Dairy",
    "FRUCTOSE": "Dairy",
    "CA.NL": "Canned",
    "SELF RAISI": "Snacks",
    "CLEARSPRING ORGANIC": "Condiments",
    "DETILIANI": "Grains",
}


def fill_brand_category_gaps(db: DatabaseManager) -> DataQualityReport:
    """Fill missing category for items that have a known brand.

    Uses a hardcoded brand→category mapping. For store brands (ALPHAMEGA),
    matches on item name substrings.

    Returns:
        DataQualityReport with counts and details.
    """
    from alibi.services.correction import update_fact_item

    report = DataQualityReport()
    conn = db.get_connection()

    rows = conn.execute(
        "SELECT fi.id, fi.name, fi.brand, f.vendor_key "
        "FROM fact_items fi "
        "JOIN facts f ON fi.fact_id = f.id "
        "WHERE fi.brand IS NOT NULL AND fi.brand != '' "
        "  AND fi.category IS NULL"
    ).fetchall()

    for row in rows:
        item_id = row["id"]
        brand = row["brand"]
        name = row["name"]
        category = None

        # Direct brand lookup
        if brand in _BRAND_CATEGORY_MAP:
            category = _BRAND_CATEGORY_MAP[brand]
        elif brand == "ALPHAMEGA" or brand == "Alphamega":
            # Store brand — match by name substring
            for substr, cat in _ALPHAMEGA_ITEM_CATEGORIES.items():
                if substr in name:
                    category = cat
                    break

        if category is None:
            continue

        try:
            ok = update_fact_item(db, item_id, {"category": category})
            if ok:
                report.units_fixed += 1
                assert report.details is not None
                report.details.append(
                    f"brand={brand!r} → category={category!r}: "
                    f"{name!r} (vendor={row['vendor_key']})"
                )
        except Exception as e:
            logger.warning("Failed to set category for %s: %s", item_id[:8], e)

    if report.total:
        logger.info("Filled %d category gaps from brand mapping", report.units_fixed)
    return report


def backfill_comparable_names(db: DatabaseManager) -> DataQualityReport:
    """Backfill missing comparable_name via Gemini name normalization.

    Items with a name but no comparable_name get English-standardized names
    from the Gemini enrichment API (or historical lookup from siblings with
    the same name_normalized).

    Returns:
        DataQualityReport with counts and details.
    """
    report = DataQualityReport()
    conn = db.get_connection()

    # First: backfill from siblings that already have comparable_name
    rows = conn.execute(
        "SELECT DISTINCT fi.id, fi.name_normalized, sib.comparable_name "
        "FROM fact_items fi "
        "JOIN fact_items sib ON sib.name_normalized = fi.name_normalized "
        "WHERE (fi.comparable_name IS NULL OR fi.comparable_name = '') "
        "  AND sib.comparable_name IS NOT NULL AND sib.comparable_name != '' "
        "  AND fi.name IS NOT NULL"
    ).fetchall()

    if rows:
        from alibi.services.correction import update_fact_item

        for row in rows:
            try:
                ok = update_fact_item(
                    db,
                    row["id"],
                    {"comparable_name": row["comparable_name"]},
                )
                if ok:
                    report.units_fixed += 1
            except Exception as e:
                logger.warning(
                    "Failed to backfill comparable_name for %s: %s",
                    row["id"][:8],
                    e,
                )

    # Count remaining gaps
    remaining = conn.execute(
        "SELECT COUNT(*) FROM fact_items "
        "WHERE (comparable_name IS NULL OR comparable_name = '') "
        "  AND name IS NOT NULL"
    ).fetchone()[0]

    if remaining > 0:
        logger.info(
            "Backfilled %d comparable_names from siblings; %d still missing "
            "(need Gemini enrichment: lt enrich gemini)",
            report.units_fixed,
            remaining,
        )
    elif report.units_fixed:
        logger.info(
            "Backfilled %d comparable_names from siblings; all items covered",
            report.units_fixed,
        )

    return report


def run_full_maintenance(
    db: DatabaseManager,
    max_history_entries: int = 20,
    stale_days: int = 90,
) -> MaintenanceReport:
    """Run all maintenance operations in sequence.

    Returns:
        MaintenanceReport with counts of all operations.
    """
    report = MaintenanceReport()
    report.templates_recalculated = recalculate_template_reliability(db)
    report.templates_marked_stale = mark_stale_templates(db, stale_days)
    report.templates_pruned = prune_confidence_history(db, max_history_entries)
    report.members_deduplicated = deduplicate_identity_members(db)
    report.orphaned_members_removed = remove_orphaned_members(db)
    backfill_comparable_names(db)
    return report
