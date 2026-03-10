"""Maintenance service facade.

Exposes learning aggregation and data cleanup operations
through the standard service layer interface.
"""

from alibi.db.connection import DatabaseManager
from alibi.maintenance.learning_aggregation import (
    DataQualityReport,
    MaintenanceReport,
    deduplicate_identity_members,
    delete_garbage_items as _delete_garbage_items,
    fill_brand_category_gaps as _fill_brand_category_gaps,
    fix_packaged_item_pricing,
    fix_weighed_item_units,
    mark_stale_templates,
    prune_confidence_history,
    recalculate_template_reliability,
    remove_orphaned_members,
    run_full_maintenance,
    stamp_extraction_source as _stamp_extraction_source,
)


def run_maintenance(
    db: DatabaseManager,
    max_history: int = 20,
    stale_days: int = 90,
) -> MaintenanceReport:
    """Run full maintenance cycle."""
    return run_full_maintenance(db, max_history, stale_days)


def recalculate_templates(db: DatabaseManager) -> int:
    """Recalculate template reliability from corrections."""
    return recalculate_template_reliability(db)


def cleanup_identities(db: DatabaseManager) -> int:
    """Deduplicate members + remove orphans. Returns total removed."""
    dedup = deduplicate_identity_members(db)
    orphans = remove_orphaned_members(db)
    return dedup + orphans


def delete_garbage(db: DatabaseManager) -> DataQualityReport:
    """Delete non-product garbage items from fact_items."""
    return _delete_garbage_items(db)


def stamp_extraction(db: DatabaseManager) -> DataQualityReport:
    """Stamp items with brand+category but missing enrichment_source."""
    return _stamp_extraction_source(db)


def fill_category_gaps(db: DatabaseManager) -> DataQualityReport:
    """Fill missing category from known brand mappings."""
    return _fill_brand_category_gaps(db)


def fix_data_quality(db: DatabaseManager) -> DataQualityReport:
    """Fix known data quality issues in fact_items.

    Currently fixes:
    - Weighed items with wrong unit (pcs -> kg)
    - Weighed items with missing unit_quantity (backfill from prices)
    - Packaged items: g -> kg conversion with per-kg unit_price
    """
    r1 = fix_weighed_item_units(db)
    r2 = fix_packaged_item_pricing(db)
    combined = DataQualityReport(
        units_fixed=r1.units_fixed + r2.units_fixed,
        unit_quantities_backfilled=(
            r1.unit_quantities_backfilled + r2.unit_quantities_backfilled
        ),
        details=(r1.details or []) + (r2.details or []),
    )
    return combined
