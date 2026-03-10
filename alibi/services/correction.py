"""Correction service for facts, clouds, and vendor assignments.

Wraps cloud correction operations and adds higher-level correction
functions that combine multiple steps (e.g., update vendor + teach
identity system).
"""

from __future__ import annotations

import logging
from typing import Any

from alibi.clouds.correction import CorrectionResult
from alibi.clouds import correction as _correction
from alibi.db.connection import DatabaseManager
from alibi.db import v2_store
from alibi.identities.matching import ensure_vendor_identity
from alibi.normalizers.vendors import normalize_vendor
from alibi.services.events import EventType, event_bus

logger = logging.getLogger(__name__)


def _record_correction(
    db: DatabaseManager,
    entity_type: str,
    entity_id: str,
    field: str,
    old_value: object,
    new_value: object,
    source: str = "system",
    user_id: str | None = None,
) -> None:
    """Record a correction event for adaptive learning.

    Silently catches errors — correction logging must never break
    the actual correction operation.
    """
    try:
        v2_store.record_correction_event(
            db,
            entity_type=entity_type,
            entity_id=entity_id,
            field=field,
            old_value=str(old_value) if old_value is not None else None,
            new_value=str(new_value) if new_value is not None else None,
            source=source,
            user_id=user_id,
        )
    except Exception as e:
        logger.debug(f"Failed to record correction event: {e}")


# Fields that callers are permitted to update on a fact via update_fact().
# This is an explicit allowlist — unknown fields are rejected.
_UPDATABLE_FIELDS = frozenset({"vendor", "amount", "date", "fact_type", "vendor_key"})

# Map service-level field names to their column names in the facts table.
_FIELD_TO_COLUMN: dict[str, str] = {
    "vendor": "vendor",
    "amount": "total_amount",
    "date": "event_date",
    "fact_type": "fact_type",
    "vendor_key": "vendor_key",
}


def move_bundle(
    db: DatabaseManager,
    bundle_id: str,
    target_cloud_id: str | None = None,
) -> CorrectionResult:
    """Move a bundle to a different cloud and re-collapse both clouds.

    Delegates directly to clouds.correction.move_bundle.

    Args:
        db: Database manager.
        bundle_id: The bundle to move.
        target_cloud_id: Cloud to move it to, or None to create a new cloud.

    Returns:
        CorrectionResult with new fact IDs after re-collapse.
    """
    result = _correction.move_bundle(db, bundle_id, target_cloud_id)
    if result.success:
        _record_correction(
            db,
            entity_type="bundle",
            entity_id=bundle_id,
            field="cloud_id",
            old_value=result.source_cloud_id,
            new_value=result.target_cloud_id,
            source="system",
        )
        event_bus.emit(
            EventType.CORRECTION_APPLIED,
            {
                "action": "move_bundle",
                "bundle_id": bundle_id,
                "target_cloud_id": result.target_cloud_id,
                "target_fact_id": result.target_fact_id,
            },
        )
    return result


def recollapse_cloud(
    db: DatabaseManager,
    cloud_id: str,
) -> CorrectionResult:
    """Force re-collapse of a cloud into a fact.

    Delegates to clouds.correction.recollapse_cloud and wraps the result
    in a CorrectionResult.

    Args:
        db: Database manager.
        cloud_id: The cloud to re-collapse.

    Returns:
        CorrectionResult with target_fact_id set if collapse succeeded.
    """
    fact_id = _correction.recollapse_cloud(db, cloud_id)
    event_bus.emit(
        EventType.CORRECTION_APPLIED,
        {
            "action": "recollapse_cloud",
            "cloud_id": cloud_id,
            "target_fact_id": fact_id,
        },
    )
    return CorrectionResult(
        success=True,
        target_cloud_id=cloud_id,
        target_fact_id=fact_id,
    )


def mark_disputed(
    db: DatabaseManager,
    cloud_id: str,
) -> CorrectionResult:
    """Mark a cloud as disputed (needs human review).

    Delegates to clouds.correction.mark_disputed and wraps the result
    in a CorrectionResult.

    Args:
        db: Database manager.
        cloud_id: The cloud to mark as disputed.

    Returns:
        CorrectionResult with success reflecting the operation outcome.
    """
    ok = _correction.mark_disputed(db, cloud_id)
    if ok:
        event_bus.emit(
            EventType.CORRECTION_APPLIED,
            {"action": "mark_disputed", "cloud_id": cloud_id},
        )
    return CorrectionResult(
        success=ok,
        source_cloud_id=cloud_id,
    )


def update_fact(
    db: DatabaseManager,
    fact_id: str,
    fields: dict[str, object],
) -> bool:
    """Update specific fields on a fact.

    Only fields in the allowlist are accepted:
        vendor, amount, date, fact_type, vendor_key

    Unknown field names raise ValueError.

    Args:
        db: Database manager.
        fact_id: ID of the fact to update.
        fields: Dict mapping field names to new values. Must not be empty
            and must contain only allowed field names.

    Returns:
        True if the fact was found and updated, False if the fact does
        not exist.

    Raises:
        ValueError: If fields is empty or contains a disallowed field name.
    """
    if not fields:
        raise ValueError("fields must not be empty")

    unknown = set(fields) - _UPDATABLE_FIELDS
    if unknown:
        raise ValueError(
            f"Disallowed field(s) for update_fact: {sorted(unknown)}. "
            f"Allowed: {sorted(_UPDATABLE_FIELDS)}"
        )

    # Guard: fact must exist
    existing = v2_store.get_fact_by_id(db, fact_id)
    if not existing:
        return False

    # Build SET clause using canonical column names
    set_parts: list[str] = []
    params: list[object] = []
    for field_name, value in fields.items():
        column = _FIELD_TO_COLUMN[field_name]
        set_parts.append(f"{column} = ?")
        params.append(value)

    params.append(fact_id)
    sql = f"UPDATE facts SET {', '.join(set_parts)} WHERE id = ?"

    with db.transaction() as cursor:
        cursor.execute(sql, tuple(params))

    # Record correction events for each changed field
    for field_name, value in fields.items():
        column = _FIELD_TO_COLUMN[field_name]
        old_val = existing.get(column)
        if str(old_val) != str(value):
            _record_correction(
                db,
                entity_type="fact",
                entity_id=fact_id,
                field=field_name,
                old_value=old_val,
                new_value=value,
            )

    # Correction-driven template refinement (best-effort)
    if "date" in fields:
        try:
            from alibi.extraction.templates import apply_correction_to_template

            vendor_key = existing.get("vendor_key")
            old_date = existing.get("event_date")
            new_date = fields["date"]
            if vendor_key and old_date and new_date:
                apply_correction_to_template(
                    db,
                    vendor_key=str(vendor_key),
                    field="date",
                    old_value=str(old_date),
                    new_value=str(new_date),
                )
        except Exception as e:
            logger.debug("Template correction from date update skipped: %s", e)

    event_bus.emit(
        EventType.FACT_UPDATED,
        {"fact_id": fact_id, "fields": list(fields.keys())},
    )
    return True


def correct_vendor(
    db: DatabaseManager,
    fact_id: str,
    new_vendor: str,
) -> bool:
    """Update the vendor on a fact and register the name in the identity system.

    Steps:
        1. Retrieve the current fact (returns False if not found).
        2. Update the vendor field on the fact row.
        3. Call ensure_vendor_identity so the new name is registered,
           preserving any existing vendor_key as a linking signal.

    Args:
        db: Database manager.
        fact_id: ID of the fact whose vendor to correct.
        new_vendor: New vendor display name.

    Returns:
        True on success, False if the fact does not exist.
    """
    existing = v2_store.get_fact_by_id(db, fact_id)
    if not existing:
        return False

    old_vendor = existing.get("vendor")

    # Normalize to a consistent display form before storing
    normalized_name = normalize_vendor(new_vendor)

    # Persist the updated vendor name
    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE facts SET vendor = ? WHERE id = ?",
            (normalized_name, fact_id),
        )

    # Record correction event
    if old_vendor != normalized_name:
        _record_correction(
            db,
            entity_type="fact",
            entity_id=fact_id,
            field="vendor",
            old_value=old_vendor,
            new_value=normalized_name,
        )

    # Teach the identity system about the new vendor name
    vendor_key = existing.get("vendor_key")
    ensure_vendor_identity(
        db,
        vendor_name=normalized_name,
        vendor_key=vendor_key,
        source="correction",
    )

    event_bus.emit(
        EventType.FACT_UPDATED,
        {
            "fact_id": fact_id,
            "fields": ["vendor"],
            "new_vendor": normalized_name,
        },
    )
    logger.info(
        "correct_vendor: fact %s vendor updated to %r (vendor_key=%r)",
        fact_id,
        normalized_name,
        vendor_key,
    )
    return True


def reconcile_forming_clouds(db: DatabaseManager) -> int:
    """Re-try collapse on all FORMING clouds.

    Useful after batch processing or when documents arrive out of order.

    Returns:
        Count of clouds newly collapsed into facts.
    """
    from alibi.db.models import CloudStatus

    rows = db.fetchall(
        "SELECT id FROM clouds WHERE status = ?",
        (CloudStatus.FORMING.value,),
    )
    if not rows:
        return 0

    collapsed = 0
    for row in rows:
        cloud_id = row["id"]
        try:
            fact_id = _correction.recollapse_cloud(db, cloud_id)
            if fact_id:
                collapsed += 1
                logger.info(
                    f"Reconciled forming cloud {cloud_id[:8]} → fact {fact_id[:8]}"
                )
        except Exception as e:
            logger.warning(f"Failed to reconcile cloud {cloud_id[:8]}: {e}")

    return collapsed


# Fields that callers are permitted to update on a fact_item.
_UPDATABLE_ITEM_FIELDS = frozenset(
    {
        "barcode",
        "brand",
        "category",
        "comparable_name",
        "name",
        "enrichment_source",
        "enrichment_confidence",
        "unit_price",
        "unit_quantity",
        "unit",
        "product_variant",
    }
)

_ITEM_FIELD_TO_COLUMN: dict[str, str] = {
    "barcode": "barcode",
    "brand": "brand",
    "category": "category",
    "comparable_name": "comparable_name",
    "name": "name",
    "enrichment_source": "enrichment_source",
    "enrichment_confidence": "enrichment_confidence",
    "unit_price": "unit_price",
    "unit_quantity": "unit_quantity",
    "unit": "unit",
    "product_variant": "product_variant",
}


def update_fact_item(
    db: DatabaseManager,
    fact_item_id: str,
    fields: dict[str, object],
) -> bool:
    """Update specific fields on a fact item.

    Only fields in the allowlist are accepted:
        barcode, brand, category, name

    When barcode is updated, also updates item identity.

    Returns:
        True if the fact item was found and updated.

    Raises:
        ValueError: If fields is empty or contains a disallowed field name.
    """
    if not fields:
        raise ValueError("fields must not be empty")

    unknown = set(fields) - _UPDATABLE_ITEM_FIELDS
    if unknown:
        raise ValueError(
            f"Disallowed field(s) for update_fact_item: {sorted(unknown)}. "
            f"Allowed: {sorted(_UPDATABLE_ITEM_FIELDS)}"
        )

    # Guard: fact_item must exist
    from alibi.services.query import get_fact_item

    existing = get_fact_item(db, fact_item_id)
    if not existing:
        return False

    # Validate numeric fields: reject non-numeric values for REAL columns.
    _NUMERIC_ITEM_FIELDS = {"unit_price", "unit_quantity", "enrichment_confidence"}
    for nf in _NUMERIC_ITEM_FIELDS:
        if nf in fields:
            v = fields[nf]
            if v is None:
                continue
            if isinstance(v, (int, float)):
                continue
            try:
                fields[nf] = float(str(v))
            except (ValueError, TypeError):
                fields[nf] = None

    # Normalize category to title case for consistency
    # (LLM returns lowercase "dairy", cloud returns "Dairy" — standardize)
    if "category" in fields and isinstance(fields["category"], str):
        fields["category"] = fields["category"].strip().title()

    # Backfill name_normalized from comparable_name when missing
    if "comparable_name" in fields and fields["comparable_name"]:
        current_norm = existing.get("name_normalized") or ""
        if not current_norm.strip():
            fields["name_normalized"] = fields["comparable_name"]

    # Build SET clause
    set_parts: list[str] = []
    params: list[object] = []
    for field_name, value in fields.items():
        column = _ITEM_FIELD_TO_COLUMN.get(field_name, field_name)
        set_parts.append(f"{column} = ?")
        params.append(value)

    params.append(fact_item_id)
    sql = f"UPDATE fact_items SET {', '.join(set_parts)} WHERE id = ?"

    with db.transaction() as cursor:
        cursor.execute(sql, tuple(params))

    # Record correction events for each changed field
    for field_name, value in fields.items():
        old_val = existing.get(field_name)
        if str(old_val) != str(value):
            _record_correction(
                db,
                entity_type="fact_item",
                entity_id=fact_item_id,
                field=field_name,
                old_value=old_val,
                new_value=value,
            )

    # If barcode was set, update item identity
    if "barcode" in fields and fields["barcode"]:
        try:
            from alibi.identities.matching import ensure_item_identity

            ensure_item_identity(
                db,
                item_name=existing.get("name"),
                barcode=str(fields["barcode"]),
                source="correction",
            )
        except Exception as e:
            logger.debug(f"Item identity update after barcode set skipped: {e}")

    # If unit_quantity was set, propagate to item identity metadata
    if "unit_quantity" in fields and fields["unit_quantity"] is not None:
        try:
            from alibi.identities.matching import ensure_item_identity

            item_name = existing.get("name")
            barcode = existing.get("barcode")
            if item_name:
                identity_id = ensure_item_identity(
                    db,
                    item_name=item_name,
                    barcode=barcode,
                    source="correction",
                )
                if identity_id:
                    _update_identity_unit_metadata(
                        db,
                        identity_id,
                        float(str(fields["unit_quantity"])),
                        str(fields.get("unit", existing.get("unit", "") or "")),
                    )
        except Exception as e:
            logger.debug(f"Identity metadata update for unit_quantity skipped: {e}")

    # Sibling propagation: spread brand/category to items with same name
    propagation_fields = {"brand", "category"}
    changed_propagatable = propagation_fields & set(fields)
    if changed_propagatable:
        try:
            name_norm = existing.get("name_normalized") or existing.get("name")
            if name_norm:
                _propagate_to_siblings(
                    db,
                    fact_item_id,
                    name_norm,
                    {k: fields[k] for k in changed_propagatable},
                )
        except Exception as e:
            logger.debug(f"Sibling propagation skipped: {e}")

    event_bus.emit(
        EventType.FACT_UPDATED,
        {"fact_item_id": fact_item_id, "fields": list(fields.keys())},
    )
    return True


def _propagate_to_siblings(
    db: DatabaseManager,
    source_item_id: str,
    name_normalized: str,
    fields: dict[str, object],
) -> int:
    """Propagate brand/category to sibling items with the same normalized name.

    Skips items already at user_confirmed enrichment source.
    Returns number of items updated.
    """
    siblings = db.fetchall(
        "SELECT id, enrichment_source FROM fact_items "
        "WHERE name_normalized = ? AND id != ? "
        "AND (enrichment_source IS NULL OR enrichment_source != 'user_confirmed')",
        (name_normalized, source_item_id),
    )
    if not siblings:
        return 0

    set_parts = []
    params: list[object] = []
    for field_name, value in fields.items():
        column = _ITEM_FIELD_TO_COLUMN.get(field_name, field_name)
        set_parts.append(f"{column} = ?")
        params.append(value)
    set_parts.append("enrichment_source = ?")
    params.append("sibling_propagation")
    set_parts.append("enrichment_confidence = ?")
    params.append(0.90)

    count = 0
    for sib in siblings:
        with db.transaction() as cursor:
            cursor.execute(
                f"UPDATE fact_items SET {', '.join(set_parts)} WHERE id = ?",
                tuple(params) + (sib["id"],),
            )
            count += 1

    if count:
        logger.info(
            f"Propagated {list(fields.keys())} to {count} siblings "
            f"of '{name_normalized}'"
        )
    return count


def _update_identity_unit_metadata(
    db: DatabaseManager,
    identity_id: str,
    unit_quantity: float,
    unit: str,
) -> None:
    """Update item identity metadata with canonical unit_quantity/unit.

    Called after a user manually sets unit_quantity on a fact item.
    Stores the authoritative unit info in the identity so future
    ingestions can apply it automatically.
    """
    import json

    row = db.fetchone(
        "SELECT metadata FROM identities WHERE id = ?",
        (identity_id,),
    )
    if not row:
        return

    metadata = json.loads(row["metadata"]) if row["metadata"] else {}
    metadata["unit_quantity"] = unit_quantity
    metadata["unit"] = unit

    with db.transaction() as cursor:
        cursor.execute(
            "UPDATE identities SET metadata = ? WHERE id = ?",
            (json.dumps(metadata), identity_id),
        )


def set_fact_location(
    db: DatabaseManager,
    fact_id: str,
    map_url: str,
) -> dict[str, Any] | None:
    """Parse a map URL and store it as a location annotation on a fact.

    If the fact already has a location annotation, it is updated.

    Args:
        db: Database manager.
        fact_id: ID of the fact.
        map_url: Google Maps URL to parse and store.

    Returns:
        Dict with lat, lng, clean_url, place_name — or None if URL
        cannot be parsed or fact does not exist.
    """
    from alibi.annotations.store import (
        add_annotation,
        get_annotations,
        update_annotation as _update_ann,
    )
    from alibi.utils.map_url import parse_map_url

    # Validate fact exists
    existing = v2_store.get_fact_by_id(db, fact_id)
    if not existing:
        return None

    parsed = parse_map_url(map_url)
    if not parsed:
        return None

    metadata = {
        "lat": parsed["lat"],
        "lng": parsed["lng"],
        "place_name": parsed.get("place_name"),
        "raw_url": map_url.strip(),
    }

    # Check for existing location annotation
    existing_anns = get_annotations(
        db,
        target_type="fact",
        target_id=fact_id,
        annotation_type="location",
    )
    if existing_anns:
        _update_ann(
            db,
            existing_anns[0]["id"],
            value=parsed["clean_url"],
            metadata=metadata,
        )
    else:
        add_annotation(
            db,
            annotation_type="location",
            target_type="fact",
            target_id=fact_id,
            key="map_url",
            value=parsed["clean_url"],
            metadata=metadata,
            source="user",
        )

    logger.info(
        "set_fact_location: fact %s -> (%s, %s)",
        fact_id[:8],
        parsed["lat"],
        parsed["lng"],
    )
    return parsed


def get_fact_location(db: DatabaseManager, fact_id: str) -> dict[str, Any] | None:
    """Get the location annotation for a fact.

    Returns:
        Dict with lat, lng, map_url, place_name — or None.
    """
    from alibi.annotations.store import get_annotations

    anns = get_annotations(
        db,
        target_type="fact",
        target_id=fact_id,
        annotation_type="location",
    )
    if not anns:
        return None

    ann = anns[0]
    meta = ann.get("metadata") or {}
    return {
        "lat": meta.get("lat"),
        "lng": meta.get("lng"),
        "map_url": ann.get("value"),
        "place_name": meta.get("place_name"),
        "annotation_id": ann["id"],
    }


def get_recent_vendor_locations(
    db: DatabaseManager, limit: int = 20
) -> list[dict[str, Any]]:
    """Get recent unique vendor+location pairs for UI picker.

    Queries facts joined with location annotations, deduplicates by
    vendor_key+coordinates, returns most recent first.

    Returns:
        List of dicts with vendor_name, vendor_key, map_url, lat, lng,
        place_name, last_used.
    """
    rows = db.fetchall(
        """
        SELECT
            f.vendor AS vendor_name,
            f.vendor_key,
            a.value AS map_url,
            a.metadata,
            MAX(a.created_at) AS last_used
        FROM annotations a
        JOIN facts f ON a.target_id = f.id
        WHERE a.annotation_type = 'location'
          AND a.target_type = 'fact'
          AND a.key = 'map_url'
        GROUP BY f.vendor_key, a.value
        ORDER BY last_used DESC
        LIMIT ?
        """,
        (limit,),
    )

    import json

    results = []
    for row in rows:
        meta = row["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        elif meta is None:
            meta = {}

        results.append(
            {
                "vendor_name": row["vendor_name"],
                "vendor_key": row["vendor_key"],
                "map_url": row["map_url"],
                "lat": meta.get("lat"),
                "lng": meta.get("lng"),
                "place_name": meta.get("place_name"),
                "last_used": row["last_used"],
            }
        )

    return results


def delete_fact_item(db: DatabaseManager, item_id: str) -> bool:
    """Delete a single fact item by ID. Returns True if it existed."""
    count = v2_store.delete_fact_items(db, [item_id])
    return count > 0


def cleanup_orphaned_annotations(db: DatabaseManager) -> int:
    """Delete annotations whose target entities no longer exist.

    Wrapper around annotations.store.cleanup_orphaned_annotations.
    """
    from alibi.annotations.store import (
        cleanup_orphaned_annotations as _cleanup,
    )

    return _cleanup(db)
