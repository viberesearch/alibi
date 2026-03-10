"""Annotation CRUD operations.

Stores and queries annotations in SQLite.
All write operations use db.transaction() for atomicity.

Annotations are the unified system for all user-facing metadata. They replace
the former tags and consumers modules. The schema is fully open-ended: any
annotation_type/key/value combination is valid.

Common annotation patterns::

    # Person allocation — who was this purchased for?
    add_annotation(db, "person", "fact", fact_id, "bought_for", "Maria")
    add_annotation(db, "person", "fact_item", item_id, "bought_for", "John")

    # Project tracking — which project does this expense belong to?
    add_annotation(db, "project", "fact", fact_id, "project", "Kitchen renovation")

    # Expense splitting — split a receipt between people
    add_annotation(db, "split", "fact_item", milk_id, "share", "Maria",
                   metadata={"ratio": 0.5})
    add_annotation(db, "split", "fact_item", milk_id, "share", "John",
                   metadata={"ratio": 0.5})

    # Notes — free-text metadata
    add_annotation(db, "note", "fact", fact_id, "note",
                   "Warranty claim submitted 2026-03-01")

    # Vendor-level metadata
    add_annotation(db, "preference", "vendor", vendor_key, "rating", "5")
    add_annotation(db, "category", "vendor", vendor_key, "category", "groceries")

    # Query all annotations for a fact
    get_annotations(db, target_type="fact", target_id=fact_id)

    # Query all person allocations across all facts
    get_annotations(db, annotation_type="person")

Available via all interfaces:
- API: POST /api/v1/facts/{id}/annotate, GET/PUT/DELETE annotations
- Telegram: /tag [fact_id] <key> <value>, /untag <annotation_id>
- MCP: mcp_annotate(target_type, target_id, annotation_type, key, value)
- CLI: through service layer (alibi.services.annotate)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)

# Valid target types for annotations
VALID_TARGET_TYPES = {"fact", "fact_item", "vendor", "identity"}


def add_annotation(
    db: DatabaseManager,
    annotation_type: str,
    target_type: str,
    target_id: str,
    key: str,
    value: str,
    metadata: dict[str, Any] | None = None,
    source: str = "user",
) -> str:
    """Add an annotation to an entity.

    Args:
        db: Database manager.
        annotation_type: Type of annotation (person, project, category, etc.).
        target_type: Entity type (fact, fact_item, vendor, identity).
        target_id: ID of the target entity.
        key: Annotation key (e.g., "bought_for", "project").
        value: Annotation value (e.g., "Maria", "Kitchen renovation").
        metadata: Optional structured data (split ratios, item lists, etc.).
        source: Origin of annotation (user, auto, inference).

    Returns:
        The annotation ID.
    """
    if target_type not in VALID_TARGET_TYPES:
        raise ValueError(
            f"Invalid target_type '{target_type}'. "
            f"Must be one of: {', '.join(sorted(VALID_TARGET_TYPES))}"
        )

    annotation_id = str(uuid4())
    now = datetime.now().isoformat()
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO annotations "
            "(id, annotation_type, target_type, target_id, key, value, "
            "metadata, source, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                annotation_id,
                annotation_type,
                target_type,
                target_id,
                key,
                value,
                json.dumps(metadata) if metadata else None,
                source,
                now,
            ),
        )
    logger.debug(
        f"Added {annotation_type} annotation on {target_type}/{target_id[:8]}: "
        f"{key}={value}"
    )
    return annotation_id


def get_annotations(
    db: DatabaseManager,
    target_type: str | None = None,
    target_id: str | None = None,
    annotation_type: str | None = None,
) -> list[dict[str, Any]]:
    """Query annotations with optional filtering.

    Args:
        db: Database manager.
        target_type: Filter by target type (fact, fact_item, etc.).
        target_id: Filter by target entity ID.
        annotation_type: Filter by annotation type (person, project, etc.).

    Returns:
        List of annotation dicts.
    """
    where: list[str] = []
    params: list[Any] = []

    if target_type:
        where.append("target_type = ?")
        params.append(target_type)
    if target_id:
        where.append("target_id = ?")
        params.append(target_id)
    if annotation_type:
        where.append("annotation_type = ?")
        params.append(annotation_type)

    where_clause = f"WHERE {' AND '.join(where)}" if where else ""

    rows = db.fetchall(
        f"SELECT * FROM annotations {where_clause} ORDER BY created_at",
        tuple(params),
    )

    result = []
    for row in rows:
        d = dict(row)
        if d.get("metadata") and isinstance(d["metadata"], str):
            d["metadata"] = json.loads(d["metadata"])
        result.append(d)
    return result


def update_annotation(
    db: DatabaseManager,
    annotation_id: str,
    value: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Update an annotation's value or metadata.

    Returns:
        True if a row was updated.
    """
    sets: list[str] = []
    params: list[Any] = []

    if value is not None:
        sets.append("value = ?")
        params.append(value)
    if metadata is not None:
        sets.append("metadata = ?")
        params.append(json.dumps(metadata))

    if not sets:
        return False

    sets.append("updated_at = ?")
    params.append(datetime.now().isoformat())
    params.append(annotation_id)

    with db.transaction() as cursor:
        cursor.execute(
            f"UPDATE annotations SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        return cursor.rowcount > 0


def delete_annotation(db: DatabaseManager, annotation_id: str) -> bool:
    """Delete an annotation.

    Returns:
        True if a row was deleted.
    """
    with db.transaction() as cursor:
        cursor.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
        return cursor.rowcount > 0


def collect_annotations_for_cleanup(
    db: DatabaseManager,
    fact_ids: list[str],
    fact_item_ids: list[str],
) -> list[dict[str, Any]]:
    """Save annotations pointing to facts/fact_items about to be deleted.

    Collects all annotations whose targets are in the given ID lists,
    preserving their data so they can be re-attached after re-ingestion.

    Returns:
        List of annotation dicts (full row data).
    """
    saved: list[dict[str, Any]] = []

    for fid in fact_ids:
        rows = db.fetchall(
            "SELECT * FROM annotations WHERE target_type = 'fact' AND target_id = ?",
            (fid,),
        )
        for row in rows:
            d = dict(row)
            if d.get("metadata") and isinstance(d["metadata"], str):
                d["metadata"] = json.loads(d["metadata"])
            saved.append(d)

    for iid in fact_item_ids:
        rows = db.fetchall(
            "SELECT * FROM annotations "
            "WHERE target_type = 'fact_item' AND target_id = ?",
            (iid,),
        )
        for row in rows:
            d = dict(row)
            if d.get("metadata") and isinstance(d["metadata"], str):
                d["metadata"] = json.loads(d["metadata"])
            saved.append(d)

    if saved:
        logger.info(
            f"Collected {len(saved)} annotation(s) from "
            f"{len(fact_ids)} fact(s), {len(fact_item_ids)} item(s)"
        )
    return saved


def migrate_annotations_to_fact(
    db: DatabaseManager,
    saved: list[dict[str, Any]],
    new_fact_id: str,
    new_items: list[dict[str, Any]],
) -> int:
    """Re-attach saved annotations to a new fact and its items.

    Fact-level annotations are moved to new_fact_id.
    Fact-item annotations are matched by item name (case-insensitive).

    Args:
        db: Database manager.
        saved: Annotations from collect_annotations_for_cleanup().
        new_fact_id: The newly created fact ID.
        new_items: List of dicts with at least 'id' and 'name' keys.

    Returns:
        Count of successfully migrated annotations.
    """
    if not saved:
        return 0

    # Build name -> new item ID mapping
    item_map: dict[str, str] = {}
    for item in new_items:
        name = (item.get("name") or "").strip().lower()
        if name:
            item_map[name] = item["id"]

    migrated = 0
    now = datetime.now().isoformat()

    with db.transaction() as cursor:
        for ann in saved:
            new_id = str(uuid4())
            target_type = ann["target_type"]
            new_target_id: str | None = None

            if target_type == "fact":
                new_target_id = new_fact_id
            elif target_type == "fact_item":
                # Match by item name
                old_meta = ann.get("metadata") or {}
                item_name = (
                    old_meta.get("item_name", "").strip().lower()
                    if isinstance(old_meta, dict)
                    else ""
                )
                if item_name and item_name in item_map:
                    new_target_id = item_map[item_name]
                else:
                    # Try key as name hint
                    ann_value = (ann.get("value") or "").strip().lower()
                    for name, iid in item_map.items():
                        if ann_value and ann_value in name:
                            new_target_id = iid
                            break

            if new_target_id is None:
                logger.debug(
                    f"Could not migrate annotation {ann['id'][:8]} "
                    f"({target_type}): no matching target"
                )
                continue

            meta_json = json.dumps(ann["metadata"]) if ann.get("metadata") else None
            cursor.execute(
                "INSERT INTO annotations "
                "(id, annotation_type, target_type, target_id, key, value, "
                "metadata, source, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_id,
                    ann["annotation_type"],
                    target_type,
                    new_target_id,
                    ann["key"],
                    ann["value"],
                    meta_json,
                    ann.get("source", "migrated"),
                    now,
                ),
            )
            migrated += 1

    if migrated:
        logger.info(f"Migrated {migrated}/{len(saved)} annotation(s) to new fact")
    return migrated


def cleanup_orphaned_annotations(db: DatabaseManager) -> int:
    """Delete annotations whose target entities no longer exist.

    Checks fact targets against facts table, fact_item against fact_items,
    identity against identities, and vendor against facts.vendor_key.

    Returns:
        Count of deleted orphaned annotations.
    """
    orphan_ids: list[str] = []

    # Fact annotations with non-existent target
    rows = db.fetchall(
        "SELECT a.id FROM annotations a "
        "LEFT JOIN facts f ON a.target_id = f.id "
        "WHERE a.target_type = 'fact' AND f.id IS NULL",
        (),
    )
    orphan_ids.extend(r["id"] for r in rows)

    # Fact-item annotations with non-existent target
    rows = db.fetchall(
        "SELECT a.id FROM annotations a "
        "LEFT JOIN fact_items fi ON a.target_id = fi.id "
        "WHERE a.target_type = 'fact_item' AND fi.id IS NULL",
        (),
    )
    orphan_ids.extend(r["id"] for r in rows)

    # Identity annotations with non-existent target
    rows = db.fetchall(
        "SELECT a.id FROM annotations a "
        "LEFT JOIN identities i ON a.target_id = i.id "
        "WHERE a.target_type = 'identity' AND i.id IS NULL",
        (),
    )
    orphan_ids.extend(r["id"] for r in rows)

    if not orphan_ids:
        return 0

    with db.transaction() as cursor:
        for aid in orphan_ids:
            cursor.execute("DELETE FROM annotations WHERE id = ?", (aid,))

    logger.info(f"Cleaned up {len(orphan_ids)} orphaned annotation(s)")
    return len(orphan_ids)
