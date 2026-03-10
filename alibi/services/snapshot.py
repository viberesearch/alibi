"""Snapshot & diff service for detecting out-of-band fact_item edits.

Workflow:
    1. take_snapshot(db)  — persist current fact_items state to disk
    2. User edits directly in TablePlus / SQL
    3. detect_changes(db, snapshot) — diff DB vs snapshot
    4. apply_changes(db, changes) — record correction_events + propagate
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)

SNAPSHOT_DIR = Path.home() / ".alibi"
SNAPSHOT_FILE = SNAPSHOT_DIR / "corrections_snapshot.json"

# Columns we track for change detection
TRACKED_COLUMNS = (
    "name",
    "name_normalized",
    "comparable_name",
    "brand",
    "category",
    "barcode",
    "unit",
    "unit_quantity",
    "enrichment_source",
    "enrichment_confidence",
)


def _normalize_value(val: object) -> str:
    """Normalize a value for comparison.  None and "" are equivalent."""
    if val is None:
        return ""
    return str(val).strip()


def take_snapshot(db: DatabaseManager) -> int:
    """Save current fact_items state to disk.

    Returns:
        Number of items captured.

    Raises:
        FileExistsError: If a snapshot already exists.
    """
    if SNAPSHOT_FILE.exists():
        raise FileExistsError(
            f"Snapshot already exists at {SNAPSHOT_FILE}. "
            "Use --clear to remove it first, or run 'detect' to process changes."
        )

    cols = ", ".join(TRACKED_COLUMNS)
    rows = db.fetchall(
        f"SELECT id, fact_id, {cols} FROM fact_items",  # noqa: S608
    )

    snapshot: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = row["id"]
        snapshot[item_id] = {
            "fact_id": row["fact_id"],
            **{col: row[col] for col in TRACKED_COLUMNS},
        }

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_FILE.write_text(json.dumps(snapshot, indent=2, default=str))
    logger.info("Snapshot captured: %d fact_items", len(snapshot))
    return len(snapshot)


def load_snapshot() -> dict[str, dict[str, Any]] | None:
    """Load snapshot from disk.  Returns None if no snapshot exists."""
    if not SNAPSHOT_FILE.exists():
        return None
    data: dict[str, dict[str, Any]] = json.loads(SNAPSHOT_FILE.read_text())
    return data


def delete_snapshot() -> bool:
    """Delete the snapshot file.  Returns True if it existed."""
    if SNAPSHOT_FILE.exists():
        SNAPSHOT_FILE.unlink()
        return True
    return False


def detect_changes(
    db: DatabaseManager,
    snapshot: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Diff current DB state against a snapshot.

    Returns a list of change dicts:
        {item_id, item_name, fact_id, field, old_value, new_value}
    """
    cols = ", ".join(TRACKED_COLUMNS)
    rows = db.fetchall(
        f"SELECT id, fact_id, name, {cols} FROM fact_items",  # noqa: S608
    )

    current: dict[str, dict[str, Any]] = {}
    for row in rows:
        current[row["id"]] = dict(row)

    changes: list[dict[str, Any]] = []

    for item_id, snap_data in snapshot.items():
        cur_data = current.get(item_id)
        if cur_data is None:
            continue  # item deleted — not our concern

        for col in TRACKED_COLUMNS:
            old = _normalize_value(snap_data.get(col))
            new = _normalize_value(cur_data.get(col))
            if old != new:
                changes.append(
                    {
                        "item_id": item_id,
                        "item_name": cur_data.get("name", ""),
                        "fact_id": snap_data.get("fact_id", ""),
                        "field": col,
                        "old_value": snap_data.get(col),
                        "new_value": cur_data.get(col),
                    }
                )

    return changes


def apply_changes(
    db: DatabaseManager,
    changes: list[dict[str, Any]],
    source: str = "tableplus",
) -> int:
    """Record correction_events for detected changes and propagate.

    Does NOT re-write fact_item rows (already written by external tool).
    Stamps enrichment_source=user_confirmed + enrichment_confidence=1.0
    on changed items.

    Returns:
        Number of correction events recorded.
    """
    from alibi.services.correction import _propagate_to_siblings, _record_correction

    recorded = 0
    stamped_ids: set[str] = set()

    for change in changes:
        _record_correction(
            db,
            entity_type="fact_item",
            entity_id=change["item_id"],
            field=change["field"],
            old_value=change["old_value"],
            new_value=change["new_value"],
            source=source,
        )
        recorded += 1

        # Stamp provenance once per item
        if change["item_id"] not in stamped_ids:
            with db.transaction() as cursor:
                cursor.execute(
                    "UPDATE fact_items SET enrichment_source = ?, "
                    "enrichment_confidence = ? WHERE id = ?",
                    ("user_confirmed", 1.0, change["item_id"]),
                )
            stamped_ids.add(change["item_id"])

        # Sibling propagation for brand/category
        if change["field"] in ("brand", "category") and change["new_value"]:
            try:
                row = db.fetchone(
                    "SELECT name_normalized, name FROM fact_items WHERE id = ?",
                    (change["item_id"],),
                )
                if row:
                    name_norm = row["name_normalized"] or row["name"]
                    if name_norm:
                        _propagate_to_siblings(
                            db,
                            change["item_id"],
                            name_norm,
                            {change["field"]: change["new_value"]},
                        )
            except Exception as e:
                logger.debug("Sibling propagation skipped: %s", e)

    return recorded
