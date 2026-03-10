"""Annotation service.

Thin wrapper around annotations.store for use by all consumers.
"""

from __future__ import annotations

import logging
from typing import Any

from alibi.annotations.store import (
    add_annotation as _add_annotation,
    delete_annotation as _delete_annotation,
    get_annotations as _get_annotations,
    update_annotation as _update_annotation,
)
from alibi.db.connection import DatabaseManager
from alibi.services.events import EventType, event_bus

logger = logging.getLogger(__name__)


def annotate(
    db: DatabaseManager,
    target_type: str,
    target_id: str,
    annotation_type: str,
    key: str,
    value: str,
    metadata: dict[str, Any] | None = None,
    source: str = "user",
) -> str:
    """Add an annotation to an entity.

    Args:
        db: Database manager.
        target_type: Entity type (fact, fact_item, vendor, identity).
        target_id: ID of the target entity.
        annotation_type: Type of annotation (person, project, category, etc.).
        key: Annotation key (e.g., "bought_for", "project").
        value: Annotation value (e.g., "Maria", "Kitchen renovation").
        metadata: Optional structured data (split ratios, item lists, etc.).
        source: Origin of annotation (user, auto, inference).

    Returns:
        The annotation ID.
    """
    annotation_id = _add_annotation(
        db,
        annotation_type=annotation_type,
        target_type=target_type,
        target_id=target_id,
        key=key,
        value=value,
        metadata=metadata,
        source=source,
    )
    event_bus.emit(
        EventType.ANNOTATION_ADDED,
        {
            "annotation_id": annotation_id,
            "target_type": target_type,
            "target_id": target_id,
            "annotation_type": annotation_type,
            "key": key,
        },
    )
    return annotation_id


def get_annotations(
    db: DatabaseManager,
    target_type: str,
    target_id: str,
    annotation_type: str | None = None,
) -> list[dict[str, Any]]:
    """Get annotations for an entity.

    Args:
        db: Database manager.
        target_type: Entity type (fact, fact_item, vendor, identity).
        target_id: ID of the target entity.
        annotation_type: Optional filter by annotation type.

    Returns:
        List of annotation dicts ordered by created_at.
    """
    return _get_annotations(
        db,
        target_type=target_type,
        target_id=target_id,
        annotation_type=annotation_type,
    )


def update_annotation(
    db: DatabaseManager,
    annotation_id: str,
    value: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Update an annotation's value or metadata.

    At least one of value or metadata must be provided; if neither is
    provided the function returns False without touching the database.

    Args:
        db: Database manager.
        annotation_id: ID of the annotation to update.
        value: New value, or None to leave unchanged.
        metadata: New metadata dict, or None to leave unchanged.

    Returns:
        True if a row was updated, False otherwise.
    """
    return _update_annotation(
        db,
        annotation_id=annotation_id,
        value=value,
        metadata=metadata,
    )


def delete_annotation(db: DatabaseManager, annotation_id: str) -> bool:
    """Delete an annotation.

    Args:
        db: Database manager.
        annotation_id: ID of the annotation to delete.

    Returns:
        True if a row was deleted, False if the ID was not found.
    """
    return _delete_annotation(db, annotation_id)
