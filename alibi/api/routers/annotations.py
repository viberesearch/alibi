"""Annotation management API endpoints.

Provides REST endpoints for:
- Adding annotations to facts
- Listing annotations on a fact (with optional type filter)
- Updating annotation value or metadata
- Deleting annotations
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from alibi.api.deps import get_database, require_user
from alibi.db.connection import DatabaseManager
from alibi.services import annotation

router = APIRouter()


class AnnotateRequest(BaseModel):
    """Request body for adding an annotation."""

    annotation_type: str
    key: str
    value: str
    metadata: Optional[dict[str, Any]] = None
    source: str = "user"


class UpdateAnnotationRequest(BaseModel):
    """Request body for updating an annotation."""

    value: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


@router.post("/facts/{fact_id}", status_code=201)
async def annotate_fact(
    fact_id: str,
    request: AnnotateRequest,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Add an annotation to a fact."""
    annotation_id = annotation.annotate(
        db,
        target_type="fact",
        target_id=fact_id,
        annotation_type=request.annotation_type,
        key=request.key,
        value=request.value,
        metadata=request.metadata,
        source=request.source,
    )
    return {"id": annotation_id, "status": "created"}


@router.get("/facts/{fact_id}")
async def get_fact_annotations(
    fact_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    annotation_type: Optional[str] = Query(
        None, description="Filter by annotation type"
    ),
) -> list[dict[str, Any]]:
    """Get all annotations on a fact."""
    return annotation.get_annotations(
        db,
        target_type="fact",
        target_id=fact_id,
        annotation_type=annotation_type,
    )


@router.put("/{annotation_id}")
async def update_annotation_endpoint(
    annotation_id: str,
    request: UpdateAnnotationRequest,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Update an annotation's value or metadata."""
    updated = annotation.update_annotation(
        db,
        annotation_id=annotation_id,
        value=request.value,
        metadata=request.metadata,
    )
    if not updated:
        raise HTTPException(
            status_code=404, detail=f"Annotation not found: {annotation_id}"
        )
    return {"id": annotation_id, "status": "updated"}


@router.delete("/{annotation_id}", status_code=204)
async def delete_annotation_endpoint(
    annotation_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> None:
    """Delete an annotation."""
    deleted = annotation.delete_annotation(db, annotation_id=annotation_id)
    if not deleted:
        raise HTTPException(
            status_code=404, detail=f"Annotation not found: {annotation_id}"
        )
