"""Document (artifact) CRUD endpoints."""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from alibi.api.deps import (
    PaginationParams,
    get_database,
    require_user,
)
from alibi.db.connection import DatabaseManager
from alibi.services import (
    delete_document,
    get_document,
    get_document_line_items,
    list_documents,
)

router = APIRouter()


class ArtifactResponse(BaseModel):
    """Document response model."""

    id: str
    file_path: str
    file_hash: str
    raw_extraction: Optional[str] = None
    created_at: Optional[str] = None


class ArtifactCreate(BaseModel):
    """Document creation request."""

    file_path: str
    file_hash: str


@router.get("", response_model=dict[str, Any])
async def list_artifacts(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    pagination: Annotated[PaginationParams, Depends()],
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
) -> dict[str, Any]:
    """List documents with optional filters."""
    filters: dict[str, Any] = {}
    if date_from:
        filters["date_from"] = date_from
    if date_to:
        filters["date_to"] = date_to

    return list_documents(
        db,
        filters=filters,
        offset=pagination.offset,
        limit=pagination.per_page,
    )


@router.get("/{artifact_id}")
async def get_artifact(
    artifact_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Get a single document by ID."""
    doc = get_document(db, artifact_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    result = dict(doc)
    raw = result.get("raw_extraction")
    if raw:
        try:
            result["extracted_data"] = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            result["extracted_data"] = None
    return result


@router.post("", status_code=201)
async def create_artifact(
    data: ArtifactCreate,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Create a new document."""
    doc_id = str(uuid.uuid4())

    with db.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO documents (id, file_path, file_hash)
            VALUES (?, ?, ?)
            """,
            (doc_id, data.file_path, data.file_hash),
        )

    return {"id": doc_id, "status": "created"}


@router.get("/{artifact_id}/line-items")
async def get_artifact_line_items(
    artifact_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> list[dict[str, Any]]:
    """Get fact items linked to a document via the bundle/cloud chain."""
    if get_document(db, artifact_id) is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    return get_document_line_items(db, artifact_id)


@router.delete("/{artifact_id}", status_code=204)
async def delete_artifact(
    artifact_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> None:
    """Delete a document."""
    if not delete_document(db, artifact_id):
        raise HTTPException(status_code=404, detail="Artifact not found")
