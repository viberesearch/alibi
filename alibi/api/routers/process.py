"""Document processing endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from alibi.api.deps import get_database, require_user
from alibi.db.connection import DatabaseManager
from alibi.db.models import DocumentType
from alibi.processing.folder_router import FolderContext
from alibi.services import ingestion

router = APIRouter()

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf", ".tiff", ".tif", ".bmp", ".webp"}

_TYPE_MAP: dict[str, DocumentType] = {
    "receipt": DocumentType.RECEIPT,
    "invoice": DocumentType.INVOICE,
    "payment": DocumentType.PAYMENT_CONFIRMATION,
    "statement": DocumentType.STATEMENT,
    "warranty": DocumentType.WARRANTY,
    "contract": DocumentType.CONTRACT,
}

_VALID_TYPES = "|".join(_TYPE_MAP.keys())


class ProcessResponse(BaseModel):
    """Response model for document processing."""

    success: bool
    document_id: Optional[str] = None
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    vendor: Optional[str] = None
    amount: Optional[str] = None
    date: Optional[str] = None
    document_type: Optional[str] = None
    items_count: int = 0
    error: Optional[str] = None


def _apply_map_url(ctx: FolderContext, map_url: Optional[str]) -> None:
    """Parse a map URL and set lat/lng on a FolderContext."""
    if not map_url:
        return
    from alibi.utils.map_url import parse_map_url

    parsed = parse_map_url(map_url)
    if parsed:
        ctx.map_url = map_url.strip()
        ctx.lat = float(parsed["lat"])  # type: ignore[arg-type]
        ctx.lng = float(parsed["lng"])  # type: ignore[arg-type]


def _build_folder_context(doc_type_str: Optional[str]) -> Optional[FolderContext]:
    """Build a FolderContext from a type string, or None for auto-detect."""
    if doc_type_str is None:
        return None
    doc_type = _TYPE_MAP.get(doc_type_str)
    if doc_type is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid type '{doc_type_str}'. "
                f"Valid values: {', '.join(_TYPE_MAP.keys())}"
            ),
        )
    return FolderContext(doc_type=doc_type)


def _result_to_response(result: Any) -> dict[str, Any]:
    """Convert a ProcessingResult to a response dict."""
    resp: dict[str, Any] = {
        "success": result.success,
        "document_id": result.document_id,
        "is_duplicate": result.is_duplicate,
        "duplicate_of": result.duplicate_of,
        "error": result.error,
        "items_count": len(result.line_items) if result.line_items else 0,
    }
    if result.extracted_data:
        resp["vendor"] = result.extracted_data.get("vendor")
        total = result.extracted_data.get("total")
        resp["amount"] = str(total) if total is not None else None
        resp["date"] = result.extracted_data.get("date")
        resp["document_type"] = result.extracted_data.get("document_type")
    return resp


@router.post("", response_model=ProcessResponse)
async def process_single(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    file: UploadFile = File(...),
    type: Optional[str] = Query(
        None,
        description=(
            "Optional document type hint. "
            "One of: receipt, invoice, payment, statement, warranty, contract."
        ),
        pattern=f"^({_VALID_TYPES})$",
    ),
    map_url: Optional[str] = Query(
        None, description="Google Maps URL to associate with this document"
    ),
) -> ProcessResponse:
    """Process a single uploaded document file."""
    folder_context = _build_folder_context(type)
    if folder_context is None:
        folder_context = FolderContext()
    folder_context.source = "api"
    folder_context.user_id = user["id"]
    _apply_map_url(folder_context, map_url)

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    data = await file.read()
    if len(data) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    filename = file.filename or "upload.bin"
    result = ingestion.process_bytes(db, data, filename, folder_context=folder_context)
    return ProcessResponse(**_result_to_response(result))


@router.post("/batch", response_model=list[ProcessResponse])
async def process_batch(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    files: list[UploadFile] = File(...),
    type: Optional[str] = Query(
        None,
        description=(
            "Optional document type hint shared across all files. "
            "One of: receipt, invoice, payment, statement, warranty, contract."
        ),
        pattern=f"^({_VALID_TYPES})$",
    ),
    map_url: Optional[str] = Query(
        None, description="Google Maps URL to associate with these documents"
    ),
) -> list[ProcessResponse]:
    """Process multiple uploaded document files."""
    folder_context = _build_folder_context(type)
    if folder_context is None:
        folder_context = FolderContext()
    folder_context.source = "api"
    folder_context.user_id = user["id"]
    _apply_map_url(folder_context, map_url)

    responses: list[ProcessResponse] = []
    for upload in files:
        ext = Path(upload.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

        data = await upload.read()
        if len(data) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="File too large (max 50MB)")

        filename = upload.filename or "upload.bin"
        result = ingestion.process_bytes(
            db, data, filename, folder_context=folder_context
        )
        responses.append(ProcessResponse(**_result_to_response(result)))
    return responses


@router.post("/group", response_model=ProcessResponse)
async def process_group(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    files: list[UploadFile] = File(...),
    type: Optional[str] = Query(
        None,
        description=(
            "Optional document type hint. "
            "One of: receipt, invoice, payment, statement, warranty, contract."
        ),
        pattern=f"^({_VALID_TYPES})$",
    ),
    map_url: Optional[str] = Query(
        None, description="Google Maps URL to associate with this document"
    ),
) -> ProcessResponse:
    """Process multiple files as pages of a single document.

    All uploaded files are treated as ordered pages of one document and
    persisted to a subfolder in the inbox. They are sent to the LLM
    together in a single call, producing a unified extraction and a
    single fact.
    """
    if not files:
        raise HTTPException(status_code=422, detail="At least one file is required.")

    folder_context = _build_folder_context(type)
    if folder_context is None:
        folder_context = FolderContext()
    folder_context.source = "api"
    folder_context.user_id = user["id"]
    _apply_map_url(folder_context, map_url)

    pages: list[tuple[bytes, str]] = []
    for i, upload in enumerate(files):
        ext = Path(upload.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

        data = await upload.read()
        if len(data) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="File too large (max 50MB)")

        filename = upload.filename or f"page_{i}.bin"
        pages.append((data, filename))

    saved_paths = ingestion.persist_upload_group(pages, folder_context)

    result = ingestion.process_document_group(
        db, saved_paths, folder_context=folder_context
    )
    return ProcessResponse(**_result_to_response(result))
