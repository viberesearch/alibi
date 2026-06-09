"""Export endpoints for data extraction."""

from __future__ import annotations

import csv
import io
import json
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse

from alibi.api.deps import get_database, require_user
from alibi.db.connection import DatabaseManager
from alibi.services import list_documents, list_fact_items_with_fact, list_facts

router = APIRouter()


@router.get("/transactions/csv")
async def export_transactions_csv(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
) -> StreamingResponse:
    """Export facts (transactions) as CSV."""
    filters: dict[str, Any] = {}
    if date_from:
        filters["date_from"] = date_from
    if date_to:
        filters["date_to"] = date_to

    result = list_facts(db, filters=filters, offset=0, limit=100000)
    facts = result["facts"]

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "id",
            "fact_type",
            "vendor",
            "amount",
            "currency",
            "date",
            "status",
        ]
    )
    for fact in facts:
        writer.writerow(
            [
                fact["id"],
                fact["fact_type"],
                fact["vendor"],
                fact["total_amount"],
                fact["currency"],
                fact["event_date"],
                fact["status"],
            ]
        )

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=transactions.csv"},
    )


@router.get("/line-items/csv")
async def export_line_items_csv(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    name: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    brand: Optional[str] = Query(None),
    vendor: Optional[str] = Query(None),
    vendor_key: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    datetime_from: Optional[str] = Query(None),
    datetime_to: Optional[str] = Query(None),
    price_min: Optional[float] = Query(None),
    price_max: Optional[float] = Query(None),
) -> StreamingResponse:
    """Export fact items (line items) as CSV, filterable along every axis."""
    filters: dict[str, Any] = {
        k: v
        for k, v in {
            "name": name,
            "category": category,
            "brand": brand,
            "vendor": vendor,
            "vendor_key": vendor_key,
            "currency": currency,
            "country": country,
            "date_from": date_from,
            "date_to": date_to,
            "datetime_from": datetime_from,
            "datetime_to": datetime_to,
            "price_min": price_min,
            "price_max": price_max,
        }.items()
        if v is not None
    }

    rows = list_fact_items_with_fact(db, filters=filters)

    output = io.StringIO()
    writer = csv.writer(output)
    cols = [
        "id",
        "name",
        "quantity",
        "unit_price",
        "total_price",
        "category",
        "brand",
        "vendor",
        "vendor_key",
        "event_date",
        "event_time",
        "currency",
        "country",
    ]
    writer.writerow(cols)
    for row in rows:
        writer.writerow([row.get(c) for c in cols])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=line_items.csv"},
    )


@router.get("/artifacts/json")
async def export_artifacts_json(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
) -> list[dict[str, Any]]:
    """Export documents as JSON."""
    filters: dict[str, Any] = {}
    if date_from:
        filters["date_from"] = date_from
    if date_to:
        filters["date_to"] = date_to

    result = list_documents(db, filters=filters, offset=0, limit=100000)
    documents = result["documents"]

    return [
        {
            "id": doc["id"],
            "file_path": doc["file_path"],
            "file_hash": doc["file_hash"],
            "created_at": doc["created_at"],
        }
        for doc in documents
    ]


# ---------------------------------------------------------------------------
# Masked export (tiered disclosure)
# ---------------------------------------------------------------------------


@router.get("/masked/transactions")
async def export_masked_transactions(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    tier: int = Query(2, ge=0, le=4, description="Disclosure tier (0=hidden, 4=exact)"),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
) -> dict[str, Any]:
    """Export transactions with tier-based masking applied.

    Tier 0: amounts hidden, vendor=category only, dates=month-year.
    Tier 1: amounts rounded, vendor=category, dates=1st of month.
    Tier 2: exact amounts/dates, vendor visible, no line items.
    Tier 3: includes line items.
    Tier 4: full provenance (unmasked).
    """
    from alibi.db.models import Tier
    from alibi.masking.service import MaskingService

    filters: dict[str, Any] = {}
    if date_from:
        filters["date_from"] = date_from
    if date_to:
        filters["date_to"] = date_to

    result = list_facts(db, filters=filters, offset=0, limit=100000)
    facts = result["facts"]

    masking_svc = MaskingService()
    tier_enum = Tier(str(tier))
    masked = masking_svc.mask_for_tier(facts, tier_enum)

    return {"tier": tier, "total": len(masked), "facts": masked}


# ---------------------------------------------------------------------------
# Anonymized export (privacy-preserving)
# ---------------------------------------------------------------------------


@router.post("/anonymized")
async def export_anonymized(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    level: str = Query(
        "pseudonymized",
        description="Anonymization level: categories_only, pseudonymized, statistical",
    ),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
) -> JSONResponse:
    """Export facts with privacy-preserving anonymization.

    - categories_only: Only categories visible, no names/amounts/dates.
    - pseudonymized: Consistent fake names, shifted amounts/dates. Reversible locally.
    - statistical: Only aggregates, no individual records.

    Returns the anonymized data and (for pseudonymized) a key file that
    enables local restoration. The key never leaves the response.
    """
    from alibi.anonymization.exporter import (
        AnonymizationLevel,
        anonymize_export,
    )

    filters: dict[str, Any] = {}
    if date_from:
        filters["date_from"] = date_from
    if date_to:
        filters["date_to"] = date_to

    result = list_facts(db, filters=filters, offset=0, limit=100000)
    facts = result["facts"]

    # Build items_by_fact lookup
    items_by_fact: dict[str, list[dict[str, Any]]] = {}
    all_items = list_fact_items_with_fact(db, filters=filters)
    for item in all_items:
        fid = item.get("fact_id", "")
        items_by_fact.setdefault(fid, []).append(item)

    anon_level = AnonymizationLevel(level)
    anonymized_data, key = anonymize_export(facts, items_by_fact, anon_level)

    response: dict[str, Any] = {
        "level": level,
        "total": len(anonymized_data),
        "data": anonymized_data,
    }

    # Include restoration key only for pseudonymized (the only reversible level)
    if anon_level == AnonymizationLevel.PSEUDONYMIZED:
        response["restoration_key"] = json.loads(key.to_json())

    return JSONResponse(content=response)
