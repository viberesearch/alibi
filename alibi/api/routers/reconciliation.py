"""Item ↔ payment reconciliation endpoint.

Classifies each transaction by which layers cover it (matched / items_only /
payment_only / empty) and reconciles the fact total against the line-item sum
and the normalised payment amount. Filters reuse the fact-level axes of the A
item filter; ``coverage`` narrows the result to one class (the payment_only /
items_only worklists).
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from alibi.api.deps import get_database, require_user
from alibi.db.connection import DatabaseManager
from alibi.services.reconciliation import _COVERAGE_CLASSES, reconcile

router = APIRouter()


@router.get("")
async def get_reconciliation(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    vendor: Optional[str] = Query(None),
    vendor_key: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    datetime_from: Optional[str] = Query(None),
    datetime_to: Optional[str] = Query(None),
    coverage: Optional[str] = Query(
        None,
        description="Restrict to one coverage class: "
        "matched | items_only | payment_only | empty",
    ),
) -> dict[str, Any]:
    """Overlay the item and payment layers; classify and reconcile each fact.

    Returns ``{summary, transactions}``. The summary counts the returned
    (filtered) set, so requesting ``coverage=payment_only`` yields the
    no-receipt worklist with a summary scoped to it.
    """
    if coverage is not None and coverage not in _COVERAGE_CLASSES:
        raise HTTPException(
            status_code=422,
            detail=f"coverage must be one of {sorted(_COVERAGE_CLASSES)}",
        )

    filters: dict[str, Any] = {
        k: v
        for k, v in {
            "vendor": vendor,
            "vendor_key": vendor_key,
            "currency": currency,
            "country": country,
            "date_from": date_from,
            "date_to": date_to,
            "datetime_from": datetime_from,
            "datetime_to": datetime_to,
            "coverage": coverage,
        }.items()
        if v is not None
    }

    return reconcile(db, filters=filters)
