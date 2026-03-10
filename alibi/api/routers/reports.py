"""Report generation endpoints."""

from __future__ import annotations

from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from alibi.api.deps import get_database, require_user
from alibi.db.connection import DatabaseManager
from alibi.services import (
    monthly_report as svc_monthly_report,
    spending_analysis as svc_spending_analysis,
)

router = APIRouter()


@router.get("/monthly/{year}/{month}")
async def monthly_report(
    year: int,
    month: int,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Generate monthly spending report."""
    return svc_monthly_report(db, year, month)


@router.get("/spending")
async def spending_analysis(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    group_by: str = Query("month", pattern="^(day|week|month|vendor|category)$"),
) -> dict[str, Any]:
    """Analyze spending patterns."""
    filters: dict[str, Any] = {}
    if date_from:
        filters["date_from"] = date_from
    if date_to:
        filters["date_to"] = date_to

    try:
        return svc_spending_analysis(db, group_by, filters)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
