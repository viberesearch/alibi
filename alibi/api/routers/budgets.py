"""Budget endpoints for scenario CRUD and comparison."""

from __future__ import annotations

import uuid
from datetime import date
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from alibi.api.deps import get_database, require_user
from alibi.budgets.service import BudgetService
from alibi.db.connection import DatabaseManager
from alibi.db.models import DataType

router = APIRouter()


class CreateScenarioRequest(BaseModel):
    name: str
    description: Optional[str] = None
    data_type: str = "target"
    parent_id: Optional[str] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None


class CreateEntryRequest(BaseModel):
    category: str
    amount: float
    currency: str = "EUR"
    period: str
    note: Optional[str] = None


@router.get("/scenarios")
async def list_scenarios(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    space_id: str = Query("default"),
) -> list[dict[str, Any]]:
    """List all budget scenarios."""
    svc = BudgetService(db)
    scenarios = svc.list_scenarios(space_id)
    return [s.model_dump(mode="json") for s in scenarios]


@router.post("/scenarios")
async def create_scenario(
    body: CreateScenarioRequest,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    space_id: str = Query("default"),
) -> dict[str, Any]:
    """Create a new budget scenario."""
    from alibi.budgets.models import BudgetScenario

    scenario = BudgetScenario(
        id=str(uuid.uuid4()),
        space_id=space_id,
        name=body.name,
        description=body.description,
        data_type=DataType(body.data_type),
        parent_id=body.parent_id,
        period_start=(
            date.fromisoformat(body.period_start) if body.period_start else None
        ),
        period_end=date.fromisoformat(body.period_end) if body.period_end else None,
    )
    svc = BudgetService(db)
    svc.create_scenario(scenario)
    return scenario.model_dump(mode="json")


@router.get("/scenarios/{scenario_id}")
async def get_scenario(
    scenario_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Get a budget scenario by ID."""
    svc = BudgetService(db)
    scenario = svc.get_scenario(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return scenario.model_dump(mode="json")


@router.delete("/scenarios/{scenario_id}")
async def delete_scenario(
    scenario_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, str]:
    """Delete a budget scenario and its entries."""
    svc = BudgetService(db)
    if not svc.delete_scenario(scenario_id):
        raise HTTPException(status_code=404, detail="Scenario not found")
    return {"status": "deleted"}


@router.get("/scenarios/{scenario_id}/entries")
async def list_entries(
    scenario_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    period: Optional[str] = Query(None, description="Filter by period (YYYY-MM)"),
) -> list[dict[str, Any]]:
    """List budget entries for a scenario."""
    svc = BudgetService(db)
    entries = svc.get_entries(scenario_id, period)
    return [e.model_dump(mode="json") for e in entries]


@router.post("/scenarios/{scenario_id}/entries")
async def add_entry(
    scenario_id: str,
    body: CreateEntryRequest,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Add a budget entry to a scenario."""
    from decimal import Decimal

    from alibi.budgets.models import BudgetEntry

    entry = BudgetEntry(
        id=str(uuid.uuid4()),
        scenario_id=scenario_id,
        category=body.category,
        amount=Decimal(str(body.amount)),
        currency=body.currency,
        period=body.period,
        note=body.note,
    )
    svc = BudgetService(db)
    svc.add_entry(entry)
    return entry.model_dump(mode="json")


@router.get("/scenarios/{scenario_id}/actual")
async def get_actual_spending(
    scenario_id: str,
    period: str = Query(..., description="Period (YYYY-MM)"),
    db: Annotated[DatabaseManager, Depends(get_database)] = None,  # type: ignore[assignment]
    user: Annotated[dict[str, Any], Depends(require_user)] = None,  # type: ignore[assignment]
) -> list[dict[str, Any]]:
    """Get actual spending from transactions for a period."""
    svc = BudgetService(db)
    scenario = svc.get_scenario(scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Scenario not found")
    entries = svc.get_actual_spending(scenario.space_id, period)
    return [e.model_dump(mode="json") for e in entries]


@router.get("/compare")
async def compare_scenarios(
    base_id: str = Query(..., description="Base scenario ID"),
    compare_id: str = Query(..., description="Comparison scenario ID"),
    period: Optional[str] = Query(None, description="Period filter (YYYY-MM)"),
    db: Annotated[DatabaseManager, Depends(get_database)] = None,  # type: ignore[assignment]
    user: Annotated[dict[str, Any], Depends(require_user)] = None,  # type: ignore[assignment]
) -> list[dict[str, Any]]:
    """Compare two budget scenarios, returning per-category variance."""
    svc = BudgetService(db)
    comparisons = svc.compare(base_id, compare_id, period)
    return [c.model_dump(mode="json") for c in comparisons]
