"""API router for MindsDB prediction endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from alibi.api.deps import get_database, require_user

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager

router = APIRouter()


class ClassifyRequest(BaseModel):
    vendor_name: str
    item_name: str
    amount: float


class TrainRequest(BaseModel):
    window: int = 6
    horizon: int = 3


# -- Training endpoints ----------------------------------------------------


@router.post("/train/forecast")
async def train_forecast(
    body: TrainRequest,
    db: Annotated["DatabaseManager", Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Train or retrain the spending forecast model."""
    from alibi.services.predictions import train_spending_forecast

    try:
        return train_spending_forecast(db, window=body.window, horizon=body.horizon)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/train/category")
async def train_category(
    db: Annotated["DatabaseManager", Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Train or retrain the category classifier model."""
    from alibi.services.predictions import train_category_classifier

    try:
        return train_category_classifier(db)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# -- Prediction endpoints -------------------------------------------------


@router.get("/forecast")
async def spending_forecast(
    db: Annotated["DatabaseManager", Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    months: int = Query(3, ge=1, le=24),
    category: str | None = Query(None),
) -> list[dict[str, Any]]:
    """Get spending forecast predictions."""
    from alibi.services.predictions import get_spending_forecast

    try:
        return get_spending_forecast(db, months=months, category=category)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/classify")
async def classify_item(
    body: ClassifyRequest,
    db: Annotated["DatabaseManager", Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Predict category for a single item."""
    from alibi.services.predictions import classify_category

    try:
        return classify_category(db, body.vendor_name, body.item_name, body.amount)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/classify/uncategorized")
async def classify_uncategorized(
    db: Annotated["DatabaseManager", Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    limit: int = Query(100, ge=1, le=1000),
    min_confidence: float = Query(0.5, ge=0.0, le=1.0),
) -> list[dict[str, Any]]:
    """Classify uncategorized items using the trained model."""
    from alibi.services.predictions import classify_uncategorized

    try:
        return classify_uncategorized(db, limit=limit, min_confidence=min_confidence)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# -- Status endpoints ------------------------------------------------------


@router.get("/models")
async def list_models(
    db: Annotated["DatabaseManager", Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> list[dict[str, Any]]:
    """List all predictor models and their status."""
    from alibi.services.predictions import list_models as _list

    try:
        return _list(db)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/models/{model_name}/status")
async def model_status(
    model_name: str,
    db: Annotated["DatabaseManager", Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Get status of a specific model."""
    from alibi.services.predictions import get_model_status

    try:
        return get_model_status(db, model_name)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/models/{model_name}")
async def delete_model(
    model_name: str,
    db: Annotated["DatabaseManager", Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, str]:
    """Delete a predictor model."""
    from alibi.services.predictions import drop_model

    try:
        return drop_model(db, model_name)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
