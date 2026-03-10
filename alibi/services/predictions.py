"""Service facade for MindsDB predictions.

Orchestrates predictor lifecycle (train, query, status) and provides
a single entry point for CLI, API, and MCP consumers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from alibi.config import get_config

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)


def _get_client() -> Any:
    """Create a MindsDB client from config.

    Returns:
        MindsDBClient instance.

    Raises:
        RuntimeError: If MindsDB is not enabled or unreachable.
    """
    from alibi.predictions.client import MindsDBClient

    config = get_config()
    if not config.mindsdb_enabled:
        raise RuntimeError("MindsDB is not enabled. Set ALIBI_MINDSDB_ENABLED=true")

    client = MindsDBClient(config.mindsdb_url)
    if not client.is_healthy():
        client.close()
        raise RuntimeError(f"MindsDB is not reachable at {config.mindsdb_url}")
    return client


# -- Training -------------------------------------------------------------


def train_spending_forecast(
    db: "DatabaseManager",
    window: int = 6,
    horizon: int = 3,
) -> dict[str, Any]:
    """Train the spending forecast model.

    Args:
        db: Database manager.
        window: Historical months per prediction.
        horizon: Future months to forecast.

    Returns:
        Dict with model name, status, and training params.
    """
    from alibi.predictions.spending import (
        MODEL_NAME,
        train_spending_forecast as _train,
    )

    with _get_client() as client:
        status = _train(client, db, window=window, horizon=horizon)
        return {
            "model": MODEL_NAME,
            "status": status,
            "window": window,
            "horizon": horizon,
        }


def train_category_classifier(
    db: "DatabaseManager",
) -> dict[str, Any]:
    """Train the category classifier model.

    Args:
        db: Database manager.

    Returns:
        Dict with model name and status.
    """
    from alibi.predictions.category import (
        MODEL_NAME,
        train_category_classifier as _train,
    )

    with _get_client() as client:
        status = _train(client, db)
        return {"model": MODEL_NAME, "status": status}


# -- Predictions -----------------------------------------------------------


def get_spending_forecast(
    db: "DatabaseManager",
    months: int = 3,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Get spending forecast predictions.

    Args:
        db: Database manager (unused but kept for interface consistency).
        months: Number of months to forecast.
        category: Optional category filter.

    Returns:
        List of forecast dicts.
    """
    from alibi.predictions.spending import forecast

    with _get_client() as client:
        return forecast(client, months=months, category=category)


def classify_category(
    db: "DatabaseManager",
    vendor_name: str,
    item_name: str,
    amount: float,
) -> dict[str, Any]:
    """Predict category for a single item.

    Args:
        db: Database manager (unused but kept for interface consistency).
        vendor_name: Vendor/store name.
        item_name: Item description.
        amount: Item or transaction amount.

    Returns:
        Dict with predicted category and confidence.
    """
    from alibi.predictions.category import classify

    with _get_client() as client:
        return classify(client, vendor_name, item_name, amount)


def classify_uncategorized(
    db: "DatabaseManager",
    limit: int = 100,
    min_confidence: float = 0.5,
) -> list[dict[str, Any]]:
    """Find uncategorized items and predict their categories.

    Args:
        db: Database manager.
        limit: Maximum items to process.
        min_confidence: Minimum confidence to include in results.

    Returns:
        List of dicts with item_id, predicted category, confidence.
    """
    from alibi.db import v2_store
    from alibi.predictions.category import classify_batch

    # Find items without categories
    items = v2_store.list_fact_items_uncategorized(db, limit=limit)
    if not items:
        return []

    # Build prediction input
    # Need fact vendor for each item
    input_data = []
    item_ids = []
    for item in items:
        fact = v2_store.inspect_fact(db, item["fact_id"])
        vendor = fact["vendor"] if fact else ""
        input_data.append(
            {
                "vendor_name": vendor,
                "item_name": item.get("name_normalized") or item.get("name", ""),
                "amount": float(item.get("total_price") or item.get("unit_price") or 0),
            }
        )
        item_ids.append(item["id"])

    with _get_client() as client:
        predictions = classify_batch(client, input_data)

    results = []
    for item_id, pred in zip(item_ids, predictions):
        conf = pred.get("category_confidence", 0.0)
        if conf >= min_confidence:
            results.append(
                {
                    "item_id": item_id,
                    "category": pred.get("category"),
                    "confidence": conf,
                    "vendor_name": pred.get("vendor_name", ""),
                    "item_name": pred.get("item_name", ""),
                }
            )

    logger.info(
        "Classified %d/%d uncategorized items (confidence >= %.2f)",
        len(results),
        len(items),
        min_confidence,
    )
    return results


# -- Status ----------------------------------------------------------------


def list_models(
    db: "DatabaseManager",
) -> list[dict[str, Any]]:
    """List all MindsDB predictor models and their status.

    Args:
        db: Database manager (unused but kept for interface consistency).

    Returns:
        List of model info dicts.
    """
    with _get_client() as client:
        models: list[dict[str, Any]] = client.list_models()
        return models


def get_model_status(
    db: "DatabaseManager",
    model_name: str,
) -> dict[str, Any]:
    """Get status of a specific model.

    Args:
        db: Database manager (unused but kept for interface consistency).
        model_name: Model name to check.

    Returns:
        Dict with model name and status.
    """
    with _get_client() as client:
        status = client.get_model_status(model_name)
        return {"model": model_name, "status": status}


def drop_model(
    db: "DatabaseManager",
    model_name: str,
) -> dict[str, str]:
    """Delete a predictor model.

    Args:
        db: Database manager (unused but kept for interface consistency).
        model_name: Model name to delete.

    Returns:
        Confirmation dict.
    """
    with _get_client() as client:
        client.drop_model(model_name)
        return {"model": model_name, "status": "deleted"}
