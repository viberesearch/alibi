"""Category inference predictor.

Trains a classification model on existing categorized items,
then predicts categories for uncategorized transactions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager
    from alibi.predictions.client import MindsDBClient

logger = logging.getLogger(__name__)

MODEL_NAME = "alibi_category_classifier"
TABLE_NAME = "alibi_category_training"

# Minimum examples per category for meaningful training
_MIN_EXAMPLES_PER_CATEGORY = 5
_MIN_TOTAL_EXAMPLES = 50


def prepare_category_training_data(
    db: "DatabaseManager",
) -> Any:
    """Build a training DataFrame from categorized fact_items.

    Joins fact_items (with category) to facts (for vendor, date) producing:
    vendor_name, item_name, amount, category.

    Returns:
        pandas DataFrame ready for MindsDB upload.
    """
    import pandas as pd  # type: ignore[import-untyped]

    from alibi.db import v2_store

    facts = v2_store.list_facts(db, fact_type="purchase", limit=50000)
    if not facts:
        return pd.DataFrame(columns=["vendor_name", "item_name", "amount", "category"])

    rows: list[dict[str, Any]] = []
    for f in facts:
        vendor = f.get("vendor") or ""
        items = v2_store.get_fact_items(db, f["id"])
        for item in items:
            cat = item.get("category")
            if not cat:
                continue
            name = item.get("name_normalized") or item.get("name") or ""
            price = item.get("total_price") or item.get("unit_price") or 0
            rows.append(
                {
                    "vendor_name": vendor,
                    "item_name": name,
                    "amount": float(price),
                    "category": cat,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["vendor_name", "item_name", "amount", "category"])

    df = pd.DataFrame(rows)

    # Filter out categories with too few examples
    cat_counts = df["category"].value_counts()
    valid_cats = cat_counts[cat_counts >= _MIN_EXAMPLES_PER_CATEGORY].index
    df = df[df["category"].isin(valid_cats)]

    logger.info(
        "Prepared category data: %d rows, %d categories",
        len(df),
        df["category"].nunique(),
    )
    return df


def train_category_classifier(
    client: "MindsDBClient",
    db: "DatabaseManager",
) -> str:
    """Train a category classification model.

    Args:
        client: MindsDB client.
        db: Database manager for reading training data.

    Returns:
        Model status after training completes.

    Raises:
        ValueError: If insufficient training data.
    """
    df = prepare_category_training_data(db)
    if len(df) < _MIN_TOTAL_EXAMPLES:
        raise ValueError(
            f"Insufficient training data: {len(df)} rows "
            f"(minimum {_MIN_TOTAL_EXAMPLES})"
        )

    # Upload training data
    client.upload_dataframe(TABLE_NAME, df)

    # Create classification model (Lightwood auto-detects column types)
    client.create_model(
        name=MODEL_NAME,
        predict="category",
        from_table=f"files.`{TABLE_NAME}`",
    )

    return client.wait_model_ready(MODEL_NAME)


def classify(
    client: "MindsDBClient",
    vendor_name: str,
    item_name: str,
    amount: float,
) -> dict[str, Any]:
    """Predict category for a single item.

    Args:
        client: MindsDB client.
        vendor_name: Vendor/store name.
        item_name: Item/product description.
        amount: Transaction or item amount.

    Returns:
        Dict with category, category_confidence, and input echo.
    """
    results = client.predict(
        MODEL_NAME,
        {
            "vendor_name": vendor_name,
            "item_name": item_name,
            "amount": amount,
        },
    )
    if not results:
        return {
            "vendor_name": vendor_name,
            "item_name": item_name,
            "amount": amount,
            "category": None,
            "category_confidence": 0.0,
        }

    result = results[0]
    return {
        "vendor_name": vendor_name,
        "item_name": item_name,
        "amount": amount,
        "category": result.get("category"),
        "category_confidence": result.get("category_confidence", 0.0),
    }


def classify_batch(
    client: "MindsDBClient",
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Predict categories for multiple items.

    Args:
        client: MindsDB client.
        items: List of dicts with vendor_name, item_name, amount.

    Returns:
        List of prediction dicts.
    """
    if not items:
        return []

    results = client.predict(MODEL_NAME, items)
    return [
        {
            "vendor_name": r.get("vendor_name", ""),
            "item_name": r.get("item_name", ""),
            "amount": r.get("amount", 0),
            "category": r.get("category"),
            "category_confidence": r.get("category_confidence", 0.0),
        }
        for r in results
    ]
