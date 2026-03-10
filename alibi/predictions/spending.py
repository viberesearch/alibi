"""Spending forecast predictor.

Trains a time-series model on monthly spending aggregated by category,
then forecasts future spending amounts.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager
    from alibi.predictions.client import MindsDBClient

logger = logging.getLogger(__name__)

MODEL_NAME = "alibi_spending_forecast"
TABLE_NAME = "alibi_spending_monthly"


def prepare_spending_training_data(
    db: "DatabaseManager",
) -> Any:
    """Build a monthly spending DataFrame from facts.

    Aggregates purchase facts by (month, category) producing columns:
    date, category, amount.

    Returns:
        pandas DataFrame ready for MindsDB upload.
    """
    import pandas as pd  # type: ignore[import-untyped]

    from alibi.db import v2_store

    facts = v2_store.list_facts(db, fact_type="purchase", limit=50000)
    if not facts:
        return pd.DataFrame(columns=["date", "category", "amount"])

    # Aggregate by (year-month, category)
    monthly: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    for f in facts:
        if f.get("total_amount") is None:
            continue
        event_date = f.get("event_date")
        if isinstance(event_date, str):
            event_date = date.fromisoformat(event_date)
        if not event_date:
            continue

        month_key = f"{event_date.year}-{event_date.month:02d}-01"
        # Derive category from items, fall back to vendor
        fact_items = v2_store.get_fact_items(db, f["id"])
        categories: set[str] = set()
        for item in fact_items:
            cat = item.get("category")
            if cat:
                categories.add(cat)

        if not categories:
            categories = {f.get("vendor") or "uncategorized"}

        amount = Decimal(str(f["total_amount"]))
        # Split amount evenly across categories when multiple
        per_cat = float(amount) / len(categories)
        for cat in categories:
            monthly[(month_key, cat)] += Decimal(str(round(per_cat, 2)))

    if not monthly:
        return pd.DataFrame(columns=["date", "category", "amount"])

    rows = [
        {"date": k[0], "category": k[1], "amount": float(v)}
        for k, v in sorted(monthly.items())
    ]

    # Fill gaps: ensure every category has a row for every month
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    all_months = sorted(df["date"].unique())
    all_cats = sorted(df["category"].unique())
    full_index = pd.MultiIndex.from_product(
        [all_months, all_cats], names=["date", "category"]
    )
    df = df.set_index(["date", "category"]).reindex(full_index, fill_value=0.0)
    df = df.reset_index()

    logger.info(
        "Prepared spending data: %d rows, %d categories, %d months",
        len(df),
        len(all_cats),
        len(all_months),
    )
    return df


def train_spending_forecast(
    client: "MindsDBClient",
    db: "DatabaseManager",
    window: int = 6,
    horizon: int = 3,
) -> str:
    """Train a time-series spending forecast model.

    Args:
        client: MindsDB client.
        db: Database manager for reading training data.
        window: Number of past months to use per prediction.
        horizon: Number of future months to predict.

    Returns:
        Model status after training completes.
    """
    df = prepare_spending_training_data(db)
    if df.empty:
        raise ValueError("No spending data available for training")

    # Upload training data
    client.upload_dataframe(TABLE_NAME, df)

    # Create time-series model
    client.create_model(
        name=MODEL_NAME,
        predict="amount",
        from_table=f"files.`{TABLE_NAME}`",
        timeseries_options={
            "order": "date",
            "group": "category",
            "window": window,
            "horizon": horizon,
        },
    )

    return client.wait_model_ready(MODEL_NAME)


def forecast(
    client: "MindsDBClient",
    months: int = 3,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """Get spending forecast predictions.

    Uses SQL JOIN to provide historical context window and get future
    predictions from the time-series model.

    Args:
        client: MindsDB client.
        months: Number of months to forecast (up to model horizon).
        category: Optional category filter.

    Returns:
        List of prediction dicts with date, category, amount.
    """
    where_clause = ""
    if category:
        where_clause = f"AND t.category = '{category}'"

    sql = f"""
    SELECT m.date AS forecast_date,
           m.category,
           m.amount AS forecast_amount,
           m.amount_confidence AS confidence
    FROM files.`{TABLE_NAME}` AS t
    JOIN mindsdb.`{MODEL_NAME}` AS m
    WHERE t.date > (
        SELECT MAX(date) FROM files.`{TABLE_NAME}`
    )
    {where_clause}
    LIMIT {months * 50}
    """

    results = client.predict_sql(sql)

    # Filter to requested number of future months per category
    if category:
        return results[:months]

    # Group by category and take first N months per category
    from collections import defaultdict

    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        cat = row.get("category", "unknown")
        if len(by_cat[cat]) < months:
            by_cat[cat].append(row)

    flat: list[dict[str, Any]] = []
    for rows in by_cat.values():
        flat.extend(rows)
    return flat
