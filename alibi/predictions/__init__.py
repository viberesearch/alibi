"""MindsDB-backed predictors for spending forecast and category inference."""

from alibi.predictions.category import (
    classify,
    prepare_category_training_data,
    train_category_classifier,
)
from alibi.predictions.client import MindsDBClient, MindsDBError
from alibi.predictions.spending import (
    forecast,
    prepare_spending_training_data,
    train_spending_forecast,
)

__all__ = [
    "MindsDBClient",
    "MindsDBError",
    "classify",
    "forecast",
    "prepare_category_training_data",
    "prepare_spending_training_data",
    "train_category_classifier",
    "train_spending_forecast",
]
