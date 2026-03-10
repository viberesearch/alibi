"""Tests for alibi.predictions.spending — spending forecast predictor."""

from __future__ import annotations

import os
from datetime import date
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.predictions.spending import (
    MODEL_NAME,
    TABLE_NAME,
    forecast,
)

# Check pandas availability for DataFrame-dependent tests
_has_pandas = False
try:
    import pandas  # noqa: F401

    _has_pandas = True
except ImportError:
    pass

_skip_no_pandas = pytest.mark.skipif(not _has_pandas, reason="pandas not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.wait_model_ready.return_value = "complete"
    return client


def _make_fact(
    fact_id: str,
    vendor: str,
    total_amount: str,
    event_date: str,
) -> dict[str, Any]:
    return {
        "id": fact_id,
        "vendor": vendor,
        "total_amount": total_amount,
        "event_date": date.fromisoformat(event_date),
    }


def _make_mock_df(rows: int = 2) -> MagicMock:
    """Return a mock that quacks like a non-empty pandas DataFrame."""
    df = MagicMock()
    df.empty = False
    df.__len__ = MagicMock(return_value=rows)
    return df


def _make_empty_mock_df() -> MagicMock:
    df = MagicMock()
    df.empty = True
    df.__len__ = MagicMock(return_value=0)
    return df


# ---------------------------------------------------------------------------
# prepare_spending_training_data() — pandas-dependent
# ---------------------------------------------------------------------------


@_skip_no_pandas
class TestPrepareSpendingTrainingData:
    def test_empty_facts_returns_empty_dataframe(self, db: Any) -> None:
        from alibi.predictions.spending import prepare_spending_training_data

        with patch("alibi.predictions.spending.v2_store") as mock_store:
            mock_store.list_facts.return_value = []
            df = prepare_spending_training_data(db)

        assert df.empty
        assert list(df.columns) == ["date", "category", "amount"]

    def test_basic_fact_produces_row(self, db: Any) -> None:
        from alibi.predictions.spending import prepare_spending_training_data

        facts = [_make_fact("f1", "Lidl", "50.00", "2025-01-15")]
        items = [{"category": "food", "total_price": "50.00"}]

        with patch("alibi.predictions.spending.v2_store") as mock_store:
            mock_store.list_facts.return_value = facts
            mock_store.get_fact_items.return_value = items

            df = prepare_spending_training_data(db)

        assert not df.empty
        assert "food" in df["category"].values
        assert "2025-01-01" in df["date"].values

    def test_fact_without_items_uses_vendor(self, db: Any) -> None:
        from alibi.predictions.spending import prepare_spending_training_data

        facts = [_make_fact("f1", "Lidl", "30.00", "2025-02-10")]

        with patch("alibi.predictions.spending.v2_store") as mock_store:
            mock_store.list_facts.return_value = facts
            mock_store.get_fact_items.return_value = []

            df = prepare_spending_training_data(db)

        assert "Lidl" in df["category"].values

    def test_fact_without_vendor_or_items_uses_uncategorized(self, db: Any) -> None:
        from alibi.predictions.spending import prepare_spending_training_data

        facts = [
            {
                "id": "f1",
                "vendor": None,
                "total_amount": "25.00",
                "event_date": date(2025, 3, 1),
            }
        ]

        with patch("alibi.predictions.spending.v2_store") as mock_store:
            mock_store.list_facts.return_value = facts
            mock_store.get_fact_items.return_value = []

            df = prepare_spending_training_data(db)

        assert "uncategorized" in df["category"].values

    def test_fills_gaps_for_missing_months(self, db: Any) -> None:
        """Two categories across two months get gap-filled to 4 rows."""
        from alibi.predictions.spending import prepare_spending_training_data

        facts = [
            _make_fact("f1", "Lidl", "50.00", "2025-01-15"),
            _make_fact("f2", "Lidl", "60.00", "2025-03-15"),
        ]
        items_food = [{"category": "food", "total_price": "50.00"}]
        items_drink = [{"category": "drink", "total_price": "60.00"}]

        def get_items(db_arg: Any, fact_id: str) -> list[dict[str, Any]]:
            return items_food if fact_id == "f1" else items_drink

        with patch("alibi.predictions.spending.v2_store") as mock_store:
            mock_store.list_facts.return_value = facts
            mock_store.get_fact_items.side_effect = get_items

            df = prepare_spending_training_data(db)

        # 2 categories x 2 months (Jan and Mar) = 4 rows with gap fill
        assert len(df) == 4
        zero_rows = df[df["amount"] == 0.0]
        assert len(zero_rows) > 0

    def test_multi_category_splits_amount(self, db: Any) -> None:
        """When a fact has two categories, amount is split evenly."""
        from alibi.predictions.spending import prepare_spending_training_data

        facts = [_make_fact("f1", "Store", "100.00", "2025-01-10")]
        items = [
            {"category": "food", "total_price": "50.00"},
            {"category": "drink", "total_price": "50.00"},
        ]

        with patch("alibi.predictions.spending.v2_store") as mock_store:
            mock_store.list_facts.return_value = facts
            mock_store.get_fact_items.return_value = items

            df = prepare_spending_training_data(db)

        food_row = df[df["category"] == "food"]
        drink_row = df[df["category"] == "drink"]
        assert len(food_row) == 1
        assert len(drink_row) == 1
        assert abs(food_row.iloc[0]["amount"] - 50.0) < 0.01
        assert abs(drink_row.iloc[0]["amount"] - 50.0) < 0.01

    def test_facts_with_none_total_amount_are_skipped(self, db: Any) -> None:
        from alibi.predictions.spending import prepare_spending_training_data

        facts = [
            {
                "id": "f1",
                "vendor": "Store",
                "total_amount": None,
                "event_date": date(2025, 1, 10),
            }
        ]

        with patch("alibi.predictions.spending.v2_store") as mock_store:
            mock_store.list_facts.return_value = facts
            mock_store.get_fact_items.return_value = []

            df = prepare_spending_training_data(db)

        assert df.empty

    def test_facts_with_none_event_date_are_skipped(self, db: Any) -> None:
        from alibi.predictions.spending import prepare_spending_training_data

        facts = [
            {
                "id": "f1",
                "vendor": "Store",
                "total_amount": "50.00",
                "event_date": None,
            }
        ]

        with patch("alibi.predictions.spending.v2_store") as mock_store:
            mock_store.list_facts.return_value = facts
            mock_store.get_fact_items.return_value = []

            df = prepare_spending_training_data(db)

        assert df.empty

    def test_string_event_date_is_parsed(self, db: Any) -> None:
        from alibi.predictions.spending import prepare_spending_training_data

        facts = [
            {
                "id": "f1",
                "vendor": "Lidl",
                "total_amount": "20.00",
                "event_date": "2025-06-15",
            }
        ]

        with patch("alibi.predictions.spending.v2_store") as mock_store:
            mock_store.list_facts.return_value = facts
            mock_store.get_fact_items.return_value = []

            df = prepare_spending_training_data(db)

        assert "2025-06-01" in df["date"].values


# ---------------------------------------------------------------------------
# train_spending_forecast() — uses mocked DataFrame so no pandas needed
# ---------------------------------------------------------------------------


class TestTrainSpendingForecast:
    def test_train_uploads_and_creates_model(
        self, db: Any, mock_client: MagicMock
    ) -> None:
        from alibi.predictions.spending import train_spending_forecast

        fake_df = _make_mock_df(rows=5)

        with patch(
            "alibi.predictions.spending.prepare_spending_training_data",
            return_value=fake_df,
        ):
            result = train_spending_forecast(mock_client, db, window=6, horizon=3)

        mock_client.upload_dataframe.assert_called_once_with(TABLE_NAME, fake_df)
        mock_client.create_model.assert_called_once()
        create_kwargs = mock_client.create_model.call_args[1]
        assert create_kwargs["name"] == MODEL_NAME
        assert create_kwargs["predict"] == "amount"
        ts_opts = create_kwargs["timeseries_options"]
        assert ts_opts["order"] == "date"
        assert ts_opts["group"] == "category"
        assert ts_opts["window"] == 6
        assert ts_opts["horizon"] == 3
        mock_client.wait_model_ready.assert_called_once_with(MODEL_NAME)
        assert result == "complete"

    def test_train_empty_data_raises_value_error(
        self, db: Any, mock_client: MagicMock
    ) -> None:
        from alibi.predictions.spending import train_spending_forecast

        empty_df = _make_empty_mock_df()
        with patch(
            "alibi.predictions.spending.prepare_spending_training_data",
            return_value=empty_df,
        ):
            with pytest.raises(ValueError, match="No spending data"):
                train_spending_forecast(mock_client, db)


# ---------------------------------------------------------------------------
# forecast() — no pandas dependency
# ---------------------------------------------------------------------------


class TestForecast:
    def test_forecast_with_category_filter(self, mock_client: MagicMock) -> None:
        rows = [
            {
                "forecast_date": "2025-04-01",
                "category": "food",
                "forecast_amount": 130.0,
                "confidence": 0.8,
            },
            {
                "forecast_date": "2025-05-01",
                "category": "food",
                "forecast_amount": 140.0,
                "confidence": 0.75,
            },
            {
                "forecast_date": "2025-06-01",
                "category": "food",
                "forecast_amount": 150.0,
                "confidence": 0.7,
            },
        ]
        mock_client.predict_sql.return_value = rows

        result = forecast(mock_client, months=2, category="food")

        assert len(result) == 2
        assert all(r["category"] == "food" for r in result)

    def test_forecast_all_categories_grouped(self, mock_client: MagicMock) -> None:
        rows = [
            {
                "forecast_date": "2025-04-01",
                "category": "food",
                "forecast_amount": 100.0,
                "confidence": 0.8,
            },
            {
                "forecast_date": "2025-04-01",
                "category": "drink",
                "forecast_amount": 40.0,
                "confidence": 0.7,
            },
            {
                "forecast_date": "2025-05-01",
                "category": "food",
                "forecast_amount": 110.0,
                "confidence": 0.75,
            },
            {
                "forecast_date": "2025-05-01",
                "category": "drink",
                "forecast_amount": 45.0,
                "confidence": 0.65,
            },
        ]
        mock_client.predict_sql.return_value = rows

        result = forecast(mock_client, months=1)

        # With months=1, each category gets at most 1 row
        cats = {r["category"] for r in result}
        assert "food" in cats
        assert "drink" in cats
        assert len(result) == 2

    def test_forecast_sql_contains_table_and_model_names(
        self, mock_client: MagicMock
    ) -> None:
        mock_client.predict_sql.return_value = []

        forecast(mock_client, months=3)

        sql_arg = mock_client.predict_sql.call_args[0][0]
        assert TABLE_NAME in sql_arg
        assert MODEL_NAME in sql_arg

    def test_forecast_category_filter_in_sql(self, mock_client: MagicMock) -> None:
        mock_client.predict_sql.return_value = []

        forecast(mock_client, months=3, category="electronics")

        sql_arg = mock_client.predict_sql.call_args[0][0]
        assert "electronics" in sql_arg

    def test_forecast_empty_result(self, mock_client: MagicMock) -> None:
        mock_client.predict_sql.return_value = []
        result = forecast(mock_client, months=3)
        assert result == []
