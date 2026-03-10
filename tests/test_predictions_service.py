"""Tests for alibi.services.predictions — service facade."""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

os.environ["ALIBI_TESTING"] = "1"

# The service does `from alibi.predictions.client import MindsDBClient` inside
# _get_client(), so we patch at the definition site.
_CLIENT_PATCH = "alibi.predictions.client.MindsDBClient"


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _mock_healthy_client() -> MagicMock:
    """Return a mock MindsDBClient that reports healthy."""
    client = MagicMock()
    client.is_healthy.return_value = True
    client.__enter__ = lambda s: s
    client.__exit__ = MagicMock(return_value=False)
    return client


@pytest.fixture
def healthy_client() -> MagicMock:
    return _mock_healthy_client()


@pytest.fixture
def enabled_config(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Set environment so MindsDB is enabled."""
    monkeypatch.setenv("ALIBI_MINDSDB_ENABLED", "true")
    monkeypatch.setenv("ALIBI_MINDSDB_URL", "http://test:47334")
    from alibi.config import reset_config

    reset_config()
    yield
    reset_config()


@pytest.fixture
def disabled_config(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Set environment so MindsDB is disabled."""
    monkeypatch.setenv("ALIBI_MINDSDB_ENABLED", "false")
    from alibi.config import reset_config

    reset_config()
    yield
    reset_config()


# ---------------------------------------------------------------------------
# _get_client() — internal helper
# ---------------------------------------------------------------------------


class TestGetClient:
    def test_raises_when_mindsdb_not_enabled(
        self, disabled_config: None, mock_db: MagicMock
    ) -> None:
        from alibi.services.predictions import _get_client

        with pytest.raises(RuntimeError, match="not enabled"):
            _get_client()

    def test_raises_when_mindsdb_unreachable(
        self, enabled_config: None, mock_db: MagicMock
    ) -> None:
        from alibi.services.predictions import _get_client

        unhealthy = MagicMock()
        unhealthy.is_healthy.return_value = False
        unhealthy.close.return_value = None

        with patch(_CLIENT_PATCH, return_value=unhealthy):
            with pytest.raises(RuntimeError, match="not reachable"):
                _get_client()

    def test_returns_client_when_healthy(
        self, enabled_config: None, mock_db: MagicMock
    ) -> None:
        from alibi.services.predictions import _get_client

        healthy = _mock_healthy_client()
        with patch(_CLIENT_PATCH, return_value=healthy):
            client = _get_client()

        assert client is not None


# ---------------------------------------------------------------------------
# train_spending_forecast()
# ---------------------------------------------------------------------------


class TestTrainSpendingForecastService:
    def test_delegates_to_predictions_spending(
        self, enabled_config: None, mock_db: MagicMock
    ) -> None:
        from alibi.predictions.spending import MODEL_NAME
        from alibi.services.predictions import train_spending_forecast

        with (
            patch(_CLIENT_PATCH, return_value=_mock_healthy_client()),
            patch(
                "alibi.predictions.spending.train_spending_forecast",
                return_value="complete",
            ),
        ):
            result = train_spending_forecast(mock_db, window=4, horizon=2)

        assert result["model"] == MODEL_NAME
        assert result["status"] == "complete"
        assert result["window"] == 4
        assert result["horizon"] == 2

    def test_propagates_value_error(
        self, enabled_config: None, mock_db: MagicMock
    ) -> None:
        from alibi.services.predictions import train_spending_forecast

        with (
            patch(_CLIENT_PATCH, return_value=_mock_healthy_client()),
            patch(
                "alibi.predictions.spending.train_spending_forecast",
                side_effect=ValueError("No data"),
            ),
        ):
            with pytest.raises(ValueError, match="No data"):
                train_spending_forecast(mock_db)


# ---------------------------------------------------------------------------
# train_category_classifier()
# ---------------------------------------------------------------------------


class TestTrainCategoryClassifierService:
    def test_delegates_to_predictions_category(
        self, enabled_config: None, mock_db: MagicMock
    ) -> None:
        from alibi.predictions.category import MODEL_NAME
        from alibi.services.predictions import train_category_classifier

        with (
            patch(_CLIENT_PATCH, return_value=_mock_healthy_client()),
            patch(
                "alibi.predictions.category.train_category_classifier",
                return_value="complete",
            ),
        ):
            result = train_category_classifier(mock_db)

        assert result["model"] == MODEL_NAME
        assert result["status"] == "complete"


# ---------------------------------------------------------------------------
# get_spending_forecast()
# ---------------------------------------------------------------------------


class TestGetSpendingForecastService:
    def test_delegates_to_spending_forecast(
        self, enabled_config: None, mock_db: MagicMock
    ) -> None:
        from alibi.services.predictions import get_spending_forecast

        fake_rows = [
            {
                "forecast_date": "2025-04-01",
                "category": "food",
                "forecast_amount": 100.0,
            }
        ]

        with (
            patch(_CLIENT_PATCH, return_value=_mock_healthy_client()),
            patch("alibi.predictions.spending.forecast", return_value=fake_rows),
        ):
            result = get_spending_forecast(mock_db, months=2, category="food")

        assert result == fake_rows


# ---------------------------------------------------------------------------
# classify_category()
# ---------------------------------------------------------------------------


class TestClassifyCategoryService:
    def test_delegates_to_category_classify(
        self, enabled_config: None, mock_db: MagicMock
    ) -> None:
        from alibi.services.predictions import classify_category

        fake_result = {
            "vendor_name": "Lidl",
            "item_name": "bread",
            "amount": 1.5,
            "category": "food",
            "category_confidence": 0.9,
        }

        with (
            patch(_CLIENT_PATCH, return_value=_mock_healthy_client()),
            patch("alibi.predictions.category.classify", return_value=fake_result),
        ):
            result = classify_category(mock_db, "Lidl", "bread", 1.5)

        assert result == fake_result


# ---------------------------------------------------------------------------
# list_models()
# ---------------------------------------------------------------------------


class TestListModelsService:
    def test_returns_model_list(self, enabled_config: None, mock_db: MagicMock) -> None:
        from alibi.services.predictions import list_models

        fake_models = [
            {
                "name": "spending_model",
                "status": "complete",
                "predict": "amount",
                "engine": "lightwood",
            }
        ]
        client = _mock_healthy_client()
        client.list_models.return_value = fake_models

        with patch(_CLIENT_PATCH, return_value=client):
            result = list_models(mock_db)

        assert result == fake_models
        client.list_models.assert_called_once()


# ---------------------------------------------------------------------------
# get_model_status()
# ---------------------------------------------------------------------------


class TestGetModelStatusService:
    def test_returns_status_dict(
        self, enabled_config: None, mock_db: MagicMock
    ) -> None:
        from alibi.services.predictions import get_model_status

        client = _mock_healthy_client()
        client.get_model_status.return_value = "training"

        with patch(_CLIENT_PATCH, return_value=client):
            result = get_model_status(mock_db, "my_model")

        assert result == {"model": "my_model", "status": "training"}
        client.get_model_status.assert_called_once_with("my_model")


# ---------------------------------------------------------------------------
# drop_model()
# ---------------------------------------------------------------------------


class TestDropModelService:
    def test_drops_and_returns_deleted(
        self, enabled_config: None, mock_db: MagicMock
    ) -> None:
        from alibi.services.predictions import drop_model

        client = _mock_healthy_client()
        client.drop_model.return_value = None

        with patch(_CLIENT_PATCH, return_value=client):
            result = drop_model(mock_db, "my_model")

        assert result == {"model": "my_model", "status": "deleted"}
        client.drop_model.assert_called_once_with("my_model")


# ---------------------------------------------------------------------------
# classify_uncategorized()
# ---------------------------------------------------------------------------


class TestClassifyUncategorizedService:
    def test_returns_empty_when_no_uncategorized(
        self, enabled_config: None, mock_db: MagicMock
    ) -> None:
        from alibi.services.predictions import classify_uncategorized

        with patch("alibi.db.v2_store.list_fact_items_uncategorized", return_value=[]):
            result = classify_uncategorized(mock_db, limit=100)

        assert result == []

    def test_filters_by_min_confidence(
        self, enabled_config: None, mock_db: MagicMock
    ) -> None:
        from alibi.services.predictions import classify_uncategorized

        items = [
            {
                "id": "i1",
                "fact_id": "f1",
                "name": "bread",
                "name_normalized": None,
                "total_price": 1.5,
                "unit_price": None,
            },
            {
                "id": "i2",
                "fact_id": "f1",
                "name": "milk",
                "name_normalized": None,
                "total_price": 0.9,
                "unit_price": None,
            },
        ]
        predictions = [
            {
                "vendor_name": "Lidl",
                "item_name": "bread",
                "amount": 1.5,
                "category": "food",
                "category_confidence": 0.8,
            },
            {
                "vendor_name": "Lidl",
                "item_name": "milk",
                "amount": 0.9,
                "category": "dairy",
                "category_confidence": 0.3,
            },
        ]
        fact = {"vendor": "Lidl", "total_amount": "5.00"}
        client = _mock_healthy_client()

        with (
            patch(
                "alibi.db.v2_store.list_fact_items_uncategorized",
                return_value=items,
            ),
            patch("alibi.db.v2_store.inspect_fact", return_value=fact),
            patch(_CLIENT_PATCH, return_value=client),
            patch(
                "alibi.predictions.category.classify_batch", return_value=predictions
            ),
        ):
            result = classify_uncategorized(mock_db, limit=100, min_confidence=0.5)

        # Only the bread row (confidence 0.8) passes the 0.5 threshold
        assert len(result) == 1
        assert result[0]["item_id"] == "i1"
        assert result[0]["category"] == "food"

    def test_disabled_raises_when_items_exist(
        self, disabled_config: None, mock_db: MagicMock
    ) -> None:
        from alibi.services.predictions import classify_uncategorized

        items = [
            {
                "id": "i1",
                "fact_id": "f1",
                "name": "x",
                "name_normalized": None,
                "total_price": 1.0,
                "unit_price": None,
            }
        ]

        with (
            patch(
                "alibi.db.v2_store.list_fact_items_uncategorized",
                return_value=items,
            ),
            patch(
                "alibi.db.v2_store.inspect_fact",
                return_value={"vendor": "V"},
            ),
        ):
            with pytest.raises(RuntimeError, match="not enabled"):
                classify_uncategorized(mock_db, limit=100)
