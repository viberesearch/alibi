"""Tests for alibi.api.routers.predictions — FastAPI prediction endpoints."""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ["ALIBI_TESTING"] = "1"

from alibi.api.app import create_app
from alibi.api.deps import get_database, require_user
from alibi.db.connection import DatabaseManager

# The router imports service functions inside each endpoint function body,
# so we patch at the service module (alibi.services.predictions.*).
_SVC = "alibi.services.predictions"

_PREDICTIONS_PREFIX = "/api/v1/predictions"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(db_manager: DatabaseManager) -> Generator[TestClient, None, None]:
    """Create a test client with DB and auth overrides."""
    app = create_app()

    def override_get_database() -> DatabaseManager:
        return db_manager

    def override_require_user() -> dict[str, Any]:
        return {"id": "test-user", "name": "Test User"}

    app.dependency_overrides[get_database] = override_get_database
    app.dependency_overrides[require_user] = override_require_user
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# POST /train/forecast
# ---------------------------------------------------------------------------


class TestTrainForecastEndpoint:
    def test_train_forecast_success(self, client: TestClient) -> None:
        fake_result = {
            "model": "alibi_spending_forecast",
            "status": "complete",
            "window": 6,
            "horizon": 3,
        }
        with patch(f"{_SVC}.train_spending_forecast", return_value=fake_result):
            resp = client.post(
                f"{_PREDICTIONS_PREFIX}/train/forecast",
                json={"window": 6, "horizon": 3},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "alibi_spending_forecast"
        assert data["status"] == "complete"

    def test_train_forecast_default_params(self, client: TestClient) -> None:
        fake_result = {
            "model": "alibi_spending_forecast",
            "status": "complete",
            "window": 6,
            "horizon": 3,
        }
        with patch(
            f"{_SVC}.train_spending_forecast", return_value=fake_result
        ) as mock_fn:
            resp = client.post(
                f"{_PREDICTIONS_PREFIX}/train/forecast",
                json={},
            )

        assert resp.status_code == 200
        # The service was called with window=6 and horizon=3 (defaults)
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["window"] == 6
        assert call_kwargs["horizon"] == 3

    def test_train_forecast_returns_400_on_runtime_error(
        self, client: TestClient
    ) -> None:
        with patch(
            f"{_SVC}.train_spending_forecast",
            side_effect=RuntimeError("MindsDB is not enabled"),
        ):
            resp = client.post(
                f"{_PREDICTIONS_PREFIX}/train/forecast",
                json={"window": 6, "horizon": 3},
            )

        assert resp.status_code == 400
        assert "MindsDB" in resp.json()["detail"]

    def test_train_forecast_returns_400_on_value_error(
        self, client: TestClient
    ) -> None:
        with patch(
            f"{_SVC}.train_spending_forecast",
            side_effect=ValueError("No spending data"),
        ):
            resp = client.post(
                f"{_PREDICTIONS_PREFIX}/train/forecast",
                json={"window": 6, "horizon": 3},
            )

        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /train/category
# ---------------------------------------------------------------------------


class TestTrainCategoryEndpoint:
    def test_train_category_success(self, client: TestClient) -> None:
        fake_result = {"model": "alibi_category_classifier", "status": "complete"}
        with patch(f"{_SVC}.train_category_classifier", return_value=fake_result):
            resp = client.post(f"{_PREDICTIONS_PREFIX}/train/category")

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "alibi_category_classifier"
        assert data["status"] == "complete"

    def test_train_category_returns_400_on_runtime_error(
        self, client: TestClient
    ) -> None:
        with patch(
            f"{_SVC}.train_category_classifier",
            side_effect=RuntimeError("MindsDB not enabled"),
        ):
            resp = client.post(f"{_PREDICTIONS_PREFIX}/train/category")

        assert resp.status_code == 400

    def test_train_category_returns_400_on_value_error(
        self, client: TestClient
    ) -> None:
        with patch(
            f"{_SVC}.train_category_classifier",
            side_effect=ValueError("Insufficient data"),
        ):
            resp = client.post(f"{_PREDICTIONS_PREFIX}/train/category")

        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /forecast
# ---------------------------------------------------------------------------


class TestForecastEndpoint:
    def test_forecast_returns_list(self, client: TestClient) -> None:
        fake_rows = [
            {
                "forecast_date": "2025-04-01",
                "category": "food",
                "forecast_amount": 120.0,
            }
        ]
        with patch(f"{_SVC}.get_spending_forecast", return_value=fake_rows):
            resp = client.get(f"{_PREDICTIONS_PREFIX}/forecast")

        assert resp.status_code == 200
        assert resp.json() == fake_rows

    def test_forecast_passes_months_and_category(self, client: TestClient) -> None:
        with patch(f"{_SVC}.get_spending_forecast", return_value=[]) as mock_fn:
            resp = client.get(f"{_PREDICTIONS_PREFIX}/forecast?months=6&category=food")

        assert resp.status_code == 200
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["months"] == 6
        assert call_kwargs["category"] == "food"

    def test_forecast_returns_400_on_runtime_error(self, client: TestClient) -> None:
        with patch(
            f"{_SVC}.get_spending_forecast",
            side_effect=RuntimeError("MindsDB not enabled"),
        ):
            resp = client.get(f"{_PREDICTIONS_PREFIX}/forecast")

        assert resp.status_code == 400

    def test_forecast_months_out_of_range_returns_422(self, client: TestClient) -> None:
        with patch(f"{_SVC}.get_spending_forecast", return_value=[]):
            resp = client.get(f"{_PREDICTIONS_PREFIX}/forecast?months=0")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /classify
# ---------------------------------------------------------------------------


class TestClassifyEndpoint:
    def test_classify_returns_prediction(self, client: TestClient) -> None:
        fake_result = {
            "vendor_name": "Lidl",
            "item_name": "bread",
            "amount": 1.5,
            "category": "food",
            "category_confidence": 0.9,
        }
        with patch(f"{_SVC}.classify_category", return_value=fake_result):
            resp = client.post(
                f"{_PREDICTIONS_PREFIX}/classify",
                json={"vendor_name": "Lidl", "item_name": "bread", "amount": 1.5},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["category"] == "food"
        assert data["vendor_name"] == "Lidl"

    def test_classify_passes_correct_args(self, client: TestClient) -> None:
        with patch(
            f"{_SVC}.classify_category",
            return_value={"category": "x", "category_confidence": 0.5},
        ) as mock_fn:
            client.post(
                f"{_PREDICTIONS_PREFIX}/classify",
                json={"vendor_name": "Aldi", "item_name": "milk", "amount": 0.89},
            )

        call_args = mock_fn.call_args[0]
        assert call_args[1] == "Aldi"
        assert call_args[2] == "milk"
        assert abs(call_args[3] - 0.89) < 0.001

    def test_classify_returns_400_on_runtime_error(self, client: TestClient) -> None:
        with patch(
            f"{_SVC}.classify_category",
            side_effect=RuntimeError("MindsDB not reachable"),
        ):
            resp = client.post(
                f"{_PREDICTIONS_PREFIX}/classify",
                json={"vendor_name": "V", "item_name": "I", "amount": 1.0},
            )

        assert resp.status_code == 400

    def test_classify_missing_fields_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            f"{_PREDICTIONS_PREFIX}/classify",
            json={"vendor_name": "V"},  # missing item_name and amount
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /classify/uncategorized
# ---------------------------------------------------------------------------


class TestClassifyUncategorizedEndpoint:
    def test_returns_list_of_suggestions(self, client: TestClient) -> None:
        fake = [
            {
                "item_id": "i1",
                "category": "food",
                "confidence": 0.85,
                "vendor_name": "Lidl",
                "item_name": "bread",
            }
        ]
        with patch(f"{_SVC}.classify_uncategorized", return_value=fake):
            resp = client.get(f"{_PREDICTIONS_PREFIX}/classify/uncategorized")

        assert resp.status_code == 200
        assert resp.json() == fake

    def test_passes_limit_and_min_confidence(self, client: TestClient) -> None:
        with patch(f"{_SVC}.classify_uncategorized", return_value=[]) as mock_fn:
            resp = client.get(
                f"{_PREDICTIONS_PREFIX}/classify/uncategorized?limit=50&min_confidence=0.7"
            )

        assert resp.status_code == 200
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["limit"] == 50
        assert abs(call_kwargs["min_confidence"] - 0.7) < 0.001

    def test_returns_400_on_runtime_error(self, client: TestClient) -> None:
        with patch(
            f"{_SVC}.classify_uncategorized",
            side_effect=RuntimeError("not enabled"),
        ):
            resp = client.get(f"{_PREDICTIONS_PREFIX}/classify/uncategorized")

        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------


class TestListModelsEndpoint:
    def test_list_models_returns_list(self, client: TestClient) -> None:
        fake_models = [
            {
                "name": "spending_model",
                "status": "complete",
                "predict": "amount",
                "engine": "lightwood",
            }
        ]
        with patch(f"{_SVC}.list_models", return_value=fake_models):
            resp = client.get(f"{_PREDICTIONS_PREFIX}/models")

        assert resp.status_code == 200
        assert resp.json() == fake_models

    def test_list_models_returns_400_on_runtime_error(self, client: TestClient) -> None:
        with patch(
            f"{_SVC}.list_models",
            side_effect=RuntimeError("not enabled"),
        ):
            resp = client.get(f"{_PREDICTIONS_PREFIX}/models")

        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /models/{name}/status
# ---------------------------------------------------------------------------


class TestModelStatusEndpoint:
    def test_model_status_returns_dict(self, client: TestClient) -> None:
        fake = {"model": "spending_model", "status": "complete"}
        with patch(f"{_SVC}.get_model_status", return_value=fake):
            resp = client.get(f"{_PREDICTIONS_PREFIX}/models/spending_model/status")

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "spending_model"
        assert data["status"] == "complete"

    def test_model_status_returns_400_on_runtime_error(
        self, client: TestClient
    ) -> None:
        with patch(
            f"{_SVC}.get_model_status",
            side_effect=RuntimeError("not enabled"),
        ):
            resp = client.get(f"{_PREDICTIONS_PREFIX}/models/unknown_model/status")

        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# DELETE /models/{name}
# ---------------------------------------------------------------------------


class TestDeleteModelEndpoint:
    def test_delete_model_returns_deleted(self, client: TestClient) -> None:
        fake = {"model": "spending_model", "status": "deleted"}
        with patch(f"{_SVC}.drop_model", return_value=fake):
            resp = client.delete(f"{_PREDICTIONS_PREFIX}/models/spending_model")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"

    def test_delete_model_returns_400_on_runtime_error(
        self, client: TestClient
    ) -> None:
        with patch(
            f"{_SVC}.drop_model",
            side_effect=RuntimeError("not enabled"),
        ):
            resp = client.delete(f"{_PREDICTIONS_PREFIX}/models/spending_model")

        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# MindsDB disabled — all endpoints return 400 via service layer
# ---------------------------------------------------------------------------


class TestMindsdbDisabledReturns400:
    """When MindsDB is not enabled, all prediction endpoints return 400."""

    def test_train_forecast_disabled(self, client: TestClient) -> None:
        with patch(
            f"{_SVC}.train_spending_forecast",
            side_effect=RuntimeError("MindsDB is not enabled"),
        ):
            resp = client.post(
                f"{_PREDICTIONS_PREFIX}/train/forecast",
                json={"window": 6, "horizon": 3},
            )
        assert resp.status_code == 400

    def test_forecast_disabled(self, client: TestClient) -> None:
        with patch(
            f"{_SVC}.get_spending_forecast",
            side_effect=RuntimeError("MindsDB is not enabled"),
        ):
            resp = client.get(f"{_PREDICTIONS_PREFIX}/forecast")
        assert resp.status_code == 400

    def test_classify_disabled(self, client: TestClient) -> None:
        with patch(
            f"{_SVC}.classify_category",
            side_effect=RuntimeError("MindsDB is not enabled"),
        ):
            resp = client.post(
                f"{_PREDICTIONS_PREFIX}/classify",
                json={"vendor_name": "V", "item_name": "I", "amount": 1.0},
            )
        assert resp.status_code == 400

    def test_list_models_disabled(self, client: TestClient) -> None:
        with patch(
            f"{_SVC}.list_models",
            side_effect=RuntimeError("MindsDB is not enabled"),
        ):
            resp = client.get(f"{_PREDICTIONS_PREFIX}/models")
        assert resp.status_code == 400
