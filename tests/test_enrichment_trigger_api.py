"""Tests for enrichment trigger API endpoints and related config/module functions.

Covers:
- Config: cloud_enrichment_enabled, anthropic_api_key, llm_enrichment_timeout
- cloud_enrichment._is_enabled() and _get_api_key()
- llm_enrichment._get_llm_timeout()
- POST /api/v1/enrichment/run/cloud
- POST /api/v1/enrichment/run/llm
"""

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
from alibi.config import Config, reset_config
from alibi.db.connection import DatabaseManager
from alibi.enrichment.cloud_enrichment import (
    CloudEnrichmentResult,
    _get_api_key,
    _is_enabled,
)
from alibi.enrichment.llm_enrichment import LlmEnrichmentResult, _get_llm_timeout

_ENRICHMENT_PREFIX = "/api/v1/enrichment"


# ---------------------------------------------------------------------------
# Client fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client(db_manager: DatabaseManager) -> Generator[TestClient, None, None]:
    """Test client with DB and auth dependency overrides."""
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
# Config tests — cloud enrichment fields
# ---------------------------------------------------------------------------


class TestConfigCloudEnrichmentDefaults:
    def test_cloud_enrichment_enabled_default_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALIBI_TESTING", "1")
        config = Config()
        assert config.cloud_enrichment_enabled is False

    def test_anthropic_api_key_default_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALIBI_TESTING", "1")
        config = Config()
        assert config.anthropic_api_key is None

    def test_llm_enrichment_timeout_default_60(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALIBI_TESTING", "1")
        config = Config()
        assert config.llm_enrichment_timeout == 60.0

    def test_cloud_enrichment_enabled_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALIBI_TESTING", "1")
        monkeypatch.setenv("ALIBI_CLOUD_ENRICHMENT_ENABLED", "true")
        config = Config()
        assert config.cloud_enrichment_enabled is True

    def test_anthropic_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALIBI_TESTING", "1")
        monkeypatch.setenv("ALIBI_ANTHROPIC_API_KEY", "sk-ant-test-key")
        config = Config()
        assert config.anthropic_api_key == "sk-ant-test-key"

    def test_llm_enrichment_timeout_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALIBI_TESTING", "1")
        monkeypatch.setenv("ALIBI_LLM_ENRICHMENT_TIMEOUT", "120.5")
        config = Config()
        assert config.llm_enrichment_timeout == pytest.approx(120.5)


# ---------------------------------------------------------------------------
# cloud_enrichment._is_enabled() tests
# ---------------------------------------------------------------------------


class TestIsEnabled:
    def test_is_enabled_reads_from_config(self) -> None:
        mock_config = MagicMock()
        mock_config.cloud_enrichment_enabled = True
        with patch("alibi.config.get_config", return_value=mock_config):
            assert _is_enabled() is True

    def test_is_enabled_default_false(self) -> None:
        mock_config = MagicMock()
        mock_config.cloud_enrichment_enabled = False
        with patch("alibi.config.get_config", return_value=mock_config):
            assert _is_enabled() is False


# ---------------------------------------------------------------------------
# cloud_enrichment._get_api_key() tests
# ---------------------------------------------------------------------------


class TestGetApiKey:
    def test_get_api_key_from_argument(self) -> None:
        result = _get_api_key("direct-key-from-arg")
        assert result == "direct-key-from-arg"

    def test_get_api_key_from_config(self) -> None:
        mock_config = MagicMock()
        mock_config.anthropic_api_key = "config-api-key"
        with patch("alibi.config.get_config", return_value=mock_config):
            result = _get_api_key(None)
        assert result == "config-api-key"

    def test_get_api_key_none_when_missing(self) -> None:
        mock_config = MagicMock()
        mock_config.anthropic_api_key = None
        with patch("alibi.config.get_config", return_value=mock_config):
            result = _get_api_key(None)
        assert result is None


# ---------------------------------------------------------------------------
# llm_enrichment._get_llm_timeout() tests
# ---------------------------------------------------------------------------


class TestGetLlmTimeout:
    def test_get_llm_timeout_from_config(self) -> None:
        mock_config = MagicMock()
        mock_config.llm_enrichment_timeout = 90.0
        with patch("alibi.config.get_config", return_value=mock_config):
            result = _get_llm_timeout()
        assert result == pytest.approx(90.0)


# ---------------------------------------------------------------------------
# POST /api/v1/enrichment/run/cloud — API endpoint tests
# ---------------------------------------------------------------------------

_CLOUD_MODULE = "alibi.enrichment.cloud_enrichment"


class TestRunCloudEnrichmentEndpoint:
    def test_run_cloud_enrichment_endpoint(self, client: TestClient) -> None:
        fake_results = [
            CloudEnrichmentResult(
                item_id="item-1", brand="BrandA", category="Dairy", success=True
            ),
            CloudEnrichmentResult(
                item_id="item-2", brand="BrandB", category="Bakery", success=True
            ),
        ]
        with patch(
            f"{_CLOUD_MODULE}.enrich_pending_by_cloud", return_value=fake_results
        ):
            resp = client.post(f"{_ENRICHMENT_PREFIX}/run/cloud")

        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "cloud_api"
        assert data["processed"] == 2
        assert data["enriched"] == 2

    def test_run_cloud_enrichment_with_limit(self, client: TestClient) -> None:
        with patch(
            f"{_CLOUD_MODULE}.enrich_pending_by_cloud", return_value=[]
        ) as mock_fn:
            resp = client.post(f"{_ENRICHMENT_PREFIX}/run/cloud?limit=25")

        assert resp.status_code == 200
        # Verify that the limit param was forwarded to the service function
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["limit"] == 25

    def test_run_cloud_enrichment_empty_results(self, client: TestClient) -> None:
        with patch(f"{_CLOUD_MODULE}.enrich_pending_by_cloud", return_value=[]):
            resp = client.post(f"{_ENRICHMENT_PREFIX}/run/cloud")

        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "cloud_api"
        assert data["processed"] == 0
        assert data["enriched"] == 0

    def test_run_cloud_enrichment_partial_success(self, client: TestClient) -> None:
        """Only items with success=True count towards enriched."""
        fake_results = [
            CloudEnrichmentResult(
                item_id="item-1", brand="BrandA", category="Dairy", success=True
            ),
            CloudEnrichmentResult(
                item_id="item-2", brand=None, category=None, success=False
            ),
            CloudEnrichmentResult(
                item_id="item-3", brand=None, category=None, success=False
            ),
        ]
        with patch(
            f"{_CLOUD_MODULE}.enrich_pending_by_cloud", return_value=fake_results
        ):
            resp = client.post(f"{_ENRICHMENT_PREFIX}/run/cloud")

        assert resp.status_code == 200
        data = resp.json()
        assert data["processed"] == 3
        assert data["enriched"] == 1

    def test_run_cloud_enrichment_limit_out_of_range_returns_422(
        self, client: TestClient
    ) -> None:
        resp = client.post(f"{_ENRICHMENT_PREFIX}/run/cloud?limit=0")
        assert resp.status_code == 422

    def test_run_cloud_enrichment_limit_too_large_returns_422(
        self, client: TestClient
    ) -> None:
        resp = client.post(f"{_ENRICHMENT_PREFIX}/run/cloud?limit=1001")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/v1/enrichment/run/llm — API endpoint tests
# ---------------------------------------------------------------------------

_LLM_MODULE = "alibi.enrichment.llm_enrichment"


class TestRunLlmEnrichmentEndpoint:
    def test_run_llm_enrichment_endpoint(self, client: TestClient) -> None:
        fake_results = [
            LlmEnrichmentResult(
                item_id="item-1", brand="LlmBrand", category="Beverages", success=True
            ),
            LlmEnrichmentResult(
                item_id="item-2", brand=None, category="Snacks", success=True
            ),
        ]
        with patch(f"{_LLM_MODULE}.enrich_pending_by_llm", return_value=fake_results):
            resp = client.post(f"{_ENRICHMENT_PREFIX}/run/llm")

        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "llm_inference"
        assert data["processed"] == 2
        assert data["enriched"] == 2

    def test_run_llm_enrichment_with_limit(self, client: TestClient) -> None:
        with patch(f"{_LLM_MODULE}.enrich_pending_by_llm", return_value=[]) as mock_fn:
            resp = client.post(f"{_ENRICHMENT_PREFIX}/run/llm?limit=50")

        assert resp.status_code == 200
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["limit"] == 50

    def test_run_llm_enrichment_empty_results(self, client: TestClient) -> None:
        with patch(f"{_LLM_MODULE}.enrich_pending_by_llm", return_value=[]):
            resp = client.post(f"{_ENRICHMENT_PREFIX}/run/llm")

        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "llm_inference"
        assert data["processed"] == 0
        assert data["enriched"] == 0

    def test_run_llm_enrichment_partial_success(self, client: TestClient) -> None:
        """Only items with success=True count towards enriched."""
        fake_results = [
            LlmEnrichmentResult(
                item_id="item-1", brand="BrandX", category="Oils", success=True
            ),
            LlmEnrichmentResult(
                item_id="item-2", brand=None, category=None, success=False
            ),
        ]
        with patch(f"{_LLM_MODULE}.enrich_pending_by_llm", return_value=fake_results):
            resp = client.post(f"{_ENRICHMENT_PREFIX}/run/llm")

        assert resp.status_code == 200
        data = resp.json()
        assert data["processed"] == 2
        assert data["enriched"] == 1

    def test_run_llm_enrichment_limit_out_of_range_returns_422(
        self, client: TestClient
    ) -> None:
        resp = client.post(f"{_ENRICHMENT_PREFIX}/run/llm?limit=0")
        assert resp.status_code == 422

    def test_run_llm_enrichment_limit_too_large_returns_422(
        self, client: TestClient
    ) -> None:
        resp = client.post(f"{_ENRICHMENT_PREFIX}/run/llm?limit=1001")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Config tests — cloud_enrichment_model field
# ---------------------------------------------------------------------------


class TestConfigCloudEnrichmentModel:
    def test_cloud_enrichment_model_default_sonnet(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALIBI_TESTING", "1")
        config = Config()
        assert config.cloud_enrichment_model == "claude-sonnet-4-6"

    def test_cloud_enrichment_model_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ALIBI_TESTING", "1")
        monkeypatch.setenv("ALIBI_CLOUD_ENRICHMENT_MODEL", "claude-opus-4-6")
        config = Config()
        assert config.cloud_enrichment_model == "claude-opus-4-6"


# ---------------------------------------------------------------------------
# cloud_enrichment._get_model() tests
# ---------------------------------------------------------------------------


class TestGetModel:
    def test_get_model_reads_from_config(self) -> None:
        from alibi.enrichment.cloud_enrichment import _get_model

        mock_config = MagicMock()
        mock_config.cloud_enrichment_model = "claude-sonnet-4-6"
        with patch("alibi.config.get_config", return_value=mock_config):
            assert _get_model() == "claude-sonnet-4-6"

    def test_get_model_override_via_argument(self) -> None:
        """infer_cloud_brand_category uses the model argument when provided."""
        import json as _json

        from alibi.enrichment.cloud_enrichment import infer_cloud_brand_category

        api_payload = {
            "content": [
                {
                    "type": "text",
                    "text": _json.dumps(
                        {"items": [{"idx": 1, "brand": None, "category": "Other"}]}
                    ),
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.json.return_value = api_payload
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            infer_cloud_brand_category(
                [{"idx": 1, "name": "Widget"}],
                api_key="test-key",
                model="claude-opus-4-6",
            )

        sent_json = mock_post.call_args.kwargs.get("json") or mock_post.call_args[
            1
        ].get("json")
        assert sent_json["model"] == "claude-opus-4-6"

    def test_get_model_default_is_haiku_for_infer(self) -> None:
        """infer_cloud_brand_category defaults to _DEFAULT_MODEL (Haiku) when model=None."""
        import json as _json

        from alibi.enrichment.cloud_enrichment import infer_cloud_brand_category

        api_payload = {
            "content": [
                {
                    "type": "text",
                    "text": _json.dumps(
                        {"items": [{"idx": 1, "brand": None, "category": "Other"}]}
                    ),
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.json.return_value = api_payload
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            infer_cloud_brand_category(
                [{"idx": 1, "name": "Widget"}],
                api_key="test-key",
                model=None,
            )

        sent_json = mock_post.call_args.kwargs.get("json") or mock_post.call_args[
            1
        ].get("json")
        assert sent_json["model"] == "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# POST /api/v1/enrichment/run/refine — API endpoint tests
# ---------------------------------------------------------------------------


class TestRunRefineEnrichmentEndpoint:
    def test_run_refine_endpoint_corrected(self, client: TestClient) -> None:
        fake_results = [
            CloudEnrichmentResult(
                item_id="item-1", brand=None, category="Fish", success=True
            ),
        ]
        with patch(
            f"{_CLOUD_MODULE}.refine_categories_by_cloud", return_value=fake_results
        ):
            resp = client.post(f"{_ENRICHMENT_PREFIX}/run/refine")

        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "cloud_refined"
        assert data["corrected"] == 1

    def test_run_refine_endpoint_no_corrections(self, client: TestClient) -> None:
        with patch(f"{_CLOUD_MODULE}.refine_categories_by_cloud", return_value=[]):
            resp = client.post(f"{_ENRICHMENT_PREFIX}/run/refine")

        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "cloud_refined"
        assert data["corrected"] == 0

    def test_run_refine_endpoint_with_limit(self, client: TestClient) -> None:
        with patch(
            f"{_CLOUD_MODULE}.refine_categories_by_cloud", return_value=[]
        ) as mock_fn:
            resp = client.post(f"{_ENRICHMENT_PREFIX}/run/refine?limit=25")

        assert resp.status_code == 200
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["limit"] == 25

    def test_run_refine_endpoint_limit_out_of_range_returns_422(
        self, client: TestClient
    ) -> None:
        resp = client.post(f"{_ENRICHMENT_PREFIX}/run/refine?limit=0")
        assert resp.status_code == 422

    def test_run_refine_endpoint_limit_too_large_returns_422(
        self, client: TestClient
    ) -> None:
        resp = client.post(f"{_ENRICHMENT_PREFIX}/run/refine?limit=1001")
        assert resp.status_code == 422
