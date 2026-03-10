"""Tests for Gemini mega-batch product enrichment."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.enrichment.gemini_enrichment import (
    EnrichmentBatchResponse,
    GeminiEnrichmentResult,
    ItemEnrichment,
    _ENRICHMENT_CONFIDENCE,
    _ENRICHMENT_SOURCE,
    enrich_items_by_gemini,
    enrich_pending_by_gemini,
    infer_batch,
    normalize_names_by_gemini,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_fact_item(
    db,
    item_id: str,
    name: str,
    brand: str | None = None,
    category: str | None = None,
    barcode: str | None = None,
    vendor: str = "Test Store",
    vendor_key: str | None = None,
    fact_id: str | None = None,
    doc_id: str | None = None,
    atom_id: str | None = None,
    cloud_id: str | None = None,
) -> None:
    """Insert a fact_item with supporting chain."""
    fact_id = fact_id or f"fact-{item_id}"
    doc_id = doc_id or f"doc-{item_id}"
    atom_id = atom_id or f"atom-{item_id}"
    cloud_id = cloud_id or f"cloud-{item_id}"

    conn = db.get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO documents (id, file_path, file_hash) "
        "VALUES (?, ?, ?)",
        (doc_id, f"/tmp/{doc_id}.jpg", f"hash-{doc_id}"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO atoms (id, document_id, atom_type, data) "
        "VALUES (?, ?, 'item', '{}')",
        (atom_id, doc_id),
    )
    conn.execute(
        "INSERT OR IGNORE INTO clouds (id, status) VALUES (?, 'collapsed')",
        (cloud_id,),
    )
    conn.execute(
        "INSERT OR IGNORE INTO facts "
        "(id, cloud_id, fact_type, vendor, vendor_key, "
        "total_amount, currency, event_date) "
        "VALUES (?, ?, 'purchase', ?, ?, 10.0, 'EUR', '2026-01-01')",
        (fact_id, cloud_id, vendor, vendor_key),
    )
    conn.execute(
        "INSERT OR IGNORE INTO fact_items "
        "(id, fact_id, atom_id, name, barcode, brand, category) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (item_id, fact_id, atom_id, name, barcode, brand, category),
    )
    conn.commit()


# ===========================================================================
# Pydantic model tests
# ===========================================================================


class TestItemEnrichment:
    def test_minimal_construction(self):
        item = ItemEnrichment(idx=1)
        assert item.idx == 1
        assert item.brand is None
        assert item.category is None
        assert item.unit_quantity is None
        assert item.unit is None
        assert item.comparable_name is None

    def test_full_construction(self):
        item = ItemEnrichment(
            idx=5,
            brand="Mlekovita",
            category="Dairy",
            unit_quantity=1000.0,
            unit="ml",
            comparable_name="Full Fat Milk 1L",
        )
        assert item.idx == 5
        assert item.brand == "Mlekovita"
        assert item.category == "Dairy"
        assert item.unit_quantity == 1000.0
        assert item.unit == "ml"
        assert item.comparable_name == "Full Fat Milk 1L"

    def test_null_fields_preserved(self):
        item = ItemEnrichment(idx=3, brand=None, category="Bakery")
        assert item.brand is None
        assert item.category == "Bakery"


class TestEnrichmentBatchResponse:
    def test_empty_items(self):
        resp = EnrichmentBatchResponse(items=[])
        assert resp.items == []

    def test_multiple_items(self):
        resp = EnrichmentBatchResponse(
            items=[
                ItemEnrichment(idx=1, brand="X", category="Dairy"),
                ItemEnrichment(idx=2, brand=None, category="Bakery"),
            ]
        )
        assert len(resp.items) == 2
        assert resp.items[0].brand == "X"
        assert resp.items[1].brand is None


class TestGeminiEnrichmentResult:
    def test_dataclass_fields(self):
        r = GeminiEnrichmentResult(
            item_id="abc123",
            brand="Brand",
            category="Dairy",
            unit_quantity=500.0,
            unit="ml",
            comparable_name="Full Fat Milk",
            success=True,
        )
        assert r.item_id == "abc123"
        assert r.success is True
        assert r.comparable_name == "Full Fat Milk"

    def test_failed_result(self):
        r = GeminiEnrichmentResult(
            item_id="xyz",
            brand=None,
            category=None,
            unit_quantity=None,
            unit=None,
            comparable_name=None,
            success=False,
        )
        assert r.success is False


# ===========================================================================
# infer_batch tests
# ===========================================================================


class TestInferBatch:
    def test_empty_items_returns_empty(self):
        result = infer_batch([], api_key="test-key")
        assert result == []

    def test_missing_api_key_returns_empty(self):
        with patch(
            "alibi.enrichment.gemini_enrichment._get_api_key", return_value=None
        ):
            result = infer_batch([{"idx": 1, "name": "Milk"}])
        assert result == []

    @patch("google.genai.Client")
    def test_successful_api_call_with_parsed_response(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = EnrichmentBatchResponse(
            items=[
                ItemEnrichment(
                    idx=1,
                    brand="Parmalat",
                    category="Dairy",
                    unit_quantity=1000.0,
                    unit="ml",
                )
            ]
        )
        mock_client.models.generate_content.return_value = mock_response

        result = infer_batch([{"idx": 1, "name": "Milk 1L"}], api_key="test-key")

        assert len(result) == 1
        assert result[0].brand == "Parmalat"
        assert result[0].category == "Dairy"
        assert result[0].unit_quantity == 1000.0
        assert result[0].unit == "ml"

    @patch("google.genai.Client")
    def test_fallback_text_parsing_when_parsed_none(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = None
        mock_response.text = '{"items": [{"idx": 1, "brand": "Nestlé", "category": "Beverages", "unit_quantity": 330.0, "unit": "ml"}]}'
        mock_client.models.generate_content.return_value = mock_response

        result = infer_batch([{"idx": 1, "name": "Nescafe Can"}], api_key="test-key")

        assert len(result) == 1
        assert result[0].brand == "Nestlé"
        assert result[0].category == "Beverages"

    @patch("google.genai.Client")
    def test_api_exception_returns_empty(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception("API error")

        result = infer_batch([{"idx": 1, "name": "Test"}], api_key="test-key")

        assert result == []

    @patch("google.genai.Client")
    def test_barcode_included_in_prompt(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = EnrichmentBatchResponse(items=[])
        mock_client.models.generate_content.return_value = mock_response

        infer_batch(
            [{"idx": 1, "name": "Milk 1L", "barcode": "5901234123457"}],
            api_key="test-key",
        )

        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs.get("contents") or call_args.args[1]
        assert "5901234123457" in contents
        assert "Milk 1L" in contents

    @patch("google.genai.Client")
    def test_item_without_barcode_no_barcode_in_prompt(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = EnrichmentBatchResponse(items=[])
        mock_client.models.generate_content.return_value = mock_response

        infer_batch(
            [{"idx": 1, "name": "Generic Bread"}],
            api_key="test-key",
        )

        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs.get("contents") or call_args.args[1]
        assert "barcode:" not in contents

    @patch("google.genai.Client")
    def test_uses_configured_model(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.parsed = EnrichmentBatchResponse(items=[])
        mock_client.models.generate_content.return_value = mock_response

        infer_batch(
            [{"idx": 1, "name": "Test"}],
            api_key="test-key",
            model="gemini-1.5-pro",
        )

        call_args = mock_client.models.generate_content.call_args
        model_arg = call_args.kwargs.get("model") or call_args.args[0]
        assert model_arg == "gemini-1.5-pro"


# ===========================================================================
# enrich_items_by_gemini tests
# ===========================================================================


class TestEnrichItemsByGemini:
    @patch("alibi.enrichment.gemini_enrichment.infer_batch")
    def test_empty_items_returns_empty(self, mock_infer, db):
        result = enrich_items_by_gemini(db, [])
        assert result == []
        mock_infer.assert_not_called()

    @patch("alibi.enrichment.gemini_enrichment.infer_batch")
    def test_successful_enrichment_updates_db(self, mock_infer, db):
        _seed_fact_item(db, "item-001", "Fresh Milk 1L")
        mock_infer.return_value = [
            ItemEnrichment(
                idx=1, brand="Arla", category="Dairy", unit_quantity=1000.0, unit="ml"
            )
        ]

        results = enrich_items_by_gemini(
            db,
            [{"id": "item-001", "name": "Fresh Milk 1L", "barcode": ""}],
            api_key="test-key",
        )

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].brand == "Arla"
        assert results[0].category == "Dairy"
        assert results[0].unit_quantity == 1000.0
        assert results[0].unit == "ml"

        # Verify DB was updated
        row = db.fetchone(
            "SELECT brand, category, unit_quantity, unit, enrichment_source, enrichment_confidence "
            "FROM fact_items WHERE id = ?",
            ("item-001",),
        )
        assert row["brand"] == "Arla"
        assert row["category"] == "Dairy"
        assert row["unit_quantity"] == 1000.0
        assert row["unit"] == "ml"
        assert row["enrichment_source"] == _ENRICHMENT_SOURCE
        assert abs(row["enrichment_confidence"] - _ENRICHMENT_CONFIDENCE) < 0.001

    @patch("alibi.enrichment.gemini_enrichment.infer_batch")
    def test_item_with_no_inferred_data_marked_failed(self, mock_infer, db):
        _seed_fact_item(db, "item-002", "Unknown Product")
        mock_infer.return_value = [ItemEnrichment(idx=1, brand=None, category=None)]

        results = enrich_items_by_gemini(
            db,
            [{"id": "item-002", "name": "Unknown Product", "barcode": ""}],
            api_key="test-key",
        )

        assert len(results) == 1
        assert results[0].success is False

    @patch("alibi.enrichment.gemini_enrichment.infer_batch")
    def test_deduplication_same_name_barcode(self, mock_infer, db):
        """Two items with the same name+barcode should result in a single API call."""
        _seed_fact_item(db, "item-003a", "Butter 250g", barcode="1234567890")
        _seed_fact_item(db, "item-003b", "Butter 250g", barcode="1234567890")
        mock_infer.return_value = [
            ItemEnrichment(idx=1, brand="Lurpak", category="Dairy")
        ]

        results = enrich_items_by_gemini(
            db,
            [
                {"id": "item-003a", "name": "Butter 250g", "barcode": "1234567890"},
                {"id": "item-003b", "name": "Butter 250g", "barcode": "1234567890"},
            ],
            api_key="test-key",
        )

        # infer_batch called with deduplicated single item
        call_args = mock_infer.call_args
        sent_items = call_args.args[0]
        assert len(sent_items) == 1

        # Both items enriched
        assert len(results) == 2
        assert all(r.success for r in results)

    @patch("alibi.enrichment.gemini_enrichment.infer_batch")
    def test_item_missing_from_response_marked_failed(self, mock_infer, db):
        """Item whose idx is absent from the API response is marked as failure."""
        _seed_fact_item(db, "item-004", "Mystery Product")
        mock_infer.return_value = []  # No results from API

        results = enrich_items_by_gemini(
            db,
            [{"id": "item-004", "name": "Mystery Product", "barcode": ""}],
            api_key="test-key",
        )

        assert len(results) == 1
        assert results[0].success is False

    @patch("alibi.enrichment.gemini_enrichment.infer_batch")
    def test_only_brand_no_category(self, mock_infer, db):
        """Brand-only result should still succeed and update DB."""
        _seed_fact_item(db, "item-005", "Brand Only Item")
        mock_infer.return_value = [
            ItemEnrichment(idx=1, brand="TestBrand", category=None)
        ]

        results = enrich_items_by_gemini(
            db,
            [{"id": "item-005", "name": "Brand Only Item", "barcode": ""}],
            api_key="test-key",
        )

        assert results[0].success is True
        assert results[0].brand == "TestBrand"
        assert results[0].category is None

    @patch("alibi.enrichment.gemini_enrichment.infer_batch")
    def test_unit_quantity_zero_not_written(self, mock_infer, db):
        """unit_quantity=0 should be treated as missing, not persisted."""
        _seed_fact_item(db, "item-006", "Bulk Item")
        mock_infer.return_value = [
            ItemEnrichment(
                idx=1, brand="Generic", category="Other", unit_quantity=0.0, unit="pcs"
            )
        ]

        results = enrich_items_by_gemini(
            db,
            [{"id": "item-006", "name": "Bulk Item", "barcode": ""}],
            api_key="test-key",
        )

        assert results[0].success is True
        # unit_quantity=0 should not be written
        row = db.fetchone(
            "SELECT unit_quantity FROM fact_items WHERE id = ?",
            ("item-006",),
        )
        assert row["unit_quantity"] is None

    @patch("alibi.enrichment.gemini_enrichment.infer_batch")
    def test_multiple_items_mixed_results(self, mock_infer, db):
        _seed_fact_item(db, "item-007", "Yogurt")
        _seed_fact_item(db, "item-008", "Unknown X")
        mock_infer.return_value = [
            ItemEnrichment(idx=1, brand="Danone", category="Dairy"),
            ItemEnrichment(idx=2, brand=None, category=None),
        ]

        results = enrich_items_by_gemini(
            db,
            [
                {"id": "item-007", "name": "Yogurt", "barcode": ""},
                {"id": "item-008", "name": "Unknown X", "barcode": ""},
            ],
            api_key="test-key",
        )

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is False


# ===========================================================================
# enrich_pending_by_gemini tests
# ===========================================================================


class TestEnrichPendingByGemini:
    def test_disabled_feature_returns_empty(self, db):
        with patch(
            "alibi.enrichment.gemini_enrichment._is_enabled", return_value=False
        ):
            result = enrich_pending_by_gemini(db, api_key="test-key")
        assert result == []

    def test_missing_api_key_returns_empty(self, db):
        with (
            patch("alibi.enrichment.gemini_enrichment._is_enabled", return_value=True),
            patch("alibi.enrichment.gemini_enrichment._get_api_key", return_value=None),
        ):
            result = enrich_pending_by_gemini(db)
        assert result == []

    def test_no_pending_items_returns_empty(self, db):
        with (
            patch("alibi.enrichment.gemini_enrichment._is_enabled", return_value=True),
            patch("alibi.enrichment.gemini_enrichment.infer_batch") as mock_infer,
        ):
            result = enrich_pending_by_gemini(db, api_key="test-key")
        assert result == []
        mock_infer.assert_not_called()

    @patch("alibi.enrichment.gemini_enrichment.infer_batch")
    def test_enriches_pending_items(self, mock_infer, db):
        _seed_fact_item(db, "pending-001", "Olive Oil 500ml")
        mock_infer.return_value = [
            ItemEnrichment(
                idx=1, brand="Cretan", category="Oils", unit_quantity=500.0, unit="ml"
            )
        ]

        with (
            patch("alibi.enrichment.gemini_enrichment._is_enabled", return_value=True),
            patch(
                "alibi.enrichment.gemini_enrichment._get_api_key",
                return_value="test-key",
            ),
        ):
            results = enrich_pending_by_gemini(db)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].brand == "Cretan"

    @patch("alibi.enrichment.gemini_enrichment.infer_batch")
    def test_already_has_brand_not_queried(self, mock_infer, db):
        """Items that already have brand AND category are not queried."""
        _seed_fact_item(
            db, "branded-001", "Milk 1L", brand="Parmalat", category="Dairy"
        )
        mock_infer.return_value = []

        with (
            patch("alibi.enrichment.gemini_enrichment._is_enabled", return_value=True),
            patch(
                "alibi.enrichment.gemini_enrichment._get_api_key",
                return_value="test-key",
            ),
        ):
            results = enrich_pending_by_gemini(db)

        # Item already has brand+category, should not be returned
        assert results == []

    @patch("alibi.enrichment.gemini_enrichment.infer_batch")
    def test_historical_unit_quantity_applied_when_gemini_missing(self, mock_infer, db):
        """When Gemini infers brand/category but no unit_quantity, historical fills the gap."""
        _seed_fact_item(db, "hist-001", "Milk 1L")
        mock_infer.return_value = [
            ItemEnrichment(
                idx=1, brand="Arla", category="Dairy", unit_quantity=None, unit=None
            )
        ]

        historical_data = {
            "unit_quantity": 1000.0,
            "unit": "ml",
            "confidence": 0.9,
            "source": "vendor",
        }

        with (
            patch("alibi.enrichment.gemini_enrichment._is_enabled", return_value=True),
            patch(
                "alibi.enrichment.gemini_enrichment._get_api_key",
                return_value="test-key",
            ),
            patch(
                "alibi.db.v2_store.get_canonical_unit_quantity",
                return_value=historical_data,
            ),
        ):
            results = enrich_pending_by_gemini(db)

        assert len(results) == 1
        assert results[0].success is True
        # Historical unit_quantity applied
        assert results[0].unit_quantity == 1000.0
        assert results[0].unit == "ml"

    @patch("alibi.enrichment.gemini_enrichment.infer_batch")
    def test_gemini_unit_quantity_not_overridden_by_historical(self, mock_infer, db):
        """When Gemini provides unit_quantity, historical should NOT override it."""
        _seed_fact_item(db, "hist-002", "Cheese 200g")
        mock_infer.return_value = [
            ItemEnrichment(
                idx=1,
                brand="President",
                category="Dairy",
                unit_quantity=200.0,
                unit="g",
            )
        ]

        historical_data = {
            "unit_quantity": 250.0,
            "unit": "g",
            "confidence": 0.8,
            "source": "identity",
        }

        with (
            patch("alibi.enrichment.gemini_enrichment._is_enabled", return_value=True),
            patch(
                "alibi.enrichment.gemini_enrichment._get_api_key",
                return_value="test-key",
            ),
            patch(
                "alibi.db.v2_store.get_canonical_unit_quantity",
                return_value=historical_data,
            ),
        ):
            results = enrich_pending_by_gemini(db)

        assert results[0].unit_quantity == 200.0  # Gemini wins, not 250.0

    @patch("alibi.enrichment.gemini_enrichment.infer_batch")
    def test_no_historical_data_gemini_only(self, mock_infer, db):
        """When there is no historical data, Gemini result is used as-is."""
        _seed_fact_item(db, "hist-003", "New Product")
        mock_infer.return_value = [
            ItemEnrichment(
                idx=1, brand="NewBrand", category="Other", unit_quantity=None, unit=None
            )
        ]

        with (
            patch("alibi.enrichment.gemini_enrichment._is_enabled", return_value=True),
            patch(
                "alibi.enrichment.gemini_enrichment._get_api_key",
                return_value="test-key",
            ),
            patch(
                "alibi.db.v2_store.get_canonical_unit_quantity",
                return_value=None,
            ),
        ):
            results = enrich_pending_by_gemini(db)

        assert results[0].success is True
        assert results[0].unit_quantity is None

    @patch("alibi.enrichment.gemini_enrichment.infer_batch")
    def test_respects_limit_parameter(self, mock_infer, db):
        """Limit parameter restricts the number of items queried from DB."""
        for i in range(10):
            _seed_fact_item(db, f"limit-item-{i:03}", f"Product {i}")
        mock_infer.return_value = []

        with (
            patch("alibi.enrichment.gemini_enrichment._is_enabled", return_value=True),
            patch(
                "alibi.enrichment.gemini_enrichment._get_api_key",
                return_value="test-key",
            ),
        ):
            enrich_pending_by_gemini(db, limit=3)

        # infer_batch called with at most 3 items
        call_args = mock_infer.call_args
        sent_items = call_args.args[0]
        assert len(sent_items) <= 3

    @patch("alibi.enrichment.gemini_enrichment.infer_batch")
    def test_comparable_name_written_to_db(self, mock_infer, db):
        """Gemini comparable_name should be persisted on the fact_item."""
        _seed_fact_item(db, "cn-001", "Γάλα Πλήρες 1L")
        mock_infer.return_value = [
            ItemEnrichment(
                idx=1,
                brand="ΔΕΛΤΑ",
                category="Dairy",
                comparable_name="Full Fat Milk 1L",
            )
        ]

        with (
            patch("alibi.enrichment.gemini_enrichment._is_enabled", return_value=True),
            patch(
                "alibi.enrichment.gemini_enrichment._get_api_key",
                return_value="test-key",
            ),
        ):
            results = enrich_pending_by_gemini(db)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].comparable_name == "Full Fat Milk 1L"

        row = db.fetchone(
            "SELECT comparable_name FROM fact_items WHERE id = ?", ("cn-001",)
        )
        assert row["comparable_name"] == "Full Fat Milk 1L"

    @patch("alibi.enrichment.gemini_enrichment.infer_batch")
    def test_null_comparable_name_does_not_overwrite(self, mock_infer, db):
        """Null comparable_name from Gemini should not clear existing value."""
        _seed_fact_item(db, "cn-002", "Milk 1L")
        # Pre-set comparable_name
        db.get_connection().execute(
            "UPDATE fact_items SET comparable_name = ? WHERE id = ?",
            ("Full Fat Milk 1L", "cn-002"),
        )
        db.get_connection().commit()

        mock_infer.return_value = [
            ItemEnrichment(idx=1, brand="Arla", category="Dairy", comparable_name=None)
        ]

        with (
            patch("alibi.enrichment.gemini_enrichment._is_enabled", return_value=True),
            patch(
                "alibi.enrichment.gemini_enrichment._get_api_key",
                return_value="test-key",
            ),
        ):
            results = enrich_pending_by_gemini(db)

        # Existing comparable_name not cleared
        row = db.fetchone(
            "SELECT comparable_name FROM fact_items WHERE id = ?", ("cn-002",)
        )
        assert row["comparable_name"] == "Full Fat Milk 1L"

    @patch("alibi.enrichment.gemini_enrichment.infer_batch")
    def test_historical_comparable_name_applied_when_gemini_missing(
        self, mock_infer, db
    ):
        """Historical comparable_name fills gap when Gemini doesn't provide one."""
        _seed_fact_item(db, "cn-003", "Γάλα 1L")
        mock_infer.return_value = [
            ItemEnrichment(
                idx=1,
                brand="ΔΕΛΤΑ",
                category="Dairy",
                comparable_name=None,
            )
        ]

        historical_cn = {
            "comparable_name": "Milk 1L",
            "confidence": 0.85,
            "source": "brand_history",
        }

        with (
            patch("alibi.enrichment.gemini_enrichment._is_enabled", return_value=True),
            patch(
                "alibi.enrichment.gemini_enrichment._get_api_key",
                return_value="test-key",
            ),
            patch(
                "alibi.db.v2_store.get_canonical_unit_quantity",
                return_value=None,
            ),
            patch(
                "alibi.db.v2_store.get_canonical_comparable_name",
                return_value=historical_cn,
            ),
        ):
            results = enrich_pending_by_gemini(db)

        assert results[0].comparable_name == "Milk 1L"


# ===========================================================================
# normalize_names_by_gemini tests
# ===========================================================================


class TestNormalizeNamesByGemini:
    def test_disabled_feature_returns_empty(self, db):
        with patch(
            "alibi.enrichment.gemini_enrichment._is_enabled", return_value=False
        ):
            result = normalize_names_by_gemini(db, api_key="test-key")
        assert result == []

    def test_missing_api_key_returns_empty(self, db):
        with (
            patch("alibi.enrichment.gemini_enrichment._is_enabled", return_value=True),
            patch("alibi.enrichment.gemini_enrichment._get_api_key", return_value=None),
        ):
            result = normalize_names_by_gemini(db)
        assert result == []

    def test_no_matching_items_returns_empty(self, db):
        with (
            patch("alibi.enrichment.gemini_enrichment._is_enabled", return_value=True),
            patch("alibi.enrichment.gemini_enrichment.infer_batch") as mock_infer,
        ):
            result = normalize_names_by_gemini(db, api_key="test-key")
        assert result == []
        mock_infer.assert_not_called()

    @patch("alibi.enrichment.gemini_enrichment.infer_batch")
    def test_selects_items_where_name_normalized_equals_name(self, mock_infer, db):
        """Items where name_normalized == name should be selected."""
        _seed_fact_item(db, "nn-001", "Γάλα Πλήρες 1L")
        # Set name_normalized = name (the condition)
        conn = db.get_connection()
        conn.execute(
            "UPDATE fact_items SET name_normalized = name WHERE id = ?",
            ("nn-001",),
        )
        conn.commit()

        mock_infer.return_value = [
            ItemEnrichment(
                idx=1,
                brand="ΔΕΛΤΑ",
                category="Dairy",
                comparable_name="Full Fat Milk 1L",
            )
        ]

        with (
            patch("alibi.enrichment.gemini_enrichment._is_enabled", return_value=True),
            patch(
                "alibi.enrichment.gemini_enrichment._get_api_key",
                return_value="test-key",
            ),
        ):
            results = normalize_names_by_gemini(db)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].comparable_name == "Full Fat Milk 1L"

    @patch("alibi.enrichment.gemini_enrichment.infer_batch")
    def test_skips_items_with_different_name_normalized(self, mock_infer, db):
        """Items where name_normalized != name should NOT be selected."""
        _seed_fact_item(db, "nn-002", "Γάλα Πλήρες 1L")
        conn = db.get_connection()
        conn.execute(
            "UPDATE fact_items SET name_normalized = ? WHERE id = ?",
            ("Full Fat Milk 1L", "nn-002"),
        )
        conn.commit()

        mock_infer.return_value = []

        with (
            patch("alibi.enrichment.gemini_enrichment._is_enabled", return_value=True),
            patch(
                "alibi.enrichment.gemini_enrichment._get_api_key",
                return_value="test-key",
            ),
        ):
            results = normalize_names_by_gemini(db)

        assert results == []
        mock_infer.assert_not_called()
