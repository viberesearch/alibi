"""Tests for LLM-based brand/category inference."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.enrichment.llm_enrichment import (
    LlmEnrichmentResult,
    _ENRICHMENT_PROMPT_TEMPLATE,
    enrich_items_by_llm,
    enrich_pending_by_llm,
    infer_brand_category,
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
# TestInferBrandCategory
# ===========================================================================


class TestInferBrandCategory:
    """Tests for infer_brand_category() — LLM call (mocked)."""

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_returns_inferred_items(self, mock_llm):
        mock_llm.return_value = {
            "items": [
                {"idx": 1, "brand": "Ferrero", "category": "Spreads"},
                {"idx": 2, "brand": None, "category": "Dairy"},
            ]
        }

        items = [
            {"idx": 1, "name": "NUTELLA 400G"},
            {"idx": 2, "name": "FRESH MILK 1L"},
        ]
        result = infer_brand_category(items, vendor_name="Supermarket")

        assert len(result) == 2
        assert result[0]["brand"] == "Ferrero"
        assert result[1]["category"] == "Dairy"

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_empty_items_returns_empty(self, mock_llm):
        result = infer_brand_category([], vendor_name="Shop")
        assert result == []
        mock_llm.assert_not_called()

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_llm_failure_returns_empty(self, mock_llm):
        mock_llm.side_effect = Exception("Ollama down")
        items = [{"idx": 1, "name": "Test Item"}]
        result = infer_brand_category(items)
        assert result == []

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_non_list_response_returns_empty(self, mock_llm):
        mock_llm.return_value = {"items": "not a list"}
        items = [{"idx": 1, "name": "Test Item"}]
        result = infer_brand_category(items)
        assert result == []

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_missing_items_key_returns_empty(self, mock_llm):
        mock_llm.return_value = {"result": "no items key"}
        items = [{"idx": 1, "name": "Test Item"}]
        result = infer_brand_category(items)
        assert result == []

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_prompt_includes_vendor_and_items(self, mock_llm):
        mock_llm.return_value = {"items": []}
        items = [
            {"idx": 1, "name": "Alpha Product"},
            {"idx": 2, "name": "Beta Product"},
        ]
        infer_brand_category(items, vendor_name="TestShop")

        call_kwargs = mock_llm.call_args
        prompt = call_kwargs.kwargs.get("emphasis_prompt", "")
        assert "TestShop" in prompt
        assert "1. Alpha Product" in prompt
        assert "2. Beta Product" in prompt

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_model_and_url_passed_through(self, mock_llm):
        mock_llm.return_value = {"items": []}
        items = [{"idx": 1, "name": "Test"}]
        infer_brand_category(
            items,
            model="custom-model",
            ollama_url="http://custom:11434",
        )

        call_kwargs = mock_llm.call_args
        assert call_kwargs.kwargs.get("model") == "custom-model"
        assert call_kwargs.kwargs.get("ollama_url") == "http://custom:11434"


# ===========================================================================
# TestEnrichItemsByLlm
# ===========================================================================


class TestEnrichItemsByLlm:
    """Tests for enrich_items_by_llm() — batch DB update."""

    @patch("alibi.enrichment.llm_enrichment.infer_brand_category")
    def test_updates_db_on_success(self, mock_infer, db):
        mock_infer.return_value = [
            {"idx": 1, "brand": "Ferrero", "category": "Spreads"},
        ]
        _seed_fact_item(db, "i1", "Nutella 400g")

        results = enrich_items_by_llm(
            db, [{"id": "i1", "name": "Nutella 400g"}], vendor_name="Shop"
        )

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].brand == "Ferrero"
        assert results[0].category == "Spreads"

        # Verify DB was updated
        row = db.fetchone("SELECT brand, category FROM fact_items WHERE id = 'i1'")
        assert row["brand"] == "Ferrero"
        assert row["category"] == "Spreads"

    @patch("alibi.enrichment.llm_enrichment.infer_brand_category")
    def test_partial_results(self, mock_infer, db):
        """LLM returns brand for one item but not another."""
        mock_infer.return_value = [
            {"idx": 1, "brand": "Known", "category": "Dairy"},
            {"idx": 2, "brand": None, "category": None},
        ]
        _seed_fact_item(db, "i1", "Known Product")
        _seed_fact_item(db, "i2", "Unknown Product")

        results = enrich_items_by_llm(
            db,
            [
                {"id": "i1", "name": "Known Product"},
                {"id": "i2", "name": "Unknown Product"},
            ],
        )

        assert results[0].success is True
        assert results[1].success is False

    @patch("alibi.enrichment.llm_enrichment.infer_brand_category")
    def test_empty_items_returns_empty(self, mock_infer, db):
        results = enrich_items_by_llm(db, [])
        assert results == []
        mock_infer.assert_not_called()

    @patch("alibi.enrichment.llm_enrichment.infer_brand_category")
    def test_llm_returns_empty_gracefully(self, mock_infer, db):
        mock_infer.return_value = []
        _seed_fact_item(db, "i1", "Test Item")

        results = enrich_items_by_llm(db, [{"id": "i1", "name": "Test Item"}])

        assert len(results) == 1
        assert results[0].success is False

    @patch("alibi.enrichment.llm_enrichment.infer_brand_category")
    def test_only_brand_updates(self, mock_infer, db):
        """When LLM returns only brand, category stays NULL."""
        mock_infer.return_value = [
            {"idx": 1, "brand": "ACME", "category": None},
        ]
        _seed_fact_item(db, "i1", "Test Item")

        results = enrich_items_by_llm(db, [{"id": "i1", "name": "Test Item"}])

        assert results[0].success is True
        assert results[0].brand == "ACME"
        row = db.fetchone("SELECT brand, category FROM fact_items WHERE id = 'i1'")
        assert row["brand"] == "ACME"
        assert row["category"] is None


# ===========================================================================
# TestEnrichPendingByLlm
# ===========================================================================


class TestEnrichPendingByLlm:
    """Tests for enrich_pending_by_llm() — batch with vendor grouping."""

    @patch("alibi.enrichment.llm_enrichment.infer_brand_category")
    def test_groups_by_vendor(self, mock_infer, db):
        """Items from different vendors should result in separate LLM calls."""
        mock_infer.return_value = [
            {"idx": 1, "brand": "Brand", "category": "Cat"},
        ]
        _seed_fact_item(db, "i1", "Item A", vendor="Shop A")
        _seed_fact_item(db, "i2", "Item B", vendor="Shop B")

        enrich_pending_by_llm(db)

        # Should have been called twice (once per vendor)
        assert mock_infer.call_count == 2
        # Check vendor names in calls
        vendors_called = [
            call.kwargs.get("vendor_name") or call.args[1]
            for call in mock_infer.call_args_list
        ]
        assert set(vendors_called) == {"Shop A", "Shop B"}

    @patch("alibi.enrichment.llm_enrichment.infer_brand_category")
    def test_skips_already_enriched(self, mock_infer, db):
        _seed_fact_item(
            db,
            "i1",
            "Enriched Item",
            brand="Known",
            category="Cat",
        )

        results = enrich_pending_by_llm(db)
        assert results == []
        mock_infer.assert_not_called()

    @patch("alibi.enrichment.llm_enrichment.infer_brand_category")
    def test_empty_db_returns_empty(self, mock_infer, db):
        results = enrich_pending_by_llm(db)
        assert results == []
        mock_infer.assert_not_called()

    @patch("alibi.enrichment.llm_enrichment.infer_brand_category")
    def test_respects_limit(self, mock_infer, db):
        mock_infer.return_value = [
            {"idx": 1, "brand": "B", "category": "C"},
        ]
        for i in range(5):
            _seed_fact_item(db, f"i{i}", f"Item {i}", vendor="Shop")

        results = enrich_pending_by_llm(db, limit=2)
        assert len(results) == 2

    @patch("alibi.enrichment.llm_enrichment.infer_brand_category")
    def test_items_with_only_brand_missing_are_picked(self, mock_infer, db):
        """Items with empty brand AND category should be picked."""
        mock_infer.return_value = [
            {"idx": 1, "brand": "Inferred", "category": "Inferred Cat"},
        ]
        _seed_fact_item(db, "i1", "Needs Enrichment", brand="", category="")

        results = enrich_pending_by_llm(db)
        assert len(results) == 1
        assert results[0].success is True


# ===========================================================================
# TestPromptTemplate
# ===========================================================================


class TestPromptTemplate:
    """Tests for the prompt template formatting."""

    def test_template_has_vendor_placeholder(self):
        assert "{vendor}" in _ENRICHMENT_PROMPT_TEMPLATE

    def test_template_has_items_placeholder(self):
        assert "{items_block}" in _ENRICHMENT_PROMPT_TEMPLATE

    def test_template_requests_json(self):
        assert '"items"' in _ENRICHMENT_PROMPT_TEMPLATE

    def test_template_includes_category_suggestions(self):
        assert "Dairy" in _ENRICHMENT_PROMPT_TEMPLATE
        assert "Beverages" in _ENRICHMENT_PROMPT_TEMPLATE
        assert "Bakery" in _ENRICHMENT_PROMPT_TEMPLATE
