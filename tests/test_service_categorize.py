"""Tests for hierarchical category enrichment (task B)."""

from __future__ import annotations

import os
from unittest.mock import patch

os.environ["ALIBI_TESTING"] = "1"

from alibi.enrichment import taxonomy
from alibi.enrichment.categorize import (
    enrich_items,
    enrich_pending_categories,
    infer_categories,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_fact_item(
    db,
    item_id: str,
    name: str,
    *,
    vendor: str = "Test Store",
    category_path: str | None = None,
) -> None:
    """Insert a fact_item with its supporting chain."""
    doc_id = f"doc-{item_id}"
    atom_id = f"atom-{item_id}"
    cloud_id = f"cloud-{item_id}"
    fact_id = f"fact-{item_id}"

    conn = db.get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO documents (id, file_path, file_hash) VALUES (?, ?, ?)",
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
        "(id, cloud_id, fact_type, vendor, total_amount, currency, event_date) "
        "VALUES (?, ?, 'purchase', ?, 10.0, 'EUR', '2026-01-01')",
        (fact_id, cloud_id, vendor),
    )
    conn.execute(
        "INSERT OR IGNORE INTO fact_items "
        "(id, fact_id, atom_id, name, category_path) VALUES (?, ?, ?, ?, ?)",
        (item_id, fact_id, atom_id, name, category_path),
    )
    conn.commit()


# ===========================================================================
# TestTaxonomy
# ===========================================================================


class TestTaxonomy:
    def test_all_paths_non_empty_and_unique(self):
        paths = taxonomy.all_paths()
        assert len(paths) > 30
        assert len(paths) == len(set(paths))

    def test_known_paths_present(self):
        for p in (
            "food > dairy > milk",
            "food > produce > vegetables",
            "household",
            "adjustment > tax",
            "transport > fuel",
        ):
            assert taxonomy.is_valid_path(p), p

    def test_intermediate_nodes_are_valid(self):
        # The LLM may stop at a branch when no exact leaf fits.
        for p in ("food", "food > dairy", "food > produce", "adjustment"):
            assert taxonomy.is_valid_path(p), p
        assert taxonomy.normalize_path("food > dairy") == "food > dairy"
        assert taxonomy.leaf_of("food > dairy") == "dairy"

    def test_invalid_path_rejected(self):
        assert not taxonomy.is_valid_path("food > nonsense")
        assert not taxonomy.is_valid_path("")
        assert not taxonomy.is_valid_path(None)

    def test_normalize_handles_drift(self):
        assert taxonomy.normalize_path("Food > Dairy > Milk") == "food > dairy > milk"
        assert taxonomy.normalize_path("food/dairy/milk") == "food > dairy > milk"
        assert taxonomy.normalize_path("food>dairy>milk") == "food > dairy > milk"
        assert taxonomy.normalize_path("  household  ") == "household"

    def test_normalize_rejects_unknown(self):
        assert taxonomy.normalize_path("food > imaginary") is None
        assert taxonomy.normalize_path("xyz") is None

    def test_leaf_of(self):
        assert taxonomy.leaf_of("food > dairy > milk") == "milk"
        assert taxonomy.leaf_of("household") == "household"

    def test_render_for_prompt_includes_branches(self):
        rendered = taxonomy.render_for_prompt()
        assert "food" in rendered
        assert "dairy" in rendered
        assert "adjustment" in rendered


# ===========================================================================
# TestInferCategories (LLM mocked)
# ===========================================================================


class TestInferCategories:
    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_keeps_valid_paths_and_answered_invalid(self, mock_llm):
        # An invalid path the model answers maps to None (answered); an item it
        # omits entirely is absent (dropped → retry later).
        mock_llm.return_value = {
            "items": [
                {"idx": 1, "category_path": "food > dairy > milk"},
                {"idx": 2, "category_path": "food > imaginary"},  # invalid -> None
            ]
        }
        items = [
            {"name": "MILK 1L"},
            {"name": "MYSTERY"},
            {"name": "DISH SOAP"},
        ]
        out = infer_categories(items, vendor_name="Shop")
        assert out == {1: "food > dairy > milk", 2: None}
        assert 3 not in out  # dropped by the model

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_empty_items_no_call(self, mock_llm):
        assert infer_categories([]) == {}
        mock_llm.assert_not_called()

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_llm_failure_returns_empty(self, mock_llm):
        mock_llm.side_effect = Exception("Ollama down")
        assert infer_categories([{"name": "X"}]) == {}

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_prompt_includes_taxonomy_and_items(self, mock_llm):
        mock_llm.return_value = {"items": []}
        infer_categories([{"name": "Alpha", "brand": "BrandX"}], vendor_name="TestShop")
        prompt = mock_llm.call_args.kwargs.get("emphasis_prompt", "")
        assert "TestShop" in prompt
        assert "1. Alpha" in prompt
        assert "food" in prompt  # taxonomy rendered into prompt


# ===========================================================================
# TestEnrichItems (write-back)
# ===========================================================================


class TestEnrichItems:
    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_writes_path_and_leaf(self, mock_llm, db):
        mock_llm.return_value = {
            "items": [{"idx": 1, "category_path": "food > dairy > milk"}]
        }
        _seed_fact_item(db, "i1", "MILK 1L")
        results = enrich_items(db, [{"id": "i1", "name": "MILK 1L"}])
        assert results[0].success
        row = db.fetchone(
            "SELECT category, category_path, enrichment_source, "
            "enrichment_confidence FROM fact_items WHERE id = 'i1'"
        )
        assert row["category_path"] == "food > dairy > milk"
        # Leaf mirrored into flat category (title-cased by update_fact_item).
        assert row["category"] == "Milk"
        assert row["enrichment_source"] == "llm_category"
        assert row["enrichment_confidence"] == 0.7

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_invalid_path_not_written(self, mock_llm, db):
        mock_llm.return_value = {
            "items": [{"idx": 1, "category_path": "food > imaginary"}]
        }
        _seed_fact_item(db, "i2", "MYSTERY")
        results = enrich_items(db, [{"id": "i2", "name": "MYSTERY"}])
        assert not results[0].success
        row = db.fetchone("SELECT category_path FROM fact_items WHERE id = 'i2'")
        assert row["category_path"] is None


# ===========================================================================
# TestEnrichPending (selection + idempotency)
# ===========================================================================


class TestEnrichPending:
    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_only_uncategorized_selected(self, mock_llm, db):
        mock_llm.return_value = {
            "items": [{"idx": 1, "category_path": "food > dairy > milk"}]
        }
        _seed_fact_item(db, "done", "CHEESE", category_path="food > dairy > cheese")
        _seed_fact_item(db, "todo", "MILK")
        results = enrich_pending_categories(db, limit=50)
        # Only the uncategorized item is processed.
        assert {r.item_id for r in results} == {"todo"}

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_idempotent_rerun_is_noop(self, mock_llm, db):
        mock_llm.return_value = {
            "items": [{"idx": 1, "category_path": "food > dairy > milk"}]
        }
        _seed_fact_item(db, "m1", "MILK")
        first = enrich_pending_categories(db, limit=50)
        assert len(first) == 1 and first[0].success
        mock_llm.reset_mock()
        second = enrich_pending_categories(db, limit=50)
        assert second == []
        mock_llm.assert_not_called()

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_unmappable_answer_not_rescanned(self, mock_llm, db):
        # A line the model maps to an invalid path stays category_path NULL but
        # is stamped with the taxonomy version, so a rerun is a no-op.
        mock_llm.return_value = {
            "items": [{"idx": 1, "category_path": "food > imaginary"}]
        }
        _seed_fact_item(db, "u1", "MYSTERY")
        first = enrich_pending_categories(db, limit=50)
        assert len(first) == 1 and not first[0].success
        row = db.fetchone(
            "SELECT category_path, category_taxonomy_version "
            "FROM fact_items WHERE id = 'u1'"
        )
        assert row["category_path"] is None
        assert row["category_taxonomy_version"] == taxonomy.TAXONOMY_VERSION
        mock_llm.reset_mock()
        second = enrich_pending_categories(db, limit=50)
        assert second == []
        mock_llm.assert_not_called()

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_dropped_item_is_retried(self, mock_llm, db):
        # An item the model omits stays unstamped and is selected again.
        mock_llm.return_value = {"items": []}
        _seed_fact_item(db, "d1", "GHOST")
        first = enrich_pending_categories(db, limit=50)
        assert len(first) == 1 and not first[0].success
        row = db.fetchone(
            "SELECT category_taxonomy_version FROM fact_items WHERE id = 'd1'"
        )
        assert row["category_taxonomy_version"] is None
        second = enrich_pending_categories(db, limit=50)
        assert {r.item_id for r in second} == {"d1"}

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_taxonomy_bump_reselects_unmapped_row(self, mock_llm, db):
        # A row stamped under an older taxonomy version becomes eligible again
        # when TAXONOMY_VERSION is bumped.
        mock_llm.return_value = {
            "items": [{"idx": 1, "category_path": "food > imaginary"}]
        }
        _seed_fact_item(db, "b1", "MYSTERY")
        enrich_pending_categories(db, limit=50)
        assert enrich_pending_categories(db, limit=50) == []  # converged at v1
        mock_llm.reset_mock()
        mock_llm.return_value = {"items": [{"idx": 1, "category_path": "household"}]}
        with patch.object(taxonomy, "TAXONOMY_VERSION", taxonomy.TAXONOMY_VERSION + 1):
            again = enrich_pending_categories(db, limit=50)
        assert {r.item_id for r in again} == {"b1"}
        mock_llm.assert_called_once()
