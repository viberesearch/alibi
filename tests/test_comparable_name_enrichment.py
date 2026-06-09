"""Tests for local-first comparable_name enrichment."""

from __future__ import annotations

import os
from unittest.mock import patch

os.environ["ALIBI_TESTING"] = "1"

from alibi.enrichment.comparable_names import (
    _clean_comparable_name,
    _tidy_comparable_name,
    enrich_items,
    enrich_pending_comparable_names,
    infer_comparable_names,
    retidy_comparable_names,
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
    comparable_name: str | None = None,
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
        "(id, fact_id, atom_id, name, comparable_name) VALUES (?, ?, ?, ?, ?)",
        (item_id, fact_id, atom_id, name, comparable_name),
    )
    conn.commit()


# ===========================================================================
# TestClean
# ===========================================================================


class TestClean:
    def test_trims_and_lowercases(self):
        assert _clean_comparable_name("  Gouda Cheese ") == "gouda cheese"

    def test_blank_and_nullish_rejected(self):
        for v in ("", "   ", "null", "None", "n/a", None, 123, ["x"]):
            assert _clean_comparable_name(v) is None

    def test_overlong_rejected(self):
        assert _clean_comparable_name("x" * 81) is None
        assert _clean_comparable_name("x" * 80) == "x" * 80

    def test_clean_also_tidies_size_tokens(self):
        # _clean_comparable_name applies the tidy, so the LLM pass never stores
        # a size/percentage token even when the model fails to strip it.
        assert _clean_comparable_name("Olive Oil 2L") == "olive oil"
        assert _clean_comparable_name("Cottage Cheese 9%") == "cottage cheese"


# ===========================================================================
# TestTidy (deterministic size/percentage stripping)
# ===========================================================================


class TestTidy:
    def test_strips_size_units(self):
        assert _tidy_comparable_name("olive oil 2l") == "olive oil"
        assert _tidy_comparable_name("octopus 1350g") == "octopus"
        assert _tidy_comparable_name("pang fillet 700g") == "pang fillet"

    def test_strips_percentages(self):
        assert _tidy_comparable_name("cottage cheese 9%") == "cottage cheese"
        assert _tidy_comparable_name("sour cream 20%") == "sour cream"
        assert _tidy_comparable_name("twin cheese 0.5%") == "twin cheese"

    def test_strips_pack_counts(self):
        assert _tidy_comparable_name("eggs large x12") == "eggs large"
        assert _tidy_comparable_name("eggs x30") == "eggs"
        assert _tidy_comparable_name("tomato paste 4x70g") == "tomato paste"

    def test_collapses_variants_to_one_bucket(self):
        # The whole point: count/size variants of one product merge.
        assert (
            _tidy_comparable_name("eggs large x12")
            == _tidy_comparable_name("eggs large x30")
            == "eggs large"
        )

    def test_is_idempotent_fixpoint(self):
        for s in ("olive oil 2l", "cottage cheese 9%", "eggs large x12", "milk"):
            once = _tidy_comparable_name(s)
            assert _tidy_comparable_name(once) == once

    def test_preserves_clean_names_and_word_content(self):
        # No size/pack/percentage token -> unchanged. Never strips words/brands.
        for s in ("gouda cheese", "extra virgin olive oil", "free range eggs"):
            assert _tidy_comparable_name(s) == s

    def test_does_not_strip_mid_word_digits(self):
        # A digit-unit inside a word is not a size token.
        assert _tidy_comparable_name("7up") == "7up"
        assert _tidy_comparable_name("omega3 oil") == "omega3 oil"


# ===========================================================================
# TestRetidy (backfill over stored rows)
# ===========================================================================


class TestRetidy:
    def test_rewrites_only_leaky_rows(self, db):
        _seed_fact_item(db, "leak", "OLIVE OIL 2L", comparable_name="olive oil 2l")
        _seed_fact_item(db, "clean", "GOUDA", comparable_name="gouda cheese")
        changes = retidy_comparable_names(db)
        assert {c.item_id for c in changes} == {"leak"}
        assert (
            db.fetchone("SELECT comparable_name FROM fact_items WHERE id = 'leak'")[
                "comparable_name"
            ]
            == "olive oil"
        )
        assert (
            db.fetchone("SELECT comparable_name FROM fact_items WHERE id = 'clean'")[
                "comparable_name"
            ]
            == "gouda cheese"
        )

    def test_rerun_is_noop(self, db):
        _seed_fact_item(db, "e1", "EGGS L X12", comparable_name="eggs large x12")
        first = retidy_comparable_names(db)
        assert len(first) == 1 and first[0].after == "eggs large"
        assert retidy_comparable_names(db) == []


# ===========================================================================
# TestInfer (LLM mocked)
# ===========================================================================


class TestInfer:
    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_keeps_clean_names_and_answered_nulls(self, mock_llm):
        # A non-product the model answers null for maps to None (answered);
        # an item it omits is absent (dropped → retry later).
        mock_llm.return_value = {
            "items": [
                {"idx": 1, "comparable_name": "Gouda Cheese"},
                {"idx": 2, "comparable_name": None},  # non-product -> answered
            ]
        }
        items = [{"name": "GOUDA"}, {"name": "TOTAL"}, {"name": "EGGS L"}]
        out = infer_comparable_names(items, vendor_name="Shop")
        assert out == {1: "gouda cheese", 2: None}
        assert 3 not in out  # dropped by the model

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_empty_items_no_call(self, mock_llm):
        assert infer_comparable_names([]) == {}
        mock_llm.assert_not_called()

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_llm_failure_returns_empty(self, mock_llm):
        mock_llm.side_effect = Exception("Ollama down")
        assert infer_comparable_names([{"name": "X"}]) == {}

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_prompt_includes_vendor_and_items(self, mock_llm):
        mock_llm.return_value = {"items": []}
        infer_comparable_names(
            [{"name": "Alpha", "brand": "BrandX"}], vendor_name="TestShop"
        )
        prompt = mock_llm.call_args.kwargs.get("emphasis_prompt", "")
        assert "TestShop" in prompt
        assert "1. Alpha" in prompt


# ===========================================================================
# TestEnrichItems (write-back)
# ===========================================================================


class TestEnrichItems:
    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_writes_comparable_name(self, mock_llm, db):
        mock_llm.return_value = {
            "items": [{"idx": 1, "comparable_name": "Gouda Cheese"}]
        }
        _seed_fact_item(db, "i1", "TYPI GOUDA 450G")
        results = enrich_items(db, [{"id": "i1", "name": "TYPI GOUDA 450G"}])
        assert results[0].success
        row = db.fetchone(
            "SELECT comparable_name, enrichment_source, enrichment_confidence "
            "FROM fact_items WHERE id = 'i1'"
        )
        assert row["comparable_name"] == "gouda cheese"
        assert row["enrichment_source"] == "llm_comparable_name"
        assert row["enrichment_confidence"] == 0.7

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_null_name_not_written(self, mock_llm, db):
        mock_llm.return_value = {"items": [{"idx": 1, "comparable_name": None}]}
        _seed_fact_item(db, "i2", "TOTAL")
        results = enrich_items(db, [{"id": "i2", "name": "TOTAL"}])
        assert not results[0].success
        row = db.fetchone("SELECT comparable_name FROM fact_items WHERE id = 'i2'")
        assert row["comparable_name"] is None


# ===========================================================================
# TestEnrichPending (selection + idempotency)
# ===========================================================================


class TestEnrichPending:
    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_only_unnamed_selected(self, mock_llm, db):
        mock_llm.return_value = {"items": [{"idx": 1, "comparable_name": "milk"}]}
        _seed_fact_item(db, "done", "GALA", comparable_name="milk")
        _seed_fact_item(db, "todo", "FRESH MILK 1L")
        results = enrich_pending_comparable_names(db, limit=50)
        assert {r.item_id for r in results} == {"todo"}

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_idempotent_rerun_is_noop(self, mock_llm, db):
        mock_llm.return_value = {"items": [{"idx": 1, "comparable_name": "milk"}]}
        _seed_fact_item(db, "m1", "MILK")
        first = enrich_pending_comparable_names(db, limit=50)
        assert len(first) == 1 and first[0].success
        mock_llm.reset_mock()
        second = enrich_pending_comparable_names(db, limit=50)
        assert second == []
        mock_llm.assert_not_called()

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_null_answer_not_rescanned(self, mock_llm, db):
        # A non-product line the model returns null for keeps a NULL
        # comparable_name but is marked processed, so a rerun is a no-op.
        mock_llm.return_value = {"items": [{"idx": 1, "comparable_name": None}]}
        _seed_fact_item(db, "t1", "TOTAL")
        first = enrich_pending_comparable_names(db, limit=50)
        assert len(first) == 1 and not first[0].success
        row = db.fetchone(
            "SELECT comparable_name, comparable_name_enriched "
            "FROM fact_items WHERE id = 't1'"
        )
        assert row["comparable_name"] is None
        assert row["comparable_name_enriched"] == 1
        mock_llm.reset_mock()
        second = enrich_pending_comparable_names(db, limit=50)
        assert second == []
        mock_llm.assert_not_called()

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_dropped_item_is_retried(self, mock_llm, db):
        # An item the model omits from its response stays unmarked and is
        # selected again on the next run.
        mock_llm.return_value = {"items": []}
        _seed_fact_item(db, "d1", "MYSTERY")
        first = enrich_pending_comparable_names(db, limit=50)
        assert len(first) == 1 and not first[0].success
        row = db.fetchone(
            "SELECT comparable_name_enriched FROM fact_items WHERE id = 'd1'"
        )
        assert row["comparable_name_enriched"] is None
        second = enrich_pending_comparable_names(db, limit=50)
        assert {r.item_id for r in second} == {"d1"}
