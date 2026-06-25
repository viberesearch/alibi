"""Tests for the LLM unit-extraction enrichment pass."""

from __future__ import annotations

import os
from unittest.mock import patch

os.environ["ALIBI_TESTING"] = "1"

from alibi.enrichment.units import (
    _clean_size,
    enrich_items,
    enrich_pending_units,
    infer_units,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_fact_item(db, item_id: str, name: str, **cols) -> None:
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
        "VALUES (?, ?, 'purchase', 'Store', 10.0, 'EUR', '2026-01-01')",
        (fact_id, cloud_id),
    )
    keys = ["id", "fact_id", "atom_id", "name", "quantity"]
    vals = [item_id, fact_id, atom_id, name, cols.pop("quantity", 1)]
    for k, v in cols.items():
        keys.append(k)
        vals.append(v)
    ph = ", ".join("?" for _ in keys)
    conn.execute(
        f"INSERT OR IGNORE INTO fact_items ({', '.join(keys)}) VALUES ({ph})",  # noqa: S608
        tuple(vals),
    )
    conn.commit()


def _get(db, item_id):
    return db.fetchone(
        "SELECT unit, unit_quantity, comparable_unit, comparable_unit_price "
        "FROM fact_items WHERE id = ?",
        (item_id,),
    )


# ===========================================================================
# TestCleanSize
# ===========================================================================


class TestCleanSize:
    def test_valid_pairs(self):
        assert _clean_size("g", 450) == ("g", __import__("decimal").Decimal("450"))
        assert _clean_size("L", "2") == ("l", __import__("decimal").Decimal("2"))
        assert _clean_size("ml", "1,5") == ("ml", __import__("decimal").Decimal("1.5"))

    def test_rejects_nulls_and_counts(self):
        assert _clean_size(None, 5) is None
        assert _clean_size("g", None) is None
        assert _clean_size("pcs", 12) is None  # count unit, not sized
        assert _clean_size("cl", 50) is None  # unsupported -> OTHER
        assert _clean_size("g", 0) is None
        assert _clean_size("g", "abc") is None


# ===========================================================================
# TestInfer (LLM mocked)
# ===========================================================================


class TestInfer:
    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_keeps_sized_and_answered_nulls(self, mock_llm):
        # An item the model answers with no size maps to None (answered);
        # an item it omits entirely is absent (dropped → retry later).
        mock_llm.return_value = {
            "items": [
                {"idx": 1, "unit": "g", "unit_quantity": 450},
                {"idx": 2, "unit": None, "unit_quantity": None},
            ]
        }
        out = infer_units(
            [{"name": "PASTA 450G"}, {"name": "BANANAS"}, {"name": "OIL 2L"}]
        )
        from decimal import Decimal

        assert out == {1: ("g", Decimal("450")), 2: None}
        assert 3 not in out  # dropped by the model

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_empty_no_call(self, mock_llm):
        assert infer_units([]) == {}
        mock_llm.assert_not_called()

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_llm_failure_returns_empty(self, mock_llm):
        mock_llm.side_effect = Exception("down")
        assert infer_units([{"name": "X 1L"}]) == {}


# ===========================================================================
# TestEnrichItems (write-back + comparable recompute)
# ===========================================================================


class TestEnrichItems:
    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_writes_unit_and_recomputes_price(self, mock_llm, db):
        mock_llm.return_value = {
            "items": [{"idx": 1, "unit": "g", "unit_quantity": 450}]
        }
        _seed_fact_item(db, "i1", "PASTA 450G", total_price=1.80, comparable_unit="pcs")
        results = enrich_items(
            db, [{"id": "i1", "name": "PASTA 450G", "total_price": 1.80, "quantity": 1}]
        )
        assert results[0].success
        row = _get(db, "i1")
        assert row["unit"] == "g"
        assert float(row["unit_quantity"]) == 450.0
        assert row["comparable_unit"] == "kg"
        assert float(row["comparable_unit_price"]) == 4.00

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_no_size_left_untouched(self, mock_llm, db):
        mock_llm.return_value = {
            "items": [{"idx": 1, "unit": None, "unit_quantity": None}]
        }
        _seed_fact_item(db, "i2", "BANANAS", total_price=1.20)
        results = enrich_items(
            db, [{"id": "i2", "name": "BANANAS", "total_price": 1.20, "quantity": 1}]
        )
        assert not results[0].success
        row = _get(db, "i2")
        assert row["unit_quantity"] is None

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_implausible_price_not_written(self, mock_llm, db):
        # tiny size + big price -> implausible per-unit; unit still set, price not.
        mock_llm.return_value = {"items": [{"idx": 1, "unit": "g", "unit_quantity": 1}]}
        _seed_fact_item(db, "i3", "CAVIAR 1G", total_price=9911.96)
        enrich_items(
            db,
            [{"id": "i3", "name": "CAVIAR 1G", "total_price": 9911.96, "quantity": 1}],
        )
        row = _get(db, "i3")
        assert row["unit"] == "g"  # unit recorded
        assert row["comparable_unit_price"] is None  # implausible -> skipped


# ===========================================================================
# TestEnrichPending (selection + idempotency)
# ===========================================================================


class TestEnrichPending:
    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_only_unsized_selected(self, mock_llm, db):
        mock_llm.return_value = {
            "items": [{"idx": 1, "unit": "g", "unit_quantity": 450}]
        }
        _seed_fact_item(db, "done", "RICE 1KG", unit_quantity=1.0)
        _seed_fact_item(db, "todo", "PASTA 450G", total_price=1.80)
        results = enrich_pending_units(db, limit=50)
        assert {r.item_id for r in results} == {"todo"}

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_idempotent_rerun(self, mock_llm, db):
        mock_llm.return_value = {
            "items": [{"idx": 1, "unit": "g", "unit_quantity": 450}]
        }
        _seed_fact_item(db, "p", "PASTA 450G", total_price=1.80)
        first = enrich_pending_units(db, limit=50)
        assert len(first) == 1 and first[0].success
        mock_llm.reset_mock()
        second = enrich_pending_units(db, limit=50)
        assert second == []
        mock_llm.assert_not_called()

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_no_size_answer_not_rescanned(self, mock_llm, db):
        # A row the model says has no size stays unit_quantity NULL but is
        # marked processed, so a rerun must not re-send it to the LLM.
        mock_llm.return_value = {
            "items": [{"idx": 1, "unit": None, "unit_quantity": None}]
        }
        _seed_fact_item(db, "n", "BANANAS", total_price=1.20)
        first = enrich_pending_units(db, limit=50)
        assert len(first) == 1 and not first[0].success
        row = _get(db, "n")
        assert row["unit_quantity"] is None
        marker = db.fetchone("SELECT unit_enriched FROM fact_items WHERE id = 'n'")
        assert marker["unit_enriched"] == 1
        mock_llm.reset_mock()
        second = enrich_pending_units(db, limit=50)
        assert second == []
        mock_llm.assert_not_called()

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_dropped_item_is_retried(self, mock_llm, db):
        # When the model omits an item from its response, it is left unmarked
        # and selected again on the next run.
        mock_llm.return_value = {"items": []}
        _seed_fact_item(db, "d", "MYSTERY ITEM", total_price=1.00)
        first = enrich_pending_units(db, limit=50)
        assert len(first) == 1 and not first[0].success
        marker = db.fetchone("SELECT unit_enriched FROM fact_items WHERE id = 'd'")
        assert marker["unit_enriched"] is None
        second = enrich_pending_units(db, limit=50)
        assert {r.item_id for r in second} == {"d"}


# ===========================================================================
# TestSchema (constrained decoding)
# ===========================================================================


class TestSchema:
    @patch("alibi.enrichment.units.call_enrichment_llm")
    def test_constrains_decoding_with_schema(self, mock_llm):
        # The response_format schema is what makes the local model unable to emit
        # malformed JSON on garbled batches — assert it is passed through.
        from alibi.enrichment.units import _RESPONSE_FORMAT

        mock_llm.return_value = [{"idx": 1, "unit": "g", "unit_quantity": 1}]
        infer_units([{"name": "a"}])
        assert mock_llm.call_args.kwargs["response_format"] is _RESPONSE_FORMAT
