"""Tests for the combined (one-call) local-LLM enrichment pass."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

os.environ["ALIBI_TESTING"] = "1"

from alibi.enrichment.combined import (
    enrich_items,
    enrich_pending_combined,
    infer_combined,
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


def _row(db, item_id):
    return db.fetchone(
        "SELECT unit, unit_quantity, comparable_name, comparable_name_enriched, "
        "       category_path, category, category_taxonomy_version, attributes, "
        "       unit_enriched, comparable_unit, comparable_unit_price "
        "FROM fact_items WHERE id = ?",
        (item_id,),
    )


# ===========================================================================
# TestInferCombined — one reply split into four answer maps
# ===========================================================================


class TestInferCombined:
    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_splits_all_four_fields(self, mock_llm):
        mock_llm.return_value = {
            "items": [
                {
                    "idx": 1,
                    "unit": "g",
                    "unit_quantity": 450,
                    "comparable_name": "gouda cheese",
                    "category_path": "food > dairy > cheese",
                    "attributes": {"type": "gouda"},
                    "pack_count": None,
                }
            ]
        }
        units, names, cats, attrs = infer_combined([{"name": "GOUDA 450G"}])
        from decimal import Decimal

        assert units == {1: ("g", Decimal("450"))}
        assert names == {1: "gouda cheese"}
        assert cats == {1: "food > dairy > cheese"}
        assert attrs == {1: ({"type": "gouda"}, None)}

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_present_null_vs_absent_per_field(self, mock_llm):
        # idx 1: answers comparable_name=null (answered-null) but omits the unit
        # keys entirely (dropped). They must be distinguishable downstream.
        mock_llm.return_value = {
            "items": [
                {"idx": 1, "comparable_name": None, "category_path": "other"},
            ]
        }
        units, names, cats, attrs = infer_combined([{"name": "TAX"}])
        assert 1 not in units  # unit keys absent -> dropped
        assert names == {1: None}  # present-null -> answered no-result
        assert cats == {1: "other"}
        assert 1 not in attrs  # attributes key absent -> dropped

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_ignores_malformed_entries(self, mock_llm):
        mock_llm.return_value = {
            "items": [
                "not a dict",
                {"no_idx": True},
                {"idx": "x", "comparable_name": "foo"},
                {"idx": 2, "comparable_name": "milk"},
            ]
        }
        _, names, _, _ = infer_combined([{"name": "A"}, {"name": "MILK"}])
        assert names == {2: "milk"}

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_empty_items_no_call(self, mock_llm):
        assert infer_combined([]) == ({}, {}, {}, {})
        mock_llm.assert_not_called()

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_llm_failure_returns_empty_maps(self, mock_llm):
        mock_llm.side_effect = Exception("down")
        assert infer_combined([{"name": "X"}]) == ({}, {}, {}, {})


# ===========================================================================
# TestEnrichItems — write-back, needs-gating, field interaction
# ===========================================================================


class TestEnrichItems:
    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_writes_every_field_for_a_fresh_item(self, mock_llm, db):
        mock_llm.return_value = {
            "items": [
                {
                    "idx": 1,
                    "unit": "g",
                    "unit_quantity": 450,
                    "comparable_name": "gouda cheese",
                    "category_path": "food > dairy > cheese",
                    "attributes": {"type": "gouda"},
                    "pack_count": None,
                }
            ]
        }
        _seed_fact_item(db, "i1", "GOUDA 450G", total_price=4.50)
        results = enrich_pending_combined(db, limit=50)
        assert len(results) == 1
        r = _row(db, "i1")
        assert r["unit"] == "g"
        assert float(r["unit_quantity"]) == 450.0
        assert r["comparable_name"] == "gouda cheese"
        assert r["category_path"] == "food > dairy > cheese"
        assert r["category"] == "Cheese"  # leaf_of title-cases the leaf
        assert json.loads(r["attributes"]) == {"type": "gouda"}
        # all four markers set
        assert r["unit_enriched"] == 1
        assert r["comparable_name_enriched"] == 1
        assert r["category_taxonomy_version"] == 1

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_does_not_overwrite_populated_fields(self, mock_llm, db):
        # Item already has a comparable_name; only category is missing. The
        # combined model returns a (different) comparable_name too, which must be
        # ignored, and comparable_name_enriched must NOT be stamped.
        mock_llm.return_value = {
            "items": [
                {
                    "idx": 1,
                    "comparable_name": "WRONG NEW NAME",
                    "category_path": "food > produce > fruit",
                    "attributes": {},
                }
            ]
        }
        _seed_fact_item(
            db,
            "i2",
            "APPLES",
            comparable_name="apples",
            unit_quantity=1.0,  # not missing -> needs_unit False
            attributes="{}",  # not missing -> needs_attr False
        )
        enrich_pending_combined(db, limit=50)
        r = _row(db, "i2")
        assert r["comparable_name"] == "apples"  # preserved
        assert r["comparable_name_enriched"] is None  # field never touched
        assert r["category_path"] == "food > produce > fruit"  # the one gap, filled

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_sized_item_suppresses_count_pack(self, mock_llm, db):
        # A sized item that ALSO comes back with a pack_count: units run first and
        # set unit_quantity, so the attributes count-pack recompute must not fire.
        mock_llm.return_value = {
            "items": [
                {
                    "idx": 1,
                    "unit": "g",
                    "unit_quantity": 450,
                    "comparable_name": "pasta",
                    "category_path": "other",
                    "attributes": {},
                    "pack_count": 12,
                }
            ]
        }
        _seed_fact_item(db, "i3", "PASTA 450G", total_price=1.80)
        enrich_pending_combined(db, limit=50)
        r = _row(db, "i3")
        assert r["unit"] == "g"  # not "pcs"
        assert float(r["unit_quantity"]) == 450.0  # not 12

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_count_pack_recomputed_per_piece(self, mock_llm, db):
        # No weight/volume size, but a pack count: attributes turns it per-piece.
        mock_llm.return_value = {
            "items": [
                {
                    "idx": 1,
                    "unit": None,
                    "unit_quantity": None,
                    "comparable_name": "eggs",
                    "category_path": "food > dairy > eggs",
                    "attributes": {"size": "L"},
                    "pack_count": 12,
                }
            ]
        }
        _seed_fact_item(db, "i4", "EGGS L X12", total_price=3.00)
        enrich_pending_combined(db, limit=50)
        r = _row(db, "i4")
        assert r["unit"] == "pcs"
        assert float(r["unit_quantity"]) == 12.0
        assert r["comparable_unit"] == "pcs"
        assert abs(float(r["comparable_unit_price"]) - 0.25) < 1e-6
        assert json.loads(r["attributes"]) == {"size": "l"}  # values lowercased
        # unit answered-null -> marked so the units pass won't re-send it
        assert r["unit_enriched"] == 1


# ===========================================================================
# TestMarkingAndSelection — idempotency + dropped-field fallback
# ===========================================================================


class TestMarkingAndSelection:
    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_dropped_field_stays_pending(self, mock_llm, db):
        # The model answers comparable_name but omits category entirely; category
        # must stay unmarked so a later run / the fallback pass retries it.
        mock_llm.return_value = {
            "items": [
                {"idx": 1, "comparable_name": "milk", "attributes": {}},
            ]
        }
        _seed_fact_item(db, "i5", "MILK", unit_quantity=1.0)
        enrich_pending_combined(db, limit=50)
        r = _row(db, "i5")
        assert r["comparable_name"] == "milk"
        assert r["comparable_name_enriched"] == 1
        assert r["category_path"] is None
        assert r["category_taxonomy_version"] is None  # dropped -> retry

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_converged_item_not_selected(self, mock_llm, db):
        mock_llm.return_value = {"items": []}
        _seed_fact_item(
            db,
            "done",
            "COMPLETE",
            unit_quantity=1.0,
            comparable_name="complete",
            category_path="other",
            category_taxonomy_version=1,
            attributes="{}",
        )
        assert enrich_pending_combined(db, limit=50) == []
        mock_llm.assert_not_called()

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_idempotent_rerun_makes_no_call(self, mock_llm, db):
        mock_llm.return_value = {
            "items": [
                {
                    "idx": 1,
                    "unit": "g",
                    "unit_quantity": 450,
                    "comparable_name": "pasta",
                    "category_path": "other",
                    "attributes": {},
                    "pack_count": None,
                }
            ]
        }
        _seed_fact_item(db, "p", "PASTA 450G", total_price=1.80)
        first = enrich_pending_combined(db, limit=50)
        assert len(first) == 1 and first[0].changed
        mock_llm.reset_mock()
        second = enrich_pending_combined(db, limit=50)
        assert second == []
        mock_llm.assert_not_called()
