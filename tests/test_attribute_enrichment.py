"""Tests for flexible product-attribute enrichment."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

os.environ["ALIBI_TESTING"] = "1"

from alibi.enrichment.attributes import (
    _clean_attributes,
    _clean_pack_count,
    _derive_variant,
    enrich_items,
    enrich_pending_attributes,
    infer_attributes,
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
        "SELECT attributes, product_variant, unit, unit_quantity, "
        "comparable_unit, comparable_unit_price FROM fact_items WHERE id = ?",
        (item_id,),
    )


# ===========================================================================
# TestNormalisation
# ===========================================================================


class TestNormalisation:
    def test_clean_attributes_coerces_types(self):
        out = _clean_attributes(
            {"size": "L", "organic": "yes", "fat_pct": "3,6", "light": False}
        )
        assert out == {"size": "l", "organic": True, "fat_pct": 3.6, "light": False}

    def test_clean_attributes_drops_junk(self):
        assert _clean_attributes(None) == {}
        assert _clean_attributes("nope") == {}
        assert _clean_attributes({"": "x", "size": ""}) == {}
        assert _clean_attributes({"note": "x" * 50}) == {}  # too long

    def test_clean_pack_count(self):
        assert _clean_pack_count(12) == 12
        assert _clean_pack_count("30") == 30
        assert _clean_pack_count(1) is None  # not a pack
        assert _clean_pack_count(0) is None
        assert _clean_pack_count(None) is None
        assert _clean_pack_count("abc") is None

    def test_derive_variant_priority(self):
        assert _derive_variant({"size": "L", "fat_pct": 3.6}) == "L"
        assert _derive_variant({"fat_pct": 3.6}) == "3.6%"
        assert _derive_variant({"type": "gouda"}) == "gouda"
        assert _derive_variant({"organic": True}) is None


# ===========================================================================
# TestInfer (LLM mocked)
# ===========================================================================


class TestInfer:
    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_returns_attrs_and_count(self, mock_llm):
        mock_llm.return_value = {
            "items": [
                {
                    "idx": 1,
                    "attributes": {"size": "L", "organic": True},
                    "pack_count": 12,
                },
                {"idx": 2, "attributes": {}, "pack_count": None},
            ]
        }
        out = infer_attributes([{"name": "EGGS L X12 ORGANIC"}, {"name": "BANANAS"}])
        assert out[1] == ({"size": "l", "organic": True}, 12)
        assert out[2] == ({}, None)

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_empty_no_call(self, mock_llm):
        assert infer_attributes([]) == {}
        mock_llm.assert_not_called()

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_llm_failure_returns_empty(self, mock_llm):
        mock_llm.side_effect = Exception("down")
        assert infer_attributes([{"name": "X"}]) == {}


# ===========================================================================
# TestEnrichItems (write-back)
# ===========================================================================


class TestEnrichItems:
    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_writes_attributes_variant_and_per_piece(self, mock_llm, db):
        mock_llm.return_value = {
            "items": [
                {
                    "idx": 1,
                    "attributes": {"size": "L", "organic": True, "free_range": True},
                    "pack_count": 12,
                }
            ]
        }
        _seed_fact_item(db, "e1", "FRESH EGGS L X12 ORGANIC", total_price=3.60)
        enrich_items(
            db,
            [
                {
                    "id": "e1",
                    "name": "FRESH EGGS L X12 ORGANIC",
                    "total_price": 3.60,
                    "quantity": 1,
                    "unit_quantity": None,
                    "product_variant": None,
                }
            ],
        )
        row = _get(db, "e1")
        attrs = json.loads(row["attributes"])
        assert attrs == {"size": "l", "organic": True, "free_range": True}
        assert row["product_variant"] == "l"  # back-filled from size
        # Pack count -> per-egg price: 3.60 / 12 = 0.30.
        assert float(row["unit_quantity"]) == 12.0
        assert row["comparable_unit"] == "pcs"
        assert float(row["comparable_unit_price"]) == 0.30

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_no_facet_writes_empty_map(self, mock_llm, db):
        mock_llm.return_value = {
            "items": [{"idx": 1, "attributes": {}, "pack_count": None}]
        }
        _seed_fact_item(db, "b1", "BANANAS", total_price=1.20)
        enrich_items(
            db, [{"id": "b1", "name": "BANANAS", "total_price": 1.20, "quantity": 1}]
        )
        row = _get(db, "b1")
        assert row["attributes"] == "{}"  # marked processed, no facets
        assert row["unit_quantity"] is None

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_does_not_clobber_existing_unit_quantity(self, mock_llm, db):
        # A weighed item already has unit_quantity; a stray pack_count must not override.
        mock_llm.return_value = {
            "items": [{"idx": 1, "attributes": {"type": "feta"}, "pack_count": 6}]
        }
        _seed_fact_item(
            db,
            "c1",
            "FETA 200G",
            total_price=2.0,
            unit_quantity=0.2,
            unit="kg",
            comparable_unit="kg",
            comparable_unit_price=10.0,
        )
        enrich_items(
            db,
            [
                {
                    "id": "c1",
                    "name": "FETA 200G",
                    "total_price": 2.0,
                    "quantity": 1,
                    "unit_quantity": 0.2,
                    "product_variant": None,
                }
            ],
        )
        row = _get(db, "c1")
        assert json.loads(row["attributes"]) == {"type": "feta"}
        assert float(row["unit_quantity"]) == 0.2  # unchanged
        assert row["comparable_unit"] == "kg"  # unchanged


# ===========================================================================
# TestEnrichPending (selection + idempotency)
# ===========================================================================


class TestEnrichPending:
    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_only_unprocessed_selected(self, mock_llm, db):
        mock_llm.return_value = {
            "items": [{"idx": 1, "attributes": {"size": "M"}, "pack_count": None}]
        }
        _seed_fact_item(db, "done", "EGGS M", attributes="{}")
        _seed_fact_item(db, "todo", "EGGS L", total_price=3.0)
        results = enrich_pending_attributes(db, limit=50)
        assert {r.item_id for r in results} == {"todo"}

    @patch("alibi.extraction.structurer.structure_ocr_text")
    def test_idempotent_rerun(self, mock_llm, db):
        mock_llm.return_value = {
            "items": [{"idx": 1, "attributes": {}, "pack_count": None}]
        }
        _seed_fact_item(db, "x", "PLAIN ITEM", total_price=2.0)
        first = enrich_pending_attributes(db, limit=50)
        assert len(first) == 1
        mock_llm.reset_mock()
        second = enrich_pending_attributes(db, limit=50)
        assert second == []
        mock_llm.assert_not_called()


# ===========================================================================
# TestSchema (constrained decoding)
# ===========================================================================


class TestSchema:
    @patch("alibi.enrichment.attributes.call_enrichment_llm")
    def test_constrains_decoding_with_schema(self, mock_llm):
        # The response_format schema is what makes the local model unable to emit
        # malformed JSON on garbled batches — assert it is passed through.
        from alibi.enrichment.attributes import _RESPONSE_FORMAT

        mock_llm.return_value = [{"idx": 1, "attributes": {}, "pack_count": None}]
        infer_attributes([{"name": "a"}])
        assert mock_llm.call_args.kwargs["response_format"] is _RESPONSE_FORMAT
