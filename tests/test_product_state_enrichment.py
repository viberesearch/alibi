"""Tests for controlled-vocabulary product-state enrichment."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

os.environ["ALIBI_TESTING"] = "1"

from alibi.enrichment.product_state import (
    STATE_VOCAB,
    _clean_state,
    _load_attributes,
    enrich_items,
    enrich_pending_states,
    infer_states,
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
    comparable_name: str | None = "thing",
    attributes: str | None = None,
    state_enriched: int | None = None,
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
        "(id, fact_id, atom_id, name, comparable_name, attributes, state_enriched) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (item_id, fact_id, atom_id, name, comparable_name, attributes, state_enriched),
    )
    conn.commit()


def _state_of(db, item_id: str) -> str | None:
    row = db.fetchone("SELECT attributes FROM fact_items WHERE id = ?", (item_id,))
    attrs = json.loads(row["attributes"]) if row and row["attributes"] else {}
    return attrs.get("state")


def _marked(db, item_id: str) -> int | None:
    row = db.fetchone("SELECT state_enriched FROM fact_items WHERE id = ?", (item_id,))
    return row["state_enriched"] if row else None


# ===========================================================================
# TestCleanState
# ===========================================================================


class TestCleanState:
    def test_vocab_members_pass_through(self):
        for v in STATE_VOCAB:
            assert _clean_state(v) == v

    def test_synonyms_map_to_canonical(self):
        assert _clean_state("raw") == "fresh"
        assert _clean_state("jarred") == "canned"
        assert _clean_state("in brine") == "canned"
        assert _clean_state("smoked") == "cured"
        assert _clean_state("toasted") == "roasted"
        assert _clean_state("ready to eat") == "cooked"
        assert _clean_state("brined") == "pickled"

    def test_case_and_whitespace_insensitive(self):
        assert _clean_state("  CANNED ") == "canned"
        assert _clean_state("Smoked") == "cured"

    def test_unknown_value_dropped(self):
        assert _clean_state("artisanal") is None
        assert _clean_state("organic") is None  # a different facet, not a state

    def test_null_like_dropped(self):
        for v in ["", "null", "none", "n/a", "unknown", None, 5, []]:
            assert _clean_state(v) is None


# ===========================================================================
# TestLoadAttributes
# ===========================================================================


class TestLoadAttributes:
    def test_none_is_empty(self):
        assert _load_attributes({"attributes": None}) == {}
        assert _load_attributes({}) == {}

    def test_parses_json_string(self):
        assert _load_attributes({"attributes": '{"organic": true}'}) == {
            "organic": True
        }

    def test_garbage_is_empty(self):
        assert _load_attributes({"attributes": "not json"}) == {}
        assert _load_attributes({"attributes": "[1,2,3]"}) == {}


# ===========================================================================
# TestInfer
# ===========================================================================


class TestInfer:
    @patch("alibi.enrichment.product_state.call_enrichment_llm")
    def test_parses_and_resolves(self, mock_llm):
        mock_llm.return_value = [
            {"idx": 1, "state": "fresh"},
            {"idx": 2, "state": "smoked"},  # synonym -> cured
            {"idx": 3, "state": None},
            {"idx": 4, "state": "gibberish"},  # dropped -> None
        ]
        out = infer_states([{"name": "a"}, {"name": "b"}, {"name": "c"}, {"name": "d"}])
        assert out == {1: "fresh", 2: "cured", 3: None, 4: None}

    @patch("alibi.enrichment.product_state.call_enrichment_llm")
    def test_dropped_idx_absent(self, mock_llm):
        mock_llm.return_value = [{"idx": 1, "state": "canned"}]
        out = infer_states([{"name": "a"}, {"name": "b"}])
        assert out == {1: "canned"}  # idx 2 absent (dropped, will be retried)

    @patch("alibi.enrichment.product_state.call_enrichment_llm")
    def test_llm_failure_returns_empty(self, mock_llm):
        mock_llm.return_value = []
        assert infer_states([{"name": "x"}]) == {}

    def test_empty_items(self):
        assert infer_states([]) == {}

    @patch("alibi.enrichment.product_state.call_enrichment_llm")
    def test_constrains_decoding_with_schema(self, mock_llm):
        # The response_format schema is what makes the local model unable to emit
        # malformed JSON on garbled batches — assert it is passed through.
        from alibi.enrichment.product_state import _RESPONSE_FORMAT

        mock_llm.return_value = [{"idx": 1, "state": "fresh"}]
        infer_states([{"name": "a"}])
        assert mock_llm.call_args.kwargs["response_format"] is _RESPONSE_FORMAT


# ===========================================================================
# TestEnrichItems
# ===========================================================================


class TestEnrichItems:
    @patch("alibi.enrichment.product_state.call_enrichment_llm")
    def test_writes_state_and_marks(self, mock_llm, db):
        _seed_fact_item(db, "a", "FRESH SALMON", comparable_name="salmon")
        mock_llm.return_value = [{"idx": 1, "state": "fresh"}]
        items = [dict(r) for r in db.fetchall("SELECT * FROM fact_items")]
        results = enrich_items(db, items)
        assert results[0].success is True
        assert _state_of(db, "a") == "fresh"
        assert _marked(db, "a") == 1

    @patch("alibi.enrichment.product_state.call_enrichment_llm")
    def test_preserves_existing_attributes(self, mock_llm, db):
        _seed_fact_item(
            db, "a", "SMOKED SALMON", attributes='{"organic": true, "size": "L"}'
        )
        mock_llm.return_value = [{"idx": 1, "state": "smoked"}]
        items = [dict(r) for r in db.fetchall("SELECT * FROM fact_items")]
        enrich_items(db, items)
        row = db.fetchone("SELECT attributes FROM fact_items WHERE id = 'a'")
        attrs = json.loads(row["attributes"])
        assert attrs == {"organic": True, "size": "L", "state": "cured"}

    @patch("alibi.enrichment.product_state.call_enrichment_llm")
    def test_null_answer_marked_but_no_state(self, mock_llm, db):
        _seed_fact_item(db, "a", "TABLE SUGAR", comparable_name="sugar")
        mock_llm.return_value = [{"idx": 1, "state": None}]
        items = [dict(r) for r in db.fetchall("SELECT * FROM fact_items")]
        results = enrich_items(db, items)
        assert results[0].success is False
        assert _state_of(db, "a") is None
        assert _marked(db, "a") == 1  # answered-null IS marked, won't be re-asked

    @patch("alibi.enrichment.product_state.call_enrichment_llm")
    def test_dropped_item_not_marked(self, mock_llm, db):
        _seed_fact_item(db, "a", "X")
        mock_llm.return_value = []  # model dropped it
        items = [dict(r) for r in db.fetchall("SELECT * FROM fact_items")]
        results = enrich_items(db, items)
        assert results[0].success is False
        assert _marked(db, "a") is None  # unmarked -> retried next run


# ===========================================================================
# TestPending
# ===========================================================================


class TestPending:
    @patch("alibi.enrichment.product_state.call_enrichment_llm")
    def test_selects_only_unmarked_real_products(self, mock_llm, db):
        _seed_fact_item(db, "prod", "CASHEWS", comparable_name="cashews")
        _seed_fact_item(
            db, "done", "OLIVES", comparable_name="olives", state_enriched=1
        )
        _seed_fact_item(db, "nonprod", "TOTAL", comparable_name=None)  # no product
        mock_llm.return_value = [{"idx": 1, "state": "roasted"}]

        results = enrich_pending_states(db)
        assert {r.item_id for r in results} == {"prod"}
        assert _state_of(db, "prod") == "roasted"

    @patch("alibi.enrichment.product_state.call_enrichment_llm")
    def test_rerun_is_noop(self, mock_llm, db):
        _seed_fact_item(db, "a", "FRESH TUNA", comparable_name="tuna")
        mock_llm.return_value = [{"idx": 1, "state": "fresh"}]
        first = enrich_pending_states(db)
        assert len(first) == 1
        # Everything answered is now marked -> nothing selected on rerun.
        assert enrich_pending_states(db) == []
