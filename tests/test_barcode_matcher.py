"""Tests for barcode-based cross-vendor product matching."""

from __future__ import annotations

import os
import uuid

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.enrichment.barcode_matcher import (
    BarcodeMatchResult,
    get_barcode_coverage,
    match_all_barcodes,
    match_by_barcode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_item(  # type: ignore[no-untyped-def]
    db,
    name,
    barcode=None,
    brand=None,
    category=None,
    comparable_name=None,
    vendor_key=None,
    enrichment_source=None,
    enrichment_confidence=None,
) -> str:
    """Create a minimal fact_item with parent records."""
    conn = db.get_connection()
    item_id = f"fi-{uuid.uuid4().hex[:8]}"
    fact_id = f"fact-{uuid.uuid4().hex[:8]}"
    doc_id = f"doc-{uuid.uuid4().hex[:8]}"
    cloud_id = f"cloud-{uuid.uuid4().hex[:8]}"
    atom_id = f"atom-{uuid.uuid4().hex[:8]}"

    conn.execute(
        "INSERT OR IGNORE INTO documents (id, file_path, file_hash) "
        "VALUES (?, ?, ?)",
        (doc_id, f"/tmp/{doc_id}.jpg", f"hash-{doc_id}"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO clouds (id, status) VALUES (?, 'collapsed')",
        (cloud_id,),
    )
    conn.execute(
        "INSERT INTO facts "
        "(id, cloud_id, fact_type, vendor, vendor_key, "
        "total_amount, currency, event_date) "
        "VALUES (?, ?, 'purchase', 'Test', ?, 10.0, 'EUR', '2026-01-01')",
        (fact_id, cloud_id, vendor_key),
    )
    conn.execute(
        "INSERT INTO atoms (id, document_id, atom_type, data) "
        "VALUES (?, ?, 'item', '{}')",
        (atom_id, doc_id),
    )
    conn.execute(
        "INSERT INTO fact_items "
        "(id, fact_id, atom_id, name, barcode, brand, category, "
        "comparable_name, enrichment_source, enrichment_confidence, "
        "quantity, total_price) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 5.0)",
        (
            item_id,
            fact_id,
            atom_id,
            name,
            barcode,
            brand,
            category,
            comparable_name,
            enrichment_source,
            enrichment_confidence,
        ),
    )
    conn.commit()
    return item_id


# ===========================================================================
# TestMatchByBarcode
# ===========================================================================


class TestMatchByBarcode:
    def test_empty_barcode_returns_empty(self, db):
        assert match_by_barcode(db, "") == []
        assert match_by_barcode(db, None) == []

    def test_single_item_no_match(self, db):
        _seed_item(db, "Milk", barcode="5000159484695", brand="Arla")
        assert match_by_barcode(db, "5000159484695") == []

    def test_two_items_same_barcode_propagates(self, db):
        _seed_item(
            db,
            "Full Cream Milk",
            barcode="5000159484695",
            brand="Arla",
            category="Dairy",
            vendor_key="vk-shop-a",
            enrichment_source="openfoodfacts",
            enrichment_confidence=0.95,
        )
        target_id = _seed_item(
            db,
            "Milk 1L",
            barcode="5000159484695",
            vendor_key="vk-shop-b",
        )

        results = match_by_barcode(db, "5000159484695")
        assert len(results) == 1
        assert results[0].item_id == target_id
        assert results[0].brand == "Arla"
        assert results[0].category == "Dairy"

    def test_already_enriched_not_overwritten(self, db):
        _seed_item(
            db,
            "Milk A",
            barcode="5000159484695",
            brand="Arla",
            category="Dairy",
            vendor_key="vk-a",
            enrichment_source="openfoodfacts",
            enrichment_confidence=0.95,
        )
        _seed_item(
            db,
            "Milk B",
            barcode="5000159484695",
            brand="Alpro",
            category="Dairy",
            vendor_key="vk-b",
            enrichment_source="openfoodfacts",
            enrichment_confidence=0.95,
        )

        results = match_by_barcode(db, "5000159484695")
        assert len(results) == 0

    def test_partial_enrichment_fills_gaps(self, db):
        _seed_item(
            db,
            "Juice",
            barcode="1234567890123",
            brand="Tropicana",
            category="Beverages",
            comparable_name="Orange Juice",
            vendor_key="vk-a",
            enrichment_source="openfoodfacts",
            enrichment_confidence=0.95,
        )
        target_id = _seed_item(
            db,
            "Juice OJ",
            barcode="1234567890123",
            brand="Tropicana",
            vendor_key="vk-b",
            enrichment_source="openfoodfacts",
            enrichment_confidence=0.95,
        )

        results = match_by_barcode(db, "1234567890123")
        assert len(results) == 1
        assert results[0].category == "Beverages"
        assert results[0].comparable_name == "Orange Juice"

    def test_three_vendors_propagates_to_all(self, db):
        _seed_item(
            db,
            "Cola",
            barcode="5449000000996",
            brand="Coca-Cola",
            category="Beverages",
            vendor_key="vk-a",
            enrichment_source="openfoodfacts",
            enrichment_confidence=0.95,
        )
        _seed_item(db, "Cola 330ml", barcode="5449000000996", vendor_key="vk-b")
        _seed_item(db, "Coca Cola", barcode="5449000000996", vendor_key="vk-c")

        results = match_by_barcode(db, "5449000000996")
        assert len(results) == 2
        for r in results:
            assert r.brand == "Coca-Cola"
            assert r.category == "Beverages"

    def test_best_source_selection(self, db):
        """Source with most fields is preferred; partial items get filled."""
        # Water A: has brand only (partial enrichment)
        _seed_item(
            db,
            "Water A",
            barcode="6001000000001",
            brand="Evian",
            vendor_key="vk-a",
            enrichment_source="gs1",
            enrichment_confidence=0.80,
        )
        # Water B: has brand + category + comparable_name (best source)
        _seed_item(
            db,
            "Water B",
            barcode="6001000000001",
            brand="Evian",
            category="Water",
            comparable_name="Mineral Water",
            vendor_key="vk-b",
            enrichment_source="openfoodfacts",
            enrichment_confidence=0.95,
        )
        # Water C: completely unenriched
        target_id = _seed_item(db, "Water", barcode="6001000000001", vendor_key="vk-c")

        results = match_by_barcode(db, "6001000000001")
        # Water A gets category+comparable, Water C gets brand+category+comparable
        assert len(results) == 2
        target_result = next(r for r in results if r.item_id == target_id)
        assert target_result.brand == "Evian"
        assert target_result.category == "Water"
        assert target_result.comparable_name == "Mineral Water"


# ===========================================================================
# TestMatchAllBarcodes
# ===========================================================================


class TestMatchAllBarcodes:
    def test_no_data_returns_empty(self, db):
        assert match_all_barcodes(db) == []

    def test_finds_multiple_barcodes(self, db):
        # Barcode 1: enriched + unenriched
        _seed_item(
            db,
            "Milk",
            barcode="1111111111111",
            brand="Arla",
            category="Dairy",
            vendor_key="vk-a",
            enrichment_source="openfoodfacts",
            enrichment_confidence=0.95,
        )
        _seed_item(db, "Milk 1L", barcode="1111111111111", vendor_key="vk-b")

        # Barcode 2: enriched + unenriched
        _seed_item(
            db,
            "Bread",
            barcode="2222222222222",
            brand="Wonder",
            category="Bakery",
            vendor_key="vk-a",
            enrichment_source="openfoodfacts",
            enrichment_confidence=0.95,
        )
        _seed_item(db, "White Bread", barcode="2222222222222", vendor_key="vk-c")

        results = match_all_barcodes(db)
        assert len(results) == 2
        barcodes = {r.barcode for r in results}
        assert "1111111111111" in barcodes
        assert "2222222222222" in barcodes

    def test_respects_limit(self, db):
        for i in range(5):
            bc = f"{i + 1:013d}"
            _seed_item(
                db,
                f"Item {i}",
                barcode=bc,
                brand=f"Brand{i}",
                category="Cat",
                vendor_key="vk-a",
                enrichment_source="openfoodfacts",
                enrichment_confidence=0.95,
            )
            _seed_item(db, f"Other {i}", barcode=bc, vendor_key="vk-b")

        results = match_all_barcodes(db, limit=2)
        assert len(results) == 2


# ===========================================================================
# TestGetBarcodeCoverage
# ===========================================================================


class TestGetBarcodeCoverage:
    def test_empty_db(self, db):
        stats = get_barcode_coverage(db)
        assert stats["total_with_barcode"] == 0
        assert stats["enriched"] == 0
        assert stats["cross_vendor_barcodes"] == 0
        assert stats["matchable"] == 0

    def test_counts_correct(self, db):
        _seed_item(
            db,
            "Enriched",
            barcode="1111111111111",
            brand="Brand",
            vendor_key="vk-a",
            enrichment_source="off",
            enrichment_confidence=0.95,
        )
        _seed_item(db, "Unenriched", barcode="1111111111111", vendor_key="vk-b")
        _seed_item(db, "No barcode", vendor_key="vk-c")

        stats = get_barcode_coverage(db)
        assert stats["total_with_barcode"] == 2
        assert stats["enriched"] == 1
        assert stats["unenriched"] == 1
        assert stats["cross_vendor_barcodes"] == 1
        assert stats["matchable"] == 1
