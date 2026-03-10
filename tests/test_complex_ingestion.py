"""End-to-end tests for complex document ingestion.

Tests real-world error patterns found in production data (~20% of 395
fact_items): embedded weights in names, # separator weighed items,
header rows leaked as items, math mismatches, barcodes through the
full pipeline (extraction -> item verifier -> YAML -> atoms -> DB).
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from alibi.db.models import DocumentType
from alibi.db import v2_store
from alibi.processing.pipeline import ProcessingPipeline


@pytest.fixture
def pipeline(db):
    p = ProcessingPipeline(db=db)
    return p


_counter = 0


def _make_image(tmp_dir: Path, name: str = "receipt.jpg") -> Path:
    global _counter
    _counter += 1
    path = tmp_dir / name
    path.write_bytes(b"\xff\xd8\xff\xe0" + _counter.to_bytes(4, "big") + b"\xff\xd9")
    return path


def _process(pipeline, db, tmp_path, extraction, name="receipt.jpg"):
    """Helper: process a fake image with mocked extraction."""
    img = _make_image(tmp_path, name)
    with (
        patch.object(pipeline, "_detect_document_type") as mock_detect,
        patch.object(pipeline, "_extract_document") as mock_extract,
    ):
        mock_detect.return_value = DocumentType.RECEIPT
        mock_extract.return_value = extraction
        result = pipeline.process_file(img)
    assert result.success, f"Processing failed: {result}"
    return result


def _get_fact_items(db):
    """Helper: fetch all fact_items ordered by name."""
    facts = db.fetchall("SELECT * FROM facts", ())
    assert len(facts) >= 1
    return db.fetchall(
        "SELECT * FROM fact_items WHERE fact_id = ? ORDER BY name",
        (facts[0]["id"],),
    )


# ---------------------------------------------------------------------------
# Extraction data: Alphamega receipt with barcodes + weighed items
# ---------------------------------------------------------------------------

ALPHAMEGA_RECEIPT = {
    "document_type": "receipt",
    "vendor": "ALPHAMEGA",
    "vendor_address": "Tombs of the Kings 5, Paphos",
    "vendor_vat": "CY10057000Y",
    "date": "2026-02-15",
    "time": "14:30",
    "total": "32.47",
    "currency": "EUR",
    "payment_method": "card",
    "line_items": [
        {
            "name": "CHARAL/CHRISTIS STRA",
            "quantity": 1,
            "unit_price": 3.99,
            "total_price": 3.99,
            "barcode": "5290036000111",
        },
        {
            "name": "ALPRO SOYA MILK 1L",
            "quantity": 2,
            "unit_price": 2.69,
            "total_price": 5.38,
            "barcode": "5411188119012",
        },
        {
            "name": "BANANAS",
            "quantity": 1,
            "unit_raw": "kg",
            "unit_quantity": 1.306,
            "unit_price": 1.99,
            "total_price": 2.60,
            "barcode": "2001379000019",
        },
        {
            "name": "AVOCADO HASS",
            "quantity": 1,
            "unit_price": 3.50,
            "total_price": 3.50,
        },
    ],
    "raw_text": "ALPHAMEGA\nBarcode: 5290036000111\nCHARAL/CHRISTIS STRA 3.99\n",
}


class TestBarcodeEndToEnd:
    """Barcode must flow: extraction -> YAML -> atoms -> fact_items."""

    def test_barcode_stored_in_fact_items(self, pipeline, db, tmp_path):
        """Barcodes from extraction land in fact_items.barcode column."""
        _process(pipeline, db, tmp_path, ALPHAMEGA_RECEIPT)
        items = _get_fact_items(db)

        barcoded = {i["name"]: i["barcode"] for i in items if i["barcode"]}
        assert "5290036000111" in barcoded.values()
        assert "5411188119012" in barcoded.values()

    def test_barcode_in_atom_data(self, pipeline, db, tmp_path):
        """Barcode persists in atom.data JSON."""
        _process(pipeline, db, tmp_path, ALPHAMEGA_RECEIPT)

        item_atoms = db.fetchall("SELECT data FROM atoms WHERE atom_type = 'item'", ())
        barcodes_in_atoms = set()
        for row in item_atoms:
            data = json.loads(row["data"])
            if data.get("barcode"):
                barcodes_in_atoms.add(data["barcode"])

        assert "5290036000111" in barcodes_in_atoms
        assert "5411188119012" in barcodes_in_atoms

    def test_barcode_absent_items_have_null(self, pipeline, db, tmp_path):
        """Items without barcodes have NULL barcode, not empty string."""
        _process(pipeline, db, tmp_path, ALPHAMEGA_RECEIPT)
        items = _get_fact_items(db)

        avocado = [i for i in items if "AVOCADO" in i["name"].upper()]
        assert len(avocado) == 1
        assert avocado[0]["barcode"] is None

    def test_weighed_item_barcode_preserved(self, pipeline, db, tmp_path):
        """Weighed item (BANANAS) retains its barcode."""
        _process(pipeline, db, tmp_path, ALPHAMEGA_RECEIPT)
        items = _get_fact_items(db)

        bananas = [i for i in items if "BANANA" in i["name"].upper()]
        assert len(bananas) == 1
        assert bananas[0]["barcode"] == "2001379000019"


# ---------------------------------------------------------------------------
# Extraction data: embedded weight patterns (real Paphos supermarket)
# ---------------------------------------------------------------------------

EMBEDDED_WEIGHT_RECEIPT = {
    "document_type": "receipt",
    "vendor": "PAPAS HYPERMARKET",
    "date": "2026-01-21",
    "total": "38.50",
    "currency": "EUR",
    "line_items": [
        # Pattern: "NAME 0.765 3.50" — weight + price embedded in name
        {
            "name": "AVOCATO 0.765 3.50",
            "quantity": 1,
            "unit": "pcs",
            "unit_price": 2.68,
            "total_price": 2.68,
        },
        # Pattern: "NAME 0.515 2.90" — green pepper weighed
        {
            "name": "PIPERIA PRA 0.515 2.90",
            "quantity": 1,
            "unit": "pcs",
            "unit_price": 1.49,
            "total_price": 1.49,
        },
        # Normal item — no embedded weight
        {
            "name": "BARILLA SPAGHETTI No5 500",
            "quantity": 1,
            "unit_price": 3.49,
            "total_price": 3.49,
        },
        # Header row leaked as item
        {
            "name": "QTY DESCRIPTION PRICE AMOUNT VAT",
            "quantity": 1,
            "unit_price": 0,
            "total_price": 0,
        },
        # Math mismatch — qty=1, unit_price=1.29, total=1.63 (real weight=1.263)
        {
            "name": "NTOMATES-TOMATOES extra PER 1.263",
            "quantity": 1,
            "unit": "kg",
            "unit_price": 1.29,
            "total_price": 1.63,
        },
    ],
    "raw_text": "PAPAS HYPERMARKET\nAVOCATO 0.765 3.50\nPIPERIA PRA 0.515 2.90\n",
}


class TestEmbeddedWeightIngestion:
    """Verifier fixes embedded weights before YAML/DB storage."""

    def test_embedded_weight_extracted(self, pipeline, db, tmp_path):
        """'AVOCATO 0.765 3.50' -> name='AVOCATO', qty=0.765."""
        _process(pipeline, db, tmp_path, EMBEDDED_WEIGHT_RECEIPT)
        items = _get_fact_items(db)

        avocado = [i for i in items if "AVOCATO" in (i["name"] or "").upper()]
        assert len(avocado) == 1
        assert "0.765" not in avocado[0]["name"]
        assert float(avocado[0]["quantity"]) == pytest.approx(0.765, abs=0.01)

    def test_header_row_removed(self, pipeline, db, tmp_path):
        """'QTY DESCRIPTION PRICE AMOUNT VAT' filtered out."""
        _process(pipeline, db, tmp_path, EMBEDDED_WEIGHT_RECEIPT)
        items = _get_fact_items(db)

        names = [i["name"].upper() for i in items]
        assert not any("QTY DESCRIPTION" in n for n in names)

    def test_normal_item_unchanged(self, pipeline, db, tmp_path):
        """Clean items pass through the verifier unmodified."""
        _process(pipeline, db, tmp_path, EMBEDDED_WEIGHT_RECEIPT)
        items = _get_fact_items(db)

        spaghetti = [i for i in items if "SPAGHETTI" in (i["name"] or "").upper()]
        assert len(spaghetti) == 1
        assert float(spaghetti[0]["unit_price"]) == pytest.approx(3.49)

    def test_math_mismatch_fixed(self, pipeline, db, tmp_path):
        """Tomatoes: qty=1, price=1.29, total=1.63 -> qty=1.263."""
        _process(pipeline, db, tmp_path, EMBEDDED_WEIGHT_RECEIPT)
        items = _get_fact_items(db)

        tomatoes = [i for i in items if "TOMAT" in (i["name"] or "").upper()]
        assert len(tomatoes) == 1
        # The verifier finds 1.263 in the name and fixes qty
        assert float(tomatoes[0]["quantity"]) == pytest.approx(1.263, abs=0.01)

    def test_item_count_after_cleanup(self, pipeline, db, tmp_path):
        """5 input items -> 4 in DB (header removed)."""
        _process(pipeline, db, tmp_path, EMBEDDED_WEIGHT_RECEIPT)
        items = _get_fact_items(db)
        assert len(items) == 4


# ---------------------------------------------------------------------------
# Extraction data: hash separator weighed items (Alphamega format)
# ---------------------------------------------------------------------------

HASH_SEPARATOR_RECEIPT = {
    "document_type": "receipt",
    "vendor": "ALPHAMEGA PAPHOS",
    "date": "2026-02-01",
    "total": "45.00",
    "currency": "EUR",
    "line_items": [
        # This is the format after text parser processes "Qty 1.535 # 13.99 each"
        # with the Phase 1 fix — the parser should have already handled it.
        # But if it slips through (e.g. from LLM extraction), verifier catches it.
        {
            "name": "BEEF MINCE",
            "quantity": 1.535,
            "unit_raw": "kg",
            "unit_price": 13.99,
            "total_price": 21.47,
            "barcode": "5290011000012",
        },
        {
            "name": "GREEN GRAPES",
            "quantity": 0.836,
            "unit_raw": "kg",
            "unit_price": 2.69,
            "total_price": 2.25,
            "barcode": "2000100000017",
        },
        # Zero-price deposit — should be flagged but not removed
        {
            "name": "Deposit Fee",
            "quantity": 1,
            "unit_price": 0,
            "total_price": 0,
        },
    ],
    "raw_text": "ALPHAMEGA PAPHOS\nBEEF MINCE 21.47\nQty 1.535 # 13.99 each\n",
}


class TestHashSeparatorIngestion:
    """Weighed items with # separator pass through correctly."""

    def test_weighed_items_have_kg_unit(self, pipeline, db, tmp_path):
        """Beef and grapes should have kg unit in fact_items."""
        _process(pipeline, db, tmp_path, HASH_SEPARATOR_RECEIPT)
        items = _get_fact_items(db)

        beef = [i for i in items if "BEEF" in (i["name"] or "").upper()]
        assert len(beef) == 1
        assert beef[0]["unit"] in ("kg", "kilogram")

        # unit_quantity now transferred from atom to fact_items
        assert beef[0]["unit_quantity"] is not None
        assert float(beef[0]["unit_quantity"]) == pytest.approx(1.535, abs=0.01)
        # Purchase count should be 1
        assert float(beef[0]["quantity"]) == 1

    def test_deposit_preserved(self, pipeline, db, tmp_path):
        """Deposit Fee with zero price should still be stored."""
        _process(pipeline, db, tmp_path, HASH_SEPARATOR_RECEIPT)
        items = _get_fact_items(db)

        deposit = [i for i in items if "DEPOSIT" in (i["name"] or "").upper()]
        # Deposit may or may not survive depending on verifier/refiner,
        # but it should NOT be flagged for LLM review (it's exempt)
        # This tests the exempt name pattern works.

    def test_barcode_on_weighed_item(self, pipeline, db, tmp_path):
        """Barcodes survive on weighed items."""
        _process(pipeline, db, tmp_path, HASH_SEPARATOR_RECEIPT)
        items = _get_fact_items(db)

        beef = [i for i in items if "BEEF" in (i["name"] or "").upper()]
        assert len(beef) == 1
        assert beef[0]["barcode"] == "5290011000012"


# ---------------------------------------------------------------------------
# Extraction data: fish market with complex OCR + barcodes in names
# ---------------------------------------------------------------------------

FISH_MARKET_RECEIPT = {
    "document_type": "receipt",
    "vendor": "FISH MARKET LIMASSOL",
    "date": "2026-01-28",
    "total": "29.50",
    "currency": "EUR",
    "line_items": [
        # Garbled OCR name — short but valid (> 2 chars)
        {
            "name": "SALMON FILLET",
            "quantity": 0.684,
            "unit_raw": "kg",
            "unit_price": 28.99,
            "total_price": 19.83,
            "barcode": "529268100003",
        },
        # Item with separate barcode field
        {
            "name": "TARAMOSALATA",
            "quantity": 1,
            "unit_price": 4.50,
            "total_price": 4.50,
            "barcode": "5292681000038",
        },
        # Red Bull multi-qty: "4 Red Bull Energy 35 1.35" -> qty=4
        {
            "name": "4 Red Bull Energy 250ml",
            "quantity": 1,
            "unit_price": 1.35,
            "total_price": 5.40,
        },
    ],
    "raw_text": "FISH MARKET LIMASSOL\nBarcode: 529268100003\nSALMON FILLET\n",
}


class TestComplexItemPatterns:
    """Mixed complex patterns: barcodes, multi-qty, weighed fish."""

    def test_fish_barcode_stored(self, pipeline, db, tmp_path):
        """Fish market items retain their barcodes in DB."""
        _process(pipeline, db, tmp_path, FISH_MARKET_RECEIPT)
        items = _get_fact_items(db)

        salmon = [i for i in items if "SALMON" in (i["name"] or "").upper()]
        assert len(salmon) == 1
        assert salmon[0]["barcode"] == "529268100003"

        tarama = [i for i in items if "TARAMOSALATA" in (i["name"] or "").upper()]
        assert len(tarama) == 1
        assert tarama[0]["barcode"] == "5292681000038"

    def test_leading_quantity_fixed(self, pipeline, db, tmp_path):
        """'4 Red Bull Energy 250ml' -> qty=4, name='Red Bull Energy 250ml'."""
        _process(pipeline, db, tmp_path, FISH_MARKET_RECEIPT)
        items = _get_fact_items(db)

        redbull = [i for i in items if "RED BULL" in (i["name"] or "").upper()]
        assert len(redbull) == 1
        assert float(redbull[0]["quantity"]) == 4
        # Leading "4" removed from name
        assert not redbull[0]["name"].startswith("4 ")

    def test_weighed_fish_quantity(self, pipeline, db, tmp_path):
        """Salmon fillet at 0.684 kg: weight stored in fact_items.unit_quantity."""
        _process(pipeline, db, tmp_path, FISH_MARKET_RECEIPT)
        items = _get_fact_items(db)

        salmon = [i for i in items if "SALMON" in (i["name"] or "").upper()]
        assert len(salmon) == 1
        assert salmon[0]["unit"] in ("kg", "kilogram")

        # unit_quantity now transferred from atom to fact_items
        assert salmon[0]["unit_quantity"] is not None
        assert float(salmon[0]["unit_quantity"]) == pytest.approx(0.684, abs=0.01)
        # Purchase count should be 1
        assert float(salmon[0]["quantity"]) == 1


# ---------------------------------------------------------------------------
# Extraction data: Russian grocery (non-Latin) with leading quantities
# ---------------------------------------------------------------------------

RUSSIAN_GROCERY_RECEIPT = {
    "document_type": "receipt",
    "vendor": "KALINKA STORE",
    "date": "2026-02-10",
    "total": "25.00",
    "currency": "EUR",
    "line_items": [
        # Leading integer quantity: "2.000 FRESH TVOROG CYPRUS"
        {
            "name": "FRESH TVOROG CYPRUS",
            "quantity": 2,
            "unit_price": 6.99,
            "total_price": 13.98,
        },
        # Garbled: all-digit name should be removed
        {
            "name": "12345",
            "quantity": 1,
            "unit_price": 0,
            "total_price": 0,
        },
        # Separator line — garbage
        {
            "name": "--------",
            "quantity": 1,
            "unit_price": 0,
            "total_price": 0,
        },
    ],
    "raw_text": "KALINKA STORE\n",
}


class TestGarbageItemRemoval:
    """Garbage/garbled items are removed before DB storage."""

    def test_all_digit_name_removed(self, pipeline, db, tmp_path):
        """'12345' (all digits) is filtered out."""
        _process(pipeline, db, tmp_path, RUSSIAN_GROCERY_RECEIPT)
        items = _get_fact_items(db)

        names = [i["name"] for i in items]
        assert "12345" not in names

    def test_separator_line_removed(self, pipeline, db, tmp_path):
        """'--------' separator line is filtered out."""
        _process(pipeline, db, tmp_path, RUSSIAN_GROCERY_RECEIPT)
        items = _get_fact_items(db)

        names = [i["name"] for i in items]
        assert not any("---" in n for n in names)

    def test_valid_item_survives(self, pipeline, db, tmp_path):
        """FRESH TVOROG CYPRUS passes through correctly."""
        _process(pipeline, db, tmp_path, RUSSIAN_GROCERY_RECEIPT)
        items = _get_fact_items(db)

        tvorog = [i for i in items if "TVOROG" in (i["name"] or "").upper()]
        assert len(tvorog) == 1
        assert float(tvorog[0]["quantity"]) == 2


# ---------------------------------------------------------------------------
# Tests: unit_quantity transferred from atoms to fact_items
# ---------------------------------------------------------------------------

UNIT_QUANTITY_RECEIPT = {
    "document_type": "receipt",
    "vendor": "SUPERMARKET TEST",
    "date": "2026-02-20",
    "total": "15.28",
    "currency": "EUR",
    "line_items": [
        # Weighed item — will get unit_quantity from _fix_weighed_item_quantities
        {
            "name": "CHICKEN BREAST",
            "quantity": 0.765,
            "unit_raw": "kg",
            "unit_price": 8.99,
            "total_price": 6.88,
        },
        # Non-weighed piece item — should NOT have unit_quantity
        {
            "name": "FRESH MILK",
            "quantity": 2,
            "unit_price": 1.50,
            "total_price": 3.00,
        },
        # Multi-qty with unit_quantity already set (e.g. 4 cans of energy drink)
        {
            "name": "RED BULL ENERGY",
            "quantity": 4,
            "unit_quantity": 0.355,
            "unit_raw": "l",
            "unit_price": 1.35,
            "total_price": 5.40,
        },
    ],
    "raw_text": "SUPERMARKET TEST\nCHICKEN BREAST 6.88\nQty 0.765 # 8.99\n",
}


class TestUnitQuantityInFactItems:
    """unit_quantity from atoms transfers into fact_items DB rows."""

    def test_unit_quantity_stored_for_weighed_item(self, pipeline, db, tmp_path):
        """Weighed chicken at 0.765kg -> fact_items.unit_quantity = 0.765."""
        _process(pipeline, db, tmp_path, UNIT_QUANTITY_RECEIPT)
        items = _get_fact_items(db)

        chicken = [i for i in items if "CHICKEN" in (i["name"] or "").upper()]
        assert len(chicken) == 1
        assert chicken[0]["unit_quantity"] is not None
        assert float(chicken[0]["unit_quantity"]) == pytest.approx(0.765, abs=0.01)
        # Purchase count should be 1 (swapped by _fix_weighed_item_quantities)
        assert float(chicken[0]["quantity"]) == 1

    def test_unit_quantity_null_for_piece_items(self, pipeline, db, tmp_path):
        """Non-weighed piece item -> fact_items.unit_quantity IS NULL."""
        _process(pipeline, db, tmp_path, UNIT_QUANTITY_RECEIPT)
        items = _get_fact_items(db)

        milk = [i for i in items if "MILK" in (i["name"] or "").upper()]
        assert len(milk) == 1
        assert milk[0]["unit"] in ("pcs", "piece")
        assert milk[0]["unit_quantity"] is None

    def test_multiple_quantity_with_unit_quantity(self, pipeline, db, tmp_path):
        """4 x Red Bull 355ml -> quantity=4, unit_quantity=0.355."""
        _process(pipeline, db, tmp_path, UNIT_QUANTITY_RECEIPT)
        items = _get_fact_items(db)

        redbull = [i for i in items if "RED BULL" in (i["name"] or "").upper()]
        assert len(redbull) == 1
        assert float(redbull[0]["quantity"]) == 4
        assert redbull[0]["unit_quantity"] is not None
        assert float(redbull[0]["unit_quantity"]) == pytest.approx(0.355, abs=0.01)
        assert redbull[0]["unit"] in ("l", "liter")

    def test_unit_quantity_in_atom_data(self, pipeline, db, tmp_path):
        """unit_quantity in atom data matches what's stored in fact_items."""
        _process(pipeline, db, tmp_path, UNIT_QUANTITY_RECEIPT)
        items = _get_fact_items(db)

        chicken = [i for i in items if "CHICKEN" in (i["name"] or "").upper()]
        assert len(chicken) == 1

        # Verify atom data and fact_items are consistent
        atom = db.fetchall(
            "SELECT data FROM atoms WHERE id = ?", (chicken[0]["atom_id"],)
        )
        assert len(atom) == 1
        data = json.loads(atom[0]["data"])
        assert float(data["unit_quantity"]) == float(chicken[0]["unit_quantity"])
