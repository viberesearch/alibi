"""Tests for atom parser — converting extraction output to atoms + bundles."""

from decimal import Decimal
from typing import Any

import pytest

from alibi.atoms.parser import (
    AtomParseResult,
    _calculate_comparable_price,
    _extract_unit_from_name,
    _fix_weighed_item_quantities,
    _infer_bundle_type,
    _infer_weighed_from_price_math,
    _is_non_item_name,
    _parse_item_tax,
    _parse_vat_analysis,
    parse_extraction,
)
from alibi.db.models import (
    Atom,
    AtomType,
    Bundle,
    BundleAtom,
    BundleAtomRole,
    BundleType,
    TaxType,
    UnitType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DOC_ID = "doc-test-001"


def _receipt_extraction() -> dict[str, Any]:
    """Minimal receipt extraction."""
    return {
        "document_type": "receipt",
        "vendor": "PAPAS HYPERMARKET",
        "vendor_address": "Panayioti Tsangari 23",
        "vendor_vat": "10355430K",
        "date": "2026-01-21",
        "time": "13:56:11",
        "subtotal": 81.61,
        "tax": 4.08,
        "total": 85.69,
        "currency": "EUR",
        "language": "el",
        "payment_method": "card",
        "card_type": "visa",
        "card_last4": "7201",
        "authorization_code": "083646",
        "line_items": [
            {
                "name": "Red Bull White 250ml",
                "quantity": 3,
                "unit_price": 1.99,
                "total_price": 5.97,
                "tax_rate": 24,
                "tax_type": "vat",
                "category": "beverages",
            },
            {
                "name": "Barilla Penne Rigate 500g",
                "quantity": 1,
                "total_price": 1.89,
                "tax_rate": 13,
                "tax_type": "vat",
                "category": "pasta",
                "brand": "Barilla",
            },
        ],
        "raw_text": "PAPAS HYPERMARKET\nPanayioti Tsangari 23\n...",
    }


def _invoice_extraction() -> dict[str, Any]:
    return {
        "document_type": "invoice",
        "vendor": "ACME Corp",
        "date": "2026-01-05",
        "total": 1000.00,
        "currency": "EUR",
        "invoice_number": "INV-001",
        "line_items": [
            {
                "name": "Widget A",
                "quantity": 10,
                "unit_price": 50.00,
                "total_price": 500.00,
            },
            {
                "name": "Widget B",
                "quantity": 5,
                "unit_price": 100.00,
                "total_price": 500.00,
            },
        ],
    }


def _statement_extraction() -> dict[str, Any]:
    return {
        "document_type": "statement",
        "institution": "Bank of Cyprus",
        "currency": "EUR",
        "transactions": [
            {"date": "2026-01-15", "description": "Purchase", "amount": 42.50},
        ],
    }


def _payment_confirmation_extraction() -> dict[str, Any]:
    return {
        "document_type": "payment_confirmation",
        "vendor": "SOME STORE",
        "total": 25.00,
        "currency": "EUR",
        "payment_method": "contactless",
        "card_last4": "1234",
        "authorization_code": "ABC123",
        "terminal_id": "T-99",
        "merchant_id": "MID-12345",
    }


# ---------------------------------------------------------------------------
# parse_extraction — full integration
# ---------------------------------------------------------------------------


class TestParseExtraction:
    """Test the main parse_extraction entry point."""

    def test_receipt_produces_correct_atom_types(self) -> None:
        result = parse_extraction(DOC_ID, _receipt_extraction())
        types = {a.atom_type for a in result.atoms}
        assert AtomType.VENDOR in types
        assert AtomType.DATETIME in types
        assert AtomType.AMOUNT in types
        assert AtomType.PAYMENT in types
        assert AtomType.ITEM in types
        assert AtomType.TAX in types

    def test_receipt_item_count(self) -> None:
        result = parse_extraction(DOC_ID, _receipt_extraction())
        item_atoms = [a for a in result.atoms if a.atom_type == AtomType.ITEM]
        assert len(item_atoms) == 2

    def test_receipt_bundle_type_is_basket(self) -> None:
        result = parse_extraction(DOC_ID, _receipt_extraction())
        assert result.bundle is not None
        assert result.bundle.bundle_type == BundleType.BASKET

    def test_receipt_bundle_atoms_link_all_atoms(self) -> None:
        result = parse_extraction(DOC_ID, _receipt_extraction())
        assert len(result.bundle_atoms) == len(result.atoms)
        atom_ids = {a.id for a in result.atoms}
        linked_ids = {ba.atom_id for ba in result.bundle_atoms}
        assert atom_ids == linked_ids

    def test_invoice_bundle_type(self) -> None:
        result = parse_extraction(DOC_ID, _invoice_extraction())
        assert result.bundle is not None
        assert result.bundle.bundle_type == BundleType.INVOICE

    def test_statement_bundle_type(self) -> None:
        result = parse_extraction(DOC_ID, _statement_extraction())
        assert result.bundle is not None
        assert result.bundle.bundle_type == BundleType.STATEMENT_LINE

    def test_payment_confirmation_bundle_type(self) -> None:
        result = parse_extraction(DOC_ID, _payment_confirmation_extraction())
        assert result.bundle is not None
        assert result.bundle.bundle_type == BundleType.PAYMENT_RECORD

    def test_vendor_atom_data(self) -> None:
        result = parse_extraction(DOC_ID, _receipt_extraction())
        vendor = next(a for a in result.atoms if a.atom_type == AtomType.VENDOR)
        assert vendor.data["name"] == "PAPAS HYPERMARKET"
        assert vendor.data["address"] == "Panayioti Tsangari 23"
        assert vendor.data["vat_number"] == "10355430K"

    def test_invoice_issuer_fallback_to_vendor(self) -> None:
        """Invoice with only issuer_* fields should create vendor atom."""
        raw = {
            "document_type": "invoice",
            "issuer": "Blue Island PLC",
            "issuer_vat": "10057000Y",
            "issuer_tax_id": "12057000A",
            "issuer_address": "10 Polyfimou str., Nicosia",
            "amount": 45.62,
            "currency": "EUR",
        }
        result = parse_extraction(DOC_ID, raw)
        vendors = [a for a in result.atoms if a.atom_type == AtomType.VENDOR]
        assert len(vendors) == 1
        v = vendors[0]
        assert v.data["name"] == "Blue Island PLC"
        assert v.data["vat_number"] == "10057000Y"
        assert v.data["tax_id"] == "12057000A"
        assert v.data["address"] == "10 Polyfimou str., Nicosia"

    def test_invoice_vendor_takes_precedence_over_issuer(self) -> None:
        """When both vendor and issuer exist, vendor fields win."""
        raw = {
            "document_type": "invoice",
            "vendor": "Vendor Name",
            "vendor_vat": "VENDORVAT",
            "issuer": "Issuer Name",
            "issuer_vat": "ISSUERVAT",
            "amount": 10.0,
            "currency": "EUR",
        }
        result = parse_extraction(DOC_ID, raw)
        v = next(a for a in result.atoms if a.atom_type == AtomType.VENDOR)
        assert v.data["name"] == "Vendor Name"
        assert v.data["vat_number"] == "VENDORVAT"

    def test_receipt_no_issuer_fallback(self) -> None:
        """Receipts should NOT fall back to issuer fields."""
        raw = {
            "document_type": "receipt",
            "issuer": "Some Issuer",
            "issuer_vat": "ISSUERVAT",
            "amount": 10.0,
            "currency": "EUR",
        }
        result = parse_extraction(DOC_ID, raw)
        vendors = [a for a in result.atoms if a.atom_type == AtomType.VENDOR]
        assert len(vendors) == 0

    def test_vendor_barcode_name_rejected(self) -> None:
        """Vendor names that look like barcodes should be rejected."""
        raw = {
            "document_type": "receipt",
            "vendor": "BARCODE: 7400/2612354",
            "amount": 10.0,
            "currency": "EUR",
        }
        result = parse_extraction(DOC_ID, raw)
        vendors = [a for a in result.atoms if a.atom_type == AtomType.VENDOR]
        assert len(vendors) == 0

    def test_vendor_pure_digits_rejected(self) -> None:
        """Vendor names that are pure digits (EAN) should be rejected."""
        raw = {
            "document_type": "receipt",
            "vendor": "5290029020351",
            "amount": 10.0,
            "currency": "EUR",
        }
        result = parse_extraction(DOC_ID, raw)
        vendors = [a for a in result.atoms if a.atom_type == AtomType.VENDOR]
        assert len(vendors) == 0

    def test_vendor_normal_name_accepted(self) -> None:
        """Normal vendor names should still be accepted."""
        raw = {
            "document_type": "receipt",
            "vendor": "ALPHAMEGA",
            "amount": 10.0,
            "currency": "EUR",
        }
        result = parse_extraction(DOC_ID, raw)
        vendors = [a for a in result.atoms if a.atom_type == AtomType.VENDOR]
        assert len(vendors) == 1
        assert vendors[0].data["name"] == "ALPHAMEGA"

    def test_datetime_suspicious_old_date_flagged(self) -> None:
        """Dates >1 year old should be flagged as suspicious."""
        raw = {
            "document_type": "receipt",
            "vendor": "SHOP",
            "date": "2020-01-15",
            "amount": 10.0,
            "currency": "EUR",
        }
        result = parse_extraction(DOC_ID, raw)
        dt = next(a for a in result.atoms if a.atom_type == AtomType.DATETIME)
        assert dt.data.get("_suspicious_date") is True

    def test_datetime_recent_date_not_flagged(self) -> None:
        """Recent dates should not be flagged."""
        raw = {
            "document_type": "receipt",
            "vendor": "SHOP",
            "date": "2026-01-15",
            "amount": 10.0,
            "currency": "EUR",
        }
        result = parse_extraction(DOC_ID, raw)
        dt = next(a for a in result.atoms if a.atom_type == AtomType.DATETIME)
        assert dt.data.get("_suspicious_date") is None

    def test_datetime_atom_includes_time(self) -> None:
        result = parse_extraction(DOC_ID, _receipt_extraction())
        dt = next(a for a in result.atoms if a.atom_type == AtomType.DATETIME)
        assert dt.data["value"] == "2026-01-21 13:56:11"
        assert dt.data["semantic_type"] == "transaction_time"

    def test_amount_atoms(self) -> None:
        result = parse_extraction(DOC_ID, _receipt_extraction())
        amounts = [a for a in result.atoms if a.atom_type == AtomType.AMOUNT]
        assert len(amounts) == 2  # total + subtotal
        semantics = {a.data["semantic_type"] for a in amounts}
        assert "total" in semantics
        assert "subtotal" in semantics

    def test_payment_atom_data(self) -> None:
        result = parse_extraction(DOC_ID, _receipt_extraction())
        pay = next(a for a in result.atoms if a.atom_type == AtomType.PAYMENT)
        assert pay.data["method"] == "card"
        assert pay.data["card_last4"] == "7201"
        assert pay.data["auth_code"] == "083646"
        assert pay.data["card_type"] == "visa"

    def test_merchant_id_passthrough(self) -> None:
        """merchant_id should pass through to payment atom data."""
        raw = _payment_confirmation_extraction()
        result = parse_extraction(DOC_ID, raw)
        pay = next(a for a in result.atoms if a.atom_type == AtomType.PAYMENT)
        assert pay.data["merchant_id"] == "MID-12345"
        assert pay.data["terminal_id"] == "T-99"

    def test_item_atom_unit_extraction(self) -> None:
        """Red Bull White 250ml should have unit extracted from name."""
        result = parse_extraction(DOC_ID, _receipt_extraction())
        items = [a for a in result.atoms if a.atom_type == AtomType.ITEM]
        red_bull = next(i for i in items if "Red Bull" in i.data["name"])
        assert red_bull.data["unit"] == "ml"
        assert red_bull.data["unit_quantity"] == "250"
        assert "250ml" not in red_bull.data["name"]

    def test_item_atom_brand(self) -> None:
        result = parse_extraction(DOC_ID, _receipt_extraction())
        items = [a for a in result.atoms if a.atom_type == AtomType.ITEM]
        barilla = next(i for i in items if "Barilla" in i.data["name"])
        assert barilla.data["brand"] == "Barilla"

    def test_item_atom_tax(self) -> None:
        result = parse_extraction(DOC_ID, _receipt_extraction())
        items = [a for a in result.atoms if a.atom_type == AtomType.ITEM]
        red_bull = next(i for i in items if "Red Bull" in i.data["name"])
        assert red_bull.data["tax_rate"] == "24"
        assert red_bull.data["tax_type"] == "vat"

    def test_item_atom_comparable_price(self) -> None:
        """250ml item should get comparable price in EUR/l."""
        result = parse_extraction(DOC_ID, _receipt_extraction())
        items = [a for a in result.atoms if a.atom_type == AtomType.ITEM]
        red_bull = next(i for i in items if "Red Bull" in i.data["name"])
        assert "comparable_unit_price" in red_bull.data
        assert red_bull.data["comparable_unit"] == "l"

    def test_name_en_produces_comparable_name(self) -> None:
        """name_en in raw item should populate both name_normalized and comparable_name."""
        raw = {
            "document_type": "receipt",
            "vendor": "Test Store",
            "total": 5.0,
            "line_items": [
                {
                    "name": "Γάλα Πλήρες",
                    "name_en": "Full Fat Milk",
                    "total_price": 2.50,
                    "quantity": 1,
                },
            ],
        }
        result = parse_extraction(DOC_ID, raw)
        items = [a for a in result.atoms if a.atom_type == AtomType.ITEM]
        assert len(items) == 1
        assert items[0].data["name_normalized"] == "Full Fat Milk"
        assert items[0].data["comparable_name"] == "Full Fat Milk"

    def test_empty_extraction(self) -> None:
        result = parse_extraction(DOC_ID, {})
        assert result.atoms == []
        assert result.bundle is not None
        assert result.bundle.bundle_type == BundleType.BASKET

    def test_no_line_items(self) -> None:
        raw = {"document_type": "receipt", "vendor": "Store", "total": 10.0}
        result = parse_extraction(DOC_ID, raw)
        item_atoms = [a for a in result.atoms if a.atom_type == AtomType.ITEM]
        assert len(item_atoms) == 0

    def test_no_payment_fields(self) -> None:
        raw = {"document_type": "receipt", "vendor": "Store", "total": 10.0}
        result = parse_extraction(DOC_ID, raw)
        pay_atoms = [a for a in result.atoms if a.atom_type == AtomType.PAYMENT]
        assert len(pay_atoms) == 0

    def test_document_id_propagated(self) -> None:
        result = parse_extraction(DOC_ID, _receipt_extraction())
        for atom in result.atoms:
            assert atom.document_id == DOC_ID
        assert result.bundle is not None
        assert result.bundle.document_id == DOC_ID

    def test_atoms_have_unique_ids(self) -> None:
        result = parse_extraction(DOC_ID, _receipt_extraction())
        ids = [a.id for a in result.atoms]
        assert len(ids) == len(set(ids))

    def test_language_propagated_to_items(self) -> None:
        result = parse_extraction(DOC_ID, _receipt_extraction())
        items = [a for a in result.atoms if a.atom_type == AtomType.ITEM]
        for item in items:
            assert item.data.get("original_language") == "el"


# ---------------------------------------------------------------------------
# Unit extraction
# ---------------------------------------------------------------------------


class TestExtractUnitFromName:
    def test_embedded_ml(self) -> None:
        r = _extract_unit_from_name("Red Bull White 250ml")
        assert r is not None
        assert r[0] == "ml"
        assert r[1] == UnitType.MILLILITER
        assert r[2] == "Red Bull White"
        assert r[3] == Decimal("250")

    def test_embedded_g(self) -> None:
        r = _extract_unit_from_name("Barilla Penne 500g")
        assert r is not None
        assert r[1] == UnitType.GRAM
        assert r[3] == Decimal("500")

    def test_embedded_kg(self) -> None:
        r = _extract_unit_from_name("Chicken 1.5kg")
        assert r is not None
        assert r[1] == UnitType.KILOGRAM
        assert r[3] == Decimal("1.5")

    def test_trailing_unit(self) -> None:
        r = _extract_unit_from_name("Avocado kg")
        assert r is not None
        assert r[1] == UnitType.KILOGRAM
        assert r[2] == "Avocado"
        assert r[3] is None

    def test_bare_number_gram(self) -> None:
        r = _extract_unit_from_name("Frozen Blueberries 500")
        assert r is not None
        assert r[1] == UnitType.GRAM
        assert r[3] == Decimal("500")

    def test_bare_number_not_common(self) -> None:
        r = _extract_unit_from_name("Widget 42")
        assert r is None

    def test_no_unit(self) -> None:
        r = _extract_unit_from_name("Bread")
        assert r is None

    def test_comma_decimal(self) -> None:
        r = _extract_unit_from_name("Olive Oil 1,5l")
        assert r is not None
        assert r[1] == UnitType.LITER
        assert r[3] == Decimal("1.5")


# ---------------------------------------------------------------------------
# Tax parsing
# ---------------------------------------------------------------------------


class TestParseItemTax:
    def test_vat_type(self) -> None:
        result = _parse_item_tax({"tax_type": "VAT", "tax_rate": 24}, {})
        assert result["tax_type"] == "vat"
        assert result["tax_rate"] == "24"

    def test_sales_tax(self) -> None:
        result = _parse_item_tax({"tax_type": "sales_tax", "tax_rate": 8.5}, {})
        assert result["tax_type"] == "sales_tax"

    def test_gst(self) -> None:
        result = _parse_item_tax({"tax_type": "gst"}, {})
        assert result["tax_type"] == "gst"

    def test_category_code_resolved(self) -> None:
        vat_map = {103: Decimal("5"), 100: Decimal("19")}
        result = _parse_item_tax({"tax_rate": 103}, vat_map)
        assert result["tax_rate"] == "5"
        assert result["tax_type"] == "vat"

    def test_category_code_unresolved(self) -> None:
        result = _parse_item_tax({"tax_rate": 103}, {})
        assert "tax_rate" not in result

    def test_fractional_rate_multiplied(self) -> None:
        result = _parse_item_tax({"tax_rate": 0.05}, {})
        assert result["tax_rate"] == "5.00"

    def test_zero_rate(self) -> None:
        result = _parse_item_tax({"tax_rate": 0}, {})
        assert result["tax_rate"] == "0"

    def test_letter_code_ignored(self) -> None:
        result = _parse_item_tax({"tax_rate": "A"}, {})
        assert "tax_rate" not in result

    def test_no_tax(self) -> None:
        result = _parse_item_tax({}, {})
        assert result["tax_type"] == "none"

    def test_rate_implies_vat(self) -> None:
        result = _parse_item_tax({"tax_rate": 24}, {})
        assert result["tax_type"] == "vat"


# ---------------------------------------------------------------------------
# VAT analysis parsing
# ---------------------------------------------------------------------------


class TestParseVatAnalysis:
    def test_standard_table(self) -> None:
        text = "Vat Rate Net Tax Gross\n103 5.00% 23.80 1.19 24.99\n100 19.00% 0.92 0.17 1.09"
        result = _parse_vat_analysis(text)
        assert result[103] == Decimal("5.00")
        assert result[100] == Decimal("19.00")

    def test_empty_text(self) -> None:
        assert _parse_vat_analysis(None) == {}
        assert _parse_vat_analysis("") == {}

    def test_no_match(self) -> None:
        assert _parse_vat_analysis("No vat table here") == {}


# ---------------------------------------------------------------------------
# Weighed item quantity fix
# ---------------------------------------------------------------------------


class TestFixWeighedItemQuantities:
    def test_conversion_factor_swap(self) -> None:
        data: dict[str, Any] = {
            "quantity": "0.76",
            "unit": "kg",
            "unit_quantity": "1000",
        }
        _fix_weighed_item_quantities(data)
        assert data["quantity"] == "1"
        assert data["unit_quantity"] == "0.76"

    def test_fractional_kg_without_unit_quantity(self) -> None:
        data: dict[str, Any] = {"quantity": "2.5", "unit": "kg"}
        _fix_weighed_item_quantities(data)
        assert data["quantity"] == "1"
        assert data["unit_quantity"] == "2.5"

    def test_duplicated_weight(self) -> None:
        data: dict[str, Any] = {
            "quantity": "0.77",
            "unit": "kg",
            "unit_quantity": "0.77",
        }
        _fix_weighed_item_quantities(data)
        assert data["quantity"] == "1"
        assert data["unit_quantity"] == "0.77"

    def test_integer_quantity_unchanged(self) -> None:
        data: dict[str, Any] = {"quantity": "3", "unit": "kg"}
        _fix_weighed_item_quantities(data)
        assert data["quantity"] == "3"

    def test_piece_unit_unchanged(self) -> None:
        data: dict[str, Any] = {"quantity": "0.5", "unit": "pcs"}
        _fix_weighed_item_quantities(data)
        assert data["quantity"] == "0.5"

    def test_no_unit_unchanged(self) -> None:
        data: dict[str, Any] = {"quantity": "2"}
        _fix_weighed_item_quantities(data)
        assert data["quantity"] == "2"


# ---------------------------------------------------------------------------
# Infer weighed from price math
# ---------------------------------------------------------------------------


class TestInferWeighedFromPriceMath:
    """Test _infer_weighed_from_price_math() — detect pcs items that are weighed."""

    def test_fractional_uq_math_matches(self) -> None:
        """1.94kg chicken at 4.95/kg = 9.60 total."""
        data: dict[str, Any] = {
            "unit": "pcs",
            "unit_quantity": "1.94",
            "unit_price": "4.95",
            "total_price": "9.60",
        }
        _infer_weighed_from_price_math(data)
        assert data["unit"] == "kg"
        assert data["unit_raw"] == "kg"

    def test_small_weight_under_one(self) -> None:
        """0.63kg fennel at 3.99/kg = 2.51 total."""
        data: dict[str, Any] = {
            "unit": "pcs",
            "unit_quantity": "0.63",
            "unit_price": "3.99",
            "total_price": "2.51",
        }
        _infer_weighed_from_price_math(data)
        assert data["unit"] == "kg"

    def test_unit_other_also_fixed(self) -> None:
        """unit='other' should also be fixed."""
        data: dict[str, Any] = {
            "unit": "other",
            "unit_quantity": "0.608",
            "unit_price": "4.79",
            "total_price": "2.91",
        }
        _infer_weighed_from_price_math(data)
        assert data["unit"] == "kg"

    def test_integer_uq_not_converted(self) -> None:
        """unit_quantity=2 (integer) should NOT be converted — could be qty=2."""
        data: dict[str, Any] = {
            "unit": "pcs",
            "unit_quantity": "2",
            "unit_price": "5.25",
            "total_price": "10.50",
        }
        _infer_weighed_from_price_math(data)
        assert data["unit"] == "pcs"

    def test_large_uq_not_converted(self) -> None:
        """unit_quantity=500 (gram value) should NOT trigger kg conversion."""
        data: dict[str, Any] = {
            "unit": "pcs",
            "unit_quantity": "500",
            "unit_price": "0.004",
            "total_price": "2.00",
        }
        _infer_weighed_from_price_math(data)
        assert data["unit"] == "pcs"

    def test_math_mismatch_not_converted(self) -> None:
        """If uq * up != total_price, don't convert."""
        data: dict[str, Any] = {
            "unit": "pcs",
            "unit_quantity": "1.94",
            "unit_price": "4.95",
            "total_price": "20.00",
        }
        _infer_weighed_from_price_math(data)
        assert data["unit"] == "pcs"

    def test_already_kg_unchanged(self) -> None:
        """unit=kg should not be touched."""
        data: dict[str, Any] = {
            "unit": "kg",
            "unit_quantity": "1.94",
            "unit_price": "4.95",
            "total_price": "9.60",
        }
        _infer_weighed_from_price_math(data)
        assert data["unit"] == "kg"

    def test_existing_unit_raw_preserved(self) -> None:
        """If unit_raw is already set, don't overwrite it."""
        data: dict[str, Any] = {
            "unit": "pcs",
            "unit_raw": "κιλά",
            "unit_quantity": "0.63",
            "unit_price": "3.99",
            "total_price": "2.51",
        }
        _infer_weighed_from_price_math(data)
        assert data["unit"] == "kg"
        assert data["unit_raw"] == "κιλά"

    def test_rounding_tolerance(self) -> None:
        """0.81 * 3.49 = 2.8269, total=2.84 — within 2% tolerance."""
        data: dict[str, Any] = {
            "unit": "pcs",
            "unit_quantity": "0.81",
            "unit_price": "3.49",
            "total_price": "2.84",
        }
        _infer_weighed_from_price_math(data)
        assert data["unit"] == "kg"

    def test_no_uq_skipped(self) -> None:
        """Items without unit_quantity should be skipped."""
        data: dict[str, Any] = {
            "unit": "pcs",
            "unit_price": "4.95",
            "total_price": "9.60",
        }
        _infer_weighed_from_price_math(data)
        assert data["unit"] == "pcs"


# ---------------------------------------------------------------------------
# Comparable price
# ---------------------------------------------------------------------------


class TestCalculateComparablePrice:
    def test_grams_to_eur_per_kg(self) -> None:
        data: dict[str, Any] = {
            "total_price": "1.89",
            "quantity": "1",
            "unit": "g",
            "unit_quantity": "500",
        }
        _calculate_comparable_price(data)
        assert data["comparable_unit"] == "kg"
        assert data["comparable_unit_price"] == "3.78"

    def test_ml_to_eur_per_l(self) -> None:
        data: dict[str, Any] = {
            "total_price": "5.97",
            "quantity": "3",
            "unit": "ml",
            "unit_quantity": "250",
        }
        _calculate_comparable_price(data)
        assert data["comparable_unit"] == "l"
        # 5.97 / (3 * 250) * 1000 = 7.96
        assert data["comparable_unit_price"] == "7.96"

    def test_kg_direct(self) -> None:
        data: dict[str, Any] = {
            "total_price": "4.99",
            "quantity": "1",
            "unit": "kg",
            "unit_quantity": "2",
        }
        _calculate_comparable_price(data)
        assert data["comparable_unit"] == "kg"
        # 4.99 / 2 = 2.495 -> 2.50
        assert data["comparable_unit_price"] == "2.50"

    def test_pieces(self) -> None:
        data: dict[str, Any] = {
            "total_price": "6.00",
            "quantity": "3",
            "unit": "pcs",
        }
        _calculate_comparable_price(data)
        assert data["comparable_unit"] == "pcs"
        assert data["comparable_unit_price"] == "2.00"

    def test_no_total_price(self) -> None:
        data: dict[str, Any] = {"quantity": "1", "unit": "kg"}
        _calculate_comparable_price(data)
        assert "comparable_unit_price" not in data

    def test_zero_quantity(self) -> None:
        data: dict[str, Any] = {"total_price": "5.00", "quantity": "0", "unit": "kg"}
        _calculate_comparable_price(data)
        assert "comparable_unit_price" not in data


# ---------------------------------------------------------------------------
# Bundle type inference
# ---------------------------------------------------------------------------


class TestInferBundleType:
    def test_receipt(self) -> None:
        assert _infer_bundle_type({"document_type": "receipt"}) == BundleType.BASKET

    def test_invoice(self) -> None:
        assert _infer_bundle_type({"document_type": "invoice"}) == BundleType.INVOICE

    def test_statement(self) -> None:
        assert (
            _infer_bundle_type({"document_type": "statement"})
            == BundleType.STATEMENT_LINE
        )

    def test_payment_confirmation(self) -> None:
        assert (
            _infer_bundle_type({"document_type": "payment_confirmation"})
            == BundleType.PAYMENT_RECORD
        )

    def test_unknown_defaults_to_basket(self) -> None:
        assert _infer_bundle_type({"document_type": "other"}) == BundleType.BASKET
        assert _infer_bundle_type({}) == BundleType.BASKET


# ---------------------------------------------------------------------------
# Price inference
# ---------------------------------------------------------------------------


class TestPriceInference:
    def test_total_from_unit_price(self) -> None:
        raw = {
            "document_type": "receipt",
            "line_items": [{"name": "Item", "quantity": 3, "unit_price": 2.50}],
        }
        result = parse_extraction(DOC_ID, raw)
        item = next(a for a in result.atoms if a.atom_type == AtomType.ITEM)
        assert Decimal(item.data["total_price"]) == Decimal("7.50")

    def test_unit_price_from_total(self) -> None:
        raw = {
            "document_type": "receipt",
            "line_items": [{"name": "Item", "quantity": 4, "total_price": 10.00}],
        }
        result = parse_extraction(DOC_ID, raw)
        item = next(a for a in result.atoms if a.atom_type == AtomType.ITEM)
        assert Decimal(item.data["unit_price"]) == Decimal("2.5")


# ---------------------------------------------------------------------------
# v2 model tests
# ---------------------------------------------------------------------------


class TestV2Models:
    """Test the new enum and model classes."""

    def test_atom_type_values(self) -> None:
        assert AtomType.ITEM.value == "item"
        assert AtomType.PAYMENT.value == "payment"
        assert AtomType.VENDOR.value == "vendor"

    def test_bundle_type_values(self) -> None:
        assert BundleType.BASKET.value == "basket"
        assert BundleType.INVOICE.value == "invoice"

    def test_atom_model_creation(self) -> None:
        atom = Atom(
            id="a1",
            document_id="d1",
            atom_type=AtomType.ITEM,
            data={"name": "Test"},
        )
        assert atom.atom_type == AtomType.ITEM
        assert atom.confidence == Decimal("1.0")

    def test_bundle_model_creation(self) -> None:
        bundle = Bundle(
            id="b1",
            document_id="d1",
            bundle_type=BundleType.BASKET,
        )
        assert bundle.bundle_type == BundleType.BASKET

    def test_bundle_atom_roles(self) -> None:
        assert BundleAtomRole.BASKET_ITEM.value == "basket_item"
        assert BundleAtomRole.VENDOR_INFO.value == "vendor_info"
        assert BundleAtomRole.PAYMENT_INFO.value == "payment_info"

    def test_parse_result_default(self) -> None:
        r = AtomParseResult()
        assert r.atoms == []
        assert r.bundle is None
        assert r.bundle_atoms == []


# ---------------------------------------------------------------------------
# Migration tests
# ---------------------------------------------------------------------------


class TestMigration009:
    """Test that migration 009 creates and removes v2 tables correctly."""

    def test_migration_up_creates_tables(self, tmp_path: Any) -> None:
        import sqlite3

        from alibi.config import Config
        from alibi.db.connection import DatabaseManager

        config = Config(db_path=tmp_path / "test.db")
        manager = DatabaseManager(config)
        manager.initialize()

        rows = manager.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in rows]
        for t in [
            "documents",
            "atoms",
            "bundles",
            "bundle_atoms",
            "clouds",
            "cloud_bundles",
            "facts",
            "fact_items",
        ]:
            assert t in tables, f"Table {t} not found"
        manager.close()

    def test_migration_down_removes_tables(self, tmp_path: Any) -> None:
        import sqlite3

        from alibi.config import Config
        from alibi.db.connection import DatabaseManager
        from alibi.db.migrate import migrate_down, migrate_up

        config = Config(db_path=tmp_path / "test.db")
        manager = DatabaseManager(config)
        manager.initialize()

        # Revert migration 009
        migrate_down(manager.get_connection(), 8)

        rows = manager.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in rows]
        for t in ["documents", "atoms", "bundles", "clouds", "facts", "fact_items"]:
            assert t not in tables, f"Table {t} still exists after down"

        # schema.sql seeds all version rows, so max version stays high
        # even after down migration — the important check is table removal above
        manager.close()

    def test_schema_version_is_11(self, tmp_path: Any) -> None:
        from alibi.config import Config
        from alibi.db.connection import DatabaseManager

        config = Config(db_path=tmp_path / "test.db")
        manager = DatabaseManager(config)
        manager.initialize()
        assert manager.get_schema_version() == 35
        manager.close()


# ---------------------------------------------------------------------------
# Statement parsing (multi-bundle)
# ---------------------------------------------------------------------------


class TestParseStatement:
    """Test parse_extraction for statement documents."""

    STATEMENT_RAW: dict[str, Any] = {
        "document_type": "statement",
        "institution": "Bank of Cyprus",
        "currency": "EUR",
        "transactions": [
            {
                "date": "2026-01-21",
                "vendor": "FRESKO HYPERMARKET",
                "amount": 85.69,
                "type": "debit",
            },
            {
                "date": "2026-01-22",
                "vendor": "LIDL",
                "amount": 32.50,
                "type": "debit",
            },
        ],
    }

    def test_statement_creates_multiple_bundles(self) -> None:
        result = parse_extraction("doc1", self.STATEMENT_RAW)
        assert len(result.bundles) == 2

    def test_statement_bundle_type_is_statement_line(self) -> None:
        result = parse_extraction("doc1", self.STATEMENT_RAW)
        for br in result.bundles:
            assert br.bundle.bundle_type == BundleType.STATEMENT_LINE

    def test_statement_atoms_per_line(self) -> None:
        result = parse_extraction("doc1", self.STATEMENT_RAW)
        # Each line should produce vendor + amount + datetime atoms = 3 per line
        assert len(result.atoms) == 6

    def test_statement_vendor_atoms(self) -> None:
        result = parse_extraction("doc1", self.STATEMENT_RAW)
        vendor_atoms = [a for a in result.atoms if a.atom_type == AtomType.VENDOR]
        names = {a.data["name"] for a in vendor_atoms}
        assert names == {"FRESKO HYPERMARKET", "LIDL"}

    def test_statement_amount_atoms(self) -> None:
        result = parse_extraction("doc1", self.STATEMENT_RAW)
        amount_atoms = [a for a in result.atoms if a.atom_type == AtomType.AMOUNT]
        values = {Decimal(a.data["value"]) for a in amount_atoms}
        assert values == {Decimal("85.69"), Decimal("32.50")}

    def test_statement_date_atoms(self) -> None:
        result = parse_extraction("doc1", self.STATEMENT_RAW)
        date_atoms = [a for a in result.atoms if a.atom_type == AtomType.DATETIME]
        dates = {a.data["value"] for a in date_atoms}
        assert dates == {"2026-01-21", "2026-01-22"}

    def test_statement_bundle_atom_links(self) -> None:
        result = parse_extraction("doc1", self.STATEMENT_RAW)
        # Each bundle should have 3 links (vendor, amount, datetime)
        for br in result.bundles:
            assert len(br.bundle_atoms) == 3
            roles = {ba.role for ba in br.bundle_atoms}
            assert roles == {
                BundleAtomRole.VENDOR_INFO,
                BundleAtomRole.TOTAL,
                BundleAtomRole.EVENT_TIME,
            }

    def test_statement_convenience_properties(self) -> None:
        result = parse_extraction("doc1", self.STATEMENT_RAW)
        # .bundle returns first bundle
        assert result.bundle is not None
        assert result.bundle.bundle_type == BundleType.STATEMENT_LINE
        # .bundle_atoms returns all links across all bundles
        assert len(result.bundle_atoms) == 6

    def test_empty_transactions_no_bundles(self) -> None:
        raw: dict[str, Any] = {
            "document_type": "statement",
            "transactions": [],
        }
        result = parse_extraction("doc1", raw)
        assert len(result.bundles) == 0
        assert result.bundle is None

    def test_statement_falls_back_to_description(self) -> None:
        """If vendor is missing, falls back to description."""
        raw: dict[str, Any] = {
            "document_type": "statement",
            "currency": "EUR",
            "transactions": [
                {
                    "date": "2026-01-21",
                    "description": "POS PAYMENT STORE XYZ",
                    "amount": 50.00,
                    "type": "debit",
                }
            ],
        }
        result = parse_extraction("doc1", raw)
        vendor_atoms = [a for a in result.atoms if a.atom_type == AtomType.VENDOR]
        assert len(vendor_atoms) == 1
        assert vendor_atoms[0].data["name"] == "POS PAYMENT STORE XYZ"

    def test_negative_debit_made_positive(self) -> None:
        """Negative debit amounts are converted to positive."""
        raw: dict[str, Any] = {
            "document_type": "statement",
            "currency": "EUR",
            "transactions": [
                {
                    "date": "2026-01-21",
                    "vendor": "STORE",
                    "amount": -50.00,
                    "type": "debit",
                }
            ],
        }
        result = parse_extraction("doc1", raw)
        amount_atoms = [a for a in result.atoms if a.atom_type == AtomType.AMOUNT]
        assert Decimal(amount_atoms[0].data["value"]) == Decimal("50")


# ---------------------------------------------------------------------------
# _is_non_item_name — defense-in-depth filter
# ---------------------------------------------------------------------------


class TestIsNonItemName:
    def test_vat_summary_line(self):
        assert _is_non_item_name("VAT1 5.00") is True
        assert _is_non_item_name("VAT 19%") is True

    def test_subtotal(self):
        assert _is_non_item_name("Subtotal") is True
        assert _is_non_item_name("SUBTOTAL") is True

    def test_total_english(self):
        assert _is_non_item_name("Total") is True
        assert _is_non_item_name("TOTAL") is True
        assert _is_non_item_name("total") is True

    def test_total_greek_variants(self):
        assert _is_non_item_name("ΣYNOAO") is True
        assert _is_non_item_name("σynoao") is True
        assert _is_non_item_name("ΣΥΝΟΛΟ") is True
        assert _is_non_item_name("σύνολο") is True

    def test_price_change_line(self):
        assert _is_non_item_name("FROM 5.99 TO 3.99") is True

    def test_qty_price_metadata(self):
        assert _is_non_item_name("3 ea 5.97") is True

    def test_empty_string(self):
        assert _is_non_item_name("") is True
        assert _is_non_item_name("   ") is True

    def test_legitimate_products_pass(self):
        assert _is_non_item_name("Red Bull White 250ml") is False
        assert _is_non_item_name("ΑΥΓΑ ΕΛΕΥΘΕΡΑΣ ΒΟΣΚΗΣ") is False
        assert _is_non_item_name("ALPHAMEGA FREE RANGE") is False
        assert _is_non_item_name("7UP ZERO") is False

    def test_non_item_filtered_in_parse(self):
        """Verify _parse_item_atom returns None for non-item names."""
        from alibi.atoms.parser import _parse_item_atom

        result = _parse_item_atom(
            "doc-1",
            {"name": "Subtotal", "total_price": 85.69},
            "EUR",
            None,
            {},
        )
        assert result is None

    def test_real_item_passes_parse(self):
        from alibi.atoms.parser import _parse_item_atom

        result = _parse_item_atom(
            "doc-1",
            {"name": "Red Bull White 250ml", "total_price": 5.97},
            "EUR",
            None,
            {},
        )
        assert result is not None
        assert result.atom_type == AtomType.ITEM
