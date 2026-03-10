"""Tests for alibi refiners module."""

import logging
from datetime import date
from decimal import Decimal

import pytest

from alibi.db.models import RecordType, TaxType, UnitType
from alibi.refiners import (
    BaseRefiner,
    DefaultRefiner,
    InsuranceRefiner,
    InvoiceRefiner,
    PaymentRefiner,
    PurchaseRefiner,
    StatementRefiner,
    WarrantyRefiner,
    get_refiner,
)
from alibi.refiners.base import (
    _normalize_amount,
    _normalize_currency,
    _normalize_date,
    _parse_quantity_unit,
)


# Tests for base helper functions


class TestNormalizeAmount:
    """Test _normalize_amount helper."""

    def test_normalize_decimal(self):
        """Test normalizing Decimal input."""
        result = _normalize_amount(Decimal("123.45"))
        assert result == Decimal("123.45")

    def test_normalize_int(self):
        """Test normalizing int input."""
        result = _normalize_amount(123)
        assert result == Decimal("123")

    def test_normalize_float(self):
        """Test normalizing float input."""
        result = _normalize_amount(123.45)
        assert result == Decimal("123.45")

    def test_normalize_string_clean(self):
        """Test normalizing clean string input."""
        result = _normalize_amount("123.45")
        assert result == Decimal("123.45")

    def test_normalize_string_with_currency(self):
        """Test normalizing string with currency symbols."""
        result = _normalize_amount("€ 123.45")
        assert result == Decimal("123.45")

    def test_normalize_string_with_comma(self):
        """Test normalizing string with comma separator."""
        result = _normalize_amount("1,234.56")
        assert result == Decimal("1234.56")

    def test_normalize_none(self):
        """Test normalizing None input."""
        result = _normalize_amount(None)
        assert result is None

    def test_normalize_invalid_string(self):
        """Test normalizing invalid string."""
        result = _normalize_amount("invalid")
        assert result is None


class TestNormalizeDate:
    """Test _normalize_date helper."""

    def test_normalize_date_object(self):
        """Test normalizing date object."""
        d = date(2024, 1, 15)
        result = _normalize_date(d)
        assert result == d

    def test_normalize_iso_format(self):
        """Test normalizing ISO format string."""
        result = _normalize_date("2024-01-15")
        assert result == date(2024, 1, 15)

    def test_normalize_european_format(self):
        """Test normalizing European format string."""
        result = _normalize_date("15/01/2024")
        assert result == date(2024, 1, 15)

    def test_normalize_us_format(self):
        """Test normalizing US format string."""
        result = _normalize_date("01/15/2024")
        assert result == date(2024, 1, 15)

    def test_normalize_dotted_format(self):
        """Test normalizing dotted format string."""
        result = _normalize_date("15.01.2024")
        assert result == date(2024, 1, 15)

    def test_normalize_none(self):
        """Test normalizing None input."""
        result = _normalize_date(None)
        assert result is None

    def test_normalize_invalid_string(self):
        """Test normalizing invalid string."""
        result = _normalize_date("invalid")
        assert result is None


class TestNormalizeCurrency:
    """Test _normalize_currency helper."""

    def test_normalize_euro_symbol(self):
        """Test normalizing Euro symbol."""
        result = _normalize_currency("€")
        assert result == "EUR"

    def test_normalize_dollar_symbol(self):
        """Test normalizing Dollar symbol."""
        result = _normalize_currency("$")
        assert result == "USD"

    def test_normalize_pound_symbol(self):
        """Test normalizing Pound symbol."""
        result = _normalize_currency("£")
        assert result == "GBP"

    def test_normalize_code(self):
        """Test normalizing currency code."""
        result = _normalize_currency("USD")
        assert result == "USD"

    def test_normalize_lowercase_code(self):
        """Test normalizing lowercase currency code."""
        result = _normalize_currency("usd")
        assert result == "USD"

    def test_normalize_none(self):
        """Test normalizing None input."""
        result = _normalize_currency(None)
        assert result == "EUR"

    def test_normalize_empty(self):
        """Test normalizing empty string."""
        result = _normalize_currency("")
        assert result == "EUR"


class TestParseQuantityUnit:
    """Test _parse_quantity_unit helper."""

    def test_parse_with_space(self):
        """Test parsing quantity with space separator."""
        quantity, unit = _parse_quantity_unit("2.5 kg")
        assert quantity == Decimal("2.5")
        assert unit == "kg"

    def test_parse_without_space(self):
        """Test parsing quantity without space separator."""
        quantity, unit = _parse_quantity_unit("500ml")
        assert quantity == Decimal("500")
        assert unit == "ml"

    def test_parse_no_unit(self):
        """Test parsing quantity without unit."""
        quantity, unit = _parse_quantity_unit("3")
        assert quantity == Decimal("3")
        assert unit is None

    def test_parse_comma_decimal(self):
        """Test parsing quantity with comma decimal."""
        quantity, unit = _parse_quantity_unit("1,5 kg")
        assert quantity == Decimal("1.5")
        assert unit == "kg"

    def test_parse_empty(self):
        """Test parsing empty string."""
        quantity, unit = _parse_quantity_unit("")
        assert quantity == Decimal("1")
        assert unit is None


# Tests for BaseRefiner


class TestBaseRefiner:
    """Test BaseRefiner abstract class."""

    def test_cannot_instantiate_abstract(self):
        """Test that BaseRefiner cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseRefiner()  # type: ignore[abstract]

    def test_default_refiner(self):
        """Test DefaultRefiner (no-op refiner)."""
        refiner = DefaultRefiner()
        raw = {
            "amount": "123.45",
            "date": "2024-01-15",
            "currency": "USD",
        }

        result = refiner.refine(raw, artifact_id="test-artifact")

        assert "id" in result
        assert result["amount"] == Decimal("123.45")
        assert result["currency"] == "USD"
        assert "provenance" in result
        assert result["provenance"]["source_id"] == "test-artifact"

    def test_build_provenance(self):
        """Test provenance building."""
        refiner = DefaultRefiner()
        provenance = refiner._build_provenance(
            artifact_id="test-artifact",
            source_type="ai_refinement",
            confidence=0.95,
        )

        assert provenance["source_type"] == "ai_refinement"
        assert provenance["source_id"] == "test-artifact"
        assert provenance["confidence"] == Decimal("0.95")
        assert provenance["processor"] == "alibi:refiner:v2"


# Tests for PaymentRefiner


class TestPaymentRefiner:
    """Test PaymentRefiner."""

    def test_basic_payment(self):
        """Test refining basic payment record."""
        refiner = PaymentRefiner()
        raw = {
            "vendor": "Amazon LLC",
            "amount": "49.99",
            "currency": "USD",
            "transaction_date": "2024-01-15",
            "payment_method": "credit card",
        }

        result = refiner.refine(raw, artifact_id="artifact-1")

        assert result["record_type"] == RecordType.PAYMENT
        assert result["vendor"] == "Amazon"
        assert result["amount"] == Decimal("49.99")
        assert result["currency"] == "USD"
        assert result["payment_method"] == "card"
        assert result["date"] == date(2024, 1, 15)

    def test_card_last4_extraction(self):
        """Test extracting card last 4 digits."""
        refiner = PaymentRefiner()
        raw = {
            "vendor": "Store",
            "amount": "10.00",
            "card_last4": "1234567890123456",
        }

        result = refiner.refine(raw)

        assert result["card_last4"] == "3456"

    def test_vendor_normalization(self):
        """Test vendor name normalization."""
        refiner = PaymentRefiner()

        assert refiner._normalize_vendor("Amazon LLC") == "Amazon"
        assert refiner._normalize_vendor("Apple Inc") == "Apple"
        assert refiner._normalize_vendor("Store GmbH") == "Store"


# Tests for PurchaseRefiner


class TestPurchaseRefiner:
    """Test PurchaseRefiner."""

    def test_basic_purchase_with_line_items(self):
        """Test refining purchase with line items."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Grocery Store",
            "amount": "25.50",
            "currency": "EUR",
            "transaction_date": "2024-01-15",
            "line_items": [
                {
                    "name": "Milk",
                    "quantity": "1 l",
                    "price": "1.50",
                },
                {
                    "name": "Bread",
                    "quantity": "1",
                    "price": "2.00",
                },
                {
                    "name": "Apples",
                    "quantity": "2.5 kg",
                    "price": "5.00",
                },
            ],
        }

        result = refiner.refine(raw, artifact_id="artifact-1")

        assert result["record_type"] == RecordType.PURCHASE
        assert result["vendor"] == "Grocery Store"
        assert len(result["line_items"]) == 3

        # Check milk item
        milk = result["line_items"][0]
        assert milk["name"] == "Milk"
        assert milk["quantity"] == Decimal("1")
        assert milk["unit"] == UnitType.LITER
        assert milk["unit_raw"] == "l"
        assert milk["total_price"] == Decimal("1.50")

        # Check apples item (2.5kg weighed → quantity=1, unit_quantity=2.5)
        apples = result["line_items"][2]
        assert apples["quantity"] == Decimal("1")
        assert apples["unit_quantity"] == Decimal("2.5")
        assert apples["unit"] == UnitType.KILOGRAM

    def test_line_item_with_tax(self):
        """Test parsing line item with tax information."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "10.00",
            "line_items": [
                {
                    "name": "Item",
                    "quantity": "1",
                    "price": "10.00",
                    "tax_type": "vat",
                    "tax_rate": "19%",
                    "tax_amount": "1.52",
                }
            ],
        }

        result = refiner.refine(raw)

        item = result["line_items"][0]
        assert item["tax_type"] == TaxType.VAT
        assert item["tax_rate"] == Decimal("19")
        assert item["tax_amount"] == Decimal("1.52")

    def test_unit_price_calculation(self):
        """Test unit price calculation from total."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "10.00",
            "line_items": [
                {
                    "name": "Item",
                    "quantity": "5",
                    "total": "10.00",
                }
            ],
        }

        result = refiner.refine(raw)

        item = result["line_items"][0]
        assert item["total_price"] == Decimal("10.00")
        assert item["unit_price"] == Decimal("2.00")

    def test_total_price_calculation(self):
        """Test total price calculation from unit price."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "10.00",
            "line_items": [
                {
                    "name": "Item",
                    "quantity": "4",
                    "unit_price": "2.50",
                }
            ],
        }

        result = refiner.refine(raw)

        item = result["line_items"][0]
        assert item["unit_price"] == Decimal("2.50")
        assert item["total_price"] == Decimal("10.00")

    def test_unit_mapping(self):
        """Test unit mapping to UnitType enum."""
        refiner = PurchaseRefiner()

        assert refiner._map_unit("kg") == UnitType.KILOGRAM
        assert refiner._map_unit("KGS") == UnitType.KILOGRAM
        assert refiner._map_unit("g") == UnitType.GRAM
        assert refiner._map_unit("ml") == UnitType.MILLILITER
        assert refiner._map_unit("pcs") == UnitType.PIECE
        assert refiner._map_unit("unknown") == UnitType.OTHER

    def test_extract_unit_from_name_trailing(self):
        """Test extracting trailing unit from name like 'Avocado kg'."""
        result = PurchaseRefiner._extract_unit_from_name("Avocado kg")
        assert result is not None
        unit_raw, unit_type, clean_name, unit_qty = result
        assert unit_raw == "kg"
        assert unit_type == UnitType.KILOGRAM
        assert clean_name == "Avocado"
        assert unit_qty is None  # Trailing unit has no quantity

    def test_extract_unit_from_name_embedded_volume(self):
        """Test extracting embedded volume like 'Red Bull White 250ml'."""
        result = PurchaseRefiner._extract_unit_from_name("Red Bull White 250ml")
        assert result is not None
        unit_raw, unit_type, clean_name, unit_qty = result
        assert unit_raw == "ml"
        assert unit_type == UnitType.MILLILITER
        assert clean_name == "Red Bull White"
        assert unit_qty == Decimal("250")

    def test_extract_unit_from_name_embedded_weight(self):
        """Test extracting embedded weight like 'Blueberries 500g'."""
        result = PurchaseRefiner._extract_unit_from_name(
            "Blue Green Wave Frozen Blueberries 500g"
        )
        assert result is not None
        unit_raw, unit_type, clean_name, unit_qty = result
        assert unit_raw == "g"
        assert unit_type == UnitType.GRAM
        assert "Blueberries" in clean_name
        assert unit_qty == Decimal("500")

    def test_extract_unit_from_name_liter(self):
        """Test extracting liter like 'Oil 1lt'."""
        result = PurchaseRefiner._extract_unit_from_name(
            "St. George Extra Virgin Oil 1lt"
        )
        assert result is not None
        unit_raw, unit_type, clean_name, unit_qty = result
        assert unit_raw == "lt"
        assert unit_type == UnitType.LITER
        assert "Oil" in clean_name
        assert unit_qty == Decimal("1")

    def test_extract_unit_from_name_no_unit(self):
        """Test name without unit returns None."""
        result = PurchaseRefiner._extract_unit_from_name("Barilla Penne Rigate")
        assert result is None

    def test_v2_unit_raw_from_llm(self):
        """Test that unit_raw from LLM (v2 prompt) is used over name parsing."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "5.00",
            "line_items": [
                {
                    "name": "Avocado",
                    "quantity": 1.1,
                    "unit_raw": "kg",
                    "unit_price": 3.59,
                    "total_price": 3.97,
                }
            ],
        }

        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["unit"] == UnitType.KILOGRAM
        assert item["unit_raw"] == "kg"
        assert item["name"] == "Avocado"

    def test_v2_name_en_passthrough(self):
        """Test that name_en from LLM becomes name_normalized."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "5.00",
            "language": "el",
            "line_items": [
                {
                    "name": "Πιπεριά Χρωματιστά",
                    "name_en": "Colored Peppers",
                    "quantity": 1,
                    "total_price": 2.98,
                }
            ],
        }

        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["name_normalized"] == "Colored Peppers"
        assert item["original_language"] == "el"

    def test_fallback_unit_extraction_from_name(self):
        """Test that unit is extracted from name when LLM doesn't provide unit_raw."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "10.00",
            "line_items": [
                {
                    "name": "Avocado kg",
                    "quantity": 1.1,
                    "unit_price": 3.59,
                    "total_price": 3.97,
                },
                {
                    "name": "Red Bull White 250ml",
                    "quantity": 1,
                    "unit_price": 1.09,
                    "total_price": 1.09,
                },
            ],
        }

        result = refiner.refine(raw)

        avocado = result["line_items"][0]
        assert avocado["unit"] == UnitType.KILOGRAM
        assert avocado["unit_raw"] == "kg"
        assert avocado["name"] == "Avocado"

        redbull = result["line_items"][1]
        assert redbull["unit"] == UnitType.MILLILITER
        assert redbull["unit_raw"] == "ml"
        assert redbull["name"] == "Red Bull White"

    def test_bare_number_common_gram_weight(self):
        """Bare trailing number matching common gram weight is extracted."""
        result = PurchaseRefiner._extract_unit_from_name(
            "Blue Green Wave Frozen Blueberries 500"
        )
        assert result is not None
        unit_raw, unit_type, clean_name, unit_qty = result
        assert unit_raw == "g"
        assert unit_type == UnitType.GRAM
        assert clean_name == "Blue Green Wave Frozen Blueberries"
        assert unit_qty == Decimal("500")

    def test_bare_number_1000g(self):
        """Bare 1000 = 1kg."""
        result = PurchaseRefiner._extract_unit_from_name("Basmati Rice 1000")
        assert result is not None
        assert result[0] == "g"
        assert result[3] == Decimal("1000")
        assert result[2] == "Basmati Rice"

    def test_bare_number_uncommon_not_extracted(self):
        """Bare trailing number not in common weights is NOT extracted."""
        result = PurchaseRefiner._extract_unit_from_name("Product Code 9876")
        assert result is None

    def test_bare_number_only_name_not_stripped(self):
        """If stripping would leave name empty, don't extract."""
        result = PurchaseRefiner._extract_unit_from_name("500")
        assert result is None

    def test_piece_unit_overridden_by_name_extraction(self):
        """When LLM returns piece unit (Stk) but name has embedded unit, prefer name."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "10.00",
            "line_items": [
                {
                    "name": "Red Bull White 250ml",
                    "quantity": 1,
                    "unit_raw": "Stk",
                    "total_price": 1.09,
                },
                {
                    "name": "Barilla Penne Rigate 500g",
                    "quantity": 1,
                    "unit_raw": "Stk",
                    "total_price": 1.95,
                },
                {
                    "name": "St. George Extra Virgin Oil 1lt",
                    "quantity": 1,
                    "unit_raw": "Stk",
                    "total_price": 7.79,
                },
            ],
        }

        result = refiner.refine(raw)

        redbull = result["line_items"][0]
        assert redbull["unit"] == UnitType.MILLILITER
        assert redbull["unit_raw"] == "ml"
        assert redbull["name"] == "Red Bull White"
        assert redbull["unit_quantity"] == Decimal("250")

        pasta = result["line_items"][1]
        assert pasta["unit"] == UnitType.GRAM
        assert pasta["unit_raw"] == "g"
        assert pasta["name"] == "Barilla Penne Rigate"
        assert pasta["unit_quantity"] == Decimal("500")

        oil = result["line_items"][2]
        assert oil["unit"] == UnitType.LITER
        assert oil["unit_raw"] == "lt"
        assert oil["name"] == "St. George Extra Virgin Oil"
        assert oil["unit_quantity"] == Decimal("1")

    def test_specific_unit_from_llm_not_overridden(self):
        """When LLM returns a specific unit (kg, ml), keep it and clean name."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "5.00",
            "line_items": [
                {
                    "name": "Avocado kg",
                    "quantity": 1.1,
                    "unit_raw": "kg",
                    "total_price": 3.97,
                },
            ],
        }

        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["unit"] == UnitType.KILOGRAM
        assert item["unit_raw"] == "kg"
        assert item["name"] == "Avocado"  # Trailing unit cleaned

    def test_piece_unit_kept_when_no_name_unit(self):
        """When LLM returns piece and name has no unit, keep piece."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "9.18",
            "line_items": [
                {
                    "name": "Charalambides Christis Strained Yogurt",
                    "quantity": 2,
                    "unit_raw": "Stk",
                    "total_price": 9.18,
                },
            ],
        }

        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["unit"] == UnitType.PIECE
        assert item["unit_raw"] == "Stk"


class TestUnitExtractionEdgeCases:
    """Tests for unit extraction from product names with edge-case formats."""

    def test_comma_decimal_unit_in_name(self):
        """'ΤΑΡΑΜΟΣΑΛΑΤΑ ΜΑΓΟΥΛΙΤΣΑ - 0,250Kg' should extract kg, qty=0.250."""
        result = PurchaseRefiner._extract_unit_from_name(
            "ΤΑΡΑΜΟΣΑΛΑΤΑ ΜΑΓΟΥΛΙΤΣΑ - 0,250Kg"
        )
        assert result is not None
        unit_raw, unit_type, clean_name, unit_qty = result
        assert unit_type == UnitType.KILOGRAM
        assert clean_name == "ΤΑΡΑΜΟΣΑΛΑΤΑ ΜΑΓΟΥΛΙΤΣΑ"
        assert unit_qty == Decimal("0.250")

    def test_leading_quantity_unit_in_name(self):
        """'1.29 kg PATATES' should extract kg, qty=1.29, name='PATATES'."""
        result = PurchaseRefiner._extract_unit_from_name("1.29 kg PATATES")
        assert result is not None
        unit_raw, unit_type, clean_name, unit_qty = result
        assert unit_type == UnitType.KILOGRAM
        assert clean_name == "PATATES"
        assert unit_qty == Decimal("1.29")

    def test_comma_decimal_no_space(self):
        """'0,250Kg' without product name extracts correctly."""
        result = PurchaseRefiner._extract_unit_from_name("0,250Kg Taramosalata")
        assert result is not None
        unit_raw, unit_type, clean_name, unit_qty = result
        assert unit_type == UnitType.KILOGRAM
        assert clean_name == "Taramosalata"
        assert unit_qty == Decimal("0.250")

    def test_refiner_comma_decimal_weight_in_name(self):
        """Full refiner: item with '0,250Kg' in name should get unit=kg."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "1.39",
            "line_items": [
                {
                    "name": "ΤΑΡΑΜΟΣΑΛΑΤΑ ΜΑΓΟΥΛΙΤΣΑ - 0,250Kg",
                    "quantity": 1,
                    "total_price": 1.39,
                },
            ],
        }

        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["unit"] == UnitType.KILOGRAM
        assert "ΤΑΡΑΜΟΣΑΛΑΤΑ" in item["name"]
        assert "Kg" not in item["name"]

    def test_refiner_leading_quantity_unit(self):
        """Full refiner: item with '1.29 kg PATATES' in name."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "2.89",
            "line_items": [
                {
                    "name": "1.29 kg PATATES",
                    "quantity": 1,
                    "total_price": 2.89,
                },
            ],
        }

        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["unit"] == UnitType.KILOGRAM
        assert item["name"] == "PATATES"

    def test_refiner_llm_returns_piece_but_name_has_weight(self):
        """LLM says 'Stk' but name has '0,250Kg' — should prefer kg from name."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "1.39",
            "line_items": [
                {
                    "name": "ΤΑΡΑΜΟΣΑΛΑΤΑ ΜΑΓΟΥΛΙΤΣΑ - 0,250Kg",
                    "quantity": 1,
                    "unit_raw": "Stk",
                    "total_price": 1.39,
                },
            ],
        }

        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["unit"] == UnitType.KILOGRAM
        assert "Kg" not in item["name"]


class TestComparableUnitPrice:
    """Tests for comparable_unit_price calculation."""

    def test_grams_normalized_to_per_kg(self):
        """400g beans at 0.99 each → EUR/kg."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Shop",
            "amount": "1.98",
            "line_items": [
                {
                    "name": "Tomato Beans 400g",
                    "quantity": 2,
                    "total_price": 1.98,
                },
            ],
        }
        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["comparable_unit"] == "kg"
        # 1.98 / (2 * 400g) * 1000 = 2.48 EUR/kg (rounded to 2dp)
        assert item["comparable_unit_price"] == Decimal("2.48")

    def test_ml_normalized_to_per_liter(self):
        """250ml Red Bull at 1.09 → EUR/l."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Shop",
            "amount": "1.09",
            "line_items": [
                {
                    "name": "Red Bull 250ml",
                    "quantity": 1,
                    "total_price": 1.09,
                },
            ],
        }
        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["comparable_unit"] == "l"
        # 1.09 / 250ml * 1000 = 4.36 EUR/l
        assert item["comparable_unit_price"] == Decimal("4.36")

    def test_kg_stays_per_kg(self):
        """1.1kg avocado at 3.97 → EUR/kg."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Shop",
            "amount": "3.97",
            "line_items": [
                {
                    "name": "Avocado kg",
                    "quantity": "1.1",
                    "total_price": 3.97,
                },
            ],
        }
        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["comparable_unit"] == "kg"
        # 3.97 / 1.1kg = 3.61 EUR/kg (rounded to 2dp)
        assert item["comparable_unit_price"] == Decimal("3.61")

    def test_liter_stays_per_liter(self):
        """1lt oil at 7.79 → EUR/l."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Shop",
            "amount": "7.79",
            "line_items": [
                {
                    "name": "Olive Oil 1lt",
                    "quantity": 1,
                    "total_price": 7.79,
                },
            ],
        }
        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["comparable_unit"] == "l"
        # 7.79 / 1l = 7.79 EUR/l
        assert item["comparable_unit_price"] == Decimal("7.79")

    def test_piece_price_per_piece(self):
        """2 yogurts at 9.18 → EUR/pcs."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Shop",
            "amount": "9.18",
            "line_items": [
                {
                    "name": "Strained Yogurt",
                    "quantity": 2,
                    "unit_raw": "Stk",
                    "total_price": 9.18,
                },
            ],
        }
        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["comparable_unit"] == "pcs"
        # 9.18 / 2 = 4.59 EUR/pcs
        assert item["comparable_unit_price"] == Decimal("4.59")

    def test_no_price_no_comparable(self):
        """Item without total_price gets no comparable price."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Shop",
            "amount": "5.00",
            "line_items": [
                {"name": "Mystery Item", "quantity": 1},
            ],
        }
        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert "comparable_unit_price" not in item

    def test_unit_quantity_from_llm_used(self):
        """LLM-provided unit_quantity used when name has no unit."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Shop",
            "amount": "4.09",
            "line_items": [
                {
                    "name": "Frozen Blueberries",
                    "quantity": 1,
                    "unit_raw": "g",
                    "unit_quantity": 500,
                    "total_price": 4.09,
                },
            ],
        }
        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["comparable_unit"] == "kg"
        # 4.09 / 500g * 1000 = 8.18 EUR/kg
        assert item["comparable_unit_price"] == Decimal("8.18")


# Tests for InvoiceRefiner


class TestInvoiceRefiner:
    """Test InvoiceRefiner."""

    def test_basic_invoice(self):
        """Test refining basic invoice record."""
        refiner = InvoiceRefiner()
        raw = {
            "issuer": "Acme Corp Inc",
            "invoice_number": "INV-2024-001",
            "amount": "1500.00",
            "currency": "EUR",
            "issue_date": "2024-01-15",
            "due_date": "2024-02-15",
        }

        result = refiner.refine(raw, artifact_id="artifact-1")

        assert result["record_type"] == RecordType.INVOICE
        assert result["issuer"] == "Acme"
        assert result["invoice_number"] == "2024-001"
        assert result["amount"] == Decimal("1500.00")
        assert result["date"] == date(2024, 1, 15)

    def test_invoice_number_normalization(self):
        """Test invoice number normalization."""
        refiner = InvoiceRefiner()

        assert refiner._normalize_invoice_number("INV-123") == "123"
        assert refiner._normalize_invoice_number("invoice-456") == "456"
        assert refiner._normalize_invoice_number("INV#789") == "789"
        assert refiner._normalize_invoice_number("abc-123") == "ABC-123"


# Tests for WarrantyRefiner


class TestWarrantyRefiner:
    """Test WarrantyRefiner."""

    def test_basic_warranty(self):
        """Test refining basic warranty record."""
        refiner = WarrantyRefiner()
        raw = {
            "product": "Laptop",
            "model": "XPS 15",
            "serial_number": "ABC123456",
            "warranty_type": "manufacturer",
            "warranty_expires": "2026-01-15",
            "vendor": "Dell Inc",
        }

        result = refiner.refine(raw, artifact_id="artifact-1")

        assert result["record_type"] == RecordType.WARRANTY
        assert result["product"] == "Laptop"
        assert result["model"] == "XPS 15"
        assert result["serial_number"] == "ABC123456"
        assert result["warranty_type"] == "manufacturer"
        assert result["vendor"] == "Dell"

    def test_warranty_type_normalization(self):
        """Test warranty type normalization."""
        refiner = WarrantyRefiner()

        assert (
            refiner._normalize_warranty_type("manufacturer warranty") == "manufacturer"
        )
        assert refiner._normalize_warranty_type("extended") == "extended"
        assert refiner._normalize_warranty_type("lifetime guarantee") == "lifetime"


# Tests for InsuranceRefiner


class TestInsuranceRefiner:
    """Test InsuranceRefiner."""

    def test_basic_insurance(self):
        """Test refining basic insurance record."""
        refiner = InsuranceRefiner()
        raw = {
            "policy_number": "POL-123456",
            "issuer": "State Farm Insurance",
            "coverage": "Home Insurance",
            "premium": "1200.00",
            "currency": "USD",
            "renewal_date": "2025-01-15",
        }

        result = refiner.refine(raw, artifact_id="artifact-1")

        assert result["record_type"] == RecordType.INSURANCE
        assert result["policy_number"] == "123456"
        assert result["issuer"] == "State Farm Insurance"
        assert result["coverage"] == "Home Insurance"
        assert result["premium"] == Decimal("1200.00")

    def test_policy_number_normalization(self):
        """Test policy number normalization."""
        refiner = InsuranceRefiner()

        assert refiner._normalize_policy_number("POL-123") == "123"
        assert refiner._normalize_policy_number("policy-456") == "456"
        assert refiner._normalize_policy_number("POL#789") == "789"


# Tests for StatementRefiner


class TestStatementRefiner:
    """Test StatementRefiner."""

    def test_basic_statement(self):
        """Test refining basic statement record."""
        refiner = StatementRefiner()
        raw = {
            "issuer": "Chase Bank",
            "account_number": "1234567890",
            "opening_balance": "1000.00",
            "closing_balance": "1500.00",
            "currency": "USD",
            "period_start": "2024-01-01",
            "period_end": "2024-01-31",
        }

        result = refiner.refine(raw, artifact_id="artifact-1")

        assert result["record_type"] == RecordType.STATEMENT
        assert result["issuer"] == "Chase Bank"
        assert result["account_number"] == "12****7890"  # Masked
        assert result["opening_balance"] == Decimal("1000.00")
        assert result["closing_balance"] == Decimal("1500.00")

    def test_statement_with_transactions(self):
        """Test refining statement with transactions."""
        refiner = StatementRefiner()
        raw = {
            "issuer": "Bank",
            "account_number": "123456",
            "currency": "EUR",
            "transactions": [
                {
                    "date": "2024-01-15",
                    "amount": "-50.00",
                    "description": "Grocery Store",
                },
                {
                    "date": "2024-01-20",
                    "amount": "1000.00",
                    "description": "Salary",
                },
            ],
        }

        result = refiner.refine(raw, artifact_id="artifact-1")

        assert len(result["transactions"]) == 2
        assert result["transactions"][0]["amount"] == Decimal("-50.00")
        assert result["transactions"][1]["amount"] == Decimal("1000.00")

    def test_account_number_masking(self):
        """Test account number masking."""
        refiner = StatementRefiner()

        assert refiner._normalize_account_number("1234567890") == "12****7890"
        assert refiner._normalize_account_number("123456") == "123456"  # Too short


# Tests for Registry


class TestRefinerRegistry:
    """Test refiner registry."""

    def test_get_payment_refiner(self):
        """Test getting PaymentRefiner from registry."""
        refiner = get_refiner(RecordType.PAYMENT)
        assert isinstance(refiner, PaymentRefiner)

    def test_get_purchase_refiner(self):
        """Test getting PurchaseRefiner from registry."""
        refiner = get_refiner(RecordType.PURCHASE)
        assert isinstance(refiner, PurchaseRefiner)

    def test_get_invoice_refiner(self):
        """Test getting InvoiceRefiner from registry."""
        refiner = get_refiner(RecordType.INVOICE)
        assert isinstance(refiner, InvoiceRefiner)

    def test_get_warranty_refiner(self):
        """Test getting WarrantyRefiner from registry."""
        refiner = get_refiner(RecordType.WARRANTY)
        assert isinstance(refiner, WarrantyRefiner)

    def test_get_insurance_refiner(self):
        """Test getting InsuranceRefiner from registry."""
        refiner = get_refiner(RecordType.INSURANCE)
        assert isinstance(refiner, InsuranceRefiner)

    def test_get_statement_refiner(self):
        """Test getting StatementRefiner from registry."""
        refiner = get_refiner(RecordType.STATEMENT)
        assert isinstance(refiner, StatementRefiner)

    def test_get_default_refiner_for_unmapped_type(self):
        """Test getting DefaultRefiner for unmapped types."""
        # REFUND is not yet mapped
        refiner = get_refiner(RecordType.REFUND)
        assert isinstance(refiner, DefaultRefiner)

    def test_all_core_types_mapped(self):
        """Test that all core record types have refiners."""
        core_types = [
            RecordType.PAYMENT,
            RecordType.PURCHASE,
            RecordType.INVOICE,
            RecordType.WARRANTY,
            RecordType.INSURANCE,
            RecordType.STATEMENT,
        ]

        for record_type in core_types:
            refiner = get_refiner(record_type)
            assert not isinstance(refiner, DefaultRefiner)


# Integration tests


class TestRefinerIntegration:
    """Integration tests for refiner workflow."""

    def test_end_to_end_payment(self):
        """Test complete payment refinement workflow."""
        raw_extraction = {
            "vendor": "Amazon.com LLC",
            "amount": "€ 49,99",
            "date": "15/01/2024",
            "payment_method": "credit card",
            "card_last4": "1234",
        }

        refiner = get_refiner(RecordType.PAYMENT)
        result = refiner.refine(raw_extraction, artifact_id="receipt-001")

        # Verify all fields refined correctly
        assert result["record_type"] == RecordType.PAYMENT
        assert result["vendor"] == "Amazon Com"
        assert result["amount"] == Decimal("49.99")
        assert result["currency"] == "EUR"
        assert result["date"] == date(2024, 1, 15)
        assert result["payment_method"] == "card"
        assert result["card_last4"] == "1234"
        assert result["provenance"]["source_id"] == "receipt-001"

    def test_end_to_end_purchase(self):
        """Test complete purchase refinement workflow."""
        raw_extraction = {
            "vendor": "Whole Foods Market Inc",
            "amount": "$45.67",
            "date": "2024-01-15",
            "line_items": [
                {
                    "name": "Organic Milk",
                    "quantity": "1 gal",
                    "unit_price": "5.99",
                    "tax_type": "sales_tax",
                    "tax_rate": "8.5",
                },
                {
                    "name": "Bread",
                    "quantity": "2 pcs",
                    "total": "6.00",
                },
            ],
        }

        refiner = get_refiner(RecordType.PURCHASE)
        result = refiner.refine(raw_extraction, artifact_id="receipt-002")

        # Verify purchase fields
        assert result["record_type"] == RecordType.PURCHASE
        assert result["vendor"] == "Whole Foods Market"
        assert result["amount"] == Decimal("45.67")
        assert result["currency"] == "USD"

        # Verify line items
        assert len(result["line_items"]) == 2

        milk = result["line_items"][0]
        assert milk["name"] == "Organic Milk"
        assert milk["quantity"] == Decimal("1")
        assert milk["unit"] == UnitType.GALLON
        assert milk["tax_type"] == TaxType.SALES_TAX
        assert milk["tax_rate"] == Decimal("8.5")

        bread = result["line_items"][1]
        assert bread["quantity"] == Decimal("2")
        assert bread["unit"] == UnitType.PIECE


class TestVatCategoryCodeResolution:
    """Tests for VAT category code detection and resolution."""

    def test_parse_vat_analysis_from_raw_text(self):
        """Test parsing VAT Analysis table from receipt text."""
        raw_text = (
            "Vat Analysis\n"
            "Vat Rate Net Tax Gross\n"
            "103 5.00% 23.80 1.19 24.99\n"
            "100 19.00% 0.92 0.17 1.09\n"
            "106 0.00% 6.95 0.00 6.95\n"
        )
        mapping = PurchaseRefiner._parse_vat_analysis(raw_text)
        assert mapping == {
            103: Decimal("5.00"),
            100: Decimal("19.00"),
            106: Decimal("0.00"),
        }

    def test_parse_vat_analysis_empty_text(self):
        """Test parsing empty text returns empty mapping."""
        assert PurchaseRefiner._parse_vat_analysis("") == {}
        assert PurchaseRefiner._parse_vat_analysis(None) == {}

    def test_category_code_resolved_via_vat_mapping(self):
        """Test that a VAT category code (103) is resolved to the actual rate (5%)."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "33.03",
            "raw_text": (
                "Vat Analysis\n"
                "103 5.00% 23.80 1.19 24.99\n"
                "100 19.00% 0.92 0.17 1.09\n"
            ),
            "line_items": [
                {
                    "name": "Yogurt",
                    "quantity": 2,
                    "total_price": 9.18,
                    "tax_rate": 103,
                    "tax_type": "vat",
                },
                {
                    "name": "Red Bull",
                    "quantity": 1,
                    "total_price": 1.09,
                    "tax_rate": 100,
                    "tax_type": "vat",
                },
            ],
        }

        result = refiner.refine(raw)

        yogurt = result["line_items"][0]
        assert yogurt["tax_rate"] == Decimal("5.00")
        assert yogurt["tax_type"] == TaxType.VAT

        redbull = result["line_items"][1]
        assert redbull["tax_rate"] == Decimal("19.00")
        assert redbull["tax_type"] == TaxType.VAT

    def test_category_code_discarded_without_mapping(self):
        """Test that a VAT category code is discarded when no mapping exists."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "10.00",
            "line_items": [
                {
                    "name": "Item",
                    "quantity": 1,
                    "total_price": 10.00,
                    "tax_rate": 103,
                    "tax_type": "vat",
                },
            ],
        }

        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert "tax_rate" not in item
        assert item["tax_type"] == TaxType.VAT

    def test_normal_tax_rate_not_affected(self):
        """Test that normal tax rates (e.g. 19, 5, 8.5) are not treated as codes."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "30.00",
            "line_items": [
                {
                    "name": "Item A",
                    "quantity": 1,
                    "total_price": 10.00,
                    "tax_rate": 19,
                    "tax_type": "vat",
                },
                {
                    "name": "Item B",
                    "quantity": 1,
                    "total_price": 10.00,
                    "tax_rate": 5,
                    "tax_type": "vat",
                },
                {
                    "name": "Item C",
                    "quantity": 1,
                    "total_price": 10.00,
                    "tax_rate": "8.5",
                    "tax_type": "sales_tax",
                },
            ],
        }

        result = refiner.refine(raw)
        assert result["line_items"][0]["tax_rate"] == Decimal("19")
        assert result["line_items"][1]["tax_rate"] == Decimal("5")
        assert result["line_items"][2]["tax_rate"] == Decimal("8.5")

    def test_zero_rate_category_resolved(self):
        """Test that a zero-rate category code (106 -> 0%) is resolved."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "5.00",
            "raw_text": "106 0.00% 5.00 0.00 5.00\n",
            "line_items": [
                {
                    "name": "Fresh Produce",
                    "quantity": 1,
                    "total_price": 5.00,
                    "tax_rate": 106,
                    "tax_type": "vat",
                },
            ],
        }

        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["tax_rate"] == Decimal("0.00")
        assert item["tax_type"] == TaxType.VAT

    def test_tax_type_inferred_when_rate_resolved(self):
        """Test that tax_type defaults to VAT when rate is resolved from mapping."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "10.00",
            "raw_text": "103 5.00% 9.52 0.48 10.00\n",
            "line_items": [
                {
                    "name": "Item",
                    "quantity": 1,
                    "total_price": 10.00,
                    "tax_rate": 103,
                },
            ],
        }

        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["tax_rate"] == Decimal("5.00")
        assert item["tax_type"] == TaxType.VAT


class TestTaxParsingRobustness:
    """Tax details are optional — items must be stored even when tax parsing fails."""

    def test_no_tax_fields_at_all(self):
        """Items without any tax info are stored normally."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "15.00",
            "line_items": [
                {"name": "Milk", "quantity": 1, "total_price": 2.50},
                {"name": "Bread", "quantity": 2, "total_price": 4.00},
            ],
        }

        result = refiner.refine(raw)
        assert len(result["line_items"]) == 2
        assert result["line_items"][0]["total_price"] == Decimal("2.50")
        assert result["line_items"][0]["tax_type"] == TaxType.NONE
        assert "tax_rate" not in result["line_items"][0]

    def test_only_tax_type_no_rate(self):
        """tax_type without tax_rate is kept."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "10.00",
            "line_items": [
                {
                    "name": "Item",
                    "quantity": 1,
                    "total_price": 10.00,
                    "tax_type": "vat",
                },
            ],
        }

        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["tax_type"] == TaxType.VAT
        assert "tax_rate" not in item
        assert item["total_price"] == Decimal("10.00")

    def test_only_tax_amount_no_rate(self):
        """tax_amount without rate is kept."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "10.00",
            "line_items": [
                {
                    "name": "Item",
                    "quantity": 1,
                    "total_price": 10.00,
                    "tax_amount": 1.52,
                },
            ],
        }

        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["tax_amount"] == Decimal("1.52")
        assert item["total_price"] == Decimal("10.00")

    def test_letter_tax_category_ignored(self):
        """Letter-based tax categories (A, B, C) are gracefully skipped."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "10.00",
            "line_items": [
                {
                    "name": "Item A",
                    "quantity": 1,
                    "total_price": 5.00,
                    "tax_rate": "A",
                    "tax_type": "vat",
                },
                {
                    "name": "Item B",
                    "quantity": 1,
                    "total_price": 5.00,
                    "tax_rate": "B",
                    "tax_type": "vat",
                },
            ],
        }

        result = refiner.refine(raw)
        assert len(result["line_items"]) == 2
        assert result["line_items"][0]["total_price"] == Decimal("5.00")
        assert "tax_rate" not in result["line_items"][0]
        assert result["line_items"][0]["tax_type"] == TaxType.VAT

    def test_garbage_tax_rate_ignored(self):
        """Completely garbage tax_rate values don't crash processing."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "10.00",
            "line_items": [
                {
                    "name": "Item",
                    "quantity": 1,
                    "total_price": 10.00,
                    "tax_rate": "N/A",
                },
            ],
        }

        result = refiner.refine(raw)
        assert len(result["line_items"]) == 1
        assert result["line_items"][0]["total_price"] == Decimal("10.00")
        assert "tax_rate" not in result["line_items"][0]

    def test_tax_rate_null_handled(self):
        """Explicit null/None tax_rate doesn't crash."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "10.00",
            "line_items": [
                {
                    "name": "Item",
                    "quantity": 1,
                    "total_price": 10.00,
                    "tax_rate": None,
                    "tax_type": None,
                },
            ],
        }

        result = refiner.refine(raw)
        assert len(result["line_items"]) == 1
        assert result["line_items"][0]["total_price"] == Decimal("10.00")

    def test_explicit_zero_tax_rate(self):
        """Explicit 0 tax rate is stored as zero, not discarded."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "10.00",
            "line_items": [
                {
                    "name": "Fresh Fruit",
                    "quantity": 1,
                    "total_price": 10.00,
                    "tax_rate": 0,
                    "tax_type": "exempt",
                },
            ],
        }

        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["tax_rate"] == Decimal("0")
        assert item["tax_type"] == TaxType.EXEMPT

    def test_inline_tax_rate_with_percent_sign(self):
        """Tax rate like '5%' or '19.00%' parsed from inline format."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "10.00",
            "line_items": [
                {
                    "name": "Item",
                    "quantity": 1,
                    "total_price": 10.00,
                    "tax_rate": "5%",
                    "tax_type": "vat",
                },
            ],
        }

        result = refiner.refine(raw)
        assert result["line_items"][0]["tax_rate"] == Decimal("5")

    def test_decimal_tax_rate_converted(self):
        """Decimal-form rate 0.19 is converted to 19%."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "10.00",
            "line_items": [
                {
                    "name": "Item",
                    "quantity": 1,
                    "total_price": 10.00,
                    "tax_rate": 0.19,
                    "tax_type": "vat",
                },
            ],
        }

        result = refiner.refine(raw)
        assert result["line_items"][0]["tax_rate"] == Decimal("19")

    def test_no_raw_text_no_crash(self):
        """Missing raw_text doesn't block VAT mapping or line item parsing."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Store",
            "amount": "10.00",
            "line_items": [
                {
                    "name": "Item",
                    "quantity": 1,
                    "total_price": 10.00,
                    "tax_rate": 19,
                },
            ],
        }

        result = refiner.refine(raw)
        assert len(result["line_items"]) == 1
        assert result["line_items"][0]["tax_rate"] == Decimal("19")


class TestWeighedItemFix:
    """Tests for weighed item unit_quantity correction.

    Bug: LLM puts weight in quantity (0.76) and conversion factor in
    unit_quantity (1000) instead of quantity=1, unit_quantity=0.76.
    """

    def test_conversion_factor_kg_fixed(self):
        """quantity=0.76 + unit=kg + unit_quantity=1000 → quantity=1, unit_quantity=0.76."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Shop",
            "amount": "0.55",
            "line_items": [
                {
                    "name": "LEMONIA KITRA",
                    "quantity": 0.76,
                    "unit_raw": "kg",
                    "unit_quantity": 1000,
                    "unit_price": 0.72,
                    "total_price": 0.55,
                },
            ],
        }
        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["quantity"] == Decimal("1")
        assert item["unit_quantity"] == Decimal("0.76")
        assert item["unit"] == UnitType.KILOGRAM

    def test_conversion_factor_liter_fixed(self):
        """quantity=1.5 + unit=l + unit_quantity=1000 → quantity=1, unit_quantity=1.5."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Shop",
            "amount": "2.30",
            "line_items": [
                {
                    "name": "Fresh Milk",
                    "quantity": 1.5,
                    "unit_raw": "l",
                    "unit_quantity": 1000,
                    "total_price": 2.30,
                },
            ],
        }
        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["quantity"] == Decimal("1")
        assert item["unit_quantity"] == Decimal("1.5")
        assert item["unit"] == UnitType.LITER

    def test_fractional_kg_no_unit_quantity_fixed(self):
        """quantity=0.76 + unit=kg + no unit_quantity → quantity=1, unit_quantity=0.76."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Shop",
            "amount": "0.55",
            "line_items": [
                {
                    "name": "LEMONIA KITRA",
                    "quantity": 0.76,
                    "unit_raw": "kg",
                    "total_price": 0.55,
                },
            ],
        }
        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["quantity"] == Decimal("1")
        assert item["unit_quantity"] == Decimal("0.76")

    def test_comparable_price_correct_after_fix(self):
        """After fix, comparable_unit_price should be correct EUR/kg."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Shop",
            "amount": "0.55",
            "line_items": [
                {
                    "name": "LEMONIA KITRA",
                    "quantity": 0.76,
                    "unit_raw": "kg",
                    "unit_quantity": 1000,
                    "total_price": 0.55,
                },
            ],
        }
        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["comparable_unit"] == "kg"
        # 0.55 / 0.76kg = 0.72 EUR/kg (rounded to 2dp)
        assert item["comparable_unit_price"] == Decimal("0.72")

    def test_packaged_item_not_affected(self):
        """quantity=2 + unit=g + unit_quantity=400 stays unchanged."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Shop",
            "amount": "1.98",
            "line_items": [
                {
                    "name": "Tomato Beans 400g",
                    "quantity": 2,
                    "total_price": 1.98,
                },
            ],
        }
        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["quantity"] == Decimal("2")
        assert item["unit_quantity"] == Decimal("400")

    def test_piece_item_not_affected(self):
        """quantity=3 + unit=pcs stays unchanged."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Shop",
            "amount": "6.00",
            "line_items": [
                {
                    "name": "Yogurt",
                    "quantity": 3,
                    "unit_raw": "pcs",
                    "total_price": 6.00,
                },
            ],
        }
        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["quantity"] == Decimal("3")

    def test_integer_kg_not_changed(self):
        """quantity=2 + unit=kg stays unchanged (ambiguous — could be 2 bags of 1kg)."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Shop",
            "amount": "4.00",
            "line_items": [
                {
                    "name": "Potatoes",
                    "quantity": 2,
                    "unit_raw": "kg",
                    "total_price": 4.00,
                },
            ],
        }
        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["quantity"] == Decimal("2")

    def test_small_gram_quantity_not_affected(self):
        """quantity=500 + unit=g + no unit_quantity stays unchanged (500g item)."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Shop",
            "amount": "4.09",
            "line_items": [
                {
                    "name": "Frozen Blueberries",
                    "quantity": 500,
                    "unit_raw": "g",
                    "total_price": 4.09,
                },
            ],
        }
        result = refiner.refine(raw)
        item = result["line_items"][0]
        # 500g is an integer count in grams — don't touch
        assert item["quantity"] == Decimal("500")

    def test_fractional_liter_no_unit_quantity_fixed(self):
        """quantity=0.5 + unit=l + no unit_quantity → quantity=1, unit_quantity=0.5."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Shop",
            "amount": "1.50",
            "line_items": [
                {
                    "name": "Orange Juice",
                    "quantity": 0.5,
                    "unit_raw": "l",
                    "total_price": 1.50,
                },
            ],
        }
        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["quantity"] == Decimal("1")
        assert item["unit_quantity"] == Decimal("0.5")

    def test_conversion_factor_with_large_quantity_fixed(self):
        """quantity=2.5 + unit=kg + unit_quantity=1000 → quantity=1, unit_quantity=2.5."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Shop",
            "amount": "7.50",
            "line_items": [
                {
                    "name": "Watermelon",
                    "quantity": 2.5,
                    "unit_raw": "kg",
                    "unit_quantity": 1000,
                    "total_price": 7.50,
                },
            ],
        }
        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["quantity"] == Decimal("1")
        assert item["unit_quantity"] == Decimal("2.5")

    def test_duplicated_weight_in_both_fields_fixed(self):
        """quantity=0.77 + unit=kg + unit_quantity=0.77 → quantity=1, unit_quantity=0.77."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Shop",
            "amount": "2.98",
            "line_items": [
                {
                    "name": "Colored Peppers",
                    "quantity": 0.77,
                    "unit_raw": "kg",
                    "unit_quantity": 0.77,
                    "total_price": 2.98,
                },
            ],
        }
        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["quantity"] == Decimal("1")
        assert item["unit_quantity"] == Decimal("0.77")
        assert item["comparable_unit"] == "kg"
        # 2.98 / 0.77 = 3.87 EUR/kg (rounded to 2dp)
        assert item["comparable_unit_price"] == Decimal("3.87")

    def test_duplicated_integer_weight_not_touched(self):
        """quantity=2 + unit=kg + unit_quantity=2 stays unchanged (could be 2x2kg)."""
        refiner = PurchaseRefiner()
        raw = {
            "vendor": "Shop",
            "amount": "8.00",
            "line_items": [
                {
                    "name": "Rice Bag",
                    "quantity": 2,
                    "unit_raw": "kg",
                    "unit_quantity": 2,
                    "total_price": 8.00,
                },
            ],
        }
        result = refiner.refine(raw)
        item = result["line_items"][0]
        assert item["quantity"] == Decimal("2")
        assert item["unit_quantity"] == Decimal("2")


class TestValidateTotal:
    """Regression tests for _validate_total (audit fix — was a no-op TODO)."""

    def test_validate_total_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """_validate_total logs warning when line items don't sum to total."""
        refiner = PurchaseRefiner()

        line_items = [
            {"total_price": Decimal("10.00")},
            {"total_price": Decimal("5.00")},
        ]
        expected_total = Decimal("50.00")  # Mismatch: items sum to 15

        with caplog.at_level(logging.WARNING):
            refiner._validate_total(expected_total, line_items)

        assert any("mismatch" in record.message.lower() for record in caplog.records)

    def test_validate_total_no_warning_when_matching(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No warning when line items sum matches total."""
        refiner = PurchaseRefiner()

        line_items = [
            {"total_price": Decimal("10.00")},
            {"total_price": Decimal("5.50")},
        ]
        expected_total = Decimal("15.50")

        with caplog.at_level(logging.WARNING):
            refiner._validate_total(expected_total, line_items)

        warning_records = [r for r in caplog.records if "mismatch" in r.message.lower()]
        assert len(warning_records) == 0
