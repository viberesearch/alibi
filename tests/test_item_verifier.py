"""Tests for the 3-layer item verifier."""

import json
import os
import tempfile
import pytest
from decimal import Decimal

from alibi.extraction.item_verifier import (
    CrossValidationResult,
    ItemCorrection,
    ItemFlag,
    ItemVerificationResult,
    _is_header_or_garbage,
    _fix_embedded_weight,
    _fix_leading_quantity,
    _validate_item_math,
    _check_zero_price,
    validate_barcode_items,
    verify_items,
    cross_validate_receipt,
)


# ---------------------------------------------------------------------------
# TestIsHeaderOrGarbage
# ---------------------------------------------------------------------------


class TestIsHeaderOrGarbage:
    def test_column_header_combo(self):
        item = {"name": "QTY DESCRIPTION PRICE AMOUNT VAT"}
        assert _is_header_or_garbage(item) is True

    def test_vat_line(self):
        item = {"name": "VAT 24.00% on 5.50"}
        assert _is_header_or_garbage(item) is True

    def test_short_name(self):
        item = {"name": "A"}
        assert _is_header_or_garbage(item) is True

    def test_all_digits(self):
        item = {"name": "12345"}
        assert _is_header_or_garbage(item) is True

    def test_separator_line(self):
        item = {"name": "-------------------"}
        assert _is_header_or_garbage(item) is True

    def test_normal_item_passes(self):
        item = {"name": "MILK 1L FULL FAT"}
        assert _is_header_or_garbage(item) is False

    def test_ticket_metadata_detected(self):
        """TICKET: footer lines from receipt printers should be garbage."""
        item = {"name": "TICKET:A11T/ DATE: 12/2/2026 TIME: 13:"}
        assert _is_header_or_garbage(item) is True

    def test_ticket_with_spaces(self):
        item = {"name": "TICKET : B22X"}
        assert _is_header_or_garbage(item) is True


# ---------------------------------------------------------------------------
# TestFixEmbeddedWeight
# ---------------------------------------------------------------------------


class TestFixEmbeddedWeight:
    def test_avocado_pattern(self):
        """Pattern 1: NAME 0.765 3.50"""
        item = {
            "name": "AVOCADO HASS 0.765 3.50",
            "quantity": "1",
            "unit": "pcs",
            "unit_price": "3.50",
            "total_price": "3.50",
        }
        result = ItemVerificationResult()
        fixed = _fix_embedded_weight(item, 0, result)

        assert fixed is True
        assert item["name"] == "AVOCADO HASS"
        assert item["quantity"] == "0.765"
        # Unit not assumed — could be kg or L depending on product
        assert "unit" not in item
        assert item["unit_price"] == "3.50"
        assert item["total_price"] == "2.68"  # 0.765 * 3.50
        assert result.items_corrected == 1

    def test_hash_separator_pattern(self):
        """Pattern 2: NAME 1.535 # 13.99 each"""
        item = {
            "name": "BEEF MINCE 1.535 # 13.99 each",
            "quantity": "1",
            "unit": "pcs",
        }
        result = ItemVerificationResult()
        fixed = _fix_embedded_weight(item, 0, result)

        assert fixed is True
        assert item["name"] == "BEEF MINCE"
        assert item["quantity"] == "1.535"
        assert "unit" not in item  # Unit not assumed
        assert item["unit_price"] == "13.99"

    def test_weight_code_prefix_pattern(self):
        """Pattern 3: 0.765 12345 NAME 3.50"""
        item = {
            "name": "0.765 20001 APPLES ROYAL 3.50",
            "quantity": "1",
            "unit": "pcs",
        }
        result = ItemVerificationResult()
        fixed = _fix_embedded_weight(item, 0, result)

        assert fixed is True
        assert item["name"] == "20001 APPLES ROYAL"
        assert item["quantity"] == "0.765"
        assert "unit" not in item  # Unit not assumed

    def test_already_has_weight_unit_skips(self):
        """Items already in kg should not be modified."""
        item = {
            "name": "BANANAS 1.200 2.99",
            "quantity": "1.200",
            "unit_raw": "kg",
            "unit": "kg",
        }
        result = ItemVerificationResult()
        fixed = _fix_embedded_weight(item, 0, result)
        assert fixed is False

    def test_already_has_volume_unit_skips(self):
        """Items already in ml/l should not be modified."""
        item = {
            "name": "JUICE 0.750 3.99",
            "quantity": "0.750",
            "unit_raw": "l",
            "unit": "l",
        }
        result = ItemVerificationResult()
        fixed = _fix_embedded_weight(item, 0, result)
        assert fixed is False

    def test_qty_not_one_skips(self):
        """Only fix qty=1 items."""
        item = {
            "name": "ORANGES 0.500 4.00",
            "quantity": "3",
            "unit": "pcs",
        }
        result = ItemVerificationResult()
        fixed = _fix_embedded_weight(item, 0, result)
        assert fixed is False

    def test_unreasonable_weight_skips(self):
        """Weight > 50 should not match."""
        item = {
            "name": "ITEM 99.999 1.00",
            "quantity": "1",
            "unit": "pcs",
        }
        result = ItemVerificationResult()
        fixed = _fix_embedded_weight(item, 0, result)
        assert fixed is False


# ---------------------------------------------------------------------------
# TestFixLeadingQuantity
# ---------------------------------------------------------------------------


class TestFixLeadingQuantity:
    def test_red_bull_pattern(self):
        """'4 Red Bull 250ml' where 4 * 0.99 = 3.96"""
        item = {
            "name": "4 Red Bull 250ml",
            "quantity": "1",
            "unit_price": "0.99",
            "total_price": "3.96",
        }
        result = ItemVerificationResult()
        fixed = _fix_leading_quantity(item, 0, result)

        assert fixed is True
        assert item["name"] == "Red Bull 250ml"
        assert item["quantity"] == "4"
        assert result.items_corrected == 1

    def test_non_matching_skips(self):
        """Leading number doesn't match math."""
        item = {
            "name": "4 Red Bull 250ml",
            "quantity": "1",
            "unit_price": "1.50",
            "total_price": "3.96",
        }
        result = ItemVerificationResult()
        fixed = _fix_leading_quantity(item, 0, result)
        assert fixed is False

    def test_no_leading_number(self):
        item = {
            "name": "Red Bull 250ml",
            "quantity": "1",
            "unit_price": "0.99",
            "total_price": "0.99",
        }
        result = ItemVerificationResult()
        fixed = _fix_leading_quantity(item, 0, result)
        assert fixed is False


# ---------------------------------------------------------------------------
# TestValidateItemMath
# ---------------------------------------------------------------------------


class TestValidateItemMath:
    def test_correct_math(self):
        item = {
            "name": "MILK",
            "quantity": "2",
            "unit_price": "1.50",
            "total_price": "3.00",
        }
        result = ItemVerificationResult()
        _validate_item_math(item, 0, result)
        assert result.items_flagged == 0
        assert result.items_corrected == 0

    def test_weight_in_name_fix(self):
        """0.230 in name, qty=1, unit_price=20.00, total=4.60 -> qty becomes 0.230"""
        item = {
            "name": "CHEESE GOUDA 0.230",
            "quantity": "1",
            "unit_price": "20.00",
            "total_price": "4.60",
        }
        result = ItemVerificationResult()
        _validate_item_math(item, 0, result)

        assert result.items_corrected == 1
        assert item["quantity"] == "0.230"
        assert "unit" not in item  # Unit not assumed — could be kg or L

    def test_unfixable_flags(self):
        item = {
            "name": "MYSTERY ITEM",
            "quantity": "1",
            "unit_price": "5.00",
            "total_price": "7.50",
        }
        result = ItemVerificationResult()
        _validate_item_math(item, 0, result)
        assert result.items_flagged == 1
        assert result.flags[0].issue == "math_mismatch"

    def test_tolerance_within_range(self):
        """0.01 difference is within 0.02 tolerance — should pass."""
        item = {
            "name": "TEA",
            "quantity": "1",
            "unit_price": "2.99",
            "total_price": "3.00",
        }
        result = ItemVerificationResult()
        _validate_item_math(item, 0, result)
        assert result.items_flagged == 0  # 0.01 within 0.02 tolerance

    def test_exact_tolerance_passes(self):
        """Exactly at tolerance boundary (0.02)."""
        item = {
            "name": "TEA",
            "quantity": "1",
            "unit_price": "2.98",
            "total_price": "3.00",
        }
        result = ItemVerificationResult()
        _validate_item_math(item, 0, result)
        assert result.items_flagged == 0  # 0.02 is within tolerance


# ---------------------------------------------------------------------------
# TestCheckUnreasonablePrice
# ---------------------------------------------------------------------------


class TestCheckUnreasonablePrice:
    def test_ocr_digit_merge_flagged(self):
        """Price > 500 flagged as likely OCR digit merge."""
        from alibi.extraction.item_verifier import _check_unreasonable_price

        item = {"name": "MILK 1L", "total_price": "5011.0"}
        result = ItemVerificationResult()
        _check_unreasonable_price(item, 0, result)
        assert result.items_flagged == 1
        assert result.flags[0].issue == "unreasonable_price"

    def test_normal_price_passes(self):
        from alibi.extraction.item_verifier import _check_unreasonable_price

        item = {"name": "MILK 1L", "total_price": "3.50"}
        result = ItemVerificationResult()
        _check_unreasonable_price(item, 0, result)
        assert result.items_flagged == 0

    def test_borderline_500_passes(self):
        from alibi.extraction.item_verifier import _check_unreasonable_price

        item = {"name": "Electronics", "total_price": "499.99"}
        result = ItemVerificationResult()
        _check_unreasonable_price(item, 0, result)
        assert result.items_flagged == 0


# ---------------------------------------------------------------------------
# TestCheckZeroPrice
# ---------------------------------------------------------------------------


class TestCheckZeroPrice:
    def test_flagged(self):
        item = {"name": "BREAD", "unit_price": "0", "total_price": "0"}
        result = ItemVerificationResult()
        _check_zero_price(item, 0, result)
        assert result.items_flagged == 1

    def test_deposit_exempt(self):
        item = {"name": "Pfand 0.25", "unit_price": "0", "total_price": "0"}
        result = ItemVerificationResult()
        _check_zero_price(item, 0, result)
        assert result.items_flagged == 0

    def test_normal_price_passes(self):
        item = {"name": "BREAD", "unit_price": "1.50", "total_price": "1.50"}
        result = ItemVerificationResult()
        _check_zero_price(item, 0, result)
        assert result.items_flagged == 0


# ---------------------------------------------------------------------------
# TestVerifyItemsIntegration
# ---------------------------------------------------------------------------


class TestVerifyItemsIntegration:
    def test_mixed_issues(self):
        """Multiple items with different issues."""
        extracted = {
            "line_items": [
                {
                    "name": "QTY DESCRIPTION PRICE AMOUNT",
                    "quantity": "1",
                    "unit_price": "0",
                },
                {
                    "name": "AVOCADO 0.765 3.50",
                    "quantity": "1",
                    "unit": "pcs",
                    "unit_price": "3.50",
                    "total_price": "3.50",
                },
                {
                    "name": "MILK",
                    "quantity": "2",
                    "unit_price": "1.50",
                    "total_price": "3.00",
                },
            ]
        }
        result = verify_items(extracted)

        # Header row removed
        assert result.items_removed == 1
        # Avocado weight fixed
        assert result.items_corrected >= 1
        # 2 items remain (header removed)
        assert len(extracted["line_items"]) == 2
        # Milk is clean
        assert extracted["line_items"][1]["name"] == "MILK"

    def test_empty_items(self):
        extracted = {"line_items": []}
        result = verify_items(extracted)
        assert result.items_verified == 0

    def test_no_items_key(self):
        extracted = {"vendor": "TEST SHOP"}
        result = verify_items(extracted)
        assert result.items_verified == 0

    def test_all_clean_receipt(self):
        extracted = {
            "line_items": [
                {
                    "name": "MILK",
                    "quantity": "1",
                    "unit_price": "1.50",
                    "total_price": "1.50",
                },
                {
                    "name": "BREAD",
                    "quantity": "2",
                    "unit_price": "2.00",
                    "total_price": "4.00",
                },
            ]
        }
        result = verify_items(extracted)
        assert result.items_verified == 2
        assert result.items_corrected == 0
        assert result.items_flagged == 0
        assert result.items_removed == 0


# ---------------------------------------------------------------------------
# TestCrossValidateReceipt
# ---------------------------------------------------------------------------


class TestCrossValidateReceipt:
    def test_matching_sum(self):
        extracted = {
            "total": "5.50",
            "line_items": [
                {"total_price": "2.50"},
                {"total_price": "3.00"},
            ],
        }
        result = cross_validate_receipt(extracted)
        assert len(result.warnings) == 0
        assert not result.needs_review

    def test_large_mismatch_needs_review(self):
        """Mismatch >50% flags needs_review."""
        extracted = {
            "total": "100.00",
            "line_items": [
                {"total_price": "10.00"},
                {"total_price": "5.00"},
            ],
        }
        result = cross_validate_receipt(extracted)
        assert len(result.warnings) == 1
        assert "85.0%" in result.warnings[0]
        assert result.needs_review

    def test_moderate_mismatch_warns_only(self):
        """Mismatch 10-50% warns but does not flag needs_review."""
        extracted = {
            "total": "100.00",
            "line_items": [
                {"total_price": "60.00"},
                {"total_price": "20.00"},
            ],
        }
        result = cross_validate_receipt(extracted)
        assert len(result.warnings) == 1
        assert "20.0%" in result.warnings[0]
        assert not result.needs_review

    def test_no_total(self):
        extracted = {
            "line_items": [
                {"total_price": "10.00"},
            ],
        }
        result = cross_validate_receipt(extracted)
        assert len(result.warnings) == 0
        assert not result.needs_review


class TestPriceSwapDetection:
    """Tests for price/total swap detection between adjacent items."""

    def test_swap_detected_and_fixed(self):
        extracted = {
            "line_items": [
                {
                    "name": "Milk 1L",
                    "quantity": "2",
                    "unit_price": "1.50",
                    "total_price": "4.00",
                },
                {
                    "name": "Bread",
                    "quantity": "1",
                    "unit_price": "4.00",
                    "total_price": "3.00",
                },
            ]
        }
        result = verify_items(extracted)
        assert result.items_corrected >= 2
        items = extracted["line_items"]
        # After swap: Milk should be 3.00, Bread should be 4.00
        assert str(items[0]["total_price"]) == "3.00"
        assert str(items[1]["total_price"]) == "4.00"

    def test_no_swap_when_math_correct(self):
        extracted = {
            "line_items": [
                {
                    "name": "Milk",
                    "quantity": "2",
                    "unit_price": "1.50",
                    "total_price": "3.00",
                },
                {
                    "name": "Bread",
                    "quantity": "1",
                    "unit_price": "2.00",
                    "total_price": "2.00",
                },
            ]
        }
        result = verify_items(extracted)
        assert result.items_corrected == 0


class TestContinuationMerge:
    """Tests for multi-line item name merge detection."""

    def test_merge_into_previous(self):
        extracted = {
            "line_items": [
                {
                    "name": "EGGS JONIS",
                    "quantity": "1",
                    "unit_price": "3.50",
                    "total_price": "3.50",
                },
                {
                    "name": "FARM (1X30)",
                    "quantity": None,
                    "unit_price": None,
                    "total_price": None,
                },
            ]
        }
        result = verify_items(extracted)
        items = extracted["line_items"]
        assert len(items) == 1
        assert "EGGS JONIS" in items[0]["name"]
        assert "FARM" in items[0]["name"]

    def test_no_merge_with_prices(self):
        extracted = {
            "line_items": [
                {
                    "name": "Milk",
                    "quantity": "1",
                    "unit_price": "1.50",
                    "total_price": "1.50",
                },
                {
                    "name": "Bread",
                    "quantity": "1",
                    "unit_price": "2.00",
                    "total_price": "2.00",
                },
            ]
        }
        result = verify_items(extracted)
        assert len(extracted["line_items"]) == 2

    def test_merge_into_next(self):
        """Priceless item at start merges into next item."""
        extracted = {
            "line_items": [
                {
                    "name": "ORGANIC",
                    "quantity": None,
                    "unit_price": None,
                    "total_price": None,
                },
                {
                    "name": "BANANAS",
                    "quantity": "1",
                    "unit_price": "2.50",
                    "total_price": "2.50",
                },
            ]
        }
        result = verify_items(extracted)
        items = extracted["line_items"]
        assert len(items) == 1
        assert "ORGANIC" in items[0]["name"]
        assert "BANANAS" in items[0]["name"]


# ---------------------------------------------------------------------------
# TestItemCountValidation
# ---------------------------------------------------------------------------


class TestItemCountValidation:
    """Tests for declared item count validation in cross_validate_receipt."""

    def test_matching_item_count(self):
        """No warning when extracted count matches declared count."""
        extracted = {
            "total": "10.00",
            "declared_item_count": 2,
            "line_items": [
                {"total_price": "5.00"},
                {"total_price": "5.00"},
            ],
        }
        result = cross_validate_receipt(extracted)
        assert not any("declares" in w for w in result.warnings)

    def test_mismatched_item_count(self):
        """Warning when extracted count differs from declared."""
        extracted = {
            "total": "10.00",
            "declared_item_count": 5,
            "line_items": [
                {"total_price": "5.00"},
                {"total_price": "5.00"},
            ],
        }
        result = cross_validate_receipt(extracted)
        mismatch_warnings = [w for w in result.warnings if "declares" in w]
        assert len(mismatch_warnings) == 1
        assert "2 items" in mismatch_warnings[0]
        assert "5 items" in mismatch_warnings[0]

    def test_no_declared_count(self):
        """No item count warning when receipt has no declared count."""
        extracted = {
            "total": "10.00",
            "line_items": [
                {"total_price": "5.00"},
                {"total_price": "5.00"},
            ],
        }
        result = cross_validate_receipt(extracted)
        assert not any("declares" in w for w in result.warnings)

    def test_item_count_mismatch_positive(self):
        """item_count_mismatch is positive when items are missing."""
        extracted = {
            "total": "15.00",
            "declared_item_count": 5,
            "line_items": [
                {"total_price": "5.00"},
                {"total_price": "5.00"},
                {"total_price": "5.00"},
            ],
        }
        result = cross_validate_receipt(extracted)
        assert result.item_count_mismatch == 2

    def test_item_count_mismatch_zero_when_match(self):
        """item_count_mismatch is 0 when counts match."""
        extracted = {
            "total": "10.00",
            "declared_item_count": 2,
            "line_items": [
                {"total_price": "5.00"},
                {"total_price": "5.00"},
            ],
        }
        result = cross_validate_receipt(extracted)
        assert result.item_count_mismatch == 0

    def test_item_count_mismatch_negative_when_extra(self):
        """item_count_mismatch is negative when more items than declared."""
        extracted = {
            "total": "15.00",
            "declared_item_count": 2,
            "line_items": [
                {"total_price": "5.00"},
                {"total_price": "5.00"},
                {"total_price": "5.00"},
            ],
        }
        result = cross_validate_receipt(extracted)
        assert result.item_count_mismatch == -1


class TestInvestigateMissingItems:
    """Tests for OCR-based missing item investigation."""

    def test_finds_unmatched_price_lines(self):
        from alibi.extraction.item_verifier import investigate_missing_items

        extracted = {
            "_ocr_text": (
                "ALPHAMEGA MILK 1L    1.50\n"
                "BARILLA SPAGHETTI    2.00\n"
                "FETA CHEESE DODONI   3.50\n"
                "TOTAL                7.00"
            ),
            "line_items": [
                {"name": "ALPHAMEGA MILK 1L"},
                {"name": "BARILLA SPAGHETTI"},
            ],
        }
        missed = investigate_missing_items(extracted, declared_count=3)
        assert len(missed) == 1
        assert "FETA CHEESE" in missed[0]

    def test_skips_total_lines(self):
        from alibi.extraction.item_verifier import investigate_missing_items

        extracted = {
            "_ocr_text": "MILK 1L    1.50\nSUBTOTAL   1.50\nTOTAL      1.50",
            "line_items": [
                {"name": "MILK 1L"},
            ],
        }
        missed = investigate_missing_items(extracted, declared_count=2)
        # Total/subtotal lines should not be returned as missed items
        assert not any("TOTAL" in m.upper() for m in missed)

    def test_empty_ocr_returns_empty(self):
        from alibi.extraction.item_verifier import investigate_missing_items

        extracted = {
            "line_items": [{"name": "MILK"}],
        }
        missed = investigate_missing_items(extracted, declared_count=3)
        assert missed == []

    def test_no_missing_returns_empty(self):
        from alibi.extraction.item_verifier import investigate_missing_items

        extracted = {
            "_ocr_text": "MILK 1.50\nBREAD 2.00",
            "line_items": [
                {"name": "MILK"},
                {"name": "BREAD"},
            ],
        }
        missed = investigate_missing_items(extracted, declared_count=2)
        assert missed == []


# ---------------------------------------------------------------------------
# TestBarcodeItemValidation
# ---------------------------------------------------------------------------


class TestBarcodeItemValidation:
    """Tests for barcode-to-item cross-validation using product_cache."""

    @pytest.fixture
    def db(self):
        """Create a temp DB with full schema."""
        from alibi.config import Config, reset_config
        from alibi.db.connection import DatabaseManager

        reset_config()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        config = Config(db_path=db_path)
        manager = DatabaseManager(config)
        if not manager.is_initialized():
            manager.initialize()
        yield manager
        manager.close()
        os.unlink(db_path)

    def test_matching_barcode_no_flag(self, db):
        """No flag when OFF product name matches item name."""
        db.execute(
            "INSERT INTO product_cache (barcode, data, source) VALUES (?, ?, ?)",
            (
                "5201054025642",
                json.dumps({"product_name": "Alphamega Full Fat Milk 1L"}),
                "openfoodfacts",
            ),
        )
        extracted = {
            "line_items": [
                {
                    "name": "ALPHAMEGA MILK 1L FULL FAT",
                    "barcode": "5201054025642",
                },
            ],
        }
        flags = validate_barcode_items(extracted, db)
        assert len(flags) == 0

    def test_valid_ean_mismatch_auto_corrects(self, db):
        """Valid EAN with low similarity auto-corrects item name."""
        db.execute(
            "INSERT INTO product_cache (barcode, data, source) VALUES (?, ?, ?)",
            (
                "5201054025642",
                json.dumps({"product_name": "Coca-Cola Zero 330ml"}),
                "openfoodfacts",
            ),
        )
        extracted = {
            "line_items": [
                {
                    "name": "ORGANIC BANANAS",
                    "barcode": "5201054025642",
                },
            ],
        }
        flags = validate_barcode_items(extracted, db)
        assert len(flags) == 1
        assert flags[0].issue == "barcode_item_corrected"
        assert "Coca-Cola" in flags[0].context
        # Item name was corrected in-place
        assert extracted["line_items"][0]["name"] == "Coca-Cola Zero 330ml"

    def test_invalid_ean_mismatch_flags_only(self, db):
        """Non-EAN barcode with low similarity flags without correction."""
        db.execute(
            "INSERT INTO product_cache (barcode, data, source) VALUES (?, ?, ?)",
            (
                "12345678",
                json.dumps({"product_name": "Coca-Cola Zero 330ml"}),
                "openfoodfacts",
            ),
        )
        extracted = {
            "line_items": [
                {
                    "name": "ORGANIC BANANAS",
                    "barcode": "12345678",
                },
            ],
        }
        flags = validate_barcode_items(extracted, db)
        assert len(flags) == 1
        assert flags[0].issue == "barcode_item_mismatch"
        # Item name NOT corrected
        assert extracted["line_items"][0]["name"] == "ORGANIC BANANAS"

    def test_short_barcode_skipped(self, db):
        """Short barcodes (product codes, not EAN) are skipped."""
        extracted = {
            "line_items": [
                {
                    "name": "MILK",
                    "barcode": "3886",
                },
            ],
        }
        flags = validate_barcode_items(extracted, db)
        assert len(flags) == 0

    def test_not_found_sentinel_skipped(self, db):
        """Negative cache entries (_not_found) are skipped."""
        db.execute(
            "INSERT INTO product_cache (barcode, data, source) VALUES (?, ?, ?)",
            (
                "5201054025642",
                json.dumps({"_not_found": True}),
                "openfoodfacts",
            ),
        )
        extracted = {
            "line_items": [
                {
                    "name": "SOME ITEM",
                    "barcode": "5201054025642",
                },
            ],
        }
        flags = validate_barcode_items(extracted, db)
        assert len(flags) == 0

    def test_no_db_returns_empty(self):
        """No flags when db is None."""
        extracted = {
            "line_items": [
                {"name": "MILK", "barcode": "5201054025642"},
            ],
        }
        flags = validate_barcode_items(extracted, db=None)
        assert len(flags) == 0
