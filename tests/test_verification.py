"""Tests for extraction verification (pure unit tests — no mocks needed)."""

from typing import Any

import pytest

from alibi.extraction.verification import (
    FALLBACK_THRESHOLD,
    RERUN_THRESHOLD,
    VerificationResult,
    _check_amount_sum,
    _check_date_valid,
    _check_item_count,
    _check_line_item_math,
    _check_required_fields,
    build_emphasis_prompt,
    verify_extraction,
)


class TestCheckAmountSum:
    """Tests for line-item sum vs total check."""

    def test_exact_match(self):
        data = {
            "total": 10.00,
            "line_items": [
                {"total_price": 4.00},
                {"total_price": 6.00},
            ],
        }
        score, flags = _check_amount_sum(data)
        assert score == 1.0
        assert flags == []

    def test_subtotal_match(self):
        data = {
            "total": 11.90,
            "subtotal": 10.00,
            "tax": 1.90,
            "line_items": [
                {"total_price": 4.00},
                {"total_price": 6.00},
            ],
        }
        score, flags = _check_amount_sum(data)
        assert score >= 0.9

    def test_sum_with_tax_added(self):
        data = {
            "total": 11.90,
            "tax": 1.90,
            "line_items": [
                {"total_price": 4.00},
                {"total_price": 6.00},
            ],
        }
        score, flags = _check_amount_sum(data)
        assert score >= 0.9

    def test_no_total(self):
        data = {"line_items": [{"total_price": 5.00}]}
        score, flags = _check_amount_sum(data)
        assert score == 0.5
        assert "no_total" in flags

    def test_no_line_items(self):
        data = {"total": 10.00, "line_items": []}
        score, flags = _check_amount_sum(data)
        assert score == 0.5

    def test_significant_mismatch(self):
        data = {
            "total": 100.00,
            "line_items": [
                {"total_price": 30.00},
                {"total_price": 20.00},
            ],
        }
        score, flags = _check_amount_sum(data)
        assert score < 0.5
        assert any("sum_off" in f for f in flags)

    def test_no_item_prices(self):
        data = {
            "total": 10.00,
            "line_items": [{"name": "Item A"}, {"name": "Item B"}],
        }
        score, flags = _check_amount_sum(data)
        assert score == 0.5
        assert "no_item_prices" in flags

    def test_close_match_within_tolerance(self):
        data = {
            "total": 10.00,
            "line_items": [
                {"total_price": 4.00},
                {"total_price": 5.97},
            ],
        }
        score, flags = _check_amount_sum(data)
        assert score >= 0.7  # Within 1%

    def test_uses_amount_field_if_no_total(self):
        data = {
            "amount": 10.00,
            "line_items": [
                {"total_price": 4.00},
                {"total_price": 6.00},
            ],
        }
        score, flags = _check_amount_sum(data)
        assert score == 1.0


class TestCheckLineItemMath:
    """Tests for qty * unit_price == total_price check."""

    def test_all_correct(self):
        data = {
            "line_items": [
                {"quantity": 2, "unit_price": 3.00, "total_price": 6.00},
                {"quantity": 1, "unit_price": 4.50, "total_price": 4.50},
            ]
        }
        score, flags = _check_line_item_math(data)
        assert score == 1.0
        assert flags == []

    def test_one_wrong(self):
        data = {
            "line_items": [
                {"quantity": 2, "unit_price": 3.00, "total_price": 6.00},
                {"quantity": 1, "unit_price": 4.50, "total_price": 9.99},
            ]
        }
        score, flags = _check_line_item_math(data)
        assert score == 0.5
        assert len(flags) == 1

    def test_no_items(self):
        data: dict[str, Any] = {"line_items": []}
        score, flags = _check_line_item_math(data)
        assert score == 0.5

    def test_missing_fields(self):
        data = {
            "line_items": [
                {"name": "Item A", "total_price": 5.00},
            ]
        }
        score, flags = _check_line_item_math(data)
        assert score == 0.5  # Not checkable

    def test_zero_quantity(self):
        data = {
            "line_items": [
                {"quantity": 0, "unit_price": 5.00, "total_price": 0.00},
            ]
        }
        score, flags = _check_line_item_math(data)
        assert score == 0.5  # Not checkable (qty=0)


class TestCheckItemCount:
    """Tests for item count in OCR text vs actual items."""

    def test_matching_count(self):
        data = {"line_items": [{"name": f"Item {i}"} for i in range(5)]}
        ocr_text = "Total: 5 items"
        score, flags = _check_item_count(data, ocr_text)
        assert score == 1.0

    def test_no_ocr_text(self):
        data = {"line_items": [{"name": "A"}]}
        score, flags = _check_item_count(data, None)
        assert score == 0.5

    def test_no_count_in_text(self):
        data = {"line_items": [{"name": "A"}]}
        score, flags = _check_item_count(data, "Thank you for shopping")
        assert score == 0.5

    def test_mismatched_count(self):
        data = {"line_items": [{"name": f"Item {i}"} for i in range(3)]}
        ocr_text = "23 Items"
        score, flags = _check_item_count(data, ocr_text)
        assert score < 0.5

    def test_close_count(self):
        data = {"line_items": [{"name": f"Item {i}"} for i in range(5)]}
        ocr_text = "6 items"
        score, flags = _check_item_count(data, ocr_text)
        assert score == 0.7

    def test_german_artikel(self):
        data = {"line_items": [{"name": f"Item {i}"} for i in range(3)]}
        ocr_text = "3 Artikel"
        score, flags = _check_item_count(data, ocr_text)
        assert score == 1.0


class TestCheckRequiredFields:
    """Tests for required field presence."""

    def test_all_present(self):
        data = {
            "vendor": "Test Store",
            "date": "2024-01-15",
            "total": 25.99,
            "currency": "EUR",
        }
        score, flags = _check_required_fields(data)
        assert score == 1.0
        assert flags == []

    def test_all_missing(self):
        data: dict[str, Any] = {}
        score, flags = _check_required_fields(data)
        assert score == 0.0
        assert "missing_vendor" in flags
        assert "missing_date" in flags
        assert "missing_total" in flags
        assert "missing_currency" in flags

    def test_partial(self):
        data = {"vendor": "Test", "total": 10.00}
        score, flags = _check_required_fields(data)
        assert score == 0.5
        assert "missing_date" in flags
        assert "missing_currency" in flags

    def test_uses_document_date_fallback(self):
        data = {
            "vendor": "Test",
            "document_date": "2024-01-15",
            "total": 10,
            "currency": "EUR",
        }
        score, flags = _check_required_fields(data)
        assert score == 1.0

    def test_uses_amount_fallback(self):
        data = {
            "vendor": "Test",
            "date": "2024-01-15",
            "amount": 10,
            "currency": "EUR",
        }
        score, flags = _check_required_fields(data)
        assert score == 1.0


class TestCheckDateValid:
    """Tests for date parsing and range check."""

    def test_valid_recent_date(self):
        from datetime import date, timedelta

        recent = (date.today() - timedelta(days=30)).isoformat()
        data = {"date": recent}
        score, flags = _check_date_valid(data)
        assert score == 1.0

    def test_no_date(self):
        data: dict[str, Any] = {}
        score, flags = _check_date_valid(data)
        assert score == 0.5

    def test_unparseable_date(self):
        data = {"date": "not-a-date"}
        score, flags = _check_date_valid(data)
        assert score == 0.0
        assert "date_unparseable" in flags

    def test_old_date(self):
        data = {"date": "2015-01-01"}
        score, flags = _check_date_valid(data)
        assert score == 0.2

    def test_dd_mm_yyyy_format(self):
        from datetime import date, timedelta

        d = date.today() - timedelta(days=10)
        data = {"date": d.strftime("%d/%m/%Y")}
        score, flags = _check_date_valid(data)
        assert score == 1.0


class TestVerifyExtraction:
    """Tests for the combined verification function."""

    def test_good_extraction(self):
        from datetime import date, timedelta

        recent = (date.today() - timedelta(days=5)).isoformat()
        data = {
            "vendor": "SuperMarket",
            "date": recent,
            "total": 15.00,
            "currency": "EUR",
            "line_items": [
                {
                    "name": "Bread",
                    "quantity": 1,
                    "unit_price": 3.00,
                    "total_price": 3.00,
                },
                {
                    "name": "Milk",
                    "quantity": 2,
                    "unit_price": 2.00,
                    "total_price": 4.00,
                },
                {
                    "name": "Cheese",
                    "quantity": 1,
                    "unit_price": 8.00,
                    "total_price": 8.00,
                },
            ],
        }
        result = verify_extraction(data)
        assert result.confidence >= 0.7
        assert result.passed is True
        assert result.rerun_recommended is False

    def test_empty_extraction(self):
        result = verify_extraction({})
        assert result.confidence == 0.0
        assert result.passed is False
        assert result.rerun_recommended is True

    def test_bad_extraction(self):
        data = {
            "total": 100.00,
            "line_items": [
                {"name": "A", "quantity": 1, "unit_price": 5.00, "total_price": 99.00},
            ],
        }
        result = verify_extraction(data)
        assert result.confidence < RERUN_THRESHOLD
        assert result.rerun_recommended is True

    def test_check_scores_populated(self):
        data = {"vendor": "Test", "date": "2024-01-15", "total": 10, "currency": "EUR"}
        result = verify_extraction(data)
        assert "amount_sum" in result.check_scores
        assert "line_item_math" in result.check_scores
        assert "required_fields" in result.check_scores
        assert "date_valid" in result.check_scores

    def test_with_ocr_text_item_count(self):
        from datetime import date, timedelta

        recent = (date.today() - timedelta(days=5)).isoformat()
        data = {
            "vendor": "Shop",
            "date": recent,
            "total": 6.00,
            "currency": "EUR",
            "line_items": [
                {"name": "A", "quantity": 1, "unit_price": 3.00, "total_price": 3.00},
                {"name": "B", "quantity": 1, "unit_price": 3.00, "total_price": 3.00},
            ],
        }
        result = verify_extraction(data, ocr_text="2 Items\nTotal: 6.00")
        assert result.check_scores["item_count"] == 1.0


class TestBuildEmphasisPrompt:
    """Tests for emphasis prompt builder."""

    def test_no_failures_returns_base(self):
        prompt = build_emphasis_prompt("some text", "receipt", {})
        assert "some text" in prompt
        assert "CORRECTIONS" not in prompt

    def test_amount_sum_failure(self):
        prompt = build_emphasis_prompt("some text", "receipt", {"amount_sum": 0.3})
        assert "total amount" in prompt.lower() or "total matches" in prompt.lower()

    def test_line_item_math_failure(self):
        prompt = build_emphasis_prompt("some text", "receipt", {"line_item_math": 0.2})
        assert "quantity" in prompt.lower() or "unit_price" in prompt.lower()

    def test_item_count_failure(self):
        prompt = build_emphasis_prompt("some text", "receipt", {"item_count": 0.1})
        assert "all line items" in prompt.lower() or "skip" in prompt.lower()
