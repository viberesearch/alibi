"""Tests for _clean_invoice_item_name() in alibi/extraction/text_parser.py.

Covers OCR-wrapped invoice table rows where columns bleed into the item name
field (line numbers, product codes, qty, unit price, currency, species codes).
"""

import pytest

from alibi.extraction.text_parser import _clean_invoice_item_name


def test_strips_leading_code():
    assert (
        _clean_invoice_item_name("1 20026 Σδαμύς Στέκ Nopβηγις")
        == "Σδαμύς Στέκ Nopβηγις"
    )


def test_strips_trailing_metadata():
    result = _clean_invoice_item_name(
        "1 20026 Σδαμύς Στέκ Nopβηγις (Salino Salar) - SAL 0.364 23.80 ΥθατιεύςNopβηγις 5.00 EUR"
    )
    assert result == "Σδαμύς Στέκ Nopβηγις (Salino Salar)"


def test_second_real_example():
    result = _clean_invoice_item_name(
        "2 60482 Καλαμύρι Αλανικίου Κατεμυγιμένο 300-500 (Lalga Spp) - SOC 0.526 19.04 ΑπτοταμψιμένοΑλειμενοΤράτα 5.00 EUR"
    )
    assert result == "Καλαμύρι Αλανικίου Κατεμυγιμένο 300-500 (Lalga Spp)"


def test_preserves_clean_name():
    assert _clean_invoice_item_name("Fresh Salmon Steak") == "Fresh Salmon Steak"


def test_preserves_size_range():
    """300-500 is a size descriptor, not metadata."""
    result = _clean_invoice_item_name(
        "2 60482 Καλαμύρι 300-500 (Lalga Spp) - SOC 0.526 19.04 Text 5.00 EUR"
    )
    assert "300-500" in result
    assert result.startswith("Καλαμύρι")


def test_empty_string():
    assert _clean_invoice_item_name("") == ""


def test_none_passthrough():
    assert _clean_invoice_item_name(None) is None


def test_strips_species_suffix_only():
    """Name without leading code but with species suffix."""
    result = _clean_invoice_item_name("Salmon Steak (Salino Salar) - SAL")
    assert result == "Salmon Steak (Salino Salar)"


def test_strips_trailing_eur_only():
    """Name with trailing EUR but no full metadata block."""
    result = _clean_invoice_item_name("Salmon Steak 5.00 EUR")
    assert result == "Salmon Steak"


def test_no_product_code_no_mutation():
    """Name starting with text (no leading number+code) is not mutated."""
    result = _clean_invoice_item_name("Atlantic Salmon Fillet 300g")
    assert result == "Atlantic Salmon Fillet 300g"


def test_large_product_code():
    """Product codes up to 13 digits are stripped."""
    result = _clean_invoice_item_name("3 1234567890123 Tuna Steak")
    assert result == "Tuna Steak"


def test_three_digit_line_number():
    """Line numbers up to 3 digits are stripped."""
    result = _clean_invoice_item_name("100 20026 Salmon Fillet")
    assert result == "Salmon Fillet"
