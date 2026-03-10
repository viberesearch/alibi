"""Tests for _clean_item_name() in collapse."""

from uuid import uuid4

from alibi.clouds.collapse import _clean_item_name
from alibi.db.models import FactItem


def _make_item(
    name: str = "Test Item",
    barcode: str | None = None,
    name_normalized: str | None = None,
) -> FactItem:
    """Helper to create a FactItem for testing."""
    item = FactItem(
        id=str(uuid4()),
        fact_id="",
        atom_id=str(uuid4()),
        name=name,
        name_normalized=name_normalized if name_normalized is not None else name,
    )
    item.barcode = barcode
    return item


class TestCleanItemName:
    def test_trailing_ea_stripped(self):
        item = _make_item("ALPHAMEGA FREE RANGE ea")
        result = _clean_item_name(item.name, item)
        assert result == "ALPHAMEGA FREE RANGE"

    def test_trailing_ea_case_insensitive(self):
        item = _make_item("WHOLE MILK EA")
        result = _clean_item_name(item.name, item)
        assert result == "WHOLE MILK"

    def test_trailing_ea_not_stripped_mid_name(self):
        item = _make_item("PEAS AND BEANS")
        result = _clean_item_name(item.name, item)
        assert result == "PEAS AND BEANS"

    def test_barcode_suffix_extracted(self):
        item = _make_item("TILDEN BASMAT Barcode: 4250370805152")
        result = _clean_item_name(item.name, item)
        assert result == "TILDEN BASMAT"
        assert item.barcode == "4250370805152"

    def test_barcode_not_overwritten_when_already_set(self):
        item = _make_item("TILDEN BASMAT Barcode: 4250370805152", barcode="9999999")
        result = _clean_item_name(item.name, item)
        assert result == "TILDEN BASMAT"
        assert item.barcode == "9999999"

    def test_barcode_case_insensitive(self):
        item = _make_item("PRODUCT NAME barcode: 1234567890")
        result = _clean_item_name(item.name, item)
        assert result == "PRODUCT NAME"
        assert item.barcode == "1234567890"

    def test_leading_sku_stripped(self):
        item = _make_item("10163 ea TILDEN BASMAT")
        result = _clean_item_name(item.name, item)
        assert result == "TILDEN BASMAT"

    def test_leading_sku_three_digits(self):
        item = _make_item("999 ea BUTTER")
        result = _clean_item_name(item.name, item)
        assert result == "BUTTER"

    def test_leading_sku_six_digits(self):
        item = _make_item("123456 ea OLIVE OIL")
        result = _clean_item_name(item.name, item)
        assert result == "OLIVE OIL"

    def test_leading_sku_and_trailing_ea_and_barcode(self):
        item = _make_item("10163 ea TILDEN BASMAT ea Barcode: 4250370805152")
        result = _clean_item_name(item.name, item)
        assert result == "TILDEN BASMAT"
        assert item.barcode == "4250370805152"

    def test_no_false_positive_7up(self):
        item = _make_item("7UP ZERO")
        result = _clean_item_name(item.name, item)
        assert result == "7UP ZERO"

    def test_no_false_positive_greek_digits(self):
        item = _make_item("3 ΑΥΓΑ")
        result = _clean_item_name(item.name, item)
        assert result == "3 ΑΥΓΑ"

    def test_no_false_positive_two_digit_sku(self):
        item = _make_item("99 ea PRODUCT")
        result = _clean_item_name(item.name, item)
        assert result == "99 ea PRODUCT"

    def test_plain_name_unchanged(self):
        item = _make_item("FRESH ORANGE JUICE 1L")
        result = _clean_item_name(item.name, item)
        assert result == "FRESH ORANGE JUICE 1L"

    def test_name_normalized_updated_when_matching(self):
        original = "ALPHAMEGA FREE RANGE ea"
        item = _make_item(original, name_normalized=original)
        cleaned = _clean_item_name(item.name, item)
        if item.name_normalized == original:
            item.name_normalized = cleaned
        assert item.name_normalized == "ALPHAMEGA FREE RANGE"

    def test_name_normalized_not_updated_when_custom(self):
        item = _make_item("ALPHAMEGA FREE RANGE ea", name_normalized="Custom Name")
        original_name = item.name
        cleaned = _clean_item_name(item.name, item)
        if item.name_normalized == original_name:
            item.name_normalized = cleaned
        assert item.name_normalized == "Custom Name"
