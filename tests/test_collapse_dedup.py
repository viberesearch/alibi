"""Tests for item deduplication and non-product filtering in collapse."""

from decimal import Decimal
from uuid import uuid4

from alibi.clouds.collapse import (
    _deduplicate_items,
    _filter_non_product_items,
    _is_non_product_item,
    _item_metadata_score,
)
from alibi.db.models import FactItem, UnitType


def _make_item(
    name: str = "Test Item",
    total_price: Decimal | None = Decimal("2.99"),
    quantity: Decimal = Decimal("1"),
    barcode: str | None = None,
    brand: str | None = None,
    category: str | None = None,
) -> FactItem:
    """Helper to create a FactItem for testing."""
    item = FactItem(
        id=str(uuid4()),
        fact_id="",
        atom_id=str(uuid4()),
        name=name,
    )
    item.total_price = total_price
    item.quantity = quantity
    item.barcode = barcode
    item.brand = brand
    item.category = category
    return item


# ---------------------------------------------------------------------------
# Non-product item detection
# ---------------------------------------------------------------------------


class TestIsNonProductItem:
    def test_price_change_annotation(self):
        item = _make_item(name="FROM 3.05 TO 2.69 -")
        assert _is_non_product_item(item) is True

    def test_percentage_off(self):
        item = _make_item(name="20% OFF -", total_price=Decimal("-0.50"))
        assert _is_non_product_item(item) is True

    def test_qty_price_metadata(self):
        item = _make_item(name="2 ea 3.05")
        assert _is_non_product_item(item) is True

    def test_discount_negative_price(self):
        item = _make_item(name="DISCOUNT", total_price=Decimal("-1.00"))
        assert _is_non_product_item(item) is True

    def test_coupon_zero_price(self):
        item = _make_item(name="MEMBER COUPON", total_price=Decimal("0"))
        assert _is_non_product_item(item) is True

    def test_discount_positive_price_kept(self):
        """A product with 'discount' in name but positive price is kept."""
        item = _make_item(name="DISCOUNT STORE PRODUCT", total_price=Decimal("5.00"))
        assert _is_non_product_item(item) is False

    def test_vat_line(self):
        item = _make_item(name="VAT2 5.00%", total_price=Decimal("0.95"))
        assert _is_non_product_item(item) is True

    def test_subtotal_line(self):
        item = _make_item(name="Subtotal", total_price=Decimal("19.02"))
        assert _is_non_product_item(item) is True

    def test_greek_total_line(self):
        item = _make_item(name="ΣYNOAO", total_price=Decimal("41.28"))
        assert _is_non_product_item(item) is True

    def test_normal_product(self):
        item = _make_item(name="MILK FRESH 1L")
        assert _is_non_product_item(item) is False

    def test_taxino_salad_not_filtered(self):
        """Greek salad product starting with TAX should not be filtered."""
        item = _make_item(name="TAXINOΣΛΛΑΤA MAΓΟΥΛΙΤΣΑ", total_price=Decimal("2.52"))
        assert _is_non_product_item(item) is False

    def test_empty_name(self):
        item = _make_item(name="")
        assert _is_non_product_item(item) is True


class TestFilterNonProductItems:
    def test_filters_non_products(self):
        items = [
            _make_item(name="MILK 1L", total_price=Decimal("2.50")),
            _make_item(name="20% OFF -", total_price=Decimal("-0.50")),
            _make_item(name="BREAD", total_price=Decimal("1.20")),
            _make_item(name="FROM 3.05 TO 2.69 -"),
        ]
        result = _filter_non_product_items(items)
        assert len(result) == 2
        names = {i.name for i in result}
        assert names == {"MILK 1L", "BREAD"}


# ---------------------------------------------------------------------------
# Item metadata score
# ---------------------------------------------------------------------------


class TestItemMetadataScore:
    def test_empty_item(self):
        item = _make_item()
        assert _item_metadata_score(item) == 0

    def test_full_metadata(self):
        item = _make_item(barcode="5901234", brand="Alphamega", category="Dairy")
        item.unit_quantity = Decimal("1.5")
        item.comparable_unit_price = Decimal("3.00")
        assert _item_metadata_score(item) == 5

    def test_partial_metadata(self):
        item = _make_item(brand="TestBrand")
        assert _item_metadata_score(item) == 1


# ---------------------------------------------------------------------------
# Item deduplication
# ---------------------------------------------------------------------------


class TestDeduplicateItems:
    def test_no_duplicates(self):
        items = [
            _make_item(name="MILK", total_price=Decimal("2.50")),
            _make_item(name="BREAD", total_price=Decimal("1.20")),
        ]
        result = _deduplicate_items(items)
        assert len(result) == 2

    def test_exact_duplicates_deduped(self):
        items = [
            _make_item(name="MILK", total_price=Decimal("2.50")),
            _make_item(name="MILK", total_price=Decimal("2.50")),
        ]
        result = _deduplicate_items(items)
        assert len(result) == 1

    def test_keeps_item_with_more_metadata(self):
        plain = _make_item(name="MILK", total_price=Decimal("2.50"))
        enriched = _make_item(
            name="MILK",
            total_price=Decimal("2.50"),
            barcode="5901234",
            brand="Fresh",
            category="Dairy",
        )
        items = [plain, enriched]
        result = _deduplicate_items(items)
        assert len(result) == 1
        assert result[0].barcode == "5901234"

    def test_case_insensitive(self):
        items = [
            _make_item(name="Milk Fresh 1L", total_price=Decimal("2.50")),
            _make_item(name="milk fresh 1l", total_price=Decimal("2.50")),
        ]
        result = _deduplicate_items(items)
        assert len(result) == 1

    def test_different_prices_not_deduped(self):
        items = [
            _make_item(name="MILK", total_price=Decimal("2.50")),
            _make_item(name="MILK", total_price=Decimal("3.00")),
        ]
        result = _deduplicate_items(items)
        assert len(result) == 2

    def test_different_quantities_not_deduped(self):
        items = [
            _make_item(name="MILK", total_price=Decimal("5.00"), quantity=Decimal("1")),
            _make_item(name="MILK", total_price=Decimal("5.00"), quantity=Decimal("2")),
        ]
        result = _deduplicate_items(items)
        assert len(result) == 2

    def test_triple_duplicates(self):
        items = [
            _make_item(name="EGGS", total_price=Decimal("3.50")),
            _make_item(name="EGGS", total_price=Decimal("3.50"), brand="Farm"),
            _make_item(name="EGGS", total_price=Decimal("3.50")),
        ]
        result = _deduplicate_items(items)
        assert len(result) == 1
        assert result[0].brand == "Farm"

    def test_empty_list(self):
        assert _deduplicate_items([]) == []
