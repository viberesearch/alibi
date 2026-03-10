"""Tests for privacy-preserving fact anonymization."""

import json
from datetime import date
from decimal import Decimal

import pytest

from alibi.anonymization.exporter import (
    AnonymizationKey,
    AnonymizationLevel,
    anonymize_export,
    generate_key,
    restore_import,
    _pseudonymize_vendor,
    _pseudonymize_item,
    _shift_amount,
    _shift_date,
)


@pytest.fixture
def key():
    """Fixed anonymization key for deterministic tests."""
    return AnonymizationKey(
        secret="test_secret_hex",
        amount_factor=1.5,
        date_offset_days=30,
    )


@pytest.fixture
def sample_facts():
    """Sample facts for anonymization tests."""
    return [
        {
            "id": "fact-001",
            "fact_type": "purchase",
            "vendor": "PAPAS HYPERMARKET",
            "total_amount": 85.69,
            "currency": "EUR",
            "event_date": "2026-01-21",
            "status": "confirmed",
            "payments": json.dumps(
                [{"method": "card", "card_last4": "7201", "amount": "85.69"}]
            ),
        },
        {
            "id": "fact-002",
            "fact_type": "purchase",
            "vendor": "LIDL",
            "total_amount": 32.50,
            "currency": "EUR",
            "event_date": "2026-01-22",
            "status": "confirmed",
            "payments": None,
        },
        {
            "id": "fact-003",
            "fact_type": "purchase",
            "vendor": "PAPAS HYPERMARKET",
            "total_amount": 45.00,
            "currency": "EUR",
            "event_date": "2026-01-28",
            "status": "confirmed",
            "payments": None,
        },
    ]


@pytest.fixture
def sample_items():
    """Sample items keyed by fact_id."""
    return {
        "fact-001": [
            {
                "name": "Red Bull 250ml",
                "category": "beverages",
                "quantity": 3,
                "unit": "pcs",
                "unit_price": 1.99,
                "total_price": 5.97,
            },
            {
                "name": "Organic Bananas",
                "category": "produce",
                "quantity": 1,
                "unit": "kg",
                "unit_price": 2.49,
                "total_price": 2.49,
            },
        ],
        "fact-002": [
            {
                "name": "Whole Wheat Bread",
                "category": "bakery",
                "quantity": 1,
                "unit": "pcs",
                "unit_price": 1.89,
                "total_price": 1.89,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Key tests
# ---------------------------------------------------------------------------


class TestAnonymizationKey:
    def test_generate_key(self):
        k = generate_key()
        assert k.secret
        assert 0.5 <= k.amount_factor < 2.0
        assert -90 <= k.date_offset_days <= 90

    def test_serialization_roundtrip(self, key):
        key.vendor_map = {"PAPAS": "VENDOR_A"}
        key.item_map = {"Red Bull": "beverages_1"}
        json_str = key.to_json()
        restored = AnonymizationKey.from_json(json_str)
        assert restored.secret == key.secret
        assert restored.amount_factor == key.amount_factor
        assert restored.date_offset_days == key.date_offset_days
        assert restored.vendor_map == key.vendor_map
        assert restored.item_map == key.item_map

    def test_two_keys_differ(self):
        k1 = generate_key()
        k2 = generate_key()
        assert k1.secret != k2.secret


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestPseudonymizeVendor:
    def test_consistent_mapping(self, key):
        p1 = _pseudonymize_vendor("PAPAS", key)
        p2 = _pseudonymize_vendor("PAPAS", key)
        assert p1 == p2  # Same vendor always gets same pseudonym

    def test_different_vendors_different_pseudonyms(self, key):
        p1 = _pseudonymize_vendor("PAPAS", key)
        p2 = _pseudonymize_vendor("LIDL", key)
        assert p1 != p2

    def test_pseudonym_format(self, key):
        p = _pseudonymize_vendor("TEST", key)
        assert p.startswith("VENDOR_")


class TestPseudonymizeItem:
    def test_uses_category(self, key):
        p = _pseudonymize_item("Red Bull 250ml", "beverages", key)
        assert p.startswith("beverages_")

    def test_consistent(self, key):
        p1 = _pseudonymize_item("Red Bull 250ml", "beverages", key)
        p2 = _pseudonymize_item("Red Bull 250ml", "beverages", key)
        assert p1 == p2

    def test_no_category_fallback(self, key):
        p = _pseudonymize_item("Unknown Item", None, key)
        assert p.startswith("item_")


class TestShiftAmount:
    def test_shifts_by_factor(self):
        result = _shift_amount(100.00, 1.5)
        assert result == "150.00"

    def test_preserves_ratios(self):
        s1 = _shift_amount(100.00, 1.5)
        s2 = _shift_amount(200.00, 1.5)
        assert s1 is not None
        assert s2 is not None
        r1 = Decimal(s1)
        r2 = Decimal(s2)
        assert r2 / r1 == Decimal("2")

    def test_none_returns_none(self):
        assert _shift_amount(None, 1.5) is None


class TestShiftDate:
    def test_shifts_forward(self):
        result = _shift_date("2026-01-15", 30)
        assert result == "2026-02-14"

    def test_shifts_backward(self):
        result = _shift_date("2026-01-15", -10)
        assert result == "2026-01-05"

    def test_preserves_intervals(self):
        sd1 = _shift_date("2026-01-01", 30)
        sd2 = _shift_date("2026-01-08", 30)
        assert sd1 is not None
        assert sd2 is not None
        d1 = date.fromisoformat(sd1)
        d2 = date.fromisoformat(sd2)
        assert (d2 - d1).days == 7  # Weekly interval preserved

    def test_none_returns_none(self):
        assert _shift_date(None, 30) is None

    def test_date_object_input(self):
        result = _shift_date(date(2026, 1, 15), 5)
        assert result == "2026-01-20"


# ---------------------------------------------------------------------------
# Pseudonymized export tests
# ---------------------------------------------------------------------------


class TestPseudonymizedExport:
    def test_vendors_pseudonymized(self, sample_facts, sample_items, key):
        result, _ = anonymize_export(
            sample_facts, sample_items, AnonymizationLevel.PSEUDONYMIZED, key
        )
        vendors = [r["vendor"] for r in result]
        assert "PAPAS HYPERMARKET" not in vendors
        assert all(v.startswith("VENDOR_") for v in vendors)

    def test_same_vendor_same_pseudonym(self, sample_facts, sample_items, key):
        result, _ = anonymize_export(
            sample_facts, sample_items, AnonymizationLevel.PSEUDONYMIZED, key
        )
        # fact-001 and fact-003 are both PAPAS HYPERMARKET
        assert result[0]["vendor"] == result[2]["vendor"]

    def test_amounts_shifted(self, sample_facts, sample_items, key):
        result, _ = anonymize_export(
            sample_facts, sample_items, AnonymizationLevel.PSEUDONYMIZED, key
        )
        # 85.69 * 1.5 = 128.535 -> 128.54
        assert Decimal(result[0]["total_amount"]) == Decimal("128.54")

    def test_dates_shifted(self, sample_facts, sample_items, key):
        result, _ = anonymize_export(
            sample_facts, sample_items, AnonymizationLevel.PSEUDONYMIZED, key
        )
        # 2026-01-21 + 30 days = 2026-02-20
        assert result[0]["event_date"] == "2026-02-20"

    def test_items_pseudonymized(self, sample_facts, sample_items, key):
        result, _ = anonymize_export(
            sample_facts, sample_items, AnonymizationLevel.PSEUDONYMIZED, key
        )
        items = result[0]["items"]
        assert len(items) == 2
        assert "Red Bull" not in items[0]["name"]
        assert items[0]["category"] == "beverages"  # Category preserved

    def test_item_prices_shifted(self, sample_facts, sample_items, key):
        result, _ = anonymize_export(
            sample_facts, sample_items, AnonymizationLevel.PSEUDONYMIZED, key
        )
        items = result[0]["items"]
        # 1.99 * 1.5 = 2.985 -> 2.98 (banker's rounding)
        assert Decimal(items[0]["unit_price"]) == Decimal("2.98")

    def test_payments_anonymized(self, sample_facts, sample_items, key):
        result, _ = anonymize_export(
            sample_facts, sample_items, AnonymizationLevel.PSEUDONYMIZED, key
        )
        payments = result[0]["payments"]
        assert len(payments) == 1
        assert payments[0]["method"] == "card"
        assert "card_last4" not in payments[0]  # Card number stripped
        # Amount shifted
        assert Decimal(payments[0]["amount"]) == Decimal("128.54")

    def test_no_original_ids(self, sample_facts, sample_items, key):
        result, _ = anonymize_export(
            sample_facts, sample_items, AnonymizationLevel.PSEUDONYMIZED, key
        )
        for r in result:
            assert "id" not in r


# ---------------------------------------------------------------------------
# Categories-only export tests
# ---------------------------------------------------------------------------


class TestCategoriesOnlyExport:
    def test_no_amounts(self, sample_facts, sample_items, key):
        result, _ = anonymize_export(
            sample_facts, sample_items, AnonymizationLevel.CATEGORIES_ONLY, key
        )
        for r in result:
            assert "total_amount" not in r

    def test_no_exact_dates(self, sample_facts, sample_items, key):
        result, _ = anonymize_export(
            sample_facts, sample_items, AnonymizationLevel.CATEGORIES_ONLY, key
        )
        for r in result:
            assert "event_date" not in r
            # Only month granularity
            if r.get("event_month"):
                assert len(r["event_month"]) == 7  # YYYY-MM format

    def test_vendors_pseudonymized(self, sample_facts, sample_items, key):
        result, _ = anonymize_export(
            sample_facts, sample_items, AnonymizationLevel.CATEGORIES_ONLY, key
        )
        for r in result:
            if r["vendor"]:
                assert r["vendor"].startswith("VENDOR_")

    def test_item_categories_only(self, sample_facts, sample_items, key):
        result, _ = anonymize_export(
            sample_facts, sample_items, AnonymizationLevel.CATEGORIES_ONLY, key
        )
        # fact-001 has 2 items
        assert result[0]["item_count"] == 2
        assert "beverages" in result[0]["item_categories"]
        assert "produce" in result[0]["item_categories"]


# ---------------------------------------------------------------------------
# Statistical export tests
# ---------------------------------------------------------------------------


class TestStatisticalExport:
    def test_no_individual_records(self, sample_facts, sample_items):
        result, _ = anonymize_export(
            sample_facts, sample_items, AnonymizationLevel.STATISTICAL
        )
        # Should be aggregates, not individual records
        for r in result:
            assert r.get("summary") == "aggregate"
            assert "vendor" not in r

    def test_aggregate_stats(self, sample_facts, sample_items):
        result, _ = anonymize_export(
            sample_facts, sample_items, AnonymizationLevel.STATISTICAL
        )
        purchase_agg = [r for r in result if r["fact_type"] == "purchase"][0]
        assert purchase_agg["count"] == 3
        assert "amount_min" in purchase_agg
        assert "amount_max" in purchase_agg
        assert "amount_mean" in purchase_agg
        assert purchase_agg["amount_min"] == 32.50
        assert purchase_agg["amount_max"] == 85.69

    def test_category_counts(self, sample_facts, sample_items):
        result, _ = anonymize_export(
            sample_facts, sample_items, AnonymizationLevel.STATISTICAL
        )
        purchase_agg = [r for r in result if r["fact_type"] == "purchase"][0]
        assert "top_categories" in purchase_agg
        assert purchase_agg["top_categories"]["beverages"] == 1

    def test_empty_facts(self):
        result, _ = anonymize_export([], {}, AnonymizationLevel.STATISTICAL)
        assert len(result) == 1
        assert result[0]["summary"] == "no_data"


# ---------------------------------------------------------------------------
# Roundtrip (anonymize + restore) tests
# ---------------------------------------------------------------------------


class TestRestoreImport:
    def test_roundtrip_vendors(self, sample_facts, sample_items, key):
        anon, key = anonymize_export(
            sample_facts, sample_items, AnonymizationLevel.PSEUDONYMIZED, key
        )
        restored = restore_import(anon, key)
        assert restored[0]["vendor"] == "PAPAS HYPERMARKET"
        assert restored[1]["vendor"] == "LIDL"

    def test_roundtrip_amounts(self, sample_facts, sample_items, key):
        anon, key = anonymize_export(
            sample_facts, sample_items, AnonymizationLevel.PSEUDONYMIZED, key
        )
        restored = restore_import(anon, key)
        # Should be close to original (rounding may cause tiny differences)
        original = Decimal("85.69")
        restored_amount = Decimal(restored[0]["total_amount"])
        assert abs(restored_amount - original) < Decimal("0.02")

    def test_roundtrip_dates(self, sample_facts, sample_items, key):
        anon, key = anonymize_export(
            sample_facts, sample_items, AnonymizationLevel.PSEUDONYMIZED, key
        )
        restored = restore_import(anon, key)
        assert restored[0]["event_date"] == "2026-01-21"

    def test_roundtrip_items(self, sample_facts, sample_items, key):
        anon, key = anonymize_export(
            sample_facts, sample_items, AnonymizationLevel.PSEUDONYMIZED, key
        )
        restored = restore_import(anon, key)
        items = restored[0]["items"]
        assert items[0]["name"] == "Red Bull 250ml"
        assert items[1]["name"] == "Organic Bananas"

    def test_roundtrip_item_prices(self, sample_facts, sample_items, key):
        anon, key = anonymize_export(
            sample_facts, sample_items, AnonymizationLevel.PSEUDONYMIZED, key
        )
        restored = restore_import(anon, key)
        items = restored[0]["items"]
        # 1.99 * 1.5 = 2.985 -> 2.99, then 2.99 / 1.5 = 1.993 -> 1.99
        assert abs(Decimal(items[0]["unit_price"]) - Decimal("1.99")) < Decimal("0.02")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_none_vendor(self, key):
        facts = [
            {
                "id": "f1",
                "fact_type": "purchase",
                "vendor": None,
                "total_amount": 10.0,
                "currency": "EUR",
                "event_date": "2026-01-01",
                "status": "confirmed",
                "payments": None,
            }
        ]
        result, _ = anonymize_export(facts, {}, AnonymizationLevel.PSEUDONYMIZED, key)
        assert result[0]["vendor"] is None

    def test_none_amount(self, key):
        facts = [
            {
                "id": "f1",
                "fact_type": "purchase",
                "vendor": "STORE",
                "total_amount": None,
                "currency": "EUR",
                "event_date": "2026-01-01",
                "status": "partial",
                "payments": None,
            }
        ]
        result, _ = anonymize_export(facts, {}, AnonymizationLevel.PSEUDONYMIZED, key)
        assert result[0]["total_amount"] is None

    def test_no_items(self, key):
        facts = [
            {
                "id": "f1",
                "fact_type": "purchase",
                "vendor": "STORE",
                "total_amount": 50.0,
                "currency": "EUR",
                "event_date": "2026-01-01",
                "status": "confirmed",
                "payments": None,
            }
        ]
        result, _ = anonymize_export(facts, {}, AnonymizationLevel.PSEUDONYMIZED, key)
        assert "items" not in result[0]

    def test_key_reuse_consistency(self, sample_facts, sample_items, key):
        """Using the same key on different exports gives consistent results."""
        r1, k1 = anonymize_export(
            sample_facts[:1], sample_items, AnonymizationLevel.PSEUDONYMIZED, key
        )
        r2, k2 = anonymize_export(
            sample_facts, sample_items, AnonymizationLevel.PSEUDONYMIZED, key
        )
        # Same vendor should get same pseudonym
        assert r1[0]["vendor"] == r2[0]["vendor"]
