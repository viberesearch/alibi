"""Tests for alibi.masking module.

Comprehensive tests covering tier-based disclosure policies, cloud AI masking,
and unmask reversal. Tests edge cases: None values, empty records, missing fields.
"""

from datetime import date
from decimal import Decimal

import pytest

from alibi.db.models import DisplayType, Tier
from alibi.masking import DisclosurePolicy, MaskingService, get_policy


# ---------------------------------------------------------------------------
# DisclosurePolicy tests
# ---------------------------------------------------------------------------


class TestDisclosurePolicyT0:
    """T0: Fully masked - most restrictive tier."""

    def setup_method(self):
        self.policy = get_policy(Tier.T0)

    def test_amount_hidden(self):
        """T0 hides all amounts."""
        assert self.policy.mask_amount(Decimal("47.50")) is None

    def test_amount_none_passthrough(self):
        """None amount stays None."""
        assert self.policy.mask_amount(None) is None

    def test_vendor_replaced_with_category(self):
        """T0 replaces vendor with category."""
        assert self.policy.mask_vendor("Lidl", "groceries") == "groceries"

    def test_vendor_no_category_shows_unknown(self):
        """T0 shows 'Unknown' when no category available."""
        assert self.policy.mask_vendor("Lidl", None) == "Unknown"

    def test_vendor_none_with_category(self):
        """T0 shows category even when vendor is None."""
        assert self.policy.mask_vendor(None, "groceries") == "groceries"

    def test_date_to_month_only(self):
        """T0 reduces dates to month-year string."""
        result = self.policy.mask_date(date(2024, 3, 15))
        assert result == "2024-03"
        assert isinstance(result, str)

    def test_date_none_passthrough(self):
        """None date stays None."""
        assert self.policy.mask_date(None) is None

    def test_no_line_items(self):
        """T0 excludes line items."""
        assert self.policy.should_include_line_items() is False

    def test_no_provenance(self):
        """T0 excludes provenance."""
        assert self.policy.should_include_provenance() is False

    def test_display_types(self):
        """T0 display types are hidden/masked."""
        assert self.policy.amount_display_type() == DisplayType.HIDDEN
        assert self.policy.vendor_display_type() == DisplayType.MASKED
        assert self.policy.date_display_type() == DisplayType.MASKED


class TestDisclosurePolicyT1:
    """T1: Rounded amounts, vendor categories, approximate dates."""

    def setup_method(self):
        self.policy = get_policy(Tier.T1)

    def test_amount_rounded_up_to_10(self):
        """T1 rounds amounts up to nearest 10."""
        assert self.policy.mask_amount(Decimal("47.50")) == Decimal("50")

    def test_amount_exact_multiple_of_10(self):
        """T1 keeps exact multiples of 10."""
        assert self.policy.mask_amount(Decimal("50.00")) == Decimal("50")

    def test_amount_small_value(self):
        """T1 rounds small amounts."""
        assert self.policy.mask_amount(Decimal("3.20")) == Decimal("10")

    def test_amount_large_value(self):
        """T1 rounds large amounts."""
        assert self.policy.mask_amount(Decimal("1234.56")) == Decimal("1240")

    def test_amount_none(self):
        """None amount stays None at T1."""
        assert self.policy.mask_amount(None) is None

    def test_vendor_shows_category(self):
        """T1 shows category instead of vendor name."""
        assert self.policy.mask_vendor("Amazon", "electronics") == "electronics"

    def test_vendor_no_category(self):
        """T1 shows Unknown when no category."""
        assert self.policy.mask_vendor("Amazon", None) == "Unknown"

    def test_date_first_of_month(self):
        """T1 reduces dates to first of month."""
        result = self.policy.mask_date(date(2024, 6, 28))
        assert result == date(2024, 6, 1)
        assert isinstance(result, date)

    def test_no_line_items(self):
        """T1 excludes line items."""
        assert self.policy.should_include_line_items() is False

    def test_no_provenance(self):
        """T1 excludes provenance."""
        assert self.policy.should_include_provenance() is False

    def test_display_types(self):
        """T1 display types are rounded/masked."""
        assert self.policy.amount_display_type() == DisplayType.ROUNDED
        assert self.policy.vendor_display_type() == DisplayType.MASKED
        assert self.policy.date_display_type() == DisplayType.ROUNDED


class TestDisclosurePolicyT2:
    """T2: Exact amounts, vendor names visible, full dates."""

    def setup_method(self):
        self.policy = get_policy(Tier.T2)

    def test_amount_exact(self):
        """T2 shows exact amounts."""
        assert self.policy.mask_amount(Decimal("47.50")) == Decimal("47.50")

    def test_vendor_shows_name(self):
        """T2 shows actual vendor name."""
        assert self.policy.mask_vendor("Lidl", "groceries") == "Lidl"

    def test_vendor_none_falls_back_to_category(self):
        """T2 falls back to category when vendor is None."""
        assert self.policy.mask_vendor(None, "groceries") == "groceries"

    def test_vendor_none_no_category(self):
        """T2 shows Unknown when both vendor and category are None."""
        assert self.policy.mask_vendor(None, None) == "Unknown"

    def test_date_exact(self):
        """T2 shows exact dates."""
        d = date(2024, 3, 15)
        assert self.policy.mask_date(d) == d

    def test_no_line_items(self):
        """T2 excludes line items."""
        assert self.policy.should_include_line_items() is False

    def test_no_provenance(self):
        """T2 excludes provenance."""
        assert self.policy.should_include_provenance() is False

    def test_display_types(self):
        """T2 display types are all exact."""
        assert self.policy.amount_display_type() == DisplayType.EXACT
        assert self.policy.vendor_display_type() == DisplayType.EXACT
        assert self.policy.date_display_type() == DisplayType.EXACT


class TestDisclosurePolicyT3:
    """T3: Full details including line items."""

    def setup_method(self):
        self.policy = get_policy(Tier.T3)

    def test_amount_exact(self):
        """T3 shows exact amounts."""
        assert self.policy.mask_amount(Decimal("99.99")) == Decimal("99.99")

    def test_vendor_shows_name(self):
        """T3 shows actual vendor name."""
        assert self.policy.mask_vendor("Apple Store", "electronics") == "Apple Store"

    def test_date_exact(self):
        """T3 shows exact dates."""
        d = date(2024, 12, 25)
        assert self.policy.mask_date(d) == d

    def test_includes_line_items(self):
        """T3 includes line items."""
        assert self.policy.should_include_line_items() is True

    def test_no_provenance(self):
        """T3 still excludes provenance."""
        assert self.policy.should_include_provenance() is False


class TestDisclosurePolicyT4:
    """T4: Full details + provenance + raw extraction data."""

    def setup_method(self):
        self.policy = get_policy(Tier.T4)

    def test_amount_exact(self):
        """T4 shows exact amounts."""
        assert self.policy.mask_amount(Decimal("12345.67")) == Decimal("12345.67")

    def test_vendor_shows_name(self):
        """T4 shows actual vendor name."""
        assert self.policy.mask_vendor("REWE", "groceries") == "REWE"

    def test_date_exact(self):
        """T4 shows exact dates."""
        d = date(2024, 1, 1)
        assert self.policy.mask_date(d) == d

    def test_includes_line_items(self):
        """T4 includes line items."""
        assert self.policy.should_include_line_items() is True

    def test_includes_provenance(self):
        """T4 includes provenance."""
        assert self.policy.should_include_provenance() is True

    def test_display_types(self):
        """T4 display types are all exact."""
        assert self.policy.amount_display_type() == DisplayType.EXACT
        assert self.policy.vendor_display_type() == DisplayType.EXACT
        assert self.policy.date_display_type() == DisplayType.EXACT


class TestGetPolicy:
    """Tests for the get_policy factory function."""

    def test_returns_correct_tier(self):
        """get_policy returns policy with matching tier."""
        for tier in Tier:
            policy = get_policy(tier)
            assert policy.tier == tier

    def test_policy_is_frozen(self):
        """DisclosurePolicy is immutable."""
        policy = get_policy(Tier.T0)
        with pytest.raises(AttributeError):
            policy.tier = Tier.T4  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MaskingService.mask_for_tier tests
# ---------------------------------------------------------------------------


class TestMaskForTier:
    """Tests for MaskingService.mask_for_tier."""

    def setup_method(self):
        self.service = MaskingService()
        self.sample_record = {
            "id": "txn-001",
            "vendor": "Lidl",
            "category": "groceries",
            "amount": Decimal("47.50"),
            "transaction_date": date(2024, 3, 15),
            "card_last4": "1234",
            "payment_method": "Visa Debit",
            "account_reference": "DE89370400440532013000",
            "line_items": [
                {"name": "Milk", "amount": Decimal("1.29")},
                {"name": "Bread", "amount": Decimal("2.49")},
            ],
            "provenance": {"source": "ocr", "processor": "ollama:qwen3-vl"},
            "extracted_data": {"raw": "some raw text"},
            "raw_text": "receipt text here",
        }

    def test_t0_masks_everything(self):
        """T0 masks amounts, vendors, dates, strips line items and provenance."""
        result = self.service.mask_for_tier([self.sample_record], Tier.T0)
        assert len(result) == 1
        r = result[0]
        assert r["amount"] is None
        assert r["vendor"] == "groceries"
        assert r["transaction_date"] == "2024-03"
        assert r["card_last4"] is None
        assert "line_items" not in r
        assert "provenance" not in r
        assert "extracted_data" not in r
        assert "raw_text" not in r
        assert "payment_method" not in r
        assert "account_reference" not in r

    def test_t1_rounds_and_masks(self):
        """T1 rounds amounts, shows category, approximate dates."""
        result = self.service.mask_for_tier([self.sample_record], Tier.T1)
        r = result[0]
        assert r["amount"] == Decimal("50")
        assert r["vendor"] == "groceries"
        assert r["transaction_date"] == date(2024, 3, 1)
        assert r["card_last4"] is None
        assert "line_items" not in r
        assert "provenance" not in r

    def test_t2_shows_exact_basics(self):
        """T2 shows exact amounts, vendor names, full dates."""
        result = self.service.mask_for_tier([self.sample_record], Tier.T2)
        r = result[0]
        assert r["amount"] == Decimal("47.50")
        assert r["vendor"] == "Lidl"
        assert r["transaction_date"] == date(2024, 3, 15)
        assert r["card_last4"] is None  # still masked below T4
        assert "line_items" not in r
        assert "provenance" not in r

    def test_t3_includes_line_items(self):
        """T3 includes line items but not provenance."""
        result = self.service.mask_for_tier([self.sample_record], Tier.T3)
        r = result[0]
        assert r["amount"] == Decimal("47.50")
        assert r["vendor"] == "Lidl"
        assert "line_items" in r
        assert len(r["line_items"]) == 2
        assert "provenance" not in r
        assert r["card_last4"] is None

    def test_t4_shows_everything(self):
        """T4 shows all details including provenance and card."""
        result = self.service.mask_for_tier([self.sample_record], Tier.T4)
        r = result[0]
        assert r["amount"] == Decimal("47.50")
        assert r["vendor"] == "Lidl"
        assert r["transaction_date"] == date(2024, 3, 15)
        assert r["card_last4"] == "1234"
        assert "line_items" in r
        assert "provenance" in r
        assert "extracted_data" in r
        assert "raw_text" in r

    def test_empty_records(self):
        """Empty record list returns empty list."""
        result = self.service.mask_for_tier([], Tier.T0)
        assert result == []

    def test_record_with_missing_fields(self):
        """Records with missing fields are handled gracefully."""
        minimal = {"id": "txn-002"}
        result = self.service.mask_for_tier([minimal], Tier.T0)
        assert len(result) == 1
        assert result[0]["id"] == "txn-002"

    def test_original_not_modified(self):
        """Masking does not mutate the original records."""
        original_amount = self.sample_record["amount"]
        original_vendor = self.sample_record["vendor"]
        self.service.mask_for_tier([self.sample_record], Tier.T0)
        assert self.sample_record["amount"] == original_amount
        assert self.sample_record["vendor"] == original_vendor

    def test_multiple_records(self):
        """Multiple records are each masked independently."""
        records = [
            {"vendor": "Lidl", "category": "groceries", "amount": Decimal("20")},
            {"vendor": "Amazon", "category": "electronics", "amount": Decimal("150")},
        ]
        result = self.service.mask_for_tier(records, Tier.T1)
        assert len(result) == 2
        assert result[0]["vendor"] == "groceries"
        assert result[1]["vendor"] == "electronics"
        assert result[0]["amount"] == Decimal("20")
        assert result[1]["amount"] == Decimal("150")

    def test_date_as_string(self):
        """Date fields stored as ISO strings are parsed and masked."""
        record = {"transaction_date": "2024-07-20"}
        result = self.service.mask_for_tier([record], Tier.T0)
        assert result[0]["transaction_date"] == "2024-07"

    def test_none_amount_at_all_tiers(self):
        """None amounts remain None at all tiers."""
        record = {"amount": None}
        for tier in Tier:
            result = self.service.mask_for_tier([record], tier)
            assert result[0]["amount"] is None


# ---------------------------------------------------------------------------
# MaskingService.mask_for_cloud_ai tests
# ---------------------------------------------------------------------------


class TestMaskForCloudAI:
    """Tests for MaskingService.mask_for_cloud_ai."""

    def setup_method(self):
        self.service = MaskingService()

    def test_vendors_replaced(self):
        """Vendor names are replaced with Merchant_A, Merchant_B, etc."""
        records = [
            {"vendor": "Lidl"},
            {"vendor": "REWE"},
            {"vendor": "Lidl"},  # Same vendor reuses same placeholder
        ]
        masked, masking_map = self.service.mask_for_cloud_ai(records)
        assert masked[0]["vendor"] == "Merchant_A"
        assert masked[1]["vendor"] == "Merchant_B"
        assert masked[2]["vendor"] == "Merchant_A"  # Same as first
        assert masking_map["vendors"]["Lidl"] == "Merchant_A"
        assert masking_map["vendors"]["REWE"] == "Merchant_B"

    def test_card_numbers_masked(self):
        """Card last4 digits are replaced with XXXX."""
        records = [{"card_last4": "1234"}]
        masked, masking_map = self.service.mask_for_cloud_ai(records)
        assert masked[0]["card_last4"] == "XXXX"
        assert masking_map["cards"]["1234"] == "XXXX"

    def test_amounts_rounded_to_ranges(self):
        """Amounts are rounded to range strings."""
        records = [{"amount": Decimal("47.50")}]
        masked, masking_map = self.service.mask_for_cloud_ai(records)
        assert isinstance(masked[0]["amount"], str)
        assert "-" in masked[0]["amount"]  # Range format like "40-50"
        assert masking_map["amounts"]["0"] == "47.50"

    def test_descriptions_masked(self):
        """Description fields are replaced with Person_A, etc."""
        records = [{"description": "Payment from John Smith"}]
        masked, masking_map = self.service.mask_for_cloud_ai(records)
        assert masked[0]["description"] == "Person_A"
        assert "Payment from John Smith" in masking_map["persons"]

    def test_payment_method_simplified(self):
        """Payment method is simplified to type only."""
        records = [
            {"payment_method": "Visa Debit Card"},
            {"payment_method": "Cash EUR"},
            {"payment_method": "Bank Transfer DE"},
            {"payment_method": "PayPal"},
        ]
        masked, _ = self.service.mask_for_cloud_ai(records)
        assert masked[0]["payment_method"] == "card"
        assert masked[1]["payment_method"] == "cash"
        assert masked[2]["payment_method"] == "transfer"
        assert masked[3]["payment_method"] == "other"

    def test_account_reference_masked(self):
        """Account references are replaced with generic placeholder."""
        records = [{"account_reference": "DE89370400440532013000"}]
        masked, _ = self.service.mask_for_cloud_ai(records)
        assert masked[0]["account_reference"] == "ACCT_MASKED"

    def test_empty_records(self):
        """Empty record list returns empty list and empty map."""
        masked, masking_map = self.service.mask_for_cloud_ai([])
        assert masked == []
        assert masking_map["vendors"] == {}
        assert masking_map["amounts"] == {}

    def test_none_vendor_not_masked(self):
        """None vendor is left as None."""
        records = [{"vendor": None}]
        masked, masking_map = self.service.mask_for_cloud_ai(records)
        assert masked[0]["vendor"] is None
        assert len(masking_map["vendors"]) == 0

    def test_original_not_modified(self):
        """Cloud AI masking does not mutate originals."""
        records = [{"vendor": "Lidl", "amount": Decimal("50")}]
        self.service.mask_for_cloud_ai(records)
        assert records[0]["vendor"] == "Lidl"
        assert records[0]["amount"] == Decimal("50")

    def test_amount_range_small(self):
        """Small amounts get 0-10 range."""
        records = [{"amount": Decimal("3.50")}]
        masked, _ = self.service.mask_for_cloud_ai(records)
        assert masked[0]["amount"] == "0-10"

    def test_amount_range_medium(self):
        """Medium amounts get appropriate 10-unit range."""
        records = [{"amount": Decimal("47.50")}]
        masked, _ = self.service.mask_for_cloud_ai(records)
        assert masked[0]["amount"] == "40-50"

    def test_amount_range_hundreds(self):
        """Amounts in hundreds get 100-unit range."""
        records = [{"amount": Decimal("247.99")}]
        masked, _ = self.service.mask_for_cloud_ai(records)
        assert masked[0]["amount"] == "200-300"

    def test_amount_range_thousands(self):
        """Amounts in thousands get 1000-unit range."""
        records = [{"amount": Decimal("1500")}]
        masked, _ = self.service.mask_for_cloud_ai(records)
        assert masked[0]["amount"] == "1000-2000"


# ---------------------------------------------------------------------------
# MaskingService.unmask tests
# ---------------------------------------------------------------------------


class TestUnmask:
    """Tests for MaskingService.unmask reversal."""

    def setup_method(self):
        self.service = MaskingService()

    def test_roundtrip_vendors(self):
        """Vendor names survive mask -> unmask roundtrip."""
        records = [{"vendor": "Lidl"}, {"vendor": "REWE"}]
        masked, masking_map = self.service.mask_for_cloud_ai(records)
        unmasked = self.service.unmask(masked, masking_map)
        assert unmasked[0]["vendor"] == "Lidl"
        assert unmasked[1]["vendor"] == "REWE"

    def test_roundtrip_amounts(self):
        """Exact amounts are restored after unmask."""
        records = [
            {"amount": Decimal("47.50")},
            {"amount": Decimal("1234.56")},
        ]
        masked, masking_map = self.service.mask_for_cloud_ai(records)
        unmasked = self.service.unmask(masked, masking_map)
        assert unmasked[0]["amount"] == Decimal("47.50")
        assert unmasked[1]["amount"] == Decimal("1234.56")

    def test_roundtrip_descriptions(self):
        """Descriptions survive roundtrip."""
        records = [{"description": "Payment from John Smith"}]
        masked, masking_map = self.service.mask_for_cloud_ai(records)
        unmasked = self.service.unmask(masked, masking_map)
        assert unmasked[0]["description"] == "Payment from John Smith"

    def test_roundtrip_card_numbers(self):
        """Card numbers survive roundtrip."""
        records = [{"card_last4": "5678"}]
        masked, masking_map = self.service.mask_for_cloud_ai(records)
        unmasked = self.service.unmask(masked, masking_map)
        assert unmasked[0]["card_last4"] == "5678"

    def test_unmask_empty(self):
        """Unmasking empty records returns empty list."""
        result = self.service.unmask(
            [], {"vendors": {}, "persons": {}, "cards": {}, "amounts": {}}
        )
        assert result == []

    def test_unmask_preserves_unmasked_fields(self):
        """Fields not in the masking map are preserved as-is."""
        records = [{"vendor": "Lidl", "category": "groceries", "id": "txn-001"}]
        masked, masking_map = self.service.mask_for_cloud_ai(records)
        # AI might add new fields
        masked[0]["ai_category"] = "food"
        unmasked = self.service.unmask(masked, masking_map)
        assert unmasked[0]["vendor"] == "Lidl"
        assert unmasked[0]["category"] == "groceries"
        assert unmasked[0]["id"] == "txn-001"
        assert unmasked[0]["ai_category"] == "food"

    def test_roundtrip_full_record(self):
        """Full record survives mask -> unmask roundtrip."""
        records = [
            {
                "id": "txn-001",
                "vendor": "Lidl",
                "category": "groceries",
                "amount": Decimal("47.50"),
                "card_last4": "1234",
                "description": "Weekly shopping",
                "transaction_date": date(2024, 3, 15),
            }
        ]
        masked, masking_map = self.service.mask_for_cloud_ai(records)
        unmasked = self.service.unmask(masked, masking_map)
        assert unmasked[0]["vendor"] == "Lidl"
        assert unmasked[0]["amount"] == Decimal("47.50")
        assert unmasked[0]["card_last4"] == "1234"
        assert unmasked[0]["description"] == "Weekly shopping"
        # Non-masked fields preserved
        assert unmasked[0]["id"] == "txn-001"
        assert unmasked[0]["category"] == "groceries"
        assert unmasked[0]["transaction_date"] == date(2024, 3, 15)
