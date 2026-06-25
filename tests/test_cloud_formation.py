"""Tests for cloud formation engine and fact collapse."""

from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from alibi.clouds.collapse import (
    CollapseResult,
    _amounts_sum_to,
    _extract_items,
    _extract_payments,
    _extract_total,
    _extract_vendor,
    infer_fact_type,
    try_collapse,
)
from alibi.clouds.formation import (
    BundleSummary,
    MatchResult,
    _amount_score,
    _date_score,
    _item_overlap_score,
    _score_match,
    _time_conflict,
    _vendor_score,
    add_bundle_to_cloud,
    create_cloud_for_bundle,
    extract_bundle_summary,
    find_cloud_for_bundle,
    normalize_vendor_name,
    vendors_match,
)
from alibi.db.models import (
    AtomType,
    BundleType,
    Cloud,
    CloudMatchType,
    CloudStatus,
    FactStatus,
    FactType,
    TaxType,
    UnitType,
)

# ---------------------------------------------------------------------------
# Vendor normalization
# ---------------------------------------------------------------------------


class TestNormalizeVendorName:
    def test_lowercase(self) -> None:
        assert normalize_vendor_name("PAPAS HYPERMARKET") == "papashypermarket"

    def test_strip_ltd(self) -> None:
        assert normalize_vendor_name("FreSko BUTANOLO LTD") == "freskobutanolo"

    def test_strip_gmbh(self) -> None:
        assert normalize_vendor_name("Siemens GmbH") == "siemens"

    def test_strip_inc(self) -> None:
        assert normalize_vendor_name("Apple Inc") == "apple"

    def test_strip_punctuation(self) -> None:
        assert normalize_vendor_name("Mc-Donald's (Cyprus)") == "mcdonaldscyprus"

    def test_empty_string(self) -> None:
        assert normalize_vendor_name("") == ""


class TestVendorsMatch:
    def test_exact_match(self) -> None:
        assert vendors_match("PAPAS HYPERMARKET", "PAPAS HYPERMARKET")

    def test_case_insensitive(self) -> None:
        assert vendors_match("papas hypermarket", "PAPAS HYPERMARKET")

    def test_substring_match(self) -> None:
        assert vendors_match("FRESKO", "FreSko BUTANOLO LTD")

    def test_short_substring_no_match(self) -> None:
        assert not vendors_match("ABC", "ABCDEF Corp")

    def test_empty_no_match(self) -> None:
        assert not vendors_match("", "Store")
        assert not vendors_match("Store", "")

    def test_different_vendors(self) -> None:
        assert not vendors_match("Store A", "Store B")


# ---------------------------------------------------------------------------
# Bundle summary extraction
# ---------------------------------------------------------------------------


class TestExtractBundleSummary:
    def test_basic_receipt(self) -> None:
        atoms = [
            {"atom_type": "vendor", "data": {"name": "PAPAS"}},
            {
                "atom_type": "amount",
                "data": {"value": "85.69", "currency": "EUR", "semantic_type": "total"},
            },
            {"atom_type": "datetime", "data": {"value": "2026-01-21 13:56"}},
            {"atom_type": "item", "data": {"name": "Red Bull"}},
            {"atom_type": "item", "data": {"name": "Bread"}},
        ]
        s = extract_bundle_summary("b1", BundleType.BASKET, atoms)
        assert s.vendor == "PAPAS"
        assert s.amount == Decimal("85.69")
        assert s.event_date == date(2026, 1, 21)
        assert len(s.item_names) == 2
        assert "red bull" in s.item_names

    def test_no_vendor(self) -> None:
        s = extract_bundle_summary("b1", BundleType.BASKET, [])
        assert s.vendor is None
        assert s.amount is None

    def test_cloud_id_preserved(self) -> None:
        s = extract_bundle_summary("b1", BundleType.BASKET, [], cloud_id="c1")
        assert s.cloud_id == "c1"

    def test_vendor_key_from_vat_number(self) -> None:
        atoms = [
            {
                "atom_type": "vendor",
                "data": {"name": "FreSko", "vat_number": "bg 123456789"},
            },
        ]
        s = extract_bundle_summary("b1", BundleType.BASKET, atoms)
        assert s.vendor == "FreSko"
        assert s.vendor_key == "BG123456789"

    def test_vendor_key_none_without_registration(self) -> None:
        atoms = [{"atom_type": "vendor", "data": {"name": "PAPAS"}}]
        s = extract_bundle_summary("b1", BundleType.BASKET, atoms)
        assert s.vendor == "PAPAS"
        assert s.vendor_key is None


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


class TestVendorScore:
    def test_exact_match(self) -> None:
        a = BundleSummary(
            bundle_id="a", bundle_type=BundleType.BASKET, vendor_normalized="papas"
        )
        b = BundleSummary(
            bundle_id="b", bundle_type=BundleType.BASKET, vendor_normalized="papas"
        )
        assert _vendor_score(a, b) == Decimal("1")

    def test_substring_match(self) -> None:
        a = BundleSummary(
            bundle_id="a", bundle_type=BundleType.BASKET, vendor_normalized="papas"
        )
        b = BundleSummary(
            bundle_id="b",
            bundle_type=BundleType.BASKET,
            vendor_normalized="papashypermarket",
        )
        assert _vendor_score(a, b) == Decimal("0.8")

    def test_no_match(self) -> None:
        a = BundleSummary(
            bundle_id="a", bundle_type=BundleType.BASKET, vendor_normalized="papas"
        )
        b = BundleSummary(
            bundle_id="b", bundle_type=BundleType.BASKET, vendor_normalized="lidl"
        )
        assert _vendor_score(a, b) == Decimal("0")

    def test_missing_vendor(self) -> None:
        a = BundleSummary(bundle_id="a", bundle_type=BundleType.BASKET)
        b = BundleSummary(
            bundle_id="b", bundle_type=BundleType.BASKET, vendor_normalized="papas"
        )
        assert _vendor_score(a, b) == Decimal("0")

    def test_vendor_key_exact_match(self) -> None:
        """Same registration ID = definitive match regardless of name."""
        a = BundleSummary(
            bundle_id="a",
            bundle_type=BundleType.BASKET,
            vendor_normalized="fresko",
            vendor_key="BG123456789",
        )
        b = BundleSummary(
            bundle_id="b",
            bundle_type=BundleType.BASKET,
            vendor_normalized="freskobutanolo",
            vendor_key="BG123456789",
        )
        assert _vendor_score(a, b) == Decimal("1")

    def test_vendor_key_mismatch_blocks(self) -> None:
        """Different registration IDs = definitive non-match even with same name."""
        a = BundleSummary(
            bundle_id="a",
            bundle_type=BundleType.BASKET,
            vendor_normalized="supermarket",
            vendor_key="BG111111111",
        )
        b = BundleSummary(
            bundle_id="b",
            bundle_type=BundleType.BASKET,
            vendor_normalized="supermarket",
            vendor_key="BG222222222",
        )
        assert _vendor_score(a, b) == Decimal("0")

    def test_vendor_key_one_side_only_falls_back(self) -> None:
        """One bundle has vendor_key, other doesn't: fall back to name matching."""
        a = BundleSummary(
            bundle_id="a",
            bundle_type=BundleType.BASKET,
            vendor_normalized="papas",
            vendor_key="BG123456789",
        )
        b = BundleSummary(
            bundle_id="b",
            bundle_type=BundleType.BASKET,
            vendor_normalized="papas",
        )
        # Falls back to name matching since only one has vendor_key
        assert _vendor_score(a, b) == Decimal("1")

    def test_vendor_key_fuzzy_match_single_char_typo(self) -> None:
        """vendor_key with a single OCR digit substitution returns 0.9."""
        # "10180201N" vs "101800201" — one digit transposed/inserted
        a = BundleSummary(
            bundle_id="a",
            bundle_type=BundleType.BASKET,
            vendor_key="10180201N",
        )
        b = BundleSummary(
            bundle_id="b",
            bundle_type=BundleType.BASKET,
            vendor_key="101800201",
        )
        assert _vendor_score(a, b) == Decimal("0.9")

    def test_vendor_key_fuzzy_no_match_too_different(self) -> None:
        """vendor_key strings that differ by more than a typo return 0."""
        a = BundleSummary(
            bundle_id="a",
            bundle_type=BundleType.BASKET,
            vendor_key="BG111111111",
        )
        b = BundleSummary(
            bundle_id="b",
            bundle_type=BundleType.BASKET,
            vendor_key="BG999999999",
        )
        assert _vendor_score(a, b) == Decimal("0")

    def test_vendor_key_prefixed_vs_bare_match(self) -> None:
        """'CY10370773Q' and '10370773Q' resolve to 1.0 after prefix stripping."""
        a = BundleSummary(
            bundle_id="a",
            bundle_type=BundleType.BASKET,
            vendor_key="CY10370773Q",
        )
        b = BundleSummary(
            bundle_id="b",
            bundle_type=BundleType.PAYMENT_RECORD,
            vendor_key="10370773Q",
        )
        assert _vendor_score(a, b) == Decimal("1")

    def test_vendor_key_both_prefixed_same_country_match(self) -> None:
        """Both prefixed with same EU code: exact match still fires first."""
        a = BundleSummary(
            bundle_id="a",
            bundle_type=BundleType.BASKET,
            vendor_key="CY10370773Q",
        )
        b = BundleSummary(
            bundle_id="b",
            bundle_type=BundleType.PAYMENT_RECORD,
            vendor_key="CY10370773Q",
        )
        assert _vendor_score(a, b) == Decimal("1")

    def test_vendor_key_different_eu_prefix_same_bare_match(self) -> None:
        """Different EU prefixes, same bare number → still match (prefix is cosmetic)."""
        a = BundleSummary(
            bundle_id="a",
            bundle_type=BundleType.BASKET,
            vendor_key="CY10370773Q",
        )
        b = BundleSummary(
            bundle_id="b",
            bundle_type=BundleType.PAYMENT_RECORD,
            vendor_key="EL10370773Q",
        )
        assert _vendor_score(a, b) == Decimal("1")

    def test_vendor_key_non_eu_prefix_not_stripped(self) -> None:
        """Non-EU prefix is not stripped; different bare numbers do not match."""
        a = BundleSummary(
            bundle_id="a",
            bundle_type=BundleType.BASKET,
            vendor_key="US12345678",
        )
        b = BundleSummary(
            bundle_id="b",
            bundle_type=BundleType.PAYMENT_RECORD,
            vendor_key="12345678",
        )
        # "US" is not an EU code, so "US12345678" stripped is still "US12345678"
        # while "12345678" stripped is "12345678" — these differ, so not an exact match.
        # Fuzzy ratio("US12345678", "12345678") ~= 0.89 → returns 0.9
        score = _vendor_score(a, b)
        assert score != Decimal("1")

    def test_vendor_key_different_bare_numbers_no_match(self) -> None:
        """Prefixed vs bare with different underlying numbers → 0."""
        a = BundleSummary(
            bundle_id="a",
            bundle_type=BundleType.BASKET,
            vendor_key="CY10370773Q",
        )
        b = BundleSummary(
            bundle_id="b",
            bundle_type=BundleType.PAYMENT_RECORD,
            vendor_key="99999999X",
        )
        assert _vendor_score(a, b) == Decimal("0")

    def test_ocr_variant_vat_rescued_by_exact_name(self) -> None:
        """Sub-fuzzy-threshold VAT variant + identical name = one merchant.

        PAPAS "10355430K" vs "103055400K" has a SequenceMatcher ratio of ~0.84,
        just under the 0.85 key-alone threshold, so without name corroboration
        the same shop split into two vendor rows. With identical trade names it
        is treated as an OCR variant (0.85), not vetoed.
        """
        a = BundleSummary(
            bundle_id="a",
            bundle_type=BundleType.BASKET,
            vendor_normalized="papashypermarket",
            vendor_key="10355430K",
        )
        b = BundleSummary(
            bundle_id="b",
            bundle_type=BundleType.BASKET,
            vendor_normalized="papashypermarket",
            vendor_key="103055400K",
        )
        assert _vendor_score(a, b) == Decimal("0.85")


class TestAmountScore:
    def test_exact_match(self) -> None:
        a = BundleSummary(
            bundle_id="a", bundle_type=BundleType.BASKET, amount=Decimal("85.69")
        )
        b = BundleSummary(
            bundle_id="b", bundle_type=BundleType.BASKET, amount=Decimal("85.69")
        )
        score, match_type = _amount_score(a, b)
        assert score == Decimal("1")
        assert match_type == CloudMatchType.EXACT_AMOUNT

    def test_different_amounts(self) -> None:
        a = BundleSummary(
            bundle_id="a", bundle_type=BundleType.BASKET, amount=Decimal("85.69")
        )
        b = BundleSummary(
            bundle_id="b", bundle_type=BundleType.BASKET, amount=Decimal("42.50")
        )
        score, _ = _amount_score(a, b)
        assert score == Decimal("0")

    def test_different_currency(self) -> None:
        a = BundleSummary(
            bundle_id="a",
            bundle_type=BundleType.BASKET,
            amount=Decimal("85.69"),
            currency="EUR",
        )
        b = BundleSummary(
            bundle_id="b",
            bundle_type=BundleType.BASKET,
            amount=Decimal("85.69"),
            currency="USD",
        )
        score, _ = _amount_score(a, b)
        assert score == Decimal("0")

    def test_missing_amount(self) -> None:
        a = BundleSummary(bundle_id="a", bundle_type=BundleType.BASKET)
        b = BundleSummary(
            bundle_id="b", bundle_type=BundleType.BASKET, amount=Decimal("85.69")
        )
        score, _ = _amount_score(a, b)
        assert score == Decimal("0")

    def test_near_match_within_tolerance(self) -> None:
        """Amounts differing by 0.01 (within 0.02 tolerance) return NEAR_AMOUNT."""
        a = BundleSummary(
            bundle_id="a", bundle_type=BundleType.BASKET, amount=Decimal("85.69")
        )
        b = BundleSummary(
            bundle_id="b", bundle_type=BundleType.BASKET, amount=Decimal("85.70")
        )
        score, match_type = _amount_score(a, b)
        assert score == Decimal("0.8")
        assert match_type == CloudMatchType.NEAR_AMOUNT

    def test_near_match_at_boundary(self) -> None:
        """Amounts differing by exactly 0.02 (at tolerance boundary) return NEAR_AMOUNT."""
        a = BundleSummary(
            bundle_id="a", bundle_type=BundleType.BASKET, amount=Decimal("85.69")
        )
        b = BundleSummary(
            bundle_id="b", bundle_type=BundleType.BASKET, amount=Decimal("85.71")
        )
        score, match_type = _amount_score(a, b)
        assert score == Decimal("0.8")
        assert match_type == CloudMatchType.NEAR_AMOUNT

    def test_beyond_tolerance(self) -> None:
        """Amounts differing by 0.03 (beyond tolerance) return zero score."""
        a = BundleSummary(
            bundle_id="a", bundle_type=BundleType.BASKET, amount=Decimal("85.69")
        )
        b = BundleSummary(
            bundle_id="b", bundle_type=BundleType.BASKET, amount=Decimal("85.72")
        )
        score, match_type = _amount_score(a, b)
        assert score == Decimal("0")
        assert match_type is None


class TestDateScore:
    def test_same_day(self) -> None:
        a = BundleSummary(
            bundle_id="a", bundle_type=BundleType.BASKET, event_date=date(2026, 1, 21)
        )
        b = BundleSummary(
            bundle_id="b", bundle_type=BundleType.BASKET, event_date=date(2026, 1, 21)
        )
        assert _date_score(a, b) == Decimal("1")

    def test_within_tolerance(self) -> None:
        a = BundleSummary(
            bundle_id="a", bundle_type=BundleType.BASKET, event_date=date(2026, 1, 21)
        )
        b = BundleSummary(
            bundle_id="b",
            bundle_type=BundleType.STATEMENT_LINE,
            event_date=date(2026, 1, 23),
        )
        score = _date_score(a, b)
        assert score > Decimal("0")
        assert score < Decimal("1")

    def test_outside_tolerance(self) -> None:
        a = BundleSummary(
            bundle_id="a", bundle_type=BundleType.BASKET, event_date=date(2026, 1, 1)
        )
        b = BundleSummary(
            bundle_id="b",
            bundle_type=BundleType.PAYMENT_RECORD,
            event_date=date(2026, 2, 1),
        )
        assert _date_score(a, b) == Decimal("0")

    def test_unknown_date(self) -> None:
        a = BundleSummary(bundle_id="a", bundle_type=BundleType.BASKET)
        b = BundleSummary(
            bundle_id="b", bundle_type=BundleType.BASKET, event_date=date(2026, 1, 21)
        )
        assert _date_score(a, b) == Decimal("0.5")  # Neutral

    def test_basket_basket_same_day_clusters(self) -> None:
        """Two receipts from same vendor on same day should cluster."""
        a = BundleSummary(
            bundle_id="a", bundle_type=BundleType.BASKET, event_date=date(2026, 1, 21)
        )
        b = BundleSummary(
            bundle_id="b", bundle_type=BundleType.BASKET, event_date=date(2026, 1, 21)
        )
        assert _date_score(a, b) == Decimal("1")

    def test_basket_basket_different_day_rejects(self) -> None:
        """Two receipts from same vendor on different days should NOT cluster."""
        a = BundleSummary(
            bundle_id="a", bundle_type=BundleType.BASKET, event_date=date(2026, 1, 21)
        )
        b = BundleSummary(
            bundle_id="b", bundle_type=BundleType.BASKET, event_date=date(2026, 1, 22)
        )
        assert _date_score(a, b) == Decimal("0")

    def test_invoice_invoice_different_day_rejects(self) -> None:
        """Two invoices on different days should NOT cluster."""
        a = BundleSummary(
            bundle_id="a", bundle_type=BundleType.INVOICE, event_date=date(2026, 3, 1)
        )
        b = BundleSummary(
            bundle_id="b", bundle_type=BundleType.INVOICE, event_date=date(2026, 3, 5)
        )
        assert _date_score(a, b) == Decimal("0")

    def test_invoice_payment_wide_tolerance(self) -> None:
        a = BundleSummary(
            bundle_id="a", bundle_type=BundleType.INVOICE, event_date=date(2026, 1, 5)
        )
        b = BundleSummary(
            bundle_id="b",
            bundle_type=BundleType.PAYMENT_RECORD,
            event_date=date(2026, 1, 20),
        )
        score = _date_score(a, b)
        assert score > Decimal("0")  # 15 days is within 60-day tolerance


class TestItemOverlapScore:
    def test_full_overlap(self) -> None:
        a = BundleSummary(
            bundle_id="a", bundle_type=BundleType.BASKET, item_names=["apple", "bread"]
        )
        b = BundleSummary(
            bundle_id="b", bundle_type=BundleType.BASKET, item_names=["apple", "bread"]
        )
        assert _item_overlap_score(a, b) == Decimal("1")

    def test_partial_overlap(self) -> None:
        a = BundleSummary(
            bundle_id="a",
            bundle_type=BundleType.BASKET,
            item_names=["apple", "bread", "milk"],
        )
        b = BundleSummary(
            bundle_id="b", bundle_type=BundleType.BASKET, item_names=["apple", "bread"]
        )
        score = _item_overlap_score(a, b)
        assert score > Decimal("0")
        assert score < Decimal("1")

    def test_no_overlap(self) -> None:
        a = BundleSummary(
            bundle_id="a", bundle_type=BundleType.BASKET, item_names=["apple"]
        )
        b = BundleSummary(
            bundle_id="b", bundle_type=BundleType.BASKET, item_names=["bread"]
        )
        assert _item_overlap_score(a, b) == Decimal("0")

    def test_empty_items(self) -> None:
        a = BundleSummary(bundle_id="a", bundle_type=BundleType.BASKET)
        b = BundleSummary(
            bundle_id="b", bundle_type=BundleType.BASKET, item_names=["apple"]
        )
        assert _item_overlap_score(a, b) == Decimal("0")


# ---------------------------------------------------------------------------
# Cloud lifecycle
# ---------------------------------------------------------------------------


class TestFindCloudForBundle:
    def _receipt_bundle(self) -> BundleSummary:
        return BundleSummary(
            bundle_id="new",
            bundle_type=BundleType.BASKET,
            vendor="PAPAS",
            vendor_normalized="papas",
            amount=Decimal("85.69"),
            event_date=date(2026, 1, 21),
        )

    def test_no_existing_bundles(self) -> None:
        result = find_cloud_for_bundle(self._receipt_bundle(), [])
        assert result.is_new_cloud

    def test_match_same_vendor_amount_date(self) -> None:
        existing = BundleSummary(
            bundle_id="old",
            bundle_type=BundleType.PAYMENT_RECORD,
            vendor="PAPAS HYPERMARKET",
            vendor_normalized="papashypermarket",
            amount=Decimal("85.69"),
            event_date=date(2026, 1, 21),
            cloud_id="cloud-1",
        )
        result = find_cloud_for_bundle(self._receipt_bundle(), [existing])
        assert not result.is_new_cloud
        assert result.cloud_id == "cloud-1"
        assert result.confidence > Decimal("0.5")

    def test_no_match_different_vendor(self) -> None:
        existing = BundleSummary(
            bundle_id="old",
            bundle_type=BundleType.PAYMENT_RECORD,
            vendor="LIDL",
            vendor_normalized="lidl",
            amount=Decimal("85.69"),
            event_date=date(2026, 1, 21),
            cloud_id="cloud-1",
        )
        result = find_cloud_for_bundle(self._receipt_bundle(), [existing])
        assert result.is_new_cloud

    def test_no_match_different_amount(self) -> None:
        existing = BundleSummary(
            bundle_id="old",
            bundle_type=BundleType.PAYMENT_RECORD,
            vendor="PAPAS",
            vendor_normalized="papas",
            amount=Decimal("42.50"),
            event_date=date(2026, 1, 21),
            cloud_id="cloud-1",
        )
        result = find_cloud_for_bundle(self._receipt_bundle(), [existing])
        assert result.is_new_cloud

    def test_skip_same_bundle(self) -> None:
        bundle = self._receipt_bundle()
        existing = BundleSummary(
            bundle_id="new",  # Same ID
            bundle_type=BundleType.BASKET,
            vendor="PAPAS",
            vendor_normalized="papas",
            amount=Decimal("85.69"),
            event_date=date(2026, 1, 21),
            cloud_id="cloud-1",
        )
        result = find_cloud_for_bundle(bundle, [existing])
        assert result.is_new_cloud


class TestCreateCloud:
    def test_creates_cloud_and_link(self) -> None:
        cloud, link = create_cloud_for_bundle("bundle-1")
        assert cloud.status == CloudStatus.FORMING
        assert link.cloud_id == cloud.id
        assert link.bundle_id == "bundle-1"
        assert link.match_confidence == Decimal("1.0")


class TestAddBundleToCloud:
    def test_creates_link(self) -> None:
        link = add_bundle_to_cloud(
            "cloud-1", "bundle-2", CloudMatchType.EXACT_AMOUNT, Decimal("0.9")
        )
        assert link.cloud_id == "cloud-1"
        assert link.bundle_id == "bundle-2"
        assert link.match_type == CloudMatchType.EXACT_AMOUNT


# ---------------------------------------------------------------------------
# Fact collapse
# ---------------------------------------------------------------------------


def _make_receipt_bundle() -> dict[str, Any]:
    return {
        "bundle_id": "b1",
        "bundle_type": "basket",
        "atoms": [
            {"atom_type": "vendor", "data": {"name": "PAPAS"}},
            {
                "atom_type": "amount",
                "data": {"value": "85.69", "currency": "EUR", "semantic_type": "total"},
            },
            {"atom_type": "datetime", "data": {"value": "2026-01-21 13:56"}},
            {
                "atom_type": "payment",
                "data": {"method": "card", "card_last4": "7201", "amount": "85.69"},
            },
            {
                "atom_type": "item",
                "id": "a1",
                "data": {
                    "name": "Red Bull",
                    "quantity": "3",
                    "unit": "ml",
                    "unit_price": "1.99",
                    "total_price": "5.97",
                    "tax_rate": "24",
                    "tax_type": "vat",
                    "category": "beverages",
                },
            },
        ],
    }


def _make_card_slip_bundle() -> dict[str, Any]:
    return {
        "bundle_id": "b2",
        "bundle_type": "payment_record",
        "atoms": [
            {"atom_type": "vendor", "data": {"name": "PAPAS"}},
            {
                "atom_type": "amount",
                "data": {"value": "85.69", "currency": "EUR", "semantic_type": "total"},
            },
            {"atom_type": "datetime", "data": {"value": "2026-01-21 14:00"}},
            {
                "atom_type": "payment",
                "data": {
                    "method": "card",
                    "card_last4": "7201",
                    "amount": "85.69",
                    "auth_code": "083646",
                },
            },
        ],
    }


class TestTryCollapseSingle:
    def test_single_receipt_collapses(self) -> None:
        cloud = Cloud(id="c1", status=CloudStatus.FORMING)
        result = try_collapse(cloud, [_make_receipt_bundle()])
        assert result.collapsed
        assert result.fact is not None
        assert result.fact.vendor == "PAPAS"
        assert result.fact.total_amount == Decimal("85.69")
        assert result.fact.currency == "EUR"
        assert result.fact.event_date == date(2026, 1, 21)
        assert result.fact.status == FactStatus.CONFIRMED
        assert result.cloud_status == CloudStatus.COLLAPSED

    def test_single_receipt_has_items(self) -> None:
        cloud = Cloud(id="c1", status=CloudStatus.FORMING)
        result = try_collapse(cloud, [_make_receipt_bundle()])
        assert len(result.items) == 1
        assert result.items[0].name == "Red Bull"
        assert result.items[0].quantity == Decimal("3")
        assert result.items[0].unit == UnitType.MILLILITER
        assert result.items[0].tax_rate == Decimal("24")
        assert result.items[0].tax_type == TaxType.VAT

    def test_single_receipt_has_payments(self) -> None:
        cloud = Cloud(id="c1", status=CloudStatus.FORMING)
        result = try_collapse(cloud, [_make_receipt_bundle()])
        assert result.fact is not None
        assert result.fact.payments is not None
        assert len(result.fact.payments) == 1
        assert result.fact.payments[0]["method"] == "card"
        assert result.fact.payments[0]["card_last4"] == "7201"

    def test_empty_bundles(self) -> None:
        cloud = Cloud(id="c1", status=CloudStatus.FORMING)
        result = try_collapse(cloud, [])
        assert not result.collapsed


class TestTryCollapseMulti:
    def test_receipt_plus_card_slip(self) -> None:
        cloud = Cloud(id="c1", status=CloudStatus.FORMING)
        bundles = [_make_receipt_bundle(), _make_card_slip_bundle()]
        result = try_collapse(cloud, bundles)
        assert result.collapsed
        assert result.fact is not None
        assert result.fact.vendor == "PAPAS"
        assert result.fact.total_amount == Decimal("85.69")
        # The receipt and the card slip record the SAME 85.69 charge (card 7201),
        # so the two payment atoms collapse to one merged entry that keeps the
        # slip's auth_code — not a doubled "paid" total.
        assert result.fact.payments is not None
        assert len(result.fact.payments) == 1
        assert result.fact.payments[0]["card_last4"] == "7201"
        assert result.fact.payments[0]["auth_code"] == "083646"

    def test_merchant_vendor_wins_over_acquirer(self) -> None:
        # Basket (real merchant) + payment slip whose vendor is the card
        # acquirer. The collapsed fact must take the merchant's name, not the
        # acquirer's, even when the acquirer bundle is listed first.
        acquirer_slip = {
            "bundle_id": "b2",
            "bundle_type": "payment_record",
            "atoms": [
                {"atom_type": "vendor", "data": {"name": "JCC PAYMENT SYSTEMS"}},
                {
                    "atom_type": "amount",
                    "data": {
                        "value": "15.32",
                        "currency": "EUR",
                        "semantic_type": "total",
                    },
                },
                {"atom_type": "datetime", "data": {"value": "2025-11-16 10:00"}},
            ],
        }
        merchant_basket = {
            "bundle_id": "b1",
            "bundle_type": "basket",
            "atoms": [
                {
                    "atom_type": "vendor",
                    "data": {
                        "name": "The Nut Cracker House",
                        "vat_number": "10123167G",
                    },
                },
                {
                    "atom_type": "amount",
                    "data": {
                        "value": "15.32",
                        "currency": "EUR",
                        "semantic_type": "total",
                    },
                },
                {"atom_type": "datetime", "data": {"value": "2025-11-16 09:58"}},
                {
                    "atom_type": "item",
                    "id": "a1",
                    "data": {"name": "Cashews", "total_price": "15.32"},
                },
            ],
        }
        cloud = Cloud(id="c1", status=CloudStatus.FORMING)
        result = try_collapse(cloud, [acquirer_slip, merchant_basket])
        assert result.collapsed
        assert result.fact is not None
        assert result.fact.vendor == "The Nut Cracker House"
        assert result.fact.vendor_key == "10123167G"

    def test_weak_evidence_stays_forming(self) -> None:
        cloud = Cloud(id="c1", status=CloudStatus.FORMING)
        # Two bundles with different vendors and amounts — weak match
        b1 = {
            "bundle_id": "b1",
            "bundle_type": "basket",
            "atoms": [
                {"atom_type": "vendor", "data": {"name": "Store A"}},
                {
                    "atom_type": "amount",
                    "data": {
                        "value": "10.00",
                        "currency": "EUR",
                        "semantic_type": "total",
                    },
                },
            ],
        }
        b2 = {
            "bundle_id": "b2",
            "bundle_type": "basket",
            "atoms": [
                {"atom_type": "vendor", "data": {"name": "Store B"}},
                {
                    "atom_type": "amount",
                    "data": {
                        "value": "20.00",
                        "currency": "EUR",
                        "semantic_type": "total",
                    },
                },
            ],
        }
        result = try_collapse(cloud, [b1, b2])
        assert not result.collapsed
        assert result.cloud_status == CloudStatus.FORMING


class TestAmountsSumTo:
    def test_exact_sum(self) -> None:
        assert _amounts_sum_to([Decimal("500"), Decimal("500")], Decimal("1000"))

    def test_within_tolerance(self) -> None:
        assert _amounts_sum_to([Decimal("500"), Decimal("500.01")], Decimal("1000"))

    def test_outside_tolerance(self) -> None:
        assert not _amounts_sum_to([Decimal("500"), Decimal("499")], Decimal("1000"))

    def test_empty_amounts(self) -> None:
        assert not _amounts_sum_to([], Decimal("100"))


class TestExtractHelpers:
    def test_extract_vendor(self) -> None:
        atoms = [{"atom_type": "vendor", "data": {"name": "PAPAS"}}]
        assert _extract_vendor(atoms) == "PAPAS"

    def test_extract_total(self) -> None:
        atoms = [
            {
                "atom_type": "amount",
                "data": {"value": "85.69", "semantic_type": "total"},
            }
        ]
        assert _extract_total(atoms) == Decimal("85.69")

    def test_extract_payments(self) -> None:
        atoms = [
            {"atom_type": "payment", "data": {"method": "card", "card_last4": "7201"}}
        ]
        payments = _extract_payments(atoms)
        assert len(payments) == 1
        assert payments[0]["method"] == "card"

    def test_extract_items(self) -> None:
        atoms = [
            {
                "atom_type": "item",
                "id": "a1",
                "data": {
                    "name": "Red Bull",
                    "quantity": "3",
                    "unit": "pcs",
                    "total_price": "5.97",
                },
            }
        ]
        items = _extract_items(atoms, "EUR")
        assert len(items) == 1
        assert items[0].name == "Red Bull"
        assert items[0].quantity == Decimal("3")
        assert items[0].total_price == Decimal("5.97")


class TestInferFactType:
    def test_basket_is_purchase(self) -> None:
        assert infer_fact_type([{"bundle_type": "basket"}]) == FactType.PURCHASE

    def test_invoice_is_purchase(self) -> None:
        assert infer_fact_type([{"bundle_type": "invoice"}]) == FactType.PURCHASE

    def test_mixed_is_purchase(self) -> None:
        bundles = [{"bundle_type": "basket"}, {"bundle_type": "payment_record"}]
        assert infer_fact_type(bundles) == FactType.PURCHASE


# ---------------------------------------------------------------------------
# Vendor legal name matching
# ---------------------------------------------------------------------------


class TestVendorScoreLegalName:
    def test_trade_name_matches_legal_name(self) -> None:
        """Trade name "fresko" should match legal name "freskobutanolo"."""
        a = BundleSummary(
            bundle_id="a",
            bundle_type=BundleType.BASKET,
            vendor_normalized="fresko",
        )
        b = BundleSummary(
            bundle_id="b",
            bundle_type=BundleType.PAYMENT_RECORD,
            vendor_legal_normalized="freskobutanolo",
        )
        assert _vendor_score(a, b) == Decimal("0.8")

    def test_legal_names_exact_match(self) -> None:
        """Both bundles have same legal name → exact match."""
        a = BundleSummary(
            bundle_id="a",
            bundle_type=BundleType.BASKET,
            vendor_normalized="fresko",
            vendor_legal_normalized="freskobutanolo",
        )
        b = BundleSummary(
            bundle_id="b",
            bundle_type=BundleType.PAYMENT_RECORD,
            vendor_normalized="fresko",
            vendor_legal_normalized="freskobutanolo",
        )
        assert _vendor_score(a, b) == Decimal("1")

    def test_trade_a_matches_legal_b(self) -> None:
        """Bundle A trade name == Bundle B legal normalized → exact match."""
        a = BundleSummary(
            bundle_id="a",
            bundle_type=BundleType.BASKET,
            vendor_normalized="freskobutanolo",
        )
        b = BundleSummary(
            bundle_id="b",
            bundle_type=BundleType.PAYMENT_RECORD,
            vendor_legal_normalized="freskobutanolo",
        )
        assert _vendor_score(a, b) == Decimal("1")

    def test_no_names_at_all(self) -> None:
        """No vendor info on either side → score 0."""
        a = BundleSummary(bundle_id="a", bundle_type=BundleType.BASKET)
        b = BundleSummary(bundle_id="b", bundle_type=BundleType.BASKET)
        assert _vendor_score(a, b) == Decimal("0")

    def test_vendor_key_overrides_legal_name(self) -> None:
        """vendor_key (registration) still takes priority over legal name."""
        a = BundleSummary(
            bundle_id="a",
            bundle_type=BundleType.BASKET,
            vendor_normalized="fresko",
            vendor_legal_normalized="freskobutanolo",
            vendor_key="10336127M",
        )
        b = BundleSummary(
            bundle_id="b",
            bundle_type=BundleType.PAYMENT_RECORD,
            vendor_normalized="fresko",
            vendor_legal_normalized="freskobutanolo",
            vendor_key="10336127M",
        )
        assert _vendor_score(a, b) == Decimal("1")


class TestExtractBundleSummaryLegalName:
    def test_legal_name_extracted(self) -> None:
        atoms = [
            {
                "atom_type": "vendor",
                "data": {
                    "name": "FRESKO",
                    "legal_name": "BUTANOLO LTD",
                    "vat_number": "10336127M",
                },
            },
        ]
        s = extract_bundle_summary("b1", BundleType.BASKET, atoms)
        assert s.vendor == "FRESKO"
        assert s.vendor_legal_name == "BUTANOLO LTD"
        assert s.vendor_legal_normalized == "butanolo"
        assert s.vendor_key == "10336127M"

    def test_no_legal_name(self) -> None:
        atoms = [{"atom_type": "vendor", "data": {"name": "PAPAS"}}]
        s = extract_bundle_summary("b1", BundleType.BASKET, atoms)
        assert s.vendor == "PAPAS"
        assert s.vendor_legal_name is None
        assert s.vendor_legal_normalized is None


# ---------------------------------------------------------------------------
# Time-aware clustering guard
# ---------------------------------------------------------------------------


def _basket(
    bid,
    time_str,
    vendor="lidlcyprus",
    amount="85.10",
    d="2025-12-29",
    btype=BundleType.BASKET,
):
    return BundleSummary(
        bundle_id=bid,
        bundle_type=btype,
        vendor=vendor,
        vendor_normalized=vendor,
        amount=Decimal(amount),
        event_date=date.fromisoformat(d),
        event_time=time_str,
    )


class TestTimeConflict:
    def test_same_type_different_times_conflict(self) -> None:
        # Two LIDL baskets, same day/amount, 6+ hours apart -> different txns.
        a = _basket("a", "12:08:35")
        b = _basket("b", "18:37:49")
        assert _time_conflict(a, b)

    def test_same_type_few_minutes_apart_conflict(self) -> None:
        # Spouses at separate tills, 5 min apart -> distinct transactions.
        a = _basket("a", "12:08:00")
        b = _basket("b", "12:13:00")
        assert _time_conflict(a, b)

    def test_same_type_same_time_no_conflict(self) -> None:
        # Same physical receipt re-scanned -> same time, may merge.
        a = _basket("a", "12:08:35")
        b = _basket("b", "12:08:35")
        assert not _time_conflict(a, b)

    def test_cross_type_few_minutes_no_conflict(self) -> None:
        # Receipt + its card slip a few minutes apart -> same transaction.
        a = _basket("a", "12:08:00", btype=BundleType.BASKET)
        b = _basket("b", "12:12:00", btype=BundleType.PAYMENT_RECORD)
        assert not _time_conflict(a, b)

    def test_missing_time_no_conflict(self) -> None:
        a = _basket("a", None)
        b = _basket("b", "12:08:35")
        assert not _time_conflict(a, b)

    def test_different_dates_no_conflict(self) -> None:
        a = _basket("a", "12:08:00", d="2025-12-29")
        b = _basket("b", "12:08:00", d="2025-12-30")
        assert not _time_conflict(a, b)


class TestScoreMatchTimeGuard:
    def test_different_times_block_merge(self) -> None:
        a = _basket("a", "12:08:35")
        b = _basket("b", "18:37:49")
        score, _ = _score_match(a, b)
        assert score == Decimal("0")

    def test_same_time_allows_merge(self) -> None:
        a = _basket("a", "12:08:35")
        b = _basket("b", "12:08:35")
        score, _ = _score_match(a, b)
        assert score > Decimal("0.5")


class TestScoreMatchDateVeto:
    """Same-type bundles beyond their date tolerance are distinct transactions.

    A supermarket never issues one payment for several days' receipts, so two
    same-vendor/same-amount baskets on different days (e.g. a weekly 3.00 coffee)
    must NOT merge — vendor+amount alone would otherwise clear the 0.5 threshold.
    """

    def test_different_day_baskets_vetoed(self) -> None:
        # Same vendor + amount, 9 days apart, no times -> distinct purchases.
        a = _basket("a", None, d="2026-02-22")
        b = _basket("b", None, d="2026-03-03")
        score, _ = _score_match(a, b)
        assert score == Decimal("0")

    def test_one_day_apart_baskets_vetoed(self) -> None:
        # BASKET↔BASKET tolerance is 0 days; even one day apart is distinct.
        a = _basket("a", None, d="2026-02-22")
        b = _basket("b", None, d="2026-02-23")
        score, _ = _score_match(a, b)
        assert score == Decimal("0")

    def test_same_day_baskets_merge(self) -> None:
        # Two scans of one receipt (same day) still cluster.
        a = _basket("a", None, d="2026-02-22")
        b = _basket("b", None, d="2026-02-22")
        score, _ = _score_match(a, b)
        assert score > Decimal("0.5")

    def test_cross_type_within_tolerance_not_vetoed(self) -> None:
        # Receipt↔slip one day apart keeps its wider asymmetric tolerance.
        a = _basket("a", None, d="2026-02-22", btype=BundleType.BASKET)
        b = _basket("b", None, d="2026-02-23", btype=BundleType.PAYMENT_RECORD)
        score, _ = _score_match(a, b)
        assert score > Decimal("0.5")


class TestExtractBundleSummaryTime:
    def test_parses_event_time(self) -> None:
        atoms = [
            {"atom_type": "datetime", "data": {"value": "2025-12-29 18:37:49"}},
        ]
        s = extract_bundle_summary("b1", BundleType.BASKET, atoms)
        assert s.event_date == date.fromisoformat("2025-12-29")
        assert s.event_time == "18:37:49"

    def test_date_only_no_time(self) -> None:
        atoms = [{"atom_type": "datetime", "data": {"value": "2025-12-29"}}]
        s = extract_bundle_summary("b1", BundleType.BASKET, atoms)
        assert s.event_date == date.fromisoformat("2025-12-29")
        assert s.event_time is None


class TestPaymentSlipCrossKeyMerge:
    """Prevention: a receipt and its card slip must cluster even when the slip
    prints a card-acquirer vendor (different name AND registration), corroborated
    by matching amount + date instead of vendor."""

    def _receipt(self) -> BundleSummary:
        return BundleSummary(
            bundle_id="receipt",
            bundle_type=BundleType.BASKET,
            vendor="PAPAS HYPERMARKET",
            vendor_normalized="papashypermarket",
            vendor_key="103055400K",
            amount=Decimal("9.31"),
            event_date=date(2025, 11, 27),
        )

    def test_acquirer_slip_merges_with_receipt(self) -> None:
        acquirer_slip = BundleSummary(
            bundle_id="slip",
            bundle_type=BundleType.PAYMENT_RECORD,
            vendor="JCC PAYMENT SYSTEMS",
            vendor_normalized="jccpaymentsystems",
            vendor_is_intermediary=True,
            amount=Decimal("9.31"),
            event_date=date(2025, 11, 27),
            cloud_id="cloud-1",
        )
        result = find_cloud_for_bundle(self._receipt(), [acquirer_slip])
        assert not result.is_new_cloud
        assert result.cloud_id == "cloud-1"
        assert result.confidence > Decimal("0.5")

    def test_acquirer_slip_does_not_merge_on_amount_mismatch(self) -> None:
        """The intermediary veto-skip still requires real corroboration: a slip
        with a different amount must not be swept into the cloud on date alone."""
        acquirer_slip = BundleSummary(
            bundle_id="slip",
            bundle_type=BundleType.PAYMENT_RECORD,
            vendor="JCC PAYMENT SYSTEMS",
            vendor_normalized="jccpaymentsystems",
            vendor_is_intermediary=True,
            amount=Decimal("42.00"),
            event_date=date(2025, 11, 27),
            cloud_id="cloud-1",
        )
        result = find_cloud_for_bundle(self._receipt(), [acquirer_slip])
        assert result.is_new_cloud

    def test_non_intermediary_different_vendor_still_vetoed(self) -> None:
        """Two real but different merchants on the same day/amount stay split —
        the veto-skip applies only to payment intermediaries."""
        other = BundleSummary(
            bundle_id="other",
            bundle_type=BundleType.PAYMENT_RECORD,
            vendor="LIDL",
            vendor_normalized="lidl",
            amount=Decimal("9.31"),
            event_date=date(2025, 11, 27),
            cloud_id="cloud-1",
        )
        result = find_cloud_for_bundle(self._receipt(), [other])
        assert result.is_new_cloud


class TestIntermediaryBundleSummary:
    """Prevention: an acquirer's printed registration must not become the
    bundle's vendor_key (it is not the merchant's VAT)."""

    def test_acquirer_vendor_flagged_and_key_suppressed(self) -> None:
        atoms = [
            {
                "atom_type": "vendor",
                "data": {"name": "JCC PAYMENT SYSTEMS", "tax_id": "10370773Q"},
            },
        ]
        s = extract_bundle_summary("slip", BundleType.PAYMENT_RECORD, atoms)
        assert s.vendor_is_intermediary is True
        assert s.vendor_key is None

    def test_merchant_vendor_keeps_key(self) -> None:
        atoms = [
            {
                "atom_type": "vendor",
                "data": {"name": "PAPAS HYPERMARKET", "tax_id": "103055400K"},
            },
        ]
        s = extract_bundle_summary("receipt", BundleType.BASKET, atoms)
        assert s.vendor_is_intermediary is False
        assert s.vendor_key == "103055400K"

    def test_literal_null_tax_id_not_a_key(self) -> None:
        """A literal 'null'/'N/A' VAT must not become a vendor_key."""
        for sentinel in ("null", "NULL", "N/A", "-", "none"):
            atoms = [
                {
                    "atom_type": "vendor",
                    "data": {"name": "SOME SHOP", "tax_id": sentinel},
                },
            ]
            s = extract_bundle_summary("b", BundleType.BASKET, atoms)
            assert s.vendor_key is None, sentinel
