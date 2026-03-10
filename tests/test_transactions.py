"""Tests for vendor matching and pattern learning."""

import pytest

from alibi.matching.transactions import (
    VendorPattern,
    fuzzy_match,
    learn_vendor_pattern,
    match_vendor_pattern,
    normalize_vendor,
)


class TestNormalizeVendor:
    """Tests for normalize_vendor function."""

    def test_lowercase(self):
        assert normalize_vendor("AMAZON") == "amazon"

    def test_remove_inc(self):
        assert normalize_vendor("Amazon Inc.") == "amazon"
        assert normalize_vendor("Amazon Inc") == "amazon"

    def test_remove_llc(self):
        assert normalize_vendor("Acme LLC") == "acme"

    def test_remove_ltd(self):
        assert normalize_vendor("Company Ltd") == "company"
        assert normalize_vendor("Company Ltd.") == "company"

    def test_remove_gmbh(self):
        assert normalize_vendor("Firma GmbH") == "firma"

    def test_remove_punctuation(self):
        assert normalize_vendor("Amazon.com") == "amazoncom"

    def test_collapse_whitespace(self):
        assert normalize_vendor("  Amazon   Store  ") == "amazonstore"

    def test_empty_string(self):
        assert normalize_vendor("") == ""

    def test_none_like(self):
        # Edge case: very short strings (slug strips nothing)
        assert normalize_vendor("A") == "a"


class TestFuzzyMatch:
    """Tests for fuzzy_match function."""

    def test_exact_match(self):
        assert fuzzy_match("amazon", "amazon") == 1.0

    def test_case_insensitive(self):
        assert fuzzy_match("Amazon", "amazon") == 1.0

    def test_similar_strings(self):
        ratio = fuzzy_match("amazon", "amazn")
        assert 0.8 < ratio < 1.0

    def test_different_strings(self):
        ratio = fuzzy_match("amazon", "walmart")
        assert ratio < 0.5

    def test_empty_strings(self):
        assert fuzzy_match("", "amazon") == 0.0
        assert fuzzy_match("amazon", "") == 0.0
        assert fuzzy_match("", "") == 0.0


class TestVendorPattern:
    """Tests for VendorPattern operations."""

    def test_learn_vendor_pattern(self):
        pattern = learn_vendor_pattern(
            vendor_raw="AMAZON.COM",
            vendor_normalized="Amazon",
            category="shopping",
            tags=["online", "retail"],
        )

        assert pattern.vendor_name == "Amazon"
        assert pattern.default_category == "shopping"
        assert "online" in pattern.default_tags
        assert pattern.confidence == 0.9

    def test_match_vendor_pattern_exact(self):
        patterns = [
            VendorPattern(
                pattern="amazon",
                vendor_name="Amazon",
                default_category="shopping",
                confidence=0.9,
            )
        ]

        result = match_vendor_pattern("Amazon.com", patterns)
        assert result is not None
        assert result.vendor_name == "Amazon"

    def test_match_vendor_pattern_fuzzy(self):
        patterns = [
            VendorPattern(
                pattern="starbucks",
                vendor_name="Starbucks",
                default_category="food",
                confidence=0.8,
            )
        ]

        result = match_vendor_pattern("STARBUCKS COFFEE", patterns)
        assert result is not None
        assert result.vendor_name == "Starbucks"

    def test_match_vendor_pattern_no_match(self):
        patterns = [
            VendorPattern(
                pattern="amazon",
                vendor_name="Amazon",
                confidence=0.9,
            )
        ]

        result = match_vendor_pattern("Walmart", patterns)
        assert result is None
