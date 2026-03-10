"""Tests for cross-vendor product matching."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from alibi.enrichment.product_matcher import (
    MatchBatchResponse,
    MatchedProductGroup,
    ProductCandidate,
    ProductMatchResult,
    _build_matching_prompt,
    find_cross_vendor_matches,
    match_products_batch,
)


class TestBuildMatchingPrompt:
    def test_basic_prompt(self) -> None:
        products = [
            ProductCandidate(item_id="a", name="Milk 1L", vendor_name="Shop A"),
            ProductCandidate(item_id="b", name="Fresh Milk 1L", vendor_name="Shop B"),
        ]
        prompt = _build_matching_prompt(products)
        assert '[0] "Milk 1L" at Shop A' in prompt
        assert '[1] "Fresh Milk 1L" at Shop B' in prompt

    def test_includes_brand(self) -> None:
        products = [
            ProductCandidate(
                item_id="a",
                name="Cola",
                vendor_name="Shop",
                brand="Coca-Cola",
            ),
        ]
        prompt = _build_matching_prompt(products)
        assert "brand=Coca-Cola" in prompt

    def test_includes_barcode(self) -> None:
        products = [
            ProductCandidate(
                item_id="a",
                name="Cola",
                vendor_name="Shop",
                barcode="5449000000996",
            ),
        ]
        prompt = _build_matching_prompt(products)
        assert "barcode=5449000000996" in prompt

    def test_includes_size(self) -> None:
        products = [
            ProductCandidate(
                item_id="a",
                name="Milk",
                vendor_name="Shop",
                unit_quantity=1.0,
                unit="L",
            ),
        ]
        prompt = _build_matching_prompt(products)
        assert "size=1.0L" in prompt

    def test_includes_comparable_name(self) -> None:
        products = [
            ProductCandidate(
                item_id="a",
                name="Γάλα",
                vendor_name="Shop",
                comparable_name="Milk",
            ),
        ]
        prompt = _build_matching_prompt(products)
        assert "en=Milk" in prompt


class TestMatchProductsBatch:
    def test_no_api_key_returns_empty(self) -> None:
        with patch(
            "alibi.enrichment.product_matcher._get_api_key",
            return_value=None,
        ):
            result = match_products_batch(
                [
                    ProductCandidate(item_id="a", name="X", vendor_name="A"),
                    ProductCandidate(item_id="b", name="Y", vendor_name="B"),
                ]
            )
        assert result == []

    def test_single_product_returns_empty(self) -> None:
        with patch(
            "alibi.enrichment.product_matcher._get_api_key",
            return_value="key",
        ):
            result = match_products_batch(
                [ProductCandidate(item_id="a", name="X", vendor_name="A")]
            )
        assert result == []

    def test_structured_response(self) -> None:
        mock_response = MagicMock()
        mock_parsed = MatchBatchResponse(
            matches=[
                ProductMatchResult(
                    product_a_idx=0,
                    product_b_idx=1,
                    confidence=0.9,
                    reasoning="Same product",
                    suggested_canonical="Full Cream Milk 1L",
                )
            ]
        )
        mock_response.parsed = mock_parsed

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        products = [
            ProductCandidate(item_id="a", name="Milk 1L", vendor_name="A"),
            ProductCandidate(item_id="b", name="Fresh Milk 1L", vendor_name="B"),
        ]

        with (
            patch(
                "alibi.enrichment.product_matcher._get_api_key",
                return_value="key",
            ),
            patch(
                "alibi.enrichment.product_matcher._get_model",
                return_value="gemini-2.0-flash",
            ),
            patch("google.genai.Client", return_value=mock_client),
        ):
            results = match_products_batch(products)

        assert len(results) == 1
        assert results[0].confidence == 0.9

    def test_api_error_returns_empty(self) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("boom")

        products = [
            ProductCandidate(item_id="a", name="X", vendor_name="A"),
            ProductCandidate(item_id="b", name="Y", vendor_name="B"),
        ]

        with (
            patch(
                "alibi.enrichment.product_matcher._get_api_key",
                return_value="key",
            ),
            patch(
                "alibi.enrichment.product_matcher._get_model",
                return_value="gemini-2.0-flash",
            ),
            patch("google.genai.Client", return_value=mock_client),
        ):
            results = match_products_batch(products)

        assert results == []


class TestFindCrossVendorMatches:
    def test_empty_db(self, mock_db: MagicMock) -> None:
        mock_db.fetchall.return_value = []
        results = find_cross_vendor_matches(mock_db)
        assert results == []

    def test_single_vendor_skipped(self, mock_db: MagicMock) -> None:
        mock_db.fetchall.return_value = [
            ("id1", "Milk", None, "Dairy", None, None, None, None, "vk1", "Shop A"),
            ("id2", "Bread", None, "Bakery", None, None, None, None, "vk1", "Shop A"),
        ]
        results = find_cross_vendor_matches(mock_db)
        assert results == []

    def test_matches_returned(self, mock_db: MagicMock) -> None:
        mock_db.fetchall.return_value = [
            ("id1", "Milk 1L", None, "Dairy", None, None, None, None, "vk1", "Shop A"),
            (
                "id2",
                "Fresh Milk 1L",
                None,
                "Dairy",
                None,
                None,
                None,
                None,
                "vk2",
                "Shop B",
            ),
        ]

        with patch(
            "alibi.enrichment.product_matcher.match_products_batch"
        ) as mock_match:
            mock_match.return_value = [
                ProductMatchResult(
                    product_a_idx=0,
                    product_b_idx=1,
                    confidence=0.9,
                    reasoning="Same product",
                    suggested_canonical="Milk 1L",
                )
            ]
            results = find_cross_vendor_matches(mock_db)

        assert len(results) == 1
        assert results[0].canonical_name == "Milk 1L"
        assert len(results[0].products) == 2

    def test_out_of_bounds_index_skipped(self, mock_db: MagicMock) -> None:
        mock_db.fetchall.return_value = [
            ("id1", "Milk", None, "Dairy", None, None, None, None, "vk1", "Shop A"),
            ("id2", "Bread", None, "Bakery", None, None, None, None, "vk2", "Shop B"),
        ]

        with patch(
            "alibi.enrichment.product_matcher.match_products_batch"
        ) as mock_match:
            mock_match.return_value = [
                ProductMatchResult(
                    product_a_idx=0,
                    product_b_idx=99,
                    confidence=0.9,
                    reasoning="Bad index",
                )
            ]
            results = find_cross_vendor_matches(mock_db)

        assert results == []

    def test_category_filter(self, mock_db: MagicMock) -> None:
        mock_db.fetchall.return_value = []
        find_cross_vendor_matches(mock_db, category="Dairy")
        call_args = mock_db.fetchall.call_args
        assert "fi.category = ?" in call_args[0][0]
