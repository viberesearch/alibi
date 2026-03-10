"""Tests for correction analytics (confusion matrix)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from alibi.analytics.corrections import (
    ConfusionMatrix,
    build_confusion_matrix,
    get_refinement_suggestions,
)


class TestBuildConfusionMatrix:
    def test_empty_db(self, mock_db: MagicMock) -> None:
        mock_db.fetchall.return_value = []
        result = build_confusion_matrix(mock_db)
        assert isinstance(result, ConfusionMatrix)
        assert result.total_corrections == 0
        assert result.category_confusions == []
        assert result.vendor_stats == []

    def test_counts_corrections(self, mock_db: MagicMock) -> None:
        # Query 1: user_confirmed items
        confirmed_rows = [
            ("id1", "Milk", "Brand", "Dairy", "user_confirmed", 1.0, "vk1", "Shop A"),
            ("id2", "Bread", "Brand", "Bakery", "user_confirmed", 1.0, "vk1", "Shop A"),
        ]
        # Query 2: low-confidence items
        low_conf_rows = [
            ("id3", "Cheese", "Dairy", "llm_inference", 0.5, "vk2", "Shop B"),
        ]
        # Query 3: category totals
        cat_totals = [("Dairy", 10), ("Bakery", 5)]

        mock_db.fetchall.side_effect = [
            confirmed_rows,
            low_conf_rows,
            cat_totals,
        ]

        result = build_confusion_matrix(mock_db, min_count=1)
        assert result.total_corrections == 3

    def test_category_confusion_detected(self, mock_db: MagicMock) -> None:
        # Same vendor with different categories = confusion
        confirmed_rows = [
            ("id1", "Item A", "B", "Dairy", "user_confirmed", 1.0, "vk1", "Shop"),
            ("id2", "Item B", "B", "Bakery", "user_confirmed", 1.0, "vk1", "Shop"),
            ("id3", "Item C", "B", "Dairy", "user_confirmed", 1.0, "vk1", "Shop"),
        ]
        low_conf_rows = []
        cat_totals = [("Dairy", 20), ("Bakery", 10)]

        mock_db.fetchall.side_effect = [
            confirmed_rows,
            low_conf_rows,
            cat_totals,
        ]

        result = build_confusion_matrix(mock_db, min_count=1)
        assert len(result.category_confusions) >= 1
        cats = {(c.original, c.corrected) for c in result.category_confusions}
        assert ("Dairy", "Bakery") in cats or ("Bakery", "Dairy") in cats

    def test_vendor_stats_grouped(self, mock_db: MagicMock) -> None:
        confirmed_rows = [
            ("id1", "Item", "B", "Cat", "user_confirmed", 1.0, "vk1", "Shop A"),
            ("id2", "Item", "B", "Cat", "user_confirmed", 1.0, "vk1", "Shop A"),
            ("id3", "Item", "B", "Cat", "user_confirmed", 1.0, "vk2", "Shop B"),
        ]
        low_conf_rows = []
        cat_totals = [("Cat", 50)]

        mock_db.fetchall.side_effect = [
            confirmed_rows,
            low_conf_rows,
            cat_totals,
        ]

        result = build_confusion_matrix(mock_db, min_count=1)
        assert len(result.vendor_stats) >= 1
        vk1 = next((v for v in result.vendor_stats if v.vendor_key == "vk1"), None)
        assert vk1 is not None
        assert vk1.total_corrections == 2

    def test_min_count_filters(self, mock_db: MagicMock) -> None:
        confirmed_rows = [
            ("id1", "Item", "B", "Cat", "user_confirmed", 1.0, "vk1", "Shop A"),
        ]
        low_conf_rows = []
        cat_totals = [("Cat", 50)]

        mock_db.fetchall.side_effect = [
            confirmed_rows,
            low_conf_rows,
            cat_totals,
        ]

        result = build_confusion_matrix(mock_db, min_count=5)
        assert result.vendor_stats == []

    def test_refinement_candidates(self, mock_db: MagicMock) -> None:
        # 5 corrections out of 10 total = 50% correction rate > 10% threshold
        confirmed_rows = [
            (
                f"id{i}",
                "Item",
                "B",
                "HighCorrection",
                "user_confirmed",
                1.0,
                "vk1",
                "Shop",
            )
            for i in range(5)
        ]
        low_conf_rows = []
        cat_totals = [("HighCorrection", 10)]

        mock_db.fetchall.side_effect = [
            confirmed_rows,
            low_conf_rows,
            cat_totals,
        ]

        result = build_confusion_matrix(mock_db, min_count=2)
        assert "HighCorrection" in result.refinement_candidates

    def test_top_corrected_fields(self, mock_db: MagicMock) -> None:
        confirmed_rows = [
            ("id1", "Item", "Brand1", "Dairy", "user_confirmed", 1.0, "vk1", "Shop"),
            ("id2", "Item", None, "Bakery", "user_confirmed", 1.0, "vk1", "Shop"),
        ]
        low_conf_rows = [
            ("id3", "Item", "Other", "cloud_refined", 0.5, "vk1", "Shop"),
        ]
        cat_totals = [("Dairy", 10), ("Bakery", 10), ("Other", 10)]

        mock_db.fetchall.side_effect = [
            confirmed_rows,
            low_conf_rows,
            cat_totals,
        ]

        result = build_confusion_matrix(mock_db, min_count=1)
        assert "category" in result.top_corrected_fields
        assert "low_confidence_enrichment" in result.top_corrected_fields


class TestGetRefinementSuggestions:
    def test_empty_db(self, mock_db: MagicMock) -> None:
        mock_db.fetchall.return_value = []
        result = get_refinement_suggestions(mock_db)
        assert result == []

    def test_suggestions_sorted_by_priority(self, mock_db: MagicMock) -> None:
        # 10 corrections for same vendor = high priority
        confirmed_rows = [
            (f"id{i}", "Item", "B", "Cat A", "user_confirmed", 1.0, "vk1", "Shop")
            for i in range(6)
        ] + [
            (f"idx{i}", "Item", "B", "Cat B", "user_confirmed", 1.0, "vk1", "Shop")
            for i in range(6)
        ]
        low_conf_rows = []
        cat_totals = [("Cat A", 10), ("Cat B", 10)]

        mock_db.fetchall.side_effect = [
            confirmed_rows,
            low_conf_rows,
            cat_totals,
        ]

        suggestions = get_refinement_suggestions(mock_db)
        assert len(suggestions) > 0
        assert suggestions[0]["priority"] == "high"

    def test_suggestion_types(self, mock_db: MagicMock) -> None:
        confirmed_rows = [
            (f"id{i}", "Item", "B", "Cat A", "user_confirmed", 1.0, "vk1", "Shop")
            for i in range(3)
        ] + [
            (f"idx{i}", "Item", "B", "Cat B", "user_confirmed", 1.0, "vk1", "Shop")
            for i in range(3)
        ]
        low_conf_rows = []
        cat_totals = [("Cat A", 5), ("Cat B", 5)]

        mock_db.fetchall.side_effect = [
            confirmed_rows,
            low_conf_rows,
            cat_totals,
        ]

        suggestions = get_refinement_suggestions(mock_db)
        types = {s["type"] for s in suggestions}
        assert "category_confusion" in types or "vendor_corrections" in types
