"""Tests for location analytics."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from alibi.analytics.location import (
    LocationSpending,
    LocationSuggestion,
    VendorBranch,
    VendorBranchComparison,
    nearby_vendor_suggestions,
    spending_by_location,
    vendor_branch_comparison,
)


def _make_location_rows(
    facts: list[dict],
) -> list[tuple]:
    """Build mock rows for _load_location_facts query."""
    rows = []
    for f in facts:
        metadata = json.dumps(
            {
                "lat": f["lat"],
                "lng": f["lng"],
                "place_name": f.get("place_name"),
            }
        )
        rows.append(
            (
                f["fact_id"],
                f.get("event_type", "purchase"),
                f.get("total_amount", 10.0),
                f.get("event_date", "2026-01-15"),
                f.get("vendor_key", "vk1"),
                metadata,
                f.get("map_url", "https://maps.google.com/?q=0,0"),
            )
        )
    return rows


class TestSpendingByLocation:
    def test_empty_db(self, mock_db: MagicMock) -> None:
        mock_db.fetchall.return_value = []
        results = spending_by_location(mock_db)
        assert results == []

    def test_single_location(self, mock_db: MagicMock) -> None:
        mock_db.fetchall.return_value = _make_location_rows(
            [
                {"fact_id": "f1", "lat": 35.0, "lng": 33.0, "total_amount": 15.0},
                {"fact_id": "f2", "lat": 35.0, "lng": 33.0, "total_amount": 25.0},
            ]
        )
        results = spending_by_location(mock_db)
        assert len(results) == 1
        assert results[0].total_amount == pytest.approx(40.0)
        assert results[0].visit_count == 2

    def test_clustering(self, mock_db: MagicMock) -> None:
        """Locations within cluster radius are merged."""
        mock_db.fetchall.return_value = _make_location_rows(
            [
                {"fact_id": "f1", "lat": 35.0000, "lng": 33.0000, "total_amount": 10.0},
                {"fact_id": "f2", "lat": 35.0001, "lng": 33.0001, "total_amount": 20.0},
            ]
        )
        results = spending_by_location(mock_db, cluster_radius_m=500)
        assert len(results) == 1

    def test_separate_clusters(self, mock_db: MagicMock) -> None:
        """Distant locations are separate clusters."""
        mock_db.fetchall.return_value = _make_location_rows(
            [
                {"fact_id": "f1", "lat": 35.0, "lng": 33.0, "total_amount": 10.0},
                {"fact_id": "f2", "lat": 36.0, "lng": 34.0, "total_amount": 20.0},
            ]
        )
        results = spending_by_location(mock_db, cluster_radius_m=100)
        assert len(results) == 2

    def test_sorted_by_total_descending(self, mock_db: MagicMock) -> None:
        mock_db.fetchall.return_value = _make_location_rows(
            [
                {"fact_id": "f1", "lat": 35.0, "lng": 33.0, "total_amount": 10.0},
                {"fact_id": "f2", "lat": 36.0, "lng": 34.0, "total_amount": 50.0},
            ]
        )
        results = spending_by_location(mock_db, cluster_radius_m=100)
        assert results[0].total_amount >= results[1].total_amount

    def test_avg_amount_calculated(self, mock_db: MagicMock) -> None:
        mock_db.fetchall.return_value = _make_location_rows(
            [
                {"fact_id": "f1", "lat": 35.0, "lng": 33.0, "total_amount": 10.0},
                {"fact_id": "f2", "lat": 35.0, "lng": 33.0, "total_amount": 30.0},
            ]
        )
        results = spending_by_location(mock_db)
        assert results[0].avg_amount == pytest.approx(20.0)

    def test_vendors_collected(self, mock_db: MagicMock) -> None:
        mock_db.fetchall.return_value = _make_location_rows(
            [
                {"fact_id": "f1", "lat": 35.0, "lng": 33.0, "vendor_key": "vk1"},
                {"fact_id": "f2", "lat": 35.0, "lng": 33.0, "vendor_key": "vk2"},
            ]
        )
        results = spending_by_location(mock_db)
        assert len(results[0].vendors) == 2


class TestVendorBranchComparison:
    def test_empty_db(self, mock_db: MagicMock) -> None:
        mock_db.fetchall.return_value = []
        results = vendor_branch_comparison(mock_db)
        assert results == []

    def test_single_location_no_branches(self, mock_db: MagicMock) -> None:
        """Vendor with only one location has no comparison (unless filtered)."""
        mock_db.fetchall.return_value = _make_location_rows(
            [
                {"fact_id": "f1", "lat": 35.0, "lng": 33.0, "vendor_key": "vk1"},
            ]
        )
        mock_db.fetchone.return_value = ("Shop A",)
        results = vendor_branch_comparison(mock_db)
        assert results == []

    def test_two_branches_detected(self, mock_db: MagicMock) -> None:
        mock_db.fetchall.return_value = _make_location_rows(
            [
                {
                    "fact_id": "f1",
                    "lat": 35.0,
                    "lng": 33.0,
                    "vendor_key": "vk1",
                    "total_amount": 10.0,
                },
                {
                    "fact_id": "f2",
                    "lat": 36.0,
                    "lng": 34.0,
                    "vendor_key": "vk1",
                    "total_amount": 20.0,
                },
            ]
        )
        mock_db.fetchone.return_value = ("Shop A",)
        results = vendor_branch_comparison(mock_db)
        assert len(results) == 1
        assert results[0].branch_count == 2

    def test_vendor_key_filter(self, mock_db: MagicMock) -> None:
        mock_db.fetchall.return_value = _make_location_rows(
            [
                {"fact_id": "f1", "lat": 35.0, "lng": 33.0, "vendor_key": "vk1"},
                {"fact_id": "f2", "lat": 35.0, "lng": 33.0, "vendor_key": "vk2"},
            ]
        )
        mock_db.fetchone.return_value = ("Shop A",)
        results = vendor_branch_comparison(mock_db, vendor_key="vk1")
        # Only vk1 results
        for r in results:
            assert r.vendor_key == "vk1"

    def test_most_visited_branch(self, mock_db: MagicMock) -> None:
        mock_db.fetchall.return_value = _make_location_rows(
            [
                {"fact_id": "f1", "lat": 35.0, "lng": 33.0, "vendor_key": "vk1"},
                {"fact_id": "f2", "lat": 35.0, "lng": 33.0, "vendor_key": "vk1"},
                {"fact_id": "f3", "lat": 36.0, "lng": 34.0, "vendor_key": "vk1"},
            ]
        )
        mock_db.fetchone.return_value = ("Shop A",)
        results = vendor_branch_comparison(mock_db)
        assert len(results) == 1
        assert results[0].most_visited is not None
        assert results[0].most_visited.visit_count == 2


class TestNearbyVendorSuggestions:
    def test_empty_db(self, mock_db: MagicMock) -> None:
        mock_db.fetchall.return_value = []
        results = nearby_vendor_suggestions(mock_db, lat=35.0, lng=33.0)
        assert results == []

    def test_nearby_vendor_found(self, mock_db: MagicMock) -> None:
        mock_db.fetchall.return_value = _make_location_rows(
            [
                {
                    "fact_id": "f1",
                    "lat": 35.0,
                    "lng": 33.0,
                    "vendor_key": "vk1",
                    "total_amount": 20.0,
                },
            ]
        )
        mock_db.fetchone.return_value = ("Shop A",)

        results = nearby_vendor_suggestions(
            mock_db, lat=35.0001, lng=33.0001, radius_m=5000
        )
        assert len(results) == 1
        assert results[0].vendor_name == "Shop A"

    def test_distant_vendor_excluded(self, mock_db: MagicMock) -> None:
        mock_db.fetchall.return_value = _make_location_rows(
            [
                {"fact_id": "f1", "lat": 36.0, "lng": 34.0, "vendor_key": "vk1"},
            ]
        )
        mock_db.fetchone.return_value = ("Shop A",)

        results = nearby_vendor_suggestions(mock_db, lat=35.0, lng=33.0, radius_m=1000)
        assert results == []

    def test_sorted_by_distance(self, mock_db: MagicMock) -> None:
        mock_db.fetchall.return_value = _make_location_rows(
            [
                {"fact_id": "f1", "lat": 35.01, "lng": 33.0, "vendor_key": "vk1"},
                {"fact_id": "f2", "lat": 35.001, "lng": 33.0, "vendor_key": "vk2"},
            ]
        )
        mock_db.fetchone.side_effect = [("Far Shop",), ("Near Shop",)]

        results = nearby_vendor_suggestions(mock_db, lat=35.0, lng=33.0, radius_m=50000)
        assert len(results) == 2
        assert results[0].distance_meters < results[1].distance_meters

    def test_limit_respected(self, mock_db: MagicMock) -> None:
        facts = [
            {
                "fact_id": f"f{i}",
                "lat": 35.0 + i * 0.001,
                "lng": 33.0,
                "vendor_key": f"vk{i}",
            }
            for i in range(5)
        ]
        mock_db.fetchall.return_value = _make_location_rows(facts)
        mock_db.fetchone.side_effect = [(f"Shop {i}",) for i in range(5)]

        results = nearby_vendor_suggestions(
            mock_db, lat=35.0, lng=33.0, radius_m=50000, limit=2
        )
        assert len(results) == 2

    def test_reason_includes_visit_count(self, mock_db: MagicMock) -> None:
        mock_db.fetchall.return_value = _make_location_rows(
            [
                {"fact_id": "f1", "lat": 35.0, "lng": 33.0, "vendor_key": "vk1"},
                {"fact_id": "f2", "lat": 35.0, "lng": 33.0, "vendor_key": "vk1"},
            ]
        )
        mock_db.fetchone.return_value = ("Shop A",)

        results = nearby_vendor_suggestions(
            mock_db, lat=35.0001, lng=33.0001, radius_m=5000
        )
        assert "2 previous visits" in results[0].reason
