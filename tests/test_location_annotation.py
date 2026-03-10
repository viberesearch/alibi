"""Tests for location annotation service and cloud formation location scoring."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

import pytest

from alibi.services import correction as svc
from alibi.utils.map_url import haversine_distance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_fact(
    db,
    fact_id="fact-1",
    vendor="Test Shop",
    vendor_key="VAT123",
    amount=10.0,
    cloud_id="cloud-1",
):
    """Insert a minimal fact + cloud for testing."""
    with db.transaction() as cur:
        cur.execute(
            "INSERT OR IGNORE INTO clouds (id, status) VALUES (?, 'collapsed')",
            (cloud_id,),
        )
        cur.execute(
            "INSERT INTO facts (id, cloud_id, vendor, vendor_key, "
            "total_amount, currency, event_date, fact_type) "
            "VALUES (?, ?, ?, ?, ?, 'EUR', '2026-01-15', 'purchase')",
            (fact_id, cloud_id, vendor, vendor_key, amount),
        )


# ---------------------------------------------------------------------------
# set_fact_location
# ---------------------------------------------------------------------------


class TestSetFactLocation:
    def test_stores_location_annotation(self, db):
        _insert_fact(db)
        url = "https://www.google.com/maps/place/Test+Shop/@34.7724,32.4218,17z"
        result = svc.set_fact_location(db, "fact-1", url)

        assert result is not None
        assert result["lat"] == pytest.approx(34.7724)
        assert result["lng"] == pytest.approx(32.4218)
        assert result["place_name"] == "Test Shop"

    def test_returns_none_for_invalid_url(self, db):
        _insert_fact(db)
        result = svc.set_fact_location(db, "fact-1", "https://example.com")
        assert result is None

    def test_returns_none_for_nonexistent_fact(self, db):
        result = svc.set_fact_location(
            db, "no-such-fact", "https://www.google.com/maps/@34.77,32.42,17z"
        )
        assert result is None

    def test_updates_existing_location(self, db):
        _insert_fact(db)
        url1 = "https://www.google.com/maps/@34.7724,32.4218,17z"
        url2 = "https://www.google.com/maps/@35.1856,33.3823,17z"

        svc.set_fact_location(db, "fact-1", url1)
        svc.set_fact_location(db, "fact-1", url2)

        loc = svc.get_fact_location(db, "fact-1")
        assert loc is not None
        assert loc["lat"] == pytest.approx(35.1856)
        assert loc["lng"] == pytest.approx(33.3823)

        # Should still be only one annotation
        from alibi.annotations.store import get_annotations

        anns = get_annotations(
            db,
            target_type="fact",
            target_id="fact-1",
            annotation_type="location",
        )
        assert len(anns) == 1

    def test_stores_raw_url_in_metadata(self, db):
        _insert_fact(db)
        raw = "  https://www.google.com/maps/@34.77,32.42,17z?utm_source=share  "
        svc.set_fact_location(db, "fact-1", raw)

        from alibi.annotations.store import get_annotations

        anns = get_annotations(
            db,
            target_type="fact",
            target_id="fact-1",
            annotation_type="location",
        )
        assert len(anns) == 1
        meta = anns[0]["metadata"]
        assert meta["raw_url"] == raw.strip()
        # Clean URL should not have utm params
        assert "utm_source" not in anns[0]["value"]


# ---------------------------------------------------------------------------
# get_fact_location
# ---------------------------------------------------------------------------


class TestGetFactLocation:
    def test_returns_location(self, db):
        _insert_fact(db)
        svc.set_fact_location(
            db, "fact-1", "https://www.google.com/maps/place/Lidl/@34.6789,33.0456,17z"
        )

        loc = svc.get_fact_location(db, "fact-1")
        assert loc is not None
        assert loc["lat"] == pytest.approx(34.6789)
        assert loc["lng"] == pytest.approx(33.0456)
        assert loc["place_name"] == "Lidl"
        assert "annotation_id" in loc

    def test_returns_none_when_no_location(self, db):
        _insert_fact(db)
        assert svc.get_fact_location(db, "fact-1") is None

    def test_returns_none_for_nonexistent_fact(self, db):
        assert svc.get_fact_location(db, "no-such-fact") is None


# ---------------------------------------------------------------------------
# get_recent_vendor_locations
# ---------------------------------------------------------------------------


class TestGetRecentVendorLocations:
    def test_returns_recent_locations(self, db):
        _insert_fact(db, "fact-1", "Shop A", "VAT-A", cloud_id="c1")
        _insert_fact(db, "fact-2", "Shop B", "VAT-B", cloud_id="c2")
        svc.set_fact_location(
            db, "fact-1", "https://www.google.com/maps/@34.77,32.42,17z"
        )
        svc.set_fact_location(
            db, "fact-2", "https://www.google.com/maps/@35.18,33.38,17z"
        )

        locations = svc.get_recent_vendor_locations(db, limit=10)
        assert len(locations) == 2
        assert locations[0]["vendor_name"] in ("Shop A", "Shop B")
        assert locations[0]["lat"] is not None

    def test_deduplicates_same_vendor_same_location(self, db):
        _insert_fact(db, "fact-1", "Shop A", "VAT-A", cloud_id="c1")
        _insert_fact(db, "fact-2", "Shop A", "VAT-A", cloud_id="c2")
        url = "https://www.google.com/maps/@34.77,32.42,17z"
        svc.set_fact_location(db, "fact-1", url)
        svc.set_fact_location(db, "fact-2", url)

        locations = svc.get_recent_vendor_locations(db, limit=10)
        assert len(locations) == 1

    def test_empty_when_no_locations(self, db):
        assert svc.get_recent_vendor_locations(db) == []

    def test_respects_limit(self, db):
        for i in range(5):
            _insert_fact(
                db,
                f"fact-{i}",
                f"Shop {i}",
                f"VAT-{i}",
                cloud_id=f"c-{i}",
            )
            svc.set_fact_location(
                db,
                f"fact-{i}",
                f"https://www.google.com/maps/@{34+i*0.01},{32+i*0.01},17z",
            )

        locations = svc.get_recent_vendor_locations(db, limit=3)
        assert len(locations) == 3


# ---------------------------------------------------------------------------
# Cloud formation: _location_score
# ---------------------------------------------------------------------------


class TestLocationScore:
    """Test the location scoring function used in cloud formation."""

    def test_same_location_returns_one(self):
        from alibi.clouds.formation import _location_score

        score = _location_score(34.77, 32.42, 34.77, 32.42)
        assert score == Decimal("1")

    def test_nearby_100m_returns_one(self):
        from alibi.clouds.formation import _location_score

        # ~50m apart
        score = _location_score(34.7700, 32.4200, 34.7705, 32.4200)
        assert score == Decimal("1")

    def test_within_500m(self):
        from alibi.clouds.formation import _location_score

        # ~300m apart
        score = _location_score(34.7700, 32.4200, 34.7730, 32.4200)
        assert score == Decimal("0.8")

    def test_within_2km(self):
        from alibi.clouds.formation import _location_score

        # ~1.1km apart
        score = _location_score(34.7700, 32.4200, 34.7800, 32.4200)
        assert score == Decimal("0.5")

    def test_within_5km(self):
        from alibi.clouds.formation import _location_score

        # ~3.5km apart
        score = _location_score(34.7700, 32.4200, 34.8000, 32.4300)
        assert score == Decimal("0.2")

    def test_far_away_returns_zero(self):
        from alibi.clouds.formation import _location_score

        # ~50km apart
        score = _location_score(34.77, 32.42, 35.18, 33.38)
        assert score == Decimal("0")

    def test_missing_coords_returns_zero(self):
        from alibi.clouds.formation import _location_score

        assert _location_score(34.77, 32.42, None, None) == Decimal("0")
        assert _location_score(None, None, 34.77, 32.42) == Decimal("0")
        assert _location_score(None, None, None, None) == Decimal("0")
