"""Tests for enrichment analytics service functions and API endpoints.

Covers get_enrichment_trends, get_vendor_coverage (service layer) and the
GET /api/v1/enrichment/analytics/trends and
GET /api/v1/enrichment/analytics/coverage endpoints.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import date
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from alibi.api.app import create_app
from alibi.api.deps import get_database
from alibi.db import v2_store
from alibi.db.connection import DatabaseManager
from alibi.db.models import (
    CloudStatus,
    Fact,
    FactStatus,
    FactType,
)
from alibi.services import enrichment_review as svc


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _make_cloud(db: DatabaseManager) -> str:
    cloud_id = str(uuid.uuid4())
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO clouds (id, status, confidence) VALUES (?, ?, ?)",
            (cloud_id, CloudStatus.COLLAPSED.value, 1.0),
        )
    return cloud_id


def _make_fact(
    db: DatabaseManager,
    vendor: str = "Test Vendor",
    vendor_key: str | None = None,
    event_date: date = date(2026, 1, 15),
) -> str:
    """Create a cloud + fact and return the fact_id.

    Each fact needs its own cloud because of UNIQUE INDEX idx_facts_cloud_unique
    on facts(cloud_id) — one fact per cloud is enforced at the DB level.
    """
    cloud_id = _make_cloud(db)
    fact_id = str(uuid.uuid4())
    fact = Fact(
        id=fact_id,
        cloud_id=cloud_id,
        fact_type=FactType.PURCHASE,
        vendor=vendor,
        vendor_key=vendor_key,
        total_amount=Decimal("10.00"),
        currency="EUR",
        event_date=event_date,
        status=FactStatus.CONFIRMED,
    )
    v2_store.store_fact(db, fact, [])
    return fact_id


def _make_doc_and_atom(db: DatabaseManager) -> str:
    doc_id = str(uuid.uuid4())
    atom_id = str(uuid.uuid4())
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO documents (id, file_path, file_hash) VALUES (?, ?, ?)",
            (doc_id, f"/tmp/test-{doc_id[:8]}.jpg", doc_id),
        )
        cursor.execute(
            "INSERT INTO atoms (id, document_id, atom_type, data) VALUES (?, ?, ?, ?)",
            (atom_id, doc_id, "item", "{}"),
        )
    return atom_id


def _make_fact_item(
    db: DatabaseManager,
    fact_id: str,
    name: str = "Olive Oil",
    brand: str | None = "BestBrand",
    category: str | None = "oils",
    enrichment_source: str | None = "openfoodfacts",
    enrichment_confidence: float | None = 0.9,
) -> str:
    atom_id = _make_doc_and_atom(db)
    item_id = str(uuid.uuid4())
    with db.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO fact_items
                (id, fact_id, atom_id, name, brand, category,
                 enrichment_source, enrichment_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                fact_id,
                atom_id,
                name,
                brand,
                category,
                enrichment_source,
                enrichment_confidence,
            ),
        )
    return item_id


# ---------------------------------------------------------------------------
# API client fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client(db_manager: DatabaseManager) -> Generator[TestClient, None, None]:
    """Test client with database dependency override (no auth required)."""
    app = create_app()

    def override_get_database() -> DatabaseManager:
        return db_manager

    app.dependency_overrides[get_database] = override_get_database
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# TestGetEnrichmentTrends — service layer
# ---------------------------------------------------------------------------


class TestGetEnrichmentTrends:
    def test_empty_db_returns_empty_periods(self, db_manager: DatabaseManager) -> None:
        result = svc.get_enrichment_trends(db_manager)
        assert result == {"periods": []}

    def test_groups_by_month_by_default(self, db_manager: DatabaseManager) -> None:
        fact_jan = _make_fact(db_manager, event_date=date(2026, 1, 10))
        fact_feb = _make_fact(db_manager, event_date=date(2026, 2, 5))
        _make_fact_item(db_manager, fact_jan, enrichment_source="openfoodfacts")
        _make_fact_item(db_manager, fact_feb, enrichment_source="openfoodfacts")

        result = svc.get_enrichment_trends(db_manager, period="month")
        period_labels = [p["period"] for p in result["periods"]]

        assert "2026-01" in period_labels
        assert "2026-02" in period_labels

    def test_groups_by_day(self, db_manager: DatabaseManager) -> None:
        fact_a = _make_fact(db_manager, event_date=date(2026, 1, 10))
        fact_b = _make_fact(db_manager, event_date=date(2026, 1, 20))
        _make_fact_item(db_manager, fact_a, enrichment_source="openfoodfacts")
        _make_fact_item(db_manager, fact_b, enrichment_source="llm_inference")

        result = svc.get_enrichment_trends(db_manager, period="day")
        period_labels = [p["period"] for p in result["periods"]]

        assert "2026-01-10" in period_labels
        assert "2026-01-20" in period_labels

    def test_filters_by_start_date(self, db_manager: DatabaseManager) -> None:
        fact_before = _make_fact(db_manager, event_date=date(2025, 12, 31))
        fact_after = _make_fact(db_manager, event_date=date(2026, 1, 15))
        _make_fact_item(db_manager, fact_before, enrichment_source="openfoodfacts")
        _make_fact_item(db_manager, fact_after, enrichment_source="openfoodfacts")

        result = svc.get_enrichment_trends(
            db_manager, start_date="2026-01-01", period="month"
        )
        period_labels = [p["period"] for p in result["periods"]]

        assert "2026-01" in period_labels
        assert "2025-12" not in period_labels

    def test_filters_by_end_date(self, db_manager: DatabaseManager) -> None:
        fact_within = _make_fact(db_manager, event_date=date(2026, 1, 5))
        fact_outside = _make_fact(db_manager, event_date=date(2026, 3, 1))
        _make_fact_item(db_manager, fact_within, enrichment_source="openfoodfacts")
        _make_fact_item(db_manager, fact_outside, enrichment_source="openfoodfacts")

        result = svc.get_enrichment_trends(
            db_manager, end_date="2026-01-31", period="month"
        )
        period_labels = [p["period"] for p in result["periods"]]

        assert "2026-01" in period_labels
        assert "2026-03" not in period_labels

    def test_filters_by_start_and_end_date(self, db_manager: DatabaseManager) -> None:
        fact_in = _make_fact(db_manager, event_date=date(2026, 2, 10))
        fact_before = _make_fact(db_manager, event_date=date(2025, 12, 1))
        fact_after = _make_fact(db_manager, event_date=date(2026, 4, 1))
        _make_fact_item(db_manager, fact_in, enrichment_source="openfoodfacts")
        _make_fact_item(db_manager, fact_before, enrichment_source="openfoodfacts")
        _make_fact_item(db_manager, fact_after, enrichment_source="openfoodfacts")

        result = svc.get_enrichment_trends(
            db_manager,
            start_date="2026-01-01",
            end_date="2026-03-31",
            period="month",
        )
        period_labels = [p["period"] for p in result["periods"]]

        assert "2026-02" in period_labels
        assert "2025-12" not in period_labels
        assert "2026-04" not in period_labels

    def test_groups_by_source_within_period(self, db_manager: DatabaseManager) -> None:
        fact_id = _make_fact(db_manager, event_date=date(2026, 1, 15))
        _make_fact_item(
            db_manager, fact_id, name="A", enrichment_source="openfoodfacts"
        )
        _make_fact_item(
            db_manager, fact_id, name="B", enrichment_source="openfoodfacts"
        )
        _make_fact_item(
            db_manager, fact_id, name="C", enrichment_source="llm_inference"
        )

        result = svc.get_enrichment_trends(db_manager, period="month")
        periods = result["periods"]

        jan = next(p for p in periods if p["period"] == "2026-01")
        assert jan["by_source"]["openfoodfacts"] == 2
        assert jan["by_source"]["llm_inference"] == 1

    def test_period_totals_sum_correctly(self, db_manager: DatabaseManager) -> None:
        fact_id = _make_fact(db_manager, event_date=date(2026, 1, 15))
        for _ in range(3):
            _make_fact_item(db_manager, fact_id, enrichment_source="openfoodfacts")
        for _ in range(2):
            _make_fact_item(db_manager, fact_id, enrichment_source="llm_inference")

        result = svc.get_enrichment_trends(db_manager, period="month")
        jan = next(p for p in result["periods"] if p["period"] == "2026-01")

        assert jan["total"] == 5
        source_sum = sum(jan["by_source"].values())
        assert source_sum == jan["total"]

    def test_excludes_unenriched_items(self, db_manager: DatabaseManager) -> None:
        fact_id = _make_fact(db_manager, event_date=date(2026, 1, 15))
        _make_fact_item(db_manager, fact_id, enrichment_source="openfoodfacts")
        # Unenriched — must not appear
        _make_fact_item(
            db_manager,
            fact_id,
            name="No Enrichment",
            enrichment_source=None,
            enrichment_confidence=None,
        )

        result = svc.get_enrichment_trends(db_manager, period="month")
        jan = next(p for p in result["periods"] if p["period"] == "2026-01")
        assert jan["total"] == 1

    def test_periods_ordered_ascending(self, db_manager: DatabaseManager) -> None:
        for month in [3, 1, 2]:
            fact_id = _make_fact(db_manager, event_date=date(2026, month, 10))
            _make_fact_item(db_manager, fact_id, enrichment_source="openfoodfacts")

        result = svc.get_enrichment_trends(db_manager, period="month")
        period_labels = [p["period"] for p in result["periods"]]
        assert period_labels == sorted(period_labels)


# ---------------------------------------------------------------------------
# TestGetVendorCoverage — service layer
# ---------------------------------------------------------------------------


class TestGetVendorCoverage:
    def test_empty_db_returns_empty_vendors(self, db_manager: DatabaseManager) -> None:
        result = svc.get_vendor_coverage(db_manager)
        assert result == {"vendors": []}

    def test_orders_by_total_items_descending(
        self, db_manager: DatabaseManager
    ) -> None:
        # Vendor A — 1 item
        fact_a = _make_fact(db_manager, vendor="Vendor A")
        _make_fact_item(db_manager, fact_a, enrichment_source="openfoodfacts")

        # Vendor B — 3 items
        fact_b = _make_fact(db_manager, vendor="Vendor B")
        for _ in range(3):
            _make_fact_item(db_manager, fact_b, enrichment_source="openfoodfacts")

        result = svc.get_vendor_coverage(db_manager)
        vendors = result["vendors"]

        assert vendors[0]["vendor"] == "Vendor B"
        assert vendors[1]["vendor"] == "Vendor A"

    def test_calculates_coverage_pct(self, db_manager: DatabaseManager) -> None:
        fact_id = _make_fact(db_manager, vendor="Market")

        # 2 enriched, 2 unenriched → 50%
        _make_fact_item(db_manager, fact_id, enrichment_source="openfoodfacts")
        _make_fact_item(db_manager, fact_id, enrichment_source="openfoodfacts")
        _make_fact_item(
            db_manager,
            fact_id,
            name="Unenriched A",
            enrichment_source=None,
            enrichment_confidence=None,
        )
        _make_fact_item(
            db_manager,
            fact_id,
            name="Unenriched B",
            enrichment_source=None,
            enrichment_confidence=None,
        )

        result = svc.get_vendor_coverage(db_manager)
        vendor = result["vendors"][0]

        assert vendor["total_items"] == 4
        assert vendor["enriched_items"] == 2
        assert vendor["coverage_pct"] == pytest.approx(50.0)

    def test_full_coverage_is_100_pct(self, db_manager: DatabaseManager) -> None:
        fact_id = _make_fact(db_manager, vendor="Fully Enriched")
        for _ in range(4):
            _make_fact_item(db_manager, fact_id, enrichment_source="openfoodfacts")

        result = svc.get_vendor_coverage(db_manager)
        vendor = result["vendors"][0]
        assert vendor["coverage_pct"] == pytest.approx(100.0)

    def test_zero_enriched_items_shows_zero_coverage(
        self, db_manager: DatabaseManager
    ) -> None:
        fact_id = _make_fact(db_manager, vendor="Bare Vendor")
        for _ in range(3):
            _make_fact_item(
                db_manager,
                fact_id,
                name="Item",
                enrichment_source=None,
                enrichment_confidence=None,
            )

        result = svc.get_vendor_coverage(db_manager)
        vendor = result["vendors"][0]
        assert vendor["enriched_items"] == 0
        assert vendor["coverage_pct"] == 0.0

    def test_includes_per_source_breakdown(self, db_manager: DatabaseManager) -> None:
        fact_id = _make_fact(db_manager, vendor="Mix Vendor")

        for _ in range(2):
            _make_fact_item(db_manager, fact_id, enrichment_source="openfoodfacts")
        _make_fact_item(db_manager, fact_id, enrichment_source="llm_inference")

        result = svc.get_vendor_coverage(db_manager)
        vendor = result["vendors"][0]

        assert vendor["sources"]["openfoodfacts"] == 2
        assert vendor["sources"]["llm_inference"] == 1

    def test_respects_limit_parameter(self, db_manager: DatabaseManager) -> None:
        for i in range(5):
            fact_id = _make_fact(db_manager, vendor=f"Vendor {i}")
            _make_fact_item(db_manager, fact_id, enrichment_source="openfoodfacts")

        result = svc.get_vendor_coverage(db_manager, limit=3)
        assert len(result["vendors"]) == 3

    def test_vendor_key_included_in_result(self, db_manager: DatabaseManager) -> None:
        fact_id = _make_fact(db_manager, vendor="VAT Vendor", vendor_key="CY12345678X")
        _make_fact_item(db_manager, fact_id, enrichment_source="openfoodfacts")

        result = svc.get_vendor_coverage(db_manager)
        vendor = result["vendors"][0]
        assert vendor["vendor_key"] == "CY12345678X"

    def test_multiple_facts_same_vendor_aggregated(
        self, db_manager: DatabaseManager
    ) -> None:
        """Items from multiple facts for the same vendor are combined."""
        fact1 = _make_fact(db_manager, vendor="Repeat Vendor", vendor_key="VAT999")
        fact2 = _make_fact(db_manager, vendor="Repeat Vendor", vendor_key="VAT999")
        _make_fact_item(db_manager, fact1, enrichment_source="openfoodfacts")
        _make_fact_item(db_manager, fact1, enrichment_source="openfoodfacts")
        _make_fact_item(
            db_manager, fact2, enrichment_source=None, enrichment_confidence=None
        )

        result = svc.get_vendor_coverage(db_manager)
        # Only one entry for this vendor
        matching = [v for v in result["vendors"] if v["vendor"] == "Repeat Vendor"]
        assert len(matching) == 1
        vendor = matching[0]
        assert vendor["total_items"] == 3
        assert vendor["enriched_items"] == 2


# ---------------------------------------------------------------------------
# TestTrendsEndpoint — API layer
# ---------------------------------------------------------------------------


class TestTrendsEndpoint:
    def test_returns_200(self, client: TestClient, db_manager: DatabaseManager) -> None:
        resp = client.get("/api/v1/enrichment/analytics/trends")
        assert resp.status_code == 200

    def test_empty_db_returns_empty_periods_list(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/enrichment/analytics/trends")
        assert resp.status_code == 200
        assert resp.json() == {"periods": []}

    def test_accepts_period_param_month(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        fact_id = _make_fact(db_manager, event_date=date(2026, 1, 10))
        _make_fact_item(db_manager, fact_id, enrichment_source="openfoodfacts")

        resp = client.get("/api/v1/enrichment/analytics/trends?period=month")
        assert resp.status_code == 200
        data = resp.json()
        assert "periods" in data
        period_labels = [p["period"] for p in data["periods"]]
        assert any(label.startswith("2026-01") for label in period_labels)

    def test_accepts_period_param_day(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        fact_id = _make_fact(db_manager, event_date=date(2026, 1, 22))
        _make_fact_item(db_manager, fact_id, enrichment_source="openfoodfacts")

        resp = client.get("/api/v1/enrichment/analytics/trends?period=day")
        assert resp.status_code == 200
        data = resp.json()
        period_labels = [p["period"] for p in data["periods"]]
        assert "2026-01-22" in period_labels

    def test_accepts_start_date_param(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        fact_old = _make_fact(db_manager, event_date=date(2025, 6, 1))
        fact_new = _make_fact(db_manager, event_date=date(2026, 1, 15))
        _make_fact_item(db_manager, fact_old, enrichment_source="openfoodfacts")
        _make_fact_item(db_manager, fact_new, enrichment_source="openfoodfacts")

        resp = client.get(
            "/api/v1/enrichment/analytics/trends" "?start_date=2026-01-01&period=month"
        )
        assert resp.status_code == 200
        data = resp.json()
        period_labels = [p["period"] for p in data["periods"]]
        assert "2025-06" not in period_labels
        assert "2026-01" in period_labels

    def test_accepts_end_date_param(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        fact_early = _make_fact(db_manager, event_date=date(2026, 1, 5))
        fact_late = _make_fact(db_manager, event_date=date(2026, 6, 1))
        _make_fact_item(db_manager, fact_early, enrichment_source="openfoodfacts")
        _make_fact_item(db_manager, fact_late, enrichment_source="openfoodfacts")

        resp = client.get(
            "/api/v1/enrichment/analytics/trends" "?end_date=2026-01-31&period=month"
        )
        assert resp.status_code == 200
        data = resp.json()
        period_labels = [p["period"] for p in data["periods"]]
        assert "2026-01" in period_labels
        assert "2026-06" not in period_labels

    def test_response_structure_per_period(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        fact_id = _make_fact(db_manager, event_date=date(2026, 2, 1))
        _make_fact_item(db_manager, fact_id, enrichment_source="openfoodfacts")

        resp = client.get("/api/v1/enrichment/analytics/trends?period=month")
        assert resp.status_code == 200
        periods = resp.json()["periods"]
        assert len(periods) >= 1
        period = periods[0]
        assert "period" in period
        assert "total" in period
        assert "by_source" in period


# ---------------------------------------------------------------------------
# TestCoverageEndpoint — API layer
# ---------------------------------------------------------------------------


class TestCoverageEndpoint:
    def test_returns_200(self, client: TestClient, db_manager: DatabaseManager) -> None:
        resp = client.get("/api/v1/enrichment/analytics/coverage")
        assert resp.status_code == 200

    def test_empty_db_returns_empty_vendors_list(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/enrichment/analytics/coverage")
        assert resp.status_code == 200
        assert resp.json() == {"vendors": []}

    def test_returns_vendor_coverage_data(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        fact_id = _make_fact(db_manager, vendor="Greenfields")
        _make_fact_item(db_manager, fact_id, enrichment_source="openfoodfacts")

        resp = client.get("/api/v1/enrichment/analytics/coverage")
        assert resp.status_code == 200
        data = resp.json()
        assert "vendors" in data
        assert len(data["vendors"]) == 1
        assert data["vendors"][0]["vendor"] == "Greenfields"

    def test_accepts_limit_param(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        for i in range(5):
            fact_id = _make_fact(db_manager, vendor=f"Vendor {i}")
            _make_fact_item(db_manager, fact_id, enrichment_source="openfoodfacts")

        resp = client.get("/api/v1/enrichment/analytics/coverage?limit=2")
        assert resp.status_code == 200
        assert len(resp.json()["vendors"]) == 2

    def test_response_structure_per_vendor(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        fact_id = _make_fact(db_manager, vendor="Freshco", vendor_key="CY00100200T")
        _make_fact_item(db_manager, fact_id, enrichment_source="openfoodfacts")
        _make_fact_item(
            db_manager,
            fact_id,
            name="Unenriched",
            enrichment_source=None,
            enrichment_confidence=None,
        )

        resp = client.get("/api/v1/enrichment/analytics/coverage")
        assert resp.status_code == 200
        vendor = resp.json()["vendors"][0]

        assert "vendor" in vendor
        assert "vendor_key" in vendor
        assert "total_items" in vendor
        assert "enriched_items" in vendor
        assert "coverage_pct" in vendor
        assert "sources" in vendor

    def test_coverage_pct_calculation(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        fact_id = _make_fact(db_manager, vendor="Precision Market")

        # 1 enriched, 3 unenriched → 25%
        _make_fact_item(db_manager, fact_id, enrichment_source="openfoodfacts")
        for _ in range(3):
            _make_fact_item(
                db_manager,
                fact_id,
                name="Unenriched",
                enrichment_source=None,
                enrichment_confidence=None,
            )

        resp = client.get("/api/v1/enrichment/analytics/coverage")
        assert resp.status_code == 200
        vendor = resp.json()["vendors"][0]
        assert vendor["coverage_pct"] == pytest.approx(25.0)

    def test_limit_default_returns_up_to_50(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        """Default limit of 50 is respected without explicit param."""
        for i in range(10):
            fact_id = _make_fact(db_manager, vendor=f"Vendor {i:02d}")
            _make_fact_item(db_manager, fact_id, enrichment_source="openfoodfacts")

        resp = client.get("/api/v1/enrichment/analytics/coverage")
        assert resp.status_code == 200
        # All 10 returned (well within default limit of 50)
        assert len(resp.json()["vendors"]) == 10
