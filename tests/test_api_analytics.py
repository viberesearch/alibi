"""Tests for the analytics API endpoints."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from alibi.api.app import create_app
from alibi.api.deps import get_database
from alibi.db.connection import DatabaseManager


@pytest.fixture
def client(db_manager: DatabaseManager) -> Generator[TestClient, None, None]:
    """Create a test client with database dependency override."""
    app = create_app()

    def override_get_database() -> DatabaseManager:
        return db_manager

    app.dependency_overrides[get_database] = override_get_database
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def seeded_db(db_manager: DatabaseManager) -> DatabaseManager:
    """Seed the database with facts for analytics tests."""
    conn = db_manager.get_connection()
    conn.execute("INSERT INTO users (id, name) VALUES (?, ?)", ("user-1", "Test User"))
    conn.execute(
        "INSERT INTO spaces (id, name, type, owner_id) VALUES (?, ?, ?, ?)",
        ("space-1", "Default", "private", "user-1"),
    )
    for i in range(5):
        cloud_id = f"cloud-{i}"
        conn.execute(
            "INSERT INTO clouds (id, status) VALUES (?, 'collapsed')", (cloud_id,)
        )
        conn.execute(
            """INSERT INTO facts (id, cloud_id, fact_type, vendor, vendor_key,
               total_amount, currency, event_date, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f"fact-{i}",
                cloud_id,
                "purchase",
                f"Vendor {i % 3}",
                f"key-{i % 3}",
                f"{(i + 1) * 10}.00",
                "EUR",
                f"2025-0{i + 1}-20",
                "confirmed",
            ),
        )
    conn.commit()
    return db_manager


class TestSpending:
    """Tests for GET /api/v1/analytics/spending."""

    def test_spending_by_month(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/analytics/spending?period=month")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_spending_by_vendor(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/analytics/spending?period=vendor")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_spending_with_dates(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get(
            "/api/v1/analytics/spending?date_from=2025-01-01&date_to=2025-06-01"
        )
        assert resp.status_code == 200

    def test_spending_invalid_period(self, client: TestClient) -> None:
        resp = client.get("/api/v1/analytics/spending?period=invalid")
        assert resp.status_code == 422


class TestSubscriptions:
    """Tests for GET /api/v1/analytics/subscriptions."""

    def test_subscriptions(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/analytics/subscriptions")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)


class TestAnomalies:
    """Tests for GET /api/v1/analytics/anomalies."""

    def test_anomalies(self, client: TestClient, seeded_db: DatabaseManager) -> None:
        resp = client.get("/api/v1/analytics/anomalies")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_anomalies_with_params(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get(
            "/api/v1/analytics/anomalies?lookback_days=30&std_threshold=1.5"
        )
        assert resp.status_code == 200


class TestVendors:
    """Tests for GET /api/v1/analytics/vendors."""

    def test_vendors(self, client: TestClient, seeded_db: DatabaseManager) -> None:
        resp = client.get("/api/v1/analytics/vendors")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_vendors" in data
        assert "aliases" in data
