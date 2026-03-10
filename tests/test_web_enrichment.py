"""Tests for enrichment review API endpoints used by the Web UI."""

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
    Cloud,
    CloudStatus,
    Fact,
    FactItem,
    FactStatus,
    FactType,
)


# ---------------------------------------------------------------------------
# Fixtures
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
# DB helpers (mirrors test_enrichment_review.py)
# ---------------------------------------------------------------------------


def _make_cloud(db: DatabaseManager) -> str:
    cloud_id = str(uuid.uuid4())
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO clouds (id, status, confidence) VALUES (?, ?, ?)",
            (cloud_id, CloudStatus.COLLAPSED.value, 1.0),
        )
    return cloud_id


def _make_fact(db: DatabaseManager, cloud_id: str, vendor: str = "Test Vendor") -> str:
    fact_id = str(uuid.uuid4())
    fact = Fact(
        id=fact_id,
        cloud_id=cloud_id,
        fact_type=FactType.PURCHASE,
        vendor=vendor,
        vendor_key=None,
        total_amount=Decimal("10.00"),
        currency="EUR",
        event_date=date(2024, 1, 1),
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
    enrichment_confidence: float | None = 0.5,
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
# GET /api/v1/enrichment/review
# ---------------------------------------------------------------------------


class TestGetReviewQueue:
    def test_returns_list(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/enrichment/review")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_returns_low_confidence_items(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)
        low_id = _make_fact_item(
            db_manager, fact_id, name="Cheap Oil", enrichment_confidence=0.3
        )
        # High confidence — should not appear
        _make_fact_item(
            db_manager, fact_id, name="Good Oil", enrichment_confidence=0.95
        )

        resp = client.get("/api/v1/enrichment/review?threshold=0.8")
        assert resp.status_code == 200
        data = resp.json()
        ids = [item["id"] for item in data]
        assert low_id in ids
        # High-confidence item absent
        assert all(item["enrichment_confidence"] < 0.8 for item in data)

    def test_includes_vendor_field(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id, vendor="SuperMart")
        _make_fact_item(db_manager, fact_id, enrichment_confidence=0.4)

        resp = client.get("/api/v1/enrichment/review")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert data[0]["vendor"] == "SuperMart"

    def test_respects_limit_param(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)
        for i in range(5):
            _make_fact_item(
                db_manager,
                fact_id,
                name=f"Item {i}",
                enrichment_confidence=0.1 + i * 0.05,
            )

        resp = client.get("/api/v1/enrichment/review?limit=2")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_empty_db_returns_empty_list(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/enrichment/review")
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# GET /api/v1/enrichment/stats
# ---------------------------------------------------------------------------


class TestGetEnrichmentStats:
    def test_returns_stats_dict(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/enrichment/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_enriched" in data
        assert "avg_confidence" in data
        assert "pending_review" in data
        assert "by_source" in data

    def test_empty_db_returns_zeros(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/enrichment/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_enriched"] == 0
        assert data["pending_review"] == 0
        assert data["by_source"] == []

    def test_counts_enriched_items(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)
        _make_fact_item(db_manager, fact_id, name="A", enrichment_confidence=0.9)
        _make_fact_item(db_manager, fact_id, name="B", enrichment_confidence=0.4)
        # Unenriched — should not count
        _make_fact_item(
            db_manager,
            fact_id,
            name="C",
            enrichment_source=None,
            enrichment_confidence=None,
        )

        resp = client.get("/api/v1/enrichment/stats")
        data = resp.json()
        assert data["total_enriched"] == 2

    def test_by_source_breakdown(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)
        for _ in range(2):
            _make_fact_item(
                db_manager,
                fact_id,
                enrichment_source="openfoodfacts",
                enrichment_confidence=0.9,
            )
        _make_fact_item(
            db_manager,
            fact_id,
            enrichment_source="llm_inference",
            enrichment_confidence=0.5,
        )

        resp = client.get("/api/v1/enrichment/stats")
        data = resp.json()
        by_source = {
            row["enrichment_source"]: row["count"] for row in data["by_source"]
        }
        assert by_source.get("openfoodfacts") == 2
        assert by_source.get("llm_inference") == 1


# ---------------------------------------------------------------------------
# POST /api/v1/enrichment/review/{id}/confirm
# ---------------------------------------------------------------------------


class TestConfirmEnrichment:
    def test_confirm_returns_confirmed_status(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)
        item_id = _make_fact_item(db_manager, fact_id, enrichment_confidence=0.5)

        resp = client.post(f"/api/v1/enrichment/review/{item_id}/confirm", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "confirmed"
        assert data["fact_item_id"] == item_id

    def test_confirm_sets_user_confirmed_source(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)
        item_id = _make_fact_item(db_manager, fact_id, enrichment_confidence=0.4)

        client.post(f"/api/v1/enrichment/review/{item_id}/confirm", json={})

        row = db_manager.fetchone(
            "SELECT enrichment_source, enrichment_confidence FROM fact_items WHERE id = ?",
            (item_id,),
        )
        assert row["enrichment_source"] == "user_confirmed"
        assert row["enrichment_confidence"] == pytest.approx(1.0)

    def test_confirm_updates_brand_and_category(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)
        item_id = _make_fact_item(
            db_manager,
            fact_id,
            brand="OldBrand",
            category="old_cat",
            enrichment_confidence=0.4,
        )

        resp = client.post(
            f"/api/v1/enrichment/review/{item_id}/confirm",
            json={"brand": "NewBrand", "category": "new_cat"},
        )
        assert resp.status_code == 200

        row = db_manager.fetchone(
            "SELECT brand, category FROM fact_items WHERE id = ?", (item_id,)
        )
        assert row["brand"] == "NewBrand"
        assert row["category"] == "New_Cat"

    def test_confirm_partial_body_only_brand(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)
        item_id = _make_fact_item(
            db_manager,
            fact_id,
            brand="Original",
            category="keep_me",
            enrichment_confidence=0.3,
        )

        resp = client.post(
            f"/api/v1/enrichment/review/{item_id}/confirm",
            json={"brand": "Updated"},
        )
        assert resp.status_code == 200

        row = db_manager.fetchone(
            "SELECT brand, category FROM fact_items WHERE id = ?", (item_id,)
        )
        assert row["brand"] == "Updated"
        # category unchanged when not in body
        assert row["category"] == "keep_me"

    def test_confirm_missing_item_returns_404(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        fake_id = str(uuid.uuid4())
        resp = client.post(f"/api/v1/enrichment/review/{fake_id}/confirm", json={})
        assert resp.status_code == 404

    def test_confirmed_item_removed_from_queue(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)
        item_id = _make_fact_item(db_manager, fact_id, enrichment_confidence=0.5)

        client.post(f"/api/v1/enrichment/review/{item_id}/confirm", json={})

        resp = client.get("/api/v1/enrichment/review?threshold=0.8")
        ids = [item["id"] for item in resp.json()]
        assert item_id not in ids


# ---------------------------------------------------------------------------
# POST /api/v1/enrichment/review/{id}/reject
# ---------------------------------------------------------------------------


class TestRejectEnrichment:
    def test_reject_returns_rejected_status(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)
        item_id = _make_fact_item(db_manager, fact_id, enrichment_confidence=0.4)

        resp = client.post(f"/api/v1/enrichment/review/{item_id}/reject")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "rejected"
        assert data["fact_item_id"] == item_id

    def test_reject_clears_enrichment_fields(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)
        item_id = _make_fact_item(
            db_manager,
            fact_id,
            brand="SomeBrand",
            category="food",
            enrichment_source="openfoodfacts",
            enrichment_confidence=0.6,
        )

        client.post(f"/api/v1/enrichment/review/{item_id}/reject")

        row = db_manager.fetchone(
            "SELECT brand, category, enrichment_source, enrichment_confidence "
            "FROM fact_items WHERE id = ?",
            (item_id,),
        )
        assert row["brand"] is None
        assert row["category"] is None
        assert row["enrichment_source"] is None
        assert row["enrichment_confidence"] is None

    def test_reject_preserves_name(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)
        item_id = _make_fact_item(
            db_manager,
            fact_id,
            name="Special Olive Oil",
            enrichment_source="llm_inference",
            enrichment_confidence=0.3,
        )

        client.post(f"/api/v1/enrichment/review/{item_id}/reject")

        row = db_manager.fetchone(
            "SELECT name FROM fact_items WHERE id = ?", (item_id,)
        )
        assert row["name"] == "Special Olive Oil"

    def test_reject_missing_item_returns_404(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        fake_id = str(uuid.uuid4())
        resp = client.post(f"/api/v1/enrichment/review/{fake_id}/reject")
        assert resp.status_code == 404

    def test_rejected_item_not_in_queue(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        cloud_id = _make_cloud(db_manager)
        fact_id = _make_fact(db_manager, cloud_id)
        item_id = _make_fact_item(db_manager, fact_id, enrichment_confidence=0.4)

        client.post(f"/api/v1/enrichment/review/{item_id}/reject")

        resp = client.get("/api/v1/enrichment/review")
        ids = [item["id"] for item in resp.json()]
        assert item_id not in ids
