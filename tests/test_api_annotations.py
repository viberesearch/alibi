"""Tests for annotation API endpoints."""

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
    """Seed database with a fact for annotation tests."""
    conn = db_manager.get_connection()
    conn.execute("INSERT INTO users (id, name) VALUES (?, ?)", ("user-1", "Test User"))
    conn.execute(
        "INSERT INTO spaces (id, name, type, owner_id) VALUES (?, ?, ?, ?)",
        ("space-1", "Default", "private", "user-1"),
    )
    cloud_id = "cloud-1"
    conn.execute("INSERT INTO clouds (id, status) VALUES (?, 'collapsed')", (cloud_id,))
    conn.execute(
        """INSERT INTO facts (id, cloud_id, fact_type, vendor, total_amount,
           currency, event_date, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "fact-1",
            cloud_id,
            "purchase",
            "Test Vendor",
            "25.00",
            "EUR",
            "2025-01-15",
            "confirmed",
        ),
    )
    conn.commit()
    return db_manager


class TestAnnotateFact:
    """POST /api/v1/annotations/facts/{fact_id}"""

    def test_annotate_fact(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.post(
            "/api/v1/annotations/facts/fact-1",
            json={
                "annotation_type": "person",
                "key": "bought_for",
                "value": "Maria",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data
        assert data["status"] == "created"

    def test_annotate_with_metadata(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.post(
            "/api/v1/annotations/facts/fact-1",
            json={
                "annotation_type": "project",
                "key": "project",
                "value": "Kitchen renovation",
                "metadata": {"budget_line": "materials"},
            },
        )
        assert resp.status_code == 201


class TestGetAnnotations:
    """GET /api/v1/annotations/facts/{fact_id}"""

    def test_get_annotations(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        client.post(
            "/api/v1/annotations/facts/fact-1",
            json={
                "annotation_type": "person",
                "key": "bought_for",
                "value": "Maria",
            },
        )
        resp = client.get("/api/v1/annotations/facts/fact-1")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["value"] == "Maria"

    def test_get_annotations_empty(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/annotations/facts/fact-1")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_filter_by_type(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        client.post(
            "/api/v1/annotations/facts/fact-1",
            json={"annotation_type": "person", "key": "for", "value": "Maria"},
        )
        client.post(
            "/api/v1/annotations/facts/fact-1",
            json={"annotation_type": "project", "key": "proj", "value": "Reno"},
        )
        resp = client.get("/api/v1/annotations/facts/fact-1?annotation_type=person")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["annotation_type"] == "person"


class TestUpdateAnnotation:
    """PUT /api/v1/annotations/{annotation_id}"""

    def test_update_annotation(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        create = client.post(
            "/api/v1/annotations/facts/fact-1",
            json={"annotation_type": "person", "key": "for", "value": "Maria"},
        )
        ann_id = create.json()["id"]

        resp = client.put(
            f"/api/v1/annotations/{ann_id}",
            json={"value": "Anna"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "updated"

        # Verify the update
        anns = client.get("/api/v1/annotations/facts/fact-1").json()
        assert anns[0]["value"] == "Anna"

    def test_update_nonexistent(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.put(
            "/api/v1/annotations/bogus-id",
            json={"value": "new"},
        )
        assert resp.status_code == 404


class TestDeleteAnnotation:
    """DELETE /api/v1/annotations/{annotation_id}"""

    def test_delete_annotation(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        create = client.post(
            "/api/v1/annotations/facts/fact-1",
            json={"annotation_type": "person", "key": "for", "value": "Maria"},
        )
        ann_id = create.json()["id"]

        resp = client.delete(f"/api/v1/annotations/{ann_id}")
        assert resp.status_code == 204

        remaining = client.get("/api/v1/annotations/facts/fact-1").json()
        assert remaining == []

    def test_delete_nonexistent(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.delete("/api/v1/annotations/bogus-id")
        assert resp.status_code == 404
