"""Tests for the identities API endpoints."""

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
    """Seed the database with identity test data."""
    conn = db_manager.get_connection()
    conn.execute("INSERT INTO users (id, name) VALUES (?, ?)", ("user-1", "Test User"))
    conn.execute(
        "INSERT INTO spaces (id, name, type, owner_id) VALUES (?, ?, ?, ?)",
        ("space-1", "Default", "private", "user-1"),
    )
    conn.execute(
        "INSERT INTO identities (id, entity_type, canonical_name) VALUES (?, ?, ?)",
        ("id-1", "vendor", "Fresko"),
    )
    conn.execute(
        "INSERT INTO identities (id, entity_type, canonical_name) VALUES (?, ?, ?)",
        ("id-2", "vendor", "Alphamega"),
    )
    conn.execute(
        "INSERT INTO identity_members (id, identity_id, member_type, value)"
        " VALUES (?, ?, ?, ?)",
        ("m-1", "id-1", "name", "Fresko"),
    )
    conn.execute(
        "INSERT INTO identity_members (id, identity_id, member_type, value)"
        " VALUES (?, ?, ?, ?)",
        ("m-2", "id-1", "vendor_key", "10057000Y"),
    )
    conn.execute(
        "INSERT INTO identity_members (id, identity_id, member_type, value)"
        " VALUES (?, ?, ?, ?)",
        ("m-3", "id-2", "name", "Alphamega"),
    )
    conn.commit()
    return db_manager


class TestListIdentities:
    """Tests for GET /api/v1/identities."""

    def test_list_identities(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/identities")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_list_by_type(self, client: TestClient, seeded_db: DatabaseManager) -> None:
        resp = client.get("/api/v1/identities?entity_type=vendor")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_list_empty_type(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/identities?entity_type=item")
        assert resp.status_code == 200
        assert resp.json() == []


class TestGetIdentity:
    """Tests for GET /api/v1/identities/{identity_id}."""

    def test_get_identity(self, client: TestClient, seeded_db: DatabaseManager) -> None:
        resp = client.get("/api/v1/identities/id-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "id-1"
        assert data["canonical_name"] == "Fresko"
        assert len(data["members"]) == 2

    def test_get_not_found(self, client: TestClient) -> None:
        resp = client.get("/api/v1/identities/bogus")
        assert resp.status_code == 404


class TestMergeIdentities:
    """Tests for POST /api/v1/identities/merge."""

    def test_merge(self, client: TestClient, seeded_db: DatabaseManager) -> None:
        resp = client.post(
            "/api/v1/identities/merge",
            json={"identity_id_a": "id-1", "identity_id_b": "id-2"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["merged_into"] == "id-1"

        # id-2 should be gone
        assert client.get("/api/v1/identities/id-2").status_code == 404

        # id-1 should have id-2's members
        merged = client.get("/api/v1/identities/id-1").json()
        values = {m["value"] for m in merged["members"]}
        assert "Alphamega" in values

    def test_merge_nonexistent(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/identities/merge",
            json={"identity_id_a": "bogus-a", "identity_id_b": "bogus-b"},
        )
        assert resp.status_code == 404


class TestResolveIdentity:
    """Tests for GET /api/v1/identities/resolve."""

    def test_resolve_by_name(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/identities/resolve?vendor_name=Fresko")
        assert resp.status_code == 200
        assert resp.json()["canonical_name"] == "Fresko"

    def test_resolve_not_found(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/identities/resolve?vendor_name=Unknown")
        assert resp.status_code == 404

    def test_resolve_no_params(self, client: TestClient) -> None:
        resp = client.get("/api/v1/identities/resolve")
        assert resp.status_code == 422
