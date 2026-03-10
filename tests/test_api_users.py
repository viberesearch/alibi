"""Tests for user management API endpoints."""

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
    """Seed database with users for tests."""
    conn = db_manager.get_connection()
    conn.execute(
        "INSERT INTO users (id, name) VALUES (?, ?)",
        ("user-1", "Alice"),
    )
    conn.execute(
        "INSERT INTO users (id, name) VALUES (?, ?)",
        ("user-2", "Bob"),
    )
    conn.commit()
    return db_manager


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------


class TestListUsers:
    """GET /api/v1/users"""

    def test_list_users(self, client: TestClient, seeded_db: DatabaseManager) -> None:
        resp = client.get("/api/v1/users")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        # seeded_db creates 2 users, but conftest may seed a system user
        names = {u["name"] for u in data}
        assert "Alice" in names
        assert "Bob" in names

    def test_list_users_empty(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/users")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)


class TestCreateUser:
    """POST /api/v1/users"""

    def test_create_user_with_name(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        resp = client.post("/api/v1/users", json={"name": "Charlie"})
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Charlie"
        assert "id" in data

    def test_create_user_no_name(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        resp = client.post("/api/v1/users", json={})
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] is None
        assert "id" in data


class TestGetUser:
    """GET /api/v1/users/{user_id}"""

    def test_get_user(self, client: TestClient, seeded_db: DatabaseManager) -> None:
        resp = client.get("/api/v1/users/user-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Alice"
        assert "contacts" in data
        assert isinstance(data["contacts"], list)

    def test_get_user_not_found(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/users/nonexistent")
        assert resp.status_code == 404


class TestUpdateUser:
    """PATCH /api/v1/users/{user_id}"""

    def test_update_user_name(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.patch("/api/v1/users/user-1", json={"name": "Alicia"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "updated"

        # Verify
        get_resp = client.get("/api/v1/users/user-1")
        assert get_resp.json()["name"] == "Alicia"

    def test_update_user_clear_name(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.patch("/api/v1/users/user-1", json={"name": None})
        assert resp.status_code == 200

    def test_update_user_not_found(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        resp = client.patch("/api/v1/users/nonexistent", json={"name": "X"})
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Contact management
# ---------------------------------------------------------------------------


class TestListContacts:
    """GET /api/v1/users/{user_id}/contacts"""

    def test_list_contacts_empty(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/users/user-1/contacts")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_contacts(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        client.post(
            "/api/v1/users/user-1/contacts",
            json={"contact_type": "telegram", "value": "12345"},
        )
        resp = client.get("/api/v1/users/user-1/contacts")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["contact_type"] == "telegram"
        assert data[0]["value"] == "12345"

    def test_list_contacts_user_not_found(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/users/nonexistent/contacts")
        assert resp.status_code == 404


class TestAddContact:
    """POST /api/v1/users/{user_id}/contacts"""

    def test_add_telegram_contact(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.post(
            "/api/v1/users/user-1/contacts",
            json={"contact_type": "telegram", "value": "12345"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["contact_type"] == "telegram"
        assert data["value"] == "12345"
        assert "id" in data

    def test_add_email_contact_with_label(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.post(
            "/api/v1/users/user-1/contacts",
            json={
                "contact_type": "email",
                "value": "alice@example.com",
                "label": "work",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["label"] == "work"

    def test_add_contact_user_not_found(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        resp = client.post(
            "/api/v1/users/nonexistent/contacts",
            json={"contact_type": "telegram", "value": "12345"},
        )
        assert resp.status_code == 404

    def test_add_duplicate_contact(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        client.post(
            "/api/v1/users/user-1/contacts",
            json={"contact_type": "telegram", "value": "12345"},
        )
        # Same contact value on different user -> 409
        resp = client.post(
            "/api/v1/users/user-2/contacts",
            json={"contact_type": "telegram", "value": "12345"},
        )
        assert resp.status_code == 409

    def test_add_multiple_contacts(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        client.post(
            "/api/v1/users/user-1/contacts",
            json={"contact_type": "telegram", "value": "111"},
        )
        client.post(
            "/api/v1/users/user-1/contacts",
            json={"contact_type": "email", "value": "a@b.com"},
        )
        resp = client.get("/api/v1/users/user-1/contacts")
        assert len(resp.json()) == 2


class TestRemoveContact:
    """DELETE /api/v1/users/{user_id}/contacts/{contact_id}"""

    def test_remove_contact(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        create = client.post(
            "/api/v1/users/user-1/contacts",
            json={"contact_type": "telegram", "value": "12345"},
        )
        contact_id = create.json()["id"]

        resp = client.delete(f"/api/v1/users/user-1/contacts/{contact_id}")
        assert resp.status_code == 204

        # Verify removed
        contacts = client.get("/api/v1/users/user-1/contacts").json()
        assert contacts == []

    def test_remove_contact_not_found(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.delete("/api/v1/users/user-1/contacts/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# API key lifecycle
# ---------------------------------------------------------------------------


class TestListKeys:
    """GET /api/v1/users/{user_id}/keys"""

    def test_list_keys_empty(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/users/user-1/keys")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_keys_user_not_found(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/users/nonexistent/keys")
        assert resp.status_code == 404


class TestCreateKey:
    """POST /api/v1/users/{user_id}/keys"""

    def test_create_key(self, client: TestClient, seeded_db: DatabaseManager) -> None:
        resp = client.post("/api/v1/users/user-1/keys", json={})
        assert resp.status_code == 201
        data = resp.json()
        assert "mnemonic" in data
        assert "id" in data
        assert "prefix" in data
        assert data["label"] == "default"

    def test_create_key_custom_label(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.post("/api/v1/users/user-1/keys", json={"label": "mobile"})
        assert resp.status_code == 201
        assert resp.json()["label"] == "mobile"

    def test_create_key_user_not_found(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        resp = client.post("/api/v1/users/nonexistent/keys", json={})
        assert resp.status_code == 404

    def test_key_appears_in_list(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        create = client.post("/api/v1/users/user-1/keys", json={})
        key_id = create.json()["id"]

        keys = client.get("/api/v1/users/user-1/keys").json()
        assert len(keys) == 1
        assert keys[0]["id"] == key_id
        # Plaintext mnemonic NOT in list response
        assert "mnemonic" not in keys[0]
        # But prefix is available
        assert "key_prefix" in keys[0]


class TestRevokeKey:
    """DELETE /api/v1/users/{user_id}/keys/{key_id}"""

    def test_revoke_key(self, client: TestClient, seeded_db: DatabaseManager) -> None:
        create = client.post("/api/v1/users/user-1/keys", json={})
        key_id = create.json()["id"]

        resp = client.delete(f"/api/v1/users/user-1/keys/{key_id}")
        assert resp.status_code == 204

        # Key still in list but inactive
        keys = client.get("/api/v1/users/user-1/keys").json()
        revoked = [k for k in keys if k["id"] == key_id]
        assert len(revoked) == 1
        assert revoked[0]["is_active"] == 0

    def test_revoke_key_not_found(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.delete("/api/v1/users/user-1/keys/nonexistent")
        assert resp.status_code == 404

    def test_revoke_already_revoked(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        create = client.post("/api/v1/users/user-1/keys", json={})
        key_id = create.json()["id"]
        client.delete(f"/api/v1/users/user-1/keys/{key_id}")

        # Second revoke should 404 (already inactive)
        resp = client.delete(f"/api/v1/users/user-1/keys/{key_id}")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Full lifecycle
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    """End-to-end user management lifecycle."""

    def test_create_user_add_contacts_create_key(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        # Create user
        user = client.post("/api/v1/users", json={"name": "Eve"}).json()
        user_id = user["id"]

        # Add contacts
        client.post(
            f"/api/v1/users/{user_id}/contacts",
            json={"contact_type": "telegram", "value": "999"},
        )
        client.post(
            f"/api/v1/users/{user_id}/contacts",
            json={"contact_type": "email", "value": "eve@test.com"},
        )

        # Create API key
        key = client.post(f"/api/v1/users/{user_id}/keys", json={}).json()
        assert "mnemonic" in key

        # Get user with contacts
        detail = client.get(f"/api/v1/users/{user_id}").json()
        assert detail["name"] == "Eve"
        assert len(detail["contacts"]) == 2

        # List keys
        keys = client.get(f"/api/v1/users/{user_id}/keys").json()
        assert len(keys) == 1

        # Revoke key
        resp = client.delete(f"/api/v1/users/{user_id}/keys/{key['id']}")
        assert resp.status_code == 204

        # Remove a contact
        contact_id = detail["contacts"][0]["id"]
        resp = client.delete(f"/api/v1/users/{user_id}/contacts/{contact_id}")
        assert resp.status_code == 204

        # Verify final state
        detail = client.get(f"/api/v1/users/{user_id}").json()
        assert len(detail["contacts"]) == 1
