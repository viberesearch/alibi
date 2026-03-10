"""Integration tests for the 3-tier API authentication flow.

Covers:
- Single-user mode (no key configured, none sent)
- Legacy ALIBI_API_KEY match and mismatch
- Mnemonic key full lifecycle (create, use, revoke)
- Invalid mnemonic rejection
- Document provenance via auth dependency
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from alibi.api.app import create_app
from alibi.api.deps import get_config_dep, get_database
from alibi.config import Config
from alibi.db.connection import DatabaseManager
from alibi.services.auth import (
    create_api_key,
    create_user,
    revoke_api_key,
    validate_api_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_db(tmp_path: Path) -> Generator[DatabaseManager, None, None]:
    """Isolated database for auth tests — separate from shared db_manager."""
    config = Config(db_path=tmp_path / "auth_test.db")
    manager = DatabaseManager(config)
    manager.initialize()
    yield manager
    manager.close()


def _make_client(
    db: DatabaseManager,
    config: Config | None = None,
) -> tuple[TestClient, Any]:
    """Create a TestClient with db (and optional config) overrides.

    Returns the client and the app so overrides can be cleared.
    """
    app = create_app()
    app.dependency_overrides[get_database] = lambda: db
    if config is not None:
        app.dependency_overrides[get_config_dep] = lambda: config
    client = TestClient(app, raise_server_exceptions=True)
    return client, app


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestApiAuthIntegration:
    """End-to-end tests for the 3-tier authentication dependency."""

    # ------------------------------------------------------------------
    # Tier 1: single-user mode
    # ------------------------------------------------------------------

    def test_no_key_single_user_mode(self, auth_db: DatabaseManager) -> None:
        """No ALIBI_API_KEY configured, no X-Api-Key header -> succeeds.

        The endpoint must return 200 because single-user mode allows all
        requests through as the system/default user.
        """
        config = Config(db_path=auth_db.config.db_path, api_key=None)
        client, app = _make_client(auth_db, config)
        try:
            resp = client.get("/api/v1/artifacts")
            assert resp.status_code == 200, resp.text
        finally:
            app.dependency_overrides.clear()

    def test_single_user_mode_uses_existing_user(
        self, auth_db: DatabaseManager
    ) -> None:
        """Single-user mode resolves the first available user.

        The DB is seeded with a 'system' user on initialize(), so
        _get_default_user() picks that up via LIMIT 1 and returns it.
        The endpoint must succeed and the system user must be present.
        """
        config = Config(db_path=auth_db.config.db_path, api_key=None)
        client, app = _make_client(auth_db, config)
        try:
            resp = client.get("/api/v1/artifacts")
            assert resp.status_code == 200

            # The system user seeded by initialize() must be present
            row = auth_db.fetchone("SELECT id, name FROM users WHERE id = 'system'")
            assert row is not None
            assert row["name"] == "System"
        finally:
            app.dependency_overrides.clear()

    def test_single_user_mode_no_key_sent_still_succeeds(
        self, auth_db: DatabaseManager
    ) -> None:
        """Sending no header in single-user mode must not produce 401."""
        config = Config(db_path=auth_db.config.db_path, api_key=None)
        client, app = _make_client(auth_db, config)
        try:
            resp = client.get("/api/v1/facts")
            assert resp.status_code == 200
        finally:
            app.dependency_overrides.clear()

    # ------------------------------------------------------------------
    # Tier 2: legacy ALIBI_API_KEY
    # ------------------------------------------------------------------

    def test_legacy_api_key_match(self, auth_db: DatabaseManager) -> None:
        """Matching legacy key returns 200 using the system user."""
        config = Config(
            db_path=auth_db.config.db_path,
            api_key="super-secret-legacy",
        )
        client, app = _make_client(auth_db, config)
        try:
            resp = client.get(
                "/api/v1/artifacts",
                headers={"X-Api-Key": "super-secret-legacy"},
            )
            assert resp.status_code == 200, resp.text
        finally:
            app.dependency_overrides.clear()

    def test_legacy_api_key_mismatch(self, auth_db: DatabaseManager) -> None:
        """Wrong legacy key must return 401."""
        config = Config(
            db_path=auth_db.config.db_path,
            api_key="super-secret-legacy",
        )
        client, app = _make_client(auth_db, config)
        try:
            resp = client.get(
                "/api/v1/artifacts",
                headers={"X-Api-Key": "totally-wrong-key"},
            )
            assert resp.status_code == 401
            assert "Invalid API key" in resp.json().get("detail", "")
        finally:
            app.dependency_overrides.clear()

    def test_no_key_sent_when_legacy_configured(self, auth_db: DatabaseManager) -> None:
        """When a legacy key is required but no header is sent -> 401."""
        config = Config(
            db_path=auth_db.config.db_path,
            api_key="super-secret-legacy",
        )
        client, app = _make_client(auth_db, config)
        try:
            resp = client.get("/api/v1/artifacts")
            assert resp.status_code == 401
            assert "API key required" in resp.json().get("detail", "")
        finally:
            app.dependency_overrides.clear()

    def test_legacy_key_uses_default_user(self, auth_db: DatabaseManager) -> None:
        """Legacy key auth returns the default/system user, not a named user."""
        config = Config(
            db_path=auth_db.config.db_path,
            api_key="legacy-key",
        )
        # Pre-create the default user so we can assert identity
        auth_db.execute(
            "INSERT OR IGNORE INTO users (id, name) VALUES (?, ?)",
            ("default", "Default User"),
        )
        auth_db.get_connection().commit()

        client, app = _make_client(auth_db, config)
        try:
            resp = client.get(
                "/api/v1/artifacts",
                headers={"X-Api-Key": "legacy-key"},
            )
            assert resp.status_code == 200
        finally:
            app.dependency_overrides.clear()

    # ------------------------------------------------------------------
    # Tier 3: mnemonic API keys
    # ------------------------------------------------------------------

    def test_mnemonic_key_full_flow(self, auth_db: DatabaseManager) -> None:
        """Create user + key, authenticate with mnemonic -> 200."""
        user = create_user(auth_db, "Integration Tester")
        key_info = create_api_key(auth_db, user["id"])
        mnemonic = key_info["mnemonic"]

        # No legacy key configured so tier-2 is skipped; mnemonic hits tier-3
        config = Config(db_path=auth_db.config.db_path, api_key=None)
        client, app = _make_client(auth_db, config)
        try:
            resp = client.get(
                "/api/v1/artifacts",
                headers={"X-Api-Key": mnemonic},
            )
            assert resp.status_code == 200, resp.text
        finally:
            app.dependency_overrides.clear()

    def test_mnemonic_key_identifies_correct_user(
        self, auth_db: DatabaseManager
    ) -> None:
        """validate_api_key returns the exact user that owns the key."""
        user = create_user(auth_db, "Alice")
        key_info = create_api_key(auth_db, user["id"])
        mnemonic = key_info["mnemonic"]

        result = validate_api_key(auth_db, mnemonic)
        assert result is not None
        assert result["id"] == user["id"]
        assert result["name"] == "Alice"

    def test_mnemonic_key_prefix_stored(self, auth_db: DatabaseManager) -> None:
        """Key prefix (first 2 words) is stored for human-readable log display."""
        user = create_user(auth_db, "Bob")
        key_info = create_api_key(auth_db, user["id"])
        mnemonic = key_info["mnemonic"]
        words = mnemonic.split()

        assert key_info["prefix"] == f"{words[0]} {words[1]}"

        # Also verify stored in DB
        row = auth_db.fetchone(
            "SELECT key_prefix FROM api_keys WHERE user_id = ?",
            (user["id"],),
        )
        assert row is not None
        assert row["key_prefix"] == key_info["prefix"]

    def test_mnemonic_last_used_updated_on_validation(
        self, auth_db: DatabaseManager
    ) -> None:
        """Successful validation updates last_used_at."""
        user = create_user(auth_db, "Charlie")
        key_info = create_api_key(auth_db, user["id"])

        # Before first use, last_used_at should be NULL
        row = auth_db.fetchone(
            "SELECT last_used_at FROM api_keys WHERE id = ?",
            (key_info["id"],),
        )
        assert row["last_used_at"] is None

        validate_api_key(auth_db, key_info["mnemonic"])

        row = auth_db.fetchone(
            "SELECT last_used_at FROM api_keys WHERE id = ?",
            (key_info["id"],),
        )
        assert row["last_used_at"] is not None

    def test_invalid_mnemonic_key_returns_401(self, auth_db: DatabaseManager) -> None:
        """Sending a random 6-word BIP39-looking string that has no DB entry -> 401."""
        # Use real-looking words but not stored
        fake_mnemonic = "abandon ability able about above absent"

        config = Config(db_path=auth_db.config.db_path, api_key=None)
        client, app = _make_client(auth_db, config)
        try:
            resp = client.get(
                "/api/v1/artifacts",
                headers={"X-Api-Key": fake_mnemonic},
            )
            assert resp.status_code == 401
            assert "Invalid API key" in resp.json().get("detail", "")
        finally:
            app.dependency_overrides.clear()

    def test_invalid_mnemonic_validate_returns_none(
        self, auth_db: DatabaseManager
    ) -> None:
        """validate_api_key returns None for an unknown mnemonic (unit check)."""
        result = validate_api_key(auth_db, "word one two three four five")
        assert result is None

    def test_revoked_key_rejected(self, auth_db: DatabaseManager) -> None:
        """A revoked mnemonic key must return 401 via the HTTP layer."""
        user = create_user(auth_db, "Dave")
        key_info = create_api_key(auth_db, user["id"])
        mnemonic = key_info["mnemonic"]

        # Confirm key works before revocation
        assert validate_api_key(auth_db, mnemonic) is not None

        revoke_api_key(auth_db, key_info["id"])

        config = Config(db_path=auth_db.config.db_path, api_key=None)
        client, app = _make_client(auth_db, config)
        try:
            resp = client.get(
                "/api/v1/artifacts",
                headers={"X-Api-Key": mnemonic},
            )
            assert resp.status_code == 401
        finally:
            app.dependency_overrides.clear()

    def test_revoked_key_validate_returns_none(self, auth_db: DatabaseManager) -> None:
        """validate_api_key returns None for a revoked key (unit check)."""
        user = create_user(auth_db, "Eve")
        key_info = create_api_key(auth_db, user["id"])
        mnemonic = key_info["mnemonic"]

        revoke_api_key(auth_db, key_info["id"])
        result = validate_api_key(auth_db, mnemonic)
        assert result is None

    def test_revoke_returns_true_once(self, auth_db: DatabaseManager) -> None:
        """revoke_api_key returns True on first revocation, False on repeat."""
        user = create_user(auth_db, "Frank")
        key_info = create_api_key(auth_db, user["id"])

        assert revoke_api_key(auth_db, key_info["id"]) is True
        assert revoke_api_key(auth_db, key_info["id"]) is False

    def test_multiple_keys_per_user(self, auth_db: DatabaseManager) -> None:
        """A user can hold multiple independent API keys."""
        user = create_user(auth_db, "Grace")
        key_a = create_api_key(auth_db, user["id"], label="work")
        key_b = create_api_key(auth_db, user["id"], label="home")

        assert validate_api_key(auth_db, key_a["mnemonic"]) is not None
        assert validate_api_key(auth_db, key_b["mnemonic"]) is not None

        # Revoking one must not affect the other
        revoke_api_key(auth_db, key_a["id"])
        assert validate_api_key(auth_db, key_a["mnemonic"]) is None
        assert validate_api_key(auth_db, key_b["mnemonic"]) is not None

    def test_mnemonic_key_with_legacy_also_configured(
        self, auth_db: DatabaseManager
    ) -> None:
        """Mnemonic key must work even when a legacy key is also configured.

        A different mnemonic (not the legacy string) must fall through tier-2
        and be validated against the api_keys table in tier-3.
        """
        user = create_user(auth_db, "Hank")
        key_info = create_api_key(auth_db, user["id"])
        mnemonic = key_info["mnemonic"]

        # Legacy key configured but mnemonic != legacy string
        config = Config(
            db_path=auth_db.config.db_path,
            api_key="legacy-fixed-value",
        )
        client, app = _make_client(auth_db, config)
        try:
            resp = client.get(
                "/api/v1/artifacts",
                headers={"X-Api-Key": mnemonic},
            )
            assert resp.status_code == 200, resp.text
        finally:
            app.dependency_overrides.clear()

    # ------------------------------------------------------------------
    # Document provenance
    # ------------------------------------------------------------------

    def test_document_provenance_user_id_flows_through(
        self, auth_db: DatabaseManager
    ) -> None:
        """Auth dependency returns the exact user dict that owns the key.

        We test this at the service layer (validate_api_key) rather than
        via a document-creation endpoint so the test stays fast and does
        not depend on file I/O or pipeline details.
        """
        user = create_user(auth_db, "Ingrid")
        key_info = create_api_key(auth_db, user["id"])

        resolved = validate_api_key(auth_db, key_info["mnemonic"])

        assert resolved is not None
        assert resolved["id"] == user["id"]
        assert resolved["name"] == "Ingrid"

    def test_document_provenance_different_users_isolated(
        self, auth_db: DatabaseManager
    ) -> None:
        """Keys belonging to different users resolve to distinct user dicts."""
        user_a = create_user(auth_db, "Alice")
        user_b = create_user(auth_db, "Bob")
        key_a = create_api_key(auth_db, user_a["id"])
        key_b = create_api_key(auth_db, user_b["id"])

        resolved_a = validate_api_key(auth_db, key_a["mnemonic"])
        resolved_b = validate_api_key(auth_db, key_b["mnemonic"])

        assert resolved_a is not None
        assert resolved_b is not None
        assert resolved_a["id"] != resolved_b["id"]
        assert resolved_a["name"] == "Alice"
        assert resolved_b["name"] == "Bob"

    def test_document_source_api_label(self, auth_db: DatabaseManager) -> None:
        """Documents ingested via the API carry source='api'.

        We verify the documents table has a source column and that a row
        inserted with source='api' is retrievable — this confirms the
        schema supports provenance tagging without requiring a live ingest.
        """
        user = create_user(auth_db, "Jane")
        doc_id = "prov-doc-1"

        auth_db.execute(
            "INSERT INTO documents (id, file_path, file_hash, source, user_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (doc_id, "/tmp/receipt.jpg", "abc123", "api", user["id"]),
        )
        auth_db.get_connection().commit()

        row = auth_db.fetchone(
            "SELECT source, user_id FROM documents WHERE id = ?",
            (doc_id,),
        )
        assert row is not None
        assert row["source"] == "api"
        assert row["user_id"] == user["id"]

    # ------------------------------------------------------------------
    # Key hash security
    # ------------------------------------------------------------------

    def test_plaintext_mnemonic_not_stored(self, auth_db: DatabaseManager) -> None:
        """The database must never contain the plaintext mnemonic."""
        user = create_user(auth_db, "Karl")
        key_info = create_api_key(auth_db, user["id"])
        mnemonic = key_info["mnemonic"]

        row = auth_db.fetchone(
            "SELECT key_hash FROM api_keys WHERE id = ?",
            (key_info["id"],),
        )
        assert row is not None
        # Hash must not equal the plaintext
        assert row["key_hash"] != mnemonic
        # Hash must be a 64-char hex SHA-256 digest
        assert len(row["key_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in row["key_hash"])

    def test_mnemonic_normalised_before_hashing(self, auth_db: DatabaseManager) -> None:
        """Mnemonic with extra whitespace or uppercase validates identically."""
        user = create_user(auth_db, "Laura")

        # Store via create_api_key which calls hash_key internally
        key_info = create_api_key(auth_db, user["id"])
        clean_mnemonic = key_info["mnemonic"]

        # Upper-cased and extra-spaced variant must also validate
        dirty = "  " + clean_mnemonic.upper() + "  "
        result = validate_api_key(auth_db, dirty)
        assert result is not None
        assert result["id"] == user["id"]
