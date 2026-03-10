"""Live API integration tests against http://127.0.0.1:3100.

Run only when the API server is up:
    uv run pytest tests/integration/test_live_api.py -v

Skip automatically when server is unreachable.
"""

from __future__ import annotations

import httpx
import pytest

from alibi.db.connection import DatabaseManager
from alibi.db.v2_store import cleanup_document
from alibi.services.auth import create_api_key, create_user, revoke_api_key

BASE_URL = "http://127.0.0.1:3100"


def _api_available() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


api_available = pytest.mark.skipif(
    not _api_available(),
    reason="API server not running at http://127.0.0.1:3100",
)


@pytest.fixture(scope="session")
def db() -> DatabaseManager:
    manager = DatabaseManager()
    if not manager.is_initialized():
        manager.initialize()
    return manager


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@pytest.mark.integration
@api_available
def test_health_schema_version():
    r = httpx.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    db_block = body["database"]
    assert db_block["initialized"] is True
    assert db_block["schema_version"] == 26
    # Counts present (may be 0 for a fresh DB)
    for key in ("documents", "facts"):
        assert key in db_block, f"Missing key '{key}' in database block"
        assert isinstance(db_block[key], int)


# ---------------------------------------------------------------------------
# Facts list — pagination
# ---------------------------------------------------------------------------


@pytest.mark.integration
@api_available
def test_facts_list_pagination():
    r = httpx.get(f"{BASE_URL}/api/v1/facts", params={"page": 1, "per_page": 5})
    assert r.status_code == 200
    body = r.json()
    for key in ("items", "total", "page", "per_page", "pages"):
        assert key in body, f"Missing pagination key '{key}'"
    assert body["page"] == 1
    assert body["per_page"] == 5
    assert isinstance(body["items"], list)
    assert len(body["items"]) <= 5
    assert isinstance(body["total"], int)
    assert isinstance(body["pages"], int)


# ---------------------------------------------------------------------------
# Single-user mode — no auth header required
# ---------------------------------------------------------------------------


@pytest.mark.integration
@api_available
def test_single_user_mode_no_header():
    # No X-API-Key header; should succeed in single-user mode
    r = httpx.get(f"{BASE_URL}/api/v1/facts")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# API key lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.integration
@api_available
def test_api_key_lifecycle(db: DatabaseManager):
    user = create_user(db, name="integration-test-user")
    user_id = user["id"]

    try:
        # Generate key
        key_info = create_api_key(db, user_id, label="live-test")
        mnemonic = key_info["mnemonic"]
        key_id = key_info["id"]

        # Valid key grants access
        r = httpx.get(
            f"{BASE_URL}/api/v1/facts",
            headers={"X-API-Key": mnemonic},
        )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"

        # Invalid key is rejected
        r_bad = httpx.get(
            f"{BASE_URL}/api/v1/facts",
            headers={"X-API-Key": "totally wrong invalid key value"},
        )
        assert r_bad.status_code == 401

        # Revoke the key
        revoked = revoke_api_key(db, key_id)
        assert revoked is True

        # Revoked key now returns 401
        r_revoked = httpx.get(
            f"{BASE_URL}/api/v1/facts",
            headers={"X-API-Key": mnemonic},
        )
        assert r_revoked.status_code == 401

    finally:
        # Clean up user and any associated keys
        db.execute("DELETE FROM api_keys WHERE user_id = ?", (user_id,))
        db.execute("DELETE FROM users WHERE id = ?", (user_id,))
        db.get_connection().commit()


# ---------------------------------------------------------------------------
# Document upload + provenance
# ---------------------------------------------------------------------------


@pytest.mark.integration
@api_available
def test_upload_provenance(db: DatabaseManager):
    """Upload a real receipt and verify source='api' on the document row."""
    from pathlib import Path

    import os

    inbox = os.environ.get("ALIBI_TEST_INBOX", "./tests/fixtures/inbox")
    test_image = Path(inbox) / "receipts" / "fresko" / "IMG_0430 Medium.jpeg"
    if not test_image.exists():
        pytest.skip("Test image not available")

    image_data = test_image.read_bytes()
    r = httpx.post(
        f"{BASE_URL}/api/v1/process",
        files={"file": ("test_receipt.jpeg", image_data, "image/jpeg")},
        timeout=180,
    )
    assert r.status_code == 200, f"Upload failed: {r.status_code} {r.text}"
    body = r.json()

    # Connection refused = Ollama not reachable from API server
    if body.get("error") and "Connection refused" in body["error"]:
        pytest.skip("Ollama not reachable from API server")

    # Duplicate detection fires before document creation
    if body.get("is_duplicate"):
        doc_id = body.get("duplicate_of")
        assert doc_id, "Duplicate detected but no duplicate_of returned"
        return

    assert body["success"] is True, f"Processing failed: {body.get('error')}"
    doc_id = body.get("document_id")
    assert doc_id, "Response missing document_id"

    try:
        row = db.fetchone(
            "SELECT source, user_id FROM documents WHERE id = ?",
            (doc_id,),
        )
        assert row is not None, f"Document {doc_id} not found in DB"
        assert row["source"] == "api", f"Expected source='api', got '{row['source']}'"
        assert row["user_id"] is not None, "user_id should not be NULL"
    finally:
        cleanup_document(db, doc_id)
