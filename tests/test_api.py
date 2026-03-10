"""Tests for the FastAPI headless layer."""

import uuid
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from alibi.api.app import create_app
from alibi.api.deps import get_database
from alibi.config import Config
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
    """Seed the database with v2 test data."""
    conn = db_manager.get_connection()

    # Create a user
    conn.execute(
        "INSERT INTO users (id, name) VALUES (?, ?)",
        ("user-1", "Test User"),
    )

    # Create a space
    conn.execute(
        "INSERT INTO spaces (id, name, type, owner_id) VALUES (?, ?, ?, ?)",
        ("space-1", "Default", "private", "user-1"),
    )

    # Create documents (v2)
    for i in range(3):
        conn.execute(
            """INSERT INTO documents (id, file_path, file_hash, created_at)
               VALUES (?, ?, ?, ?)""",
            (
                f"doc-{i}",
                f"/receipts/receipt_{i}.jpg",
                f"hash{i}",
                f"2025-0{i + 1}-15T10:00:00",
            ),
        )

    # Create clouds and facts (v2)
    for i in range(5):
        cloud_id = f"cloud-{i}"
        conn.execute(
            "INSERT INTO clouds (id, status) VALUES (?, 'collapsed')",
            (cloud_id,),
        )
        conn.execute(
            """INSERT INTO facts (id, cloud_id, fact_type, vendor, total_amount,
                                   currency, event_date, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f"fact-{i}",
                cloud_id,
                "purchase",
                f"Vendor {i % 3}",
                f"{(i + 1) * 10}.00",
                "EUR",
                f"2025-0{i + 1}-20",
                "confirmed",
            ),
        )

    # Create atoms for fact_items FK
    for i in range(3):
        conn.execute(
            """INSERT INTO atoms (id, document_id, atom_type, data)
               VALUES (?, ?, ?, ?)""",
            (f"atom-li-{i}", "doc-0", "item", "{}"),
        )

    # Create fact items (v2 line items)
    for i in range(3):
        conn.execute(
            """INSERT INTO fact_items (id, fact_id, atom_id, name, quantity,
                                       unit_price, total_price, category)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f"fi-{i}",
                "fact-0",
                f"atom-li-{i}",
                f"Item {i}",
                "1",
                f"{(i + 1) * 5}.00",
                f"{(i + 1) * 5}.00",
                "groceries" if i < 2 else "dairy",
            ),
        )

    # Create items (assets — items table stays in v2)
    conn.execute(
        """INSERT INTO items (id, space_id, name, category, purchase_price,
                              currency, status, warranty_expires)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "item-1",
            "space-1",
            "Laptop",
            "electronics",
            "1200.00",
            "EUR",
            "active",
            "2026-06-15",
        ),
    )
    conn.execute(
        """INSERT INTO items (id, space_id, name, category, purchase_price,
                              currency, status)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        ("item-2", "space-1", "Headphones", "electronics", "99.99", "EUR", "active"),
    )

    # Link document to cloud via bundles
    conn.execute(
        "INSERT INTO bundles (id, document_id, bundle_type) VALUES (?, ?, ?)",
        ("bundle-0", "doc-0", "basket"),
    )
    conn.execute(
        "INSERT INTO cloud_bundles (cloud_id, bundle_id, match_type) "
        "VALUES (?, ?, ?)",
        ("cloud-0", "bundle-0", "exact_amount"),
    )

    conn.commit()
    return db_manager


class TestHealth:
    """Tests for health endpoint."""

    def test_health_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert data["database"]["initialized"] is True

    def test_health_includes_stats(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/health")
        data = resp.json()
        assert "documents" in data["database"]


class TestArtifacts:
    """Tests for document endpoints."""

    def test_list_empty(self, client: TestClient) -> None:
        resp = client.get("/api/v1/artifacts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["documents"] == []

    def test_list_with_data(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/artifacts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3

    def test_get_artifact(self, client: TestClient, seeded_db: DatabaseManager) -> None:
        resp = client.get("/api/v1/artifacts/doc-0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "doc-0"
        assert data["file_path"] == "/receipts/receipt_0.jpg"

    def test_get_artifact_not_found(self, client: TestClient) -> None:
        resp = client.get("/api/v1/artifacts/nonexistent")
        assert resp.status_code == 404

    def test_create_artifact(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.post(
            "/api/v1/artifacts",
            json={
                "file_path": "/test/file.jpg",
                "file_hash": "abc123",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data
        assert data["status"] == "created"

    def test_delete_artifact(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.delete("/api/v1/artifacts/doc-0")
        assert resp.status_code == 204

        # Verify deleted
        resp = client.get("/api/v1/artifacts/doc-0")
        assert resp.status_code == 404

    def test_get_artifact_line_items(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/artifacts/doc-0/line-items")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3

    def test_pagination(self, client: TestClient, seeded_db: DatabaseManager) -> None:
        resp = client.get("/api/v1/artifacts?per_page=2&page=1")
        data = resp.json()
        assert len(data["documents"]) == 2
        assert data["total"] == 3


class TestLineItems:
    """Tests for fact item endpoints."""

    def test_list_line_items(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/line-items")
        data = resp.json()
        assert data["total"] == 3

    def test_filter_by_category(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/line-items?category=groceries")
        data = resp.json()
        assert data["total"] == 2

    def test_filter_by_name(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/line-items?name=Item 1")
        data = resp.json()
        assert data["total"] == 1

    def test_get_line_item(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/line-items/fi-0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Item 0"

    def test_get_line_item_not_found(self, client: TestClient) -> None:
        resp = client.get("/api/v1/line-items/nonexistent")
        assert resp.status_code == 404


class TestItems:
    """Tests for item (asset) endpoints."""

    def test_list_items(self, client: TestClient, seeded_db: DatabaseManager) -> None:
        resp = client.get("/api/v1/items")
        data = resp.json()
        assert data["total"] == 2

    def test_filter_by_category(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/items?category=electronics")
        data = resp.json()
        assert data["total"] == 2

    def test_get_item_with_relations(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/items/item-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Laptop"
        assert "documents" in data
        assert "facts" in data

    def test_create_item(self, client: TestClient, seeded_db: DatabaseManager) -> None:
        resp = client.post(
            "/api/v1/items",
            json={
                "space_id": "space-1",
                "name": "New Item",
                "category": "appliances",
                "purchase_price": "299.99",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data

    def test_update_item(self, client: TestClient, seeded_db: DatabaseManager) -> None:
        resp = client.patch(
            "/api/v1/items/item-1",
            json={
                "current_value": "800.00",
            },
        )
        assert resp.status_code == 200

    def test_delete_item(self, client: TestClient, seeded_db: DatabaseManager) -> None:
        resp = client.delete("/api/v1/items/item-1")
        assert resp.status_code == 204


class TestReports:
    """Tests for report endpoints."""

    def test_monthly_report(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/reports/monthly/2025/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["period"]["year"] == 2025
        assert data["period"]["month"] == 1
        assert "expenses" in data
        assert "income" in data
        assert "top_vendors" in data

    def test_monthly_report_december(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/reports/monthly/2025/12")
        assert resp.status_code == 200

    def test_spending_analysis(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/reports/spending")
        assert resp.status_code == 200
        data = resp.json()
        assert data["group_by"] == "month"
        assert "data" in data

    def test_spending_by_vendor(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/reports/spending?group_by=vendor")
        data = resp.json()
        assert data["group_by"] == "vendor"


class TestSearch:
    """Tests for search endpoint."""

    def test_search_documents(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/search?q=receipt")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] > 0

    def test_search_with_type_filter(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/search?q=Vendor&type=fact")
        data = resp.json()
        # Should only return facts
        for item in data["results"]:
            assert item["result_type"] == "fact"

    def test_search_items(self, client: TestClient, seeded_db: DatabaseManager) -> None:
        resp = client.get("/api/v1/search?q=Laptop&type=item")
        data = resp.json()
        assert data["total"] == 1

    def test_search_empty(self, client: TestClient, seeded_db: DatabaseManager) -> None:
        resp = client.get("/api/v1/search?q=nonexistent_xyz")
        data = resp.json()
        assert data["total"] == 0


class TestExport:
    """Tests for export endpoints."""

    def test_export_transactions_csv(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/export/transactions/csv")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/csv; charset=utf-8"
        content = resp.text
        lines = content.strip().split("\n")
        assert len(lines) == 6  # header + 5 facts

    def test_export_transactions_csv_filtered(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get(
            "/api/v1/export/transactions/csv?date_from=2025-03-01&date_to=2025-05-31"
        )
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        assert len(lines) == 4  # header + 3 facts

    def test_export_line_items_csv(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/export/line-items/csv")
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        assert len(lines) == 4  # header + 3 fact items

    def test_export_artifacts_json(
        self, client: TestClient, seeded_db: DatabaseManager
    ) -> None:
        resp = client.get("/api/v1/export/artifacts/json")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3


class TestAuth:
    """Tests for authentication."""

    def test_no_auth_required_in_single_user_mode(
        self, db_manager: DatabaseManager
    ) -> None:
        """Single-user mode (no api_key configured) should not require auth."""
        app = create_app()
        app.dependency_overrides[get_database] = lambda: db_manager
        with TestClient(app) as c:
            resp = c.get("/api/v1/artifacts")
            assert resp.status_code == 200
        app.dependency_overrides.clear()

    def test_auth_required_with_api_key_configured(self, tmp_path: Path) -> None:
        """When api_key is set, requests without key should fail."""
        from alibi.api.deps import get_config_dep

        config = Config(
            db_path=tmp_path / "test_auth.db",
            api_key="test-secret-key",
        )
        manager = DatabaseManager(config)
        manager.initialize()

        app = create_app()
        app.dependency_overrides[get_database] = lambda: manager
        app.dependency_overrides[get_config_dep] = lambda: config

        with TestClient(app) as c:
            # Without key
            resp = c.get("/api/v1/artifacts")
            assert resp.status_code == 401

            # With wrong key
            resp = c.get(
                "/api/v1/artifacts",
                headers={"X-Api-Key": "wrong-key"},
            )
            assert resp.status_code == 401

            # With correct key
            resp = c.get(
                "/api/v1/artifacts",
                headers={"X-Api-Key": "test-secret-key"},
            )
            assert resp.status_code == 200

        app.dependency_overrides.clear()
        manager.close()


class TestDeps:
    """Tests for dependency injection utilities."""

    def test_pagination_defaults(self) -> None:
        from alibi.api.deps import PaginationParams

        p = PaginationParams(page=1, per_page=50)
        assert p.page == 1
        assert p.per_page == 50
        assert p.offset == 0

    def test_pagination_custom(self) -> None:
        from alibi.api.deps import PaginationParams

        p = PaginationParams(page=3, per_page=10)
        assert p.page == 3
        assert p.per_page == 10
        assert p.offset == 20

    def test_paginate_empty(self) -> None:
        from alibi.api.deps import PaginationParams, paginate

        result = paginate([], PaginationParams(page=1, per_page=50))
        assert result["total"] == 0
        assert result["items"] == []
        assert result["pages"] == 0

    def test_paginate_with_items(self) -> None:
        from alibi.api.deps import PaginationParams, paginate

        items = list(range(25))
        result = paginate(items, PaginationParams(page=1, per_page=10))
        assert result["total"] == 25
        assert len(result["items"]) == 10
        assert result["pages"] == 3


class TestAppFactory:
    """Tests for app factory."""

    def test_create_app(self) -> None:
        app = create_app()
        assert app.title == "Alibi"
        # Should have routes registered
        assert len(app.routes) > 10

    def test_openapi_schema(self, client: TestClient) -> None:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["info"]["title"] == "Alibi"
        assert "/api/v1/artifacts" in schema["paths"]
        assert "/api/v1/facts" in schema["paths"]
        assert "/health" in schema["paths"]
