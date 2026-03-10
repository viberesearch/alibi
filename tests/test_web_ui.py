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


class TestWebIndex:
    """Tests for the Web UI index endpoint."""

    def test_web_index_returns_html(self, client: TestClient) -> None:
        """GET /web returns 200 with HTML content type."""
        response = client.get("/web")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_web_index_contains_alibi_title(self, client: TestClient) -> None:
        """Response body contains Alibi title."""
        response = client.get("/web")
        assert response.status_code == 200
        content = response.text
        assert "Alibi" in content
        assert "<title>" in content

    def test_web_index_contains_app_header(self, client: TestClient) -> None:
        """Response body contains app-logo header."""
        response = client.get("/web")
        assert response.status_code == 200
        content = response.text
        assert "app-logo" in content
        assert "app-header" in content

    def test_web_index_contains_dashboard_view(self, client: TestClient) -> None:
        """Response body contains view-dashboard id."""
        response = client.get("/web")
        assert response.status_code == 200
        content = response.text
        assert "view-dashboard" in content


class TestStaticMount:
    """Tests for the static files mount."""

    def test_index_html_served_via_mount(self, client: TestClient) -> None:
        """GET /web/index.html returns 200 with HTML."""
        response = client.get("/web/index.html")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Alibi" in response.text

    def test_nonexistent_static_returns_404(self, client: TestClient) -> None:
        """GET /web/nonexistent.css returns 404."""
        response = client.get("/web/nonexistent.css")
        assert response.status_code == 404


class TestApiUnaffected:
    """Tests to ensure API endpoints still work alongside Web UI."""

    def test_health_endpoint_still_works(self, client: TestClient) -> None:
        """GET /health returns 200 with status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_users_api_still_works(self, client: TestClient) -> None:
        """GET /api/v1/users returns 200."""
        response = client.get("/api/v1/users")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
