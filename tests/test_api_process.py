"""Tests for the document processing API endpoints."""

from __future__ import annotations

import io
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from alibi.api.app import create_app
from alibi.api.deps import get_database
from alibi.db.connection import DatabaseManager
from alibi.db.models import DocumentType
from alibi.processing.folder_router import FolderContext


def _make_mock_result(
    success: bool = True,
    document_id: str = "doc-123",
    is_duplicate: bool = False,
    duplicate_of: str | None = None,
    error: str | None = None,
    extracted_data: dict[str, Any] | None = None,
    line_items: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Build a mock ProcessingResult."""
    result = MagicMock()
    result.success = success
    result.document_id = document_id
    result.is_duplicate = is_duplicate
    result.duplicate_of = duplicate_of
    result.error = error
    result.extracted_data = extracted_data or {
        "vendor": "Test Vendor",
        "total": "25.50",
        "date": "2025-01-15",
        "document_type": "receipt",
    }
    result.line_items = line_items if line_items is not None else [{"name": "item1"}]
    return result


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


class TestProcessSingle:
    """Tests for POST /api/v1/process."""

    def test_process_single_file(self, client: TestClient) -> None:
        mock_result = _make_mock_result()
        with patch(
            "alibi.api.routers.process.ingestion.process_bytes",
            return_value=mock_result,
        ):
            resp = client.post(
                "/api/v1/process",
                files={
                    "file": ("receipt.jpg", io.BytesIO(b"fake image"), "image/jpeg")
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["document_id"] == "doc-123"
        assert data["vendor"] == "Test Vendor"
        assert data["amount"] == "25.50"
        assert data["items_count"] == 1

    def test_process_with_type(self, client: TestClient) -> None:
        mock_result = _make_mock_result()
        captured: list[Any] = []

        def capture(
            db: Any,
            data: bytes,
            filename: str,
            folder_context: FolderContext | None = None,
        ) -> MagicMock:
            captured.append(folder_context)
            return mock_result

        with patch(
            "alibi.api.routers.process.ingestion.process_bytes",
            side_effect=capture,
        ):
            resp = client.post(
                "/api/v1/process?type=receipt",
                files={"file": ("r.jpg", io.BytesIO(b"data"), "image/jpeg")},
            )

        assert resp.status_code == 200
        assert len(captured) == 1
        assert captured[0] is not None
        assert captured[0].doc_type == DocumentType.RECEIPT

    def test_process_all_valid_types(self, client: TestClient) -> None:
        for t in ["receipt", "invoice", "payment", "statement", "warranty", "contract"]:
            mock_result = _make_mock_result()
            with patch(
                "alibi.api.routers.process.ingestion.process_bytes",
                return_value=mock_result,
            ):
                resp = client.post(
                    f"/api/v1/process?type={t}",
                    files={"file": ("d.jpg", io.BytesIO(b"d"), "image/jpeg")},
                )
            assert resp.status_code == 200, f"Failed for type={t}"

    def test_process_invalid_type(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/process?type=invalid",
            files={"file": ("d.jpg", io.BytesIO(b"d"), "image/jpeg")},
        )
        assert resp.status_code == 422

    def test_process_no_file(self, client: TestClient) -> None:
        resp = client.post("/api/v1/process")
        assert resp.status_code == 422

    def test_process_without_type_passes_none(self, client: TestClient) -> None:
        mock_result = _make_mock_result()
        captured: list[Any] = []

        def capture(
            db: Any,
            data: bytes,
            filename: str,
            folder_context: FolderContext | None = None,
        ) -> MagicMock:
            captured.append(folder_context)
            return mock_result

        with patch(
            "alibi.api.routers.process.ingestion.process_bytes",
            side_effect=capture,
        ):
            resp = client.post(
                "/api/v1/process",
                files={"file": ("d.jpg", io.BytesIO(b"d"), "image/jpeg")},
            )

        assert resp.status_code == 200
        # No doc_type specified, but provenance is always set
        ctx = captured[0]
        assert ctx is not None
        assert ctx.doc_type is None
        assert ctx.source == "api"

    def test_process_failure_result(self, client: TestClient) -> None:
        mock_result = _make_mock_result(
            success=False,
            document_id=None,
            error="Pipeline failure",
            extracted_data=None,
            line_items=[],
        )
        with patch(
            "alibi.api.routers.process.ingestion.process_bytes",
            return_value=mock_result,
        ):
            resp = client.post(
                "/api/v1/process",
                files={"file": ("b.jpg", io.BytesIO(b"d"), "image/jpeg")},
            )

        data = resp.json()
        assert data["success"] is False
        assert data["error"] == "Pipeline failure"
        assert data["items_count"] == 0


class TestProcessBatch:
    """Tests for POST /api/v1/process/batch."""

    def test_process_batch(self, client: TestClient) -> None:
        mock_result = _make_mock_result()
        with patch(
            "alibi.api.routers.process.ingestion.process_bytes",
            return_value=mock_result,
        ):
            resp = client.post(
                "/api/v1/process/batch",
                files=[
                    ("files", ("a.jpg", io.BytesIO(b"d1"), "image/jpeg")),
                    ("files", ("b.jpg", io.BytesIO(b"d2"), "image/jpeg")),
                ],
            )

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_process_batch_with_type(self, client: TestClient) -> None:
        mock_result = _make_mock_result()
        captured: list[Any] = []

        def capture(
            db: Any,
            data: bytes,
            filename: str,
            folder_context: FolderContext | None = None,
        ) -> MagicMock:
            captured.append(folder_context)
            return mock_result

        with patch(
            "alibi.api.routers.process.ingestion.process_bytes",
            side_effect=capture,
        ):
            resp = client.post(
                "/api/v1/process/batch?type=invoice",
                files=[
                    ("files", ("a.pdf", io.BytesIO(b"d1"), "application/pdf")),
                    ("files", ("b.pdf", io.BytesIO(b"d2"), "application/pdf")),
                ],
            )

        assert resp.status_code == 200
        assert len(captured) == 2
        for ctx in captured:
            assert ctx is not None
            assert ctx.doc_type == DocumentType.INVOICE

    def test_process_batch_invalid_type(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/process/batch?type=unknown",
            files=[("files", ("a.jpg", io.BytesIO(b"d"), "image/jpeg"))],
        )
        assert resp.status_code == 422

    def test_process_batch_no_files(self, client: TestClient) -> None:
        resp = client.post("/api/v1/process/batch")
        assert resp.status_code == 422


class TestProcessGroup:
    """Tests for POST /api/v1/process/group."""

    def test_process_group_success(self, client: TestClient) -> None:
        mock_result = _make_mock_result()
        with patch(
            "alibi.api.routers.process.ingestion.process_document_group",
            return_value=mock_result,
        ):
            resp = client.post(
                "/api/v1/process/group",
                files=[
                    ("files", ("page1.jpg", io.BytesIO(b"p1"), "image/jpeg")),
                    ("files", ("page2.jpg", io.BytesIO(b"p2"), "image/jpeg")),
                ],
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["document_id"] == "doc-123"
        assert data["vendor"] == "Test Vendor"
        assert data["amount"] == "25.50"
        assert data["items_count"] == 1

    def test_process_group_with_type(self, client: TestClient) -> None:
        mock_result = _make_mock_result()
        captured: list[Any] = []

        def capture(
            db: Any,
            paths: Any,
            folder_context: FolderContext | None = None,
        ) -> MagicMock:
            captured.append(folder_context)
            return mock_result

        with patch(
            "alibi.api.routers.process.ingestion.process_document_group",
            side_effect=capture,
        ):
            resp = client.post(
                "/api/v1/process/group?type=invoice",
                files=[
                    ("files", ("page1.pdf", io.BytesIO(b"p1"), "application/pdf")),
                    ("files", ("page2.pdf", io.BytesIO(b"p2"), "application/pdf")),
                ],
            )

        assert resp.status_code == 200
        assert len(captured) == 1
        assert captured[0] is not None
        assert captured[0].doc_type == DocumentType.INVOICE

    def test_process_group_single_file(self, client: TestClient) -> None:
        mock_result = _make_mock_result()
        with patch(
            "alibi.api.routers.process.ingestion.process_document_group",
            return_value=mock_result,
        ):
            resp = client.post(
                "/api/v1/process/group",
                files=[
                    ("files", ("only.jpg", io.BytesIO(b"data"), "image/jpeg")),
                ],
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_process_group_invalid_type(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/process/group?type=invalid",
            files=[("files", ("a.jpg", io.BytesIO(b"d"), "image/jpeg"))],
        )
        assert resp.status_code == 422

    def test_process_group_no_files(self, client: TestClient) -> None:
        resp = client.post("/api/v1/process/group")
        assert resp.status_code == 422

    def test_process_group_provenance(self, client: TestClient) -> None:
        mock_result = _make_mock_result()
        captured: list[Any] = []

        def capture(
            db: Any,
            paths: Any,
            folder_context: FolderContext | None = None,
        ) -> MagicMock:
            captured.append(folder_context)
            return mock_result

        with patch(
            "alibi.api.routers.process.ingestion.process_document_group",
            side_effect=capture,
        ):
            resp = client.post(
                "/api/v1/process/group",
                files=[
                    ("files", ("doc.jpg", io.BytesIO(b"data"), "image/jpeg")),
                ],
            )

        assert resp.status_code == 200
        ctx = captured[0]
        assert ctx is not None
        assert ctx.source == "api"
        assert ctx.user_id is not None

    def test_process_group_paths_ordered(self, client: TestClient) -> None:
        mock_result = _make_mock_result()
        captured_paths: list[Any] = []

        def capture(
            db: Any,
            paths: Any,
            folder_context: FolderContext | None = None,
        ) -> MagicMock:
            captured_paths.extend(paths)
            return mock_result

        with patch(
            "alibi.api.routers.process.ingestion.process_document_group",
            side_effect=capture,
        ):
            resp = client.post(
                "/api/v1/process/group",
                files=[
                    ("files", ("first.jpg", io.BytesIO(b"p0"), "image/jpeg")),
                    ("files", ("second.jpg", io.BytesIO(b"p1"), "image/jpeg")),
                    ("files", ("third.jpg", io.BytesIO(b"p2"), "image/jpeg")),
                ],
            )

        assert resp.status_code == 200
        assert len(captured_paths) == 3
        names = [p.name for p in captured_paths]
        assert names[0] == "page_000.jpg"
        assert names[1] == "page_001.jpg"
        assert names[2] == "page_002.jpg"

    def test_process_group_failure(self, client: TestClient) -> None:
        mock_result = _make_mock_result(
            success=False,
            document_id=None,
            error="Group processing failed",
            extracted_data=None,
            line_items=[],
        )
        with patch(
            "alibi.api.routers.process.ingestion.process_document_group",
            return_value=mock_result,
        ):
            resp = client.post(
                "/api/v1/process/group",
                files=[
                    ("files", ("page1.jpg", io.BytesIO(b"p1"), "image/jpeg")),
                    ("files", ("page2.jpg", io.BytesIO(b"p2"), "image/jpeg")),
                ],
            )

        data = resp.json()
        assert data["success"] is False
        assert data["error"] == "Group processing failed"
        assert data["items_count"] == 0
