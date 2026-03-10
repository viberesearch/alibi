"""Tests for document provenance (source + user_id on documents table)."""

from alibi.db.models import Document
from alibi.db.v2_store import get_document_by_hash, store_document


class TestDocumentProvenance:
    def test_store_with_source_and_user(self, db):
        doc = Document(
            id="doc-1",
            file_path="/tmp/test.jpg",
            file_hash="abc123",
            source="telegram",
            user_id="system",
        )
        store_document(db, doc)

        retrieved = get_document_by_hash(db, "abc123")
        assert retrieved is not None
        assert retrieved["source"] == "telegram"
        assert retrieved["user_id"] == "system"

    def test_store_without_provenance(self, db):
        doc = Document(
            id="doc-2",
            file_path="/tmp/test2.jpg",
            file_hash="def456",
        )
        store_document(db, doc)

        retrieved = get_document_by_hash(db, "def456")
        assert retrieved is not None
        assert retrieved["source"] is None
        assert retrieved["user_id"] is None

    def test_different_sources(self, db):
        sources = ["cli", "api", "telegram", "watcher", "mcp"]
        for i, src in enumerate(sources):
            doc = Document(
                id=f"doc-{i+10}",
                file_path=f"/tmp/test{i}.jpg",
                file_hash=f"hash{i}",
                source=src,
                user_id="system",
            )
            store_document(db, doc)

            retrieved = get_document_by_hash(db, f"hash{i}")
            assert retrieved["source"] == src


class TestFolderContextProvenance:
    def test_folder_context_has_source_and_user(self):
        from alibi.processing.folder_router import FolderContext

        ctx = FolderContext(source="telegram", user_id="user-1")
        assert ctx.source == "telegram"
        assert ctx.user_id == "user-1"

    def test_folder_context_defaults_to_none(self):
        from alibi.processing.folder_router import FolderContext

        ctx = FolderContext()
        assert ctx.source is None
        assert ctx.user_id is None


class TestProcessingResultProvenance:
    def test_result_carries_provenance(self):
        from pathlib import Path

        from alibi.processing.pipeline import ProcessingResult

        result = ProcessingResult(
            success=True,
            file_path=Path("/tmp/test.jpg"),
            source="api",
            user_id="user-1",
        )
        assert result.source == "api"
        assert result.user_id == "user-1"
