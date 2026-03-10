"""Tests for alibi.services.ingestion."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alibi.processing.pipeline import ProcessingResult
from alibi.processing.folder_router import FolderContext
from alibi.services import ingestion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PIPELINE_TARGET = "alibi.services.ingestion.ProcessingPipeline"


def _make_result(path: Path, success: bool = True) -> ProcessingResult:
    return ProcessingResult(success=success, file_path=path)


# ---------------------------------------------------------------------------
# process_file
# ---------------------------------------------------------------------------


class TestProcessFile:
    def test_delegates_to_pipeline(self, tmp_path):
        doc = tmp_path / "receipt.jpg"
        doc.write_bytes(b"fake")
        expected = _make_result(doc)

        with patch(PIPELINE_TARGET) as MockPipeline:
            instance = MockPipeline.return_value
            instance.process_file.return_value = expected

            db = MagicMock()
            result = ingestion.process_file(db, doc)

        MockPipeline.assert_called_once_with(db)
        instance.process_file.assert_called_once_with(doc, folder_context=None)
        assert result is expected

    def test_converts_string_path(self, tmp_path):
        doc = tmp_path / "invoice.pdf"
        doc.write_bytes(b"fake")
        expected = _make_result(doc)

        with patch(PIPELINE_TARGET) as MockPipeline:
            instance = MockPipeline.return_value
            instance.process_file.return_value = expected

            db = MagicMock()
            result = ingestion.process_file(db, str(doc))

        call_args = instance.process_file.call_args
        assert isinstance(call_args.args[0], Path)
        assert result is expected

    def test_passes_folder_context(self, tmp_path):
        doc = tmp_path / "receipt.jpg"
        doc.write_bytes(b"fake")
        ctx = FolderContext()
        expected = _make_result(doc)

        with patch(PIPELINE_TARGET) as MockPipeline:
            instance = MockPipeline.return_value
            instance.process_file.return_value = expected

            ingestion.process_file(MagicMock(), doc, folder_context=ctx)

        instance.process_file.assert_called_once_with(doc, folder_context=ctx)


# ---------------------------------------------------------------------------
# process_batch
# ---------------------------------------------------------------------------


class TestProcessBatch:
    def test_converts_strings_to_paths(self, tmp_path):
        files = [tmp_path / f"doc{i}.jpg" for i in range(3)]
        for f in files:
            f.write_bytes(b"fake")

        expected = [_make_result(f) for f in files]

        with patch(PIPELINE_TARGET) as MockPipeline:
            instance = MockPipeline.return_value
            instance.process_batch.return_value = expected

            db = MagicMock()
            result = ingestion.process_batch(db, [str(f) for f in files])

        call_args = instance.process_batch.call_args
        passed_paths = call_args.args[0]
        assert all(isinstance(p, Path) for p in passed_paths)
        assert result is expected

    def test_no_folder_context_passes_none(self, tmp_path):
        doc = tmp_path / "doc.jpg"
        doc.write_bytes(b"fake")

        with patch(PIPELINE_TARGET) as MockPipeline:
            instance = MockPipeline.return_value
            instance.process_batch.return_value = [_make_result(doc)]

            ingestion.process_batch(MagicMock(), [doc])

        call_args = instance.process_batch.call_args
        assert call_args.kwargs.get("folder_contexts") is None

    def test_shared_folder_context_replicated(self, tmp_path):
        files = [tmp_path / f"doc{i}.jpg" for i in range(2)]
        for f in files:
            f.write_bytes(b"fake")

        ctx = FolderContext()

        with patch(PIPELINE_TARGET) as MockPipeline:
            instance = MockPipeline.return_value
            instance.process_batch.return_value = [_make_result(f) for f in files]

            ingestion.process_batch(MagicMock(), files, folder_context=ctx)

        call_args = instance.process_batch.call_args
        contexts = call_args.kwargs.get("folder_contexts")
        assert contexts == [ctx, ctx]

    def test_empty_batch(self):
        with patch(PIPELINE_TARGET) as MockPipeline:
            instance = MockPipeline.return_value
            instance.process_batch.return_value = []

            result = ingestion.process_batch(MagicMock(), [])

        instance.process_batch.assert_called_once_with([], folder_contexts=None)
        assert result == []


# ---------------------------------------------------------------------------
# process_document_group
# ---------------------------------------------------------------------------


class TestProcessDocumentGroup:
    def test_delegates_with_parent_as_folder_path(self, tmp_path):
        subdir = tmp_path / "scan"
        subdir.mkdir()
        pages = [subdir / f"page{i}.jpg" for i in range(2)]
        for p in pages:
            p.write_bytes(b"fake")

        expected = _make_result(subdir)

        with patch(PIPELINE_TARGET) as MockPipeline:
            instance = MockPipeline.return_value
            instance.process_document_group.return_value = expected

            db = MagicMock()
            result = ingestion.process_document_group(db, pages)

        instance.process_document_group.assert_called_once_with(
            subdir, pages, folder_context=None
        )
        assert result is expected

    def test_converts_string_paths(self, tmp_path):
        subdir = tmp_path / "scan"
        subdir.mkdir()
        pages = [subdir / f"page{i}.jpg" for i in range(2)]
        for p in pages:
            p.write_bytes(b"fake")

        expected = _make_result(subdir)

        with patch(PIPELINE_TARGET) as MockPipeline:
            instance = MockPipeline.return_value
            instance.process_document_group.return_value = expected

            ingestion.process_document_group(MagicMock(), [str(p) for p in pages])

        call_args = instance.process_document_group.call_args
        passed_files = call_args.args[1]
        assert all(isinstance(p, Path) for p in passed_files)

    def test_empty_list_still_delegates(self):
        with patch(PIPELINE_TARGET) as MockPipeline:
            instance = MockPipeline.return_value
            instance.process_document_group.return_value = ProcessingResult(
                success=False,
                file_path=Path("."),
                error="No supported files in folder",
            )

            result = ingestion.process_document_group(MagicMock(), [])

        instance.process_document_group.assert_called_once()
        assert result.success is False


# ---------------------------------------------------------------------------
# process_bytes
# ---------------------------------------------------------------------------


class TestProcessBytes:
    def test_persists_to_inbox_and_delegates(self, monkeypatch):
        monkeypatch.delenv("ALIBI_VAULT_PATH", raising=False)
        payload = b"fake image data"
        expected = ProcessingResult(success=True, file_path=Path("/tmp/x.jpg"))

        with patch(PIPELINE_TARGET) as MockPipeline:
            instance = MockPipeline.return_value
            instance.process_file.return_value = expected

            db = MagicMock()
            result = ingestion.process_bytes(db, payload, "photo.jpg")

        assert result is expected
        call_args = instance.process_file.call_args
        called_path = call_args.args[0]
        assert called_path.suffix == ".jpg"
        assert called_path.exists(), "Persisted file should remain on disk"
        assert called_path.read_bytes() == payload
        # Cleanup
        called_path.unlink(missing_ok=True)

    def test_file_persists_after_processing(self, monkeypatch):
        monkeypatch.delenv("ALIBI_VAULT_PATH", raising=False)
        captured: list[Path] = []

        def fake_process_file(path, folder_context=None):
            captured.append(path)
            return ProcessingResult(success=True, file_path=path)

        with patch(PIPELINE_TARGET) as MockPipeline:
            instance = MockPipeline.return_value
            instance.process_file.side_effect = fake_process_file

            ingestion.process_bytes(MagicMock(), b"data", "doc.pdf")

        assert len(captured) == 1
        assert captured[0].exists(), "Persisted file should remain on disk"
        captured[0].unlink(missing_ok=True)

    def test_preserves_extension_from_filename(self, monkeypatch):
        monkeypatch.delenv("ALIBI_VAULT_PATH", raising=False)
        for ext in (".jpg", ".png", ".pdf"):
            with patch(PIPELINE_TARGET) as MockPipeline:
                instance = MockPipeline.return_value
                instance.process_file.return_value = ProcessingResult(
                    success=True, file_path=Path(f"/tmp/x{ext}")
                )

                ingestion.process_bytes(MagicMock(), b"x", f"document{ext}")

            call_args = instance.process_file.call_args
            path = call_args.args[0]
            assert path.suffix == ext
            path.unlink(missing_ok=True)

    def test_passes_folder_context_to_process_file(self, monkeypatch):
        monkeypatch.delenv("ALIBI_VAULT_PATH", raising=False)
        ctx = FolderContext(source="api")
        with patch(PIPELINE_TARGET) as MockPipeline:
            instance = MockPipeline.return_value
            instance.process_file.return_value = ProcessingResult(
                success=True, file_path=Path("/tmp/x.jpg")
            )

            ingestion.process_bytes(MagicMock(), b"data", "img.jpg", folder_context=ctx)

        call_args = instance.process_file.call_args
        assert call_args.kwargs.get("folder_context") is ctx
        call_args.args[0].unlink(missing_ok=True)

    def test_unknown_extension_uses_jpg_fallback(self, monkeypatch):
        monkeypatch.delenv("ALIBI_VAULT_PATH", raising=False)
        with patch(PIPELINE_TARGET) as MockPipeline:
            instance = MockPipeline.return_value
            instance.process_file.return_value = ProcessingResult(
                success=True, file_path=Path("/tmp/x.jpg")
            )

            ingestion.process_bytes(MagicMock(), b"data", "noextension")

        call_args = instance.process_file.call_args
        path = call_args.args[0]
        assert path.suffix == ".jpg"
        path.unlink(missing_ok=True)

    def test_source_prefix_in_filename(self, monkeypatch):
        monkeypatch.delenv("ALIBI_VAULT_PATH", raising=False)
        ctx = FolderContext(source="api")
        with patch(PIPELINE_TARGET) as MockPipeline:
            instance = MockPipeline.return_value
            instance.process_file.return_value = ProcessingResult(
                success=True, file_path=Path("/tmp/x.jpg")
            )

            ingestion.process_bytes(
                MagicMock(), b"data", "scan.jpg", folder_context=ctx
            )

        call_args = instance.process_file.call_args
        path = call_args.args[0]
        assert path.name.startswith("api_")
        path.unlink(missing_ok=True)
