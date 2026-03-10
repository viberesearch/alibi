"""Tests for processing modules."""

import tempfile
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from alibi.processing.watcher import (
    SUPPORTED_EXTENSIONS,
    DocumentEventHandler,
    InboxWatcher,
    is_supported_file,
    scan_inbox,
)


class TestProcessGroupYamlCacheRegression:
    """Regression test for NameError when YAML cache exists (audit fix)."""

    def test_process_group_yaml_cache_hit(self, db_manager, tmp_path: Path) -> None:
        """process_document_group() succeeds when YAML cache exists.

        Previously raised NameError because extracted_data was not set
        in the cached code path.
        """
        from alibi.processing.pipeline import ProcessingPipeline, ProcessingResult

        pipeline = ProcessingPipeline(db=db_manager)

        # Create test files
        group_dir = tmp_path / "doc_group"
        group_dir.mkdir()
        img1 = group_dir / "page1.jpg"
        img1.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # Minimal JPEG

        cached_data = {
            "vendor": "Test Store",
            "total": "42.50",
            "document_type": "receipt",
            "line_items": [{"name": "Item A", "total_price": "42.50"}],
        }
        cached_meta = {"version": 5, "file_hash": "hash123"}

        with (
            patch(
                "alibi.processing.pipeline.read_yaml_with_meta",
                return_value=(cached_data, cached_meta),
            ),
            patch(
                "alibi.processing.pipeline.compute_file_hash",
                return_value="hash123",
            ),
            patch(
                "alibi.processing.pipeline.compute_yaml_hash",
                return_value="yaml-hash-123",
            ),
            patch(
                "alibi.processing.pipeline.v2_store.get_document_by_hash",
                return_value=None,
            ),
            patch.object(
                pipeline,
                "_detect_document_type",
                return_value=MagicMock(value="receipt"),
            ),
            patch.object(
                pipeline,
                "_ingest_from_yaml",
                return_value=ProcessingResult(
                    success=True,
                    file_path=group_dir,
                    document_id="doc-123",
                    extracted_data=cached_data,
                ),
            ),
            patch.object(
                pipeline,
                "_commit_yaml_versioning",
            ),
        ):
            result = pipeline.process_document_group(group_dir, [img1])

        assert result.success is True
        assert result.extracted_data is not None


class TestSupportedFiles:
    """Tests for file type detection."""

    def test_image_files_supported(self):
        """Test that image files are supported."""
        for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]:
            assert is_supported_file(Path(f"test{ext}"))
            assert is_supported_file(Path(f"test{ext.upper()}"))

    def test_pdf_supported(self):
        """Test that PDF files are supported."""
        assert is_supported_file(Path("document.pdf"))
        assert is_supported_file(Path("DOCUMENT.PDF"))

    def test_unsupported_files(self):
        """Test that unsupported files are rejected."""
        for ext in [".txt", ".doc", ".xlsx", ".mp3", ".mp4"]:
            assert not is_supported_file(Path(f"test{ext}"))


class TestScanInbox:
    """Tests for inbox scanning."""

    def test_scan_empty_directory(self, tmp_path):
        """Test scanning an empty directory."""
        files = scan_inbox(tmp_path)
        assert files == []

    def test_scan_finds_supported_files(self, tmp_path):
        """Test that scan finds supported files."""
        # Create test files
        (tmp_path / "receipt.jpg").touch()
        (tmp_path / "invoice.pdf").touch()
        (tmp_path / "notes.txt").touch()  # Should be ignored

        files = scan_inbox(tmp_path)

        assert len(files) == 2
        names = {f.name for f in files}
        assert "receipt.jpg" in names
        assert "invoice.pdf" in names
        assert "notes.txt" not in names

    def test_scan_recursive(self, tmp_path):
        """Test that scan finds files in subdirectories."""
        subdir = tmp_path / "2025" / "receipts"
        subdir.mkdir(parents=True)

        (subdir / "receipt.png").touch()
        (tmp_path / "doc.pdf").touch()

        files = scan_inbox(tmp_path)

        assert len(files) == 2

    def test_scan_sorted_by_mtime(self, tmp_path):
        """Test that files are sorted by modification time."""
        import time

        (tmp_path / "old.jpg").touch()
        time.sleep(0.01)
        (tmp_path / "new.jpg").touch()

        files = scan_inbox(tmp_path)

        assert len(files) == 2
        assert files[0].name == "old.jpg"
        assert files[1].name == "new.jpg"

    def test_scan_nonexistent_directory(self, tmp_path):
        """Test scanning a nonexistent directory."""
        nonexistent = tmp_path / "does_not_exist"
        files = scan_inbox(nonexistent)
        assert files == []


class TestDocumentEventHandler:
    """Tests for the document event handler."""

    def test_handler_calls_callback_on_create(self, tmp_path):
        """Test that handler calls callback when file is created."""
        callback = MagicMock()
        handler = DocumentEventHandler(callback)

        test_file = tmp_path / "test.jpg"
        test_file.touch()

        event = MagicMock()
        event.is_directory = False
        event.src_path = str(test_file)

        handler.on_created(event)

        callback.assert_called_once_with(test_file)

    def test_handler_ignores_directories(self, tmp_path):
        """Test that handler ignores directory events."""
        callback = MagicMock()
        handler = DocumentEventHandler(callback)

        event = MagicMock()
        event.is_directory = True
        event.src_path = str(tmp_path)

        handler.on_created(event)

        callback.assert_not_called()

    def test_handler_ignores_unsupported_files(self, tmp_path):
        """Test that handler ignores unsupported files."""
        callback = MagicMock()
        handler = DocumentEventHandler(callback)

        test_file = tmp_path / "notes.txt"
        test_file.touch()

        event = MagicMock()
        event.is_directory = False
        event.src_path = str(test_file)

        handler.on_created(event)

        callback.assert_not_called()

    def test_handler_debounces_rapid_events(self, tmp_path):
        """Test that handler debounces rapid events for same file."""
        callback = MagicMock()
        handler = DocumentEventHandler(callback, debounce_seconds=1.0)

        test_file = tmp_path / "test.jpg"
        test_file.touch()

        event = MagicMock()
        event.is_directory = False
        event.src_path = str(test_file)

        # First event should trigger
        handler.on_created(event)
        # Second event should be debounced
        handler.on_created(event)

        assert callback.call_count == 1


class TestInboxWatcher:
    """Tests for the inbox watcher."""

    def test_watcher_raises_without_config(self):
        """Test that watcher raises error when no inbox configured."""
        with patch("alibi.processing.watcher.get_config") as mock_config:
            mock_config.return_value.get_inbox_path.return_value = None

            with pytest.raises(ValueError, match="No inbox path"):
                InboxWatcher()

    def test_watcher_context_manager(self, tmp_path):
        """Test watcher as context manager."""
        with patch("alibi.processing.watcher.get_config") as mock_config:
            mock_config.return_value.get_inbox_path.return_value = tmp_path

            watcher = InboxWatcher(inbox_path=tmp_path)
            assert not watcher.is_running()

            with watcher:
                assert watcher.is_running()

            assert not watcher.is_running()


class TestProcessingPipeline:
    """Tests for the processing pipeline."""

    def test_pipeline_rejects_nonexistent_file(self, tmp_path):
        """Test that pipeline rejects nonexistent file."""
        from alibi.processing.pipeline import ProcessingPipeline

        pipeline = ProcessingPipeline()
        result = pipeline.process_file(tmp_path / "nonexistent.jpg")

        assert not result.success
        assert result.error is not None
        assert "not found" in result.error.lower()
        pipeline.close()

    def test_pipeline_rejects_unsupported_file(self, tmp_path):
        """Test that pipeline rejects unsupported file."""
        from alibi.processing.pipeline import ProcessingPipeline

        test_file = tmp_path / "notes.txt"
        test_file.write_text("Hello")

        pipeline = ProcessingPipeline()
        result = pipeline.process_file(test_file)

        assert not result.success
        assert result.error is not None
        assert "unsupported" in result.error.lower()
        pipeline.close()

    def test_processing_result_dataclass(self):
        """Test ProcessingResult dataclass."""
        from alibi.processing.pipeline import ProcessingResult

        result = ProcessingResult(
            success=True,
            file_path=Path("/test/file.jpg"),
            document_id="123",
        )

        assert result.success
        assert result.document_id == "123"
        assert not result.is_duplicate
        assert result.error is None

    def test_fill_locale_gaps_sets_currency_from_inbox_config(self):
        """Currency filled from inbox config when extraction didn't provide it."""
        from alibi.processing.config_loader import InboxConfig
        from alibi.processing.folder_router import FolderContext
        from alibi.processing.pipeline import ProcessingPipeline

        pipeline = ProcessingPipeline()
        data: dict[str, Any] = {"vendor": "Test", "total": 10.0}
        ctx = FolderContext(inbox_config=InboxConfig(default_currency="EUR"))
        pipeline._fill_locale_gaps(data, ctx)
        assert data["currency"] == "EUR"
        pipeline.close()

    def test_fill_locale_gaps_does_not_override_existing(self):
        """Currency from extraction takes priority over inbox config."""
        from alibi.processing.config_loader import InboxConfig
        from alibi.processing.folder_router import FolderContext
        from alibi.processing.pipeline import ProcessingPipeline

        pipeline = ProcessingPipeline()
        data: dict[str, Any] = {"vendor": "Test", "currency": "GBP"}
        ctx = FolderContext(inbox_config=InboxConfig(default_currency="EUR"))
        pipeline._fill_locale_gaps(data, ctx)
        assert data["currency"] == "GBP"
        pipeline.close()

    def test_fill_locale_gaps_no_context(self):
        """No-op when folder context is None."""
        from alibi.processing.pipeline import ProcessingPipeline

        pipeline = ProcessingPipeline()
        data: dict[str, Any] = {"vendor": "Test"}
        pipeline._fill_locale_gaps(data, None)
        assert "currency" not in data
        pipeline.close()


class TestDocumentGroupProcessing:
    """Tests for multi-page document group processing."""

    def test_group_empty_folder_returns_error(self, tmp_path: Path) -> None:
        """Empty folder should return error result."""
        from alibi.processing.pipeline import ProcessingPipeline

        pipeline = ProcessingPipeline()
        result = pipeline.process_document_group(tmp_path, [])

        assert not result.success
        assert "No supported files" in (result.error or "")
        pipeline.close()

    def test_group_processes_multiple_images(self, tmp_path: Path) -> None:
        """Document group should send all images to LLM and create one artifact."""
        from alibi.config import Config
        from alibi.db.connection import DatabaseManager
        from alibi.processing.pipeline import ProcessingPipeline

        # Create isolated DB
        config = Config(db_path=tmp_path / "test.db", _env_file=None)
        db = DatabaseManager(config)
        db.initialize()

        # Create fake image files in a subfolder
        doc_dir = tmp_path / "receipt-pages"
        doc_dir.mkdir()
        for i in range(3):
            (doc_dir / f"page{i + 1}.jpg").write_bytes(b"\xff\xd8\xff" + bytes(100))

        files = sorted(doc_dir.glob("*.jpg"))

        with (
            patch("alibi.processing.pipeline.extract_from_images") as mock_extract,
            patch("alibi.processing.pipeline.compute_file_hash") as mock_hash,
            patch(
                "alibi.processing.pipeline.vision_detect_document_type",
                return_value="receipt",
            ),
        ):
            mock_hash.side_effect = lambda p: f"hash-{p.name}"
            mock_extract.return_value = {
                "vendor": "Test Store",
                "total": 42.50,
                "currency": "EUR",
                "date": "2026-01-15",
                "line_items": [{"name": "Item 1", "total": 42.50}],
                "raw_text": "Page 1\nPage 2\nPage 3",
            }

            pipeline = ProcessingPipeline(db=db)
            result = pipeline.process_document_group(doc_dir, files)

        assert result.success
        assert result.document_id is not None
        assert not result.is_duplicate
        # Verify extract_from_images was called with all 3 files
        mock_extract.assert_called_once()
        call_args = mock_extract.call_args
        assert len(call_args[0][0]) == 3

    def test_group_detects_duplicates(self, tmp_path: Path) -> None:
        """Processing same folder twice should detect duplicate."""
        from alibi.config import Config
        from alibi.db.connection import DatabaseManager
        from alibi.processing.pipeline import ProcessingPipeline

        # Create isolated DB
        config = Config(db_path=tmp_path / "test.db", _env_file=None)
        db = DatabaseManager(config)
        db.initialize()

        doc_dir = tmp_path / "pages"
        doc_dir.mkdir()
        (doc_dir / "page1.jpg").write_bytes(b"\xff\xd8\xff" + bytes(50))
        files = sorted(doc_dir.glob("*.jpg"))

        with (
            patch("alibi.processing.pipeline.extract_from_images") as mock_extract,
            patch("alibi.processing.pipeline.compute_file_hash") as mock_hash,
            patch(
                "alibi.processing.pipeline.vision_detect_document_type",
                return_value="receipt",
            ),
        ):
            mock_hash.return_value = "fixed-hash"
            mock_extract.return_value = {
                "vendor": "Store",
                "total": 10.0,
                "currency": "EUR",
                "raw_text": "test",
            }

            pipeline = ProcessingPipeline(db=db)
            result1 = pipeline.process_document_group(doc_dir, files)
            result2 = pipeline.process_document_group(doc_dir, files)

        assert result1.success
        assert not result1.is_duplicate
        assert result2.success
        assert result2.is_duplicate

    def test_type_to_str_mapping(self) -> None:
        """_type_to_str should map DocumentType to extraction string."""
        from alibi.db.models import DocumentType
        from alibi.processing.pipeline import ProcessingPipeline

        assert ProcessingPipeline._type_to_str(DocumentType.RECEIPT) == "receipt"
        assert ProcessingPipeline._type_to_str(DocumentType.INVOICE) == "invoice"
        assert ProcessingPipeline._type_to_str(DocumentType.OTHER) == "receipt"

    def test_merge_extraction_combines_line_items(self) -> None:
        """_merge_extraction should combine line items from multiple pages."""
        from alibi.processing.pipeline import ProcessingPipeline

        base: dict[str, Any] = {
            "vendor": "Store",
            "total": 100.0,
            "line_items": [{"name": "Item A"}],
            "raw_text": "Page 1 text",
        }
        extra: dict[str, Any] = {
            "line_items": [{"name": "Item B"}, {"name": "Item C"}],
            "raw_text": "Page 2 text",
        }

        ProcessingPipeline._merge_extraction(base, extra)

        assert len(base["line_items"]) == 3
        assert base["line_items"][0]["name"] == "Item A"
        assert base["line_items"][2]["name"] == "Item C"
        assert "Page 1 text" in base["raw_text"]
        assert "Page 2 text" in base["raw_text"]
        assert base["vendor"] == "Store"  # Not overwritten

    def test_merge_extraction_fills_missing_fields(self) -> None:
        """_merge_extraction should fill missing scalar fields from extra."""
        from alibi.processing.pipeline import ProcessingPipeline

        base: dict[str, Any] = {"vendor": None, "total": None, "raw_text": ""}
        extra: dict[str, Any] = {"vendor": "Found Store", "total": 55.0}

        ProcessingPipeline._merge_extraction(base, extra)

        assert base["vendor"] == "Found Store"
        assert base["total"] == 55.0


class TestDateValidation:
    """Tests for date plausibility validation."""

    def _pipeline(self) -> Any:
        from alibi.processing.pipeline import ProcessingPipeline

        return ProcessingPipeline()

    def test_valid_date_in_raw_text(self):
        """Date matching raw text is kept."""
        p = self._pipeline()
        result = p._validate_date(date(2026, 2, 17), "Receipt 17/02/2026 12:38:37")
        assert result == date(2026, 2, 17)
        p.close()

    def test_wrong_date_corrected_by_raw_text(self):
        """LLM date not in raw text is replaced by raw text date."""
        p = self._pipeline()
        result = p._validate_date(
            date(2023, 3, 13),  # LLM hallucinated
            "Terminal ID 0045930-01-3-13 13062\n17/02/2026 12:38:37",
        )
        assert result == date(2026, 2, 17)
        p.close()

    def test_ancient_date_rejected_no_raw_text(self):
        """Date >2 years old with no raw text is rejected."""
        p = self._pipeline()
        result = p._validate_date(date(2010, 1, 1), None)
        assert result is None
        p.close()

    def test_no_raw_text_sane_date_kept(self):
        """Recent date with no raw text is kept."""
        p = self._pipeline()
        recent = date.today()
        result = p._validate_date(recent, None)
        assert result == recent
        p.close()

    def test_none_date_passthrough(self):
        """None date stays None."""
        p = self._pipeline()
        result = p._validate_date(None, "some text 01/01/2026")
        assert result is None
        p.close()

    def test_extract_dates_from_text(self):
        """Multiple date formats extracted from raw text."""
        from alibi.processing.pipeline import ProcessingPipeline

        dates = ProcessingPipeline._extract_dates_from_text(
            "Date: 17/02/2026 and 2025-12-15 and 01.06.2025"
        )
        assert date(2026, 2, 17) in dates
        assert date(2025, 12, 15) in dates
        assert date(2025, 6, 1) in dates

    def test_extract_dates_ignores_non_dates(self):
        """Number sequences that aren't valid dates are ignored."""
        from alibi.processing.pipeline import ProcessingPipeline

        dates = ProcessingPipeline._extract_dates_from_text(
            "ID: 45930-01-3-13 amount: 99.99"
        )
        # 01-3-13 is not a valid YYYY-MM-DD pattern (year too short)
        assert len(dates) == 0

    def test_multiple_raw_dates_picks_most_recent(self):
        """When multiple valid dates in raw text, picks the most recent."""
        p = self._pipeline()
        result = p._validate_date(
            date(2020, 1, 1),  # Wrong
            "Old date: 01/01/2025\nNew date: 15/06/2025",
        )
        assert result == date(2025, 6, 15)
        p.close()


class TestExtractFromImages:
    """Tests for multi-image vision extraction."""

    def test_extract_from_images_single_delegates(self, tmp_path: Path) -> None:
        """Single image should delegate to extract_from_image."""
        from alibi.extraction.vision import extract_from_images

        img = tmp_path / "single.jpg"
        img.write_bytes(b"\xff\xd8\xff" + bytes(50))

        with patch("alibi.extraction.vision.extract_from_image") as mock_single:
            mock_single.return_value = {"vendor": "Test"}
            result = extract_from_images([img])

        mock_single.assert_called_once_with(
            img,
            "receipt",
            None,
            None,
            180.0,
            skip_llm_threshold=None,
            country=None,
            hints=None,
        )
        assert result["vendor"] == "Test"

    def test_extract_from_images_empty_raises(self) -> None:
        """Empty image list should raise VisionExtractionError."""
        from alibi.extraction.vision import VisionExtractionError, extract_from_images

        with pytest.raises(VisionExtractionError, match="No image paths"):
            extract_from_images([])

    def test_extract_from_images_missing_file_raises(self, tmp_path: Path) -> None:
        """Missing file should raise VisionExtractionError."""
        from alibi.extraction.vision import VisionExtractionError, extract_from_images

        with pytest.raises(VisionExtractionError, match="not found"):
            extract_from_images([tmp_path / "missing1.jpg", tmp_path / "missing2.jpg"])

    def test_extract_from_images_multi_calls_ollama(self, tmp_path: Path) -> None:
        """Multiple images: OCR each page, parse, structure combined text."""
        from alibi.extraction.vision import extract_from_images

        for i in range(3):
            (tmp_path / f"page{i}.jpg").write_bytes(b"\xff\xd8\xff" + bytes(20 + i))

        files = sorted(tmp_path.glob("*.jpg"))

        with (
            patch("alibi.extraction.ocr.ocr_image_with_retry") as mock_ocr,
            patch("alibi.extraction.structurer._call_ollama_text") as mock_text,
        ):
            mock_ocr.side_effect = [
                ("Page 1 text content", False),
                ("Page 2 text content", False),
                ("Page 3 text content", False),
            ]
            mock_text.return_value = {
                "response": '{"vendor": "Multi-Page Store", "total": 99.0}'
            }

            result = extract_from_images(files)

        # OCR called once per page via ocr_image_with_retry
        assert mock_ocr.call_count == 3
        # Short text → parser confidence low → full LLM structuring
        assert mock_text.call_count == 1
        # Prompt should contain page markers
        prompt_arg = mock_text.call_args[0][2]
        assert "PAGE 1" in prompt_arg
        assert result["vendor"] == "Multi-Page Store"
