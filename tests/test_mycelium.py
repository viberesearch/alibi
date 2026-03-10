"""Tests for Mycelium integration module."""

from __future__ import annotations

import os
import tempfile
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alibi.db.models import Artifact, DocumentStatus, DocumentType


class TestArtifactNoteGeneration:
    """Tests for artifact note generation."""

    def test_generate_artifact_note_basic(self) -> None:
        """Test basic artifact note generation."""
        from alibi.mycelium.notes import generate_artifact_note

        artifact = Artifact(
            id="test-123",
            space_id="default",
            type=DocumentType.RECEIPT,
            file_path="/path/to/receipt.jpg",
            file_hash="abc123",
            vendor="Test Store",
            document_date=date(2024, 1, 15),
            amount=Decimal("42.50"),
            currency="EUR",
            status=DocumentStatus.PROCESSED,
        )

        note = generate_artifact_note(artifact)

        assert "type: document" in note
        assert 'artifact_id: "test-123"' in note
        assert "document_type: receipt" in note
        assert 'vendor: "Test Store"' in note
        assert "date: 2024-01-15" in note
        assert "receipt.jpg" in note
        # Amount is formatted with Euro symbol
        assert "42.50" in note

    def test_generate_artifact_note_with_tags(self) -> None:
        """Test artifact note with tags."""
        from alibi.mycelium.notes import generate_artifact_note

        artifact = Artifact(
            id="test-456",
            space_id="default",
            type=DocumentType.INVOICE,
            file_path="/path/to/invoice.pdf",
            file_hash="def456",
            vendor="Vendor Co",
            status=DocumentStatus.PROCESSED,
        )

        note = generate_artifact_note(artifact, tags=["work", "expenses"])

        assert '"work"' in note
        assert '"expenses"' in note

    def test_generate_artifact_note_with_extracted_data(self) -> None:
        """Test artifact note includes extracted data."""
        from alibi.mycelium.notes import generate_artifact_note

        artifact = Artifact(
            id="test-789",
            space_id="default",
            type=DocumentType.WARRANTY,
            file_path="/path/to/warranty.pdf",
            file_hash="ghi789",
            vendor="Electronics Store",
            extracted_data={
                "vendor": "Electronics Store",
                "warranty_period": "2 years",
                "product_name": "Laptop",
            },
            status=DocumentStatus.PROCESSED,
        )

        note = generate_artifact_note(artifact)

        assert "Warranty Period" in note
        assert "2 years" in note
        assert "Product Name" in note
        assert "Laptop" in note

    def test_get_artifact_note_filename(self) -> None:
        """Test artifact filename generation."""
        from alibi.mycelium.notes import get_artifact_note_filename

        artifact = Artifact(
            id="test-123",
            space_id="default",
            type=DocumentType.RECEIPT,
            file_path="/path/to/file.jpg",
            file_hash="abc",
            vendor="Test Store",
            document_date=date(2024, 3, 20),
            status=DocumentStatus.PROCESSED,
        )

        filename = get_artifact_note_filename(artifact)

        assert filename == "2024-03-20_receipt_Test Store"

    def test_get_artifact_note_filename_sanitizes(self) -> None:
        """Test filename sanitization."""
        from alibi.mycelium.notes import get_artifact_note_filename

        artifact = Artifact(
            id="test-123",
            space_id="default",
            type=DocumentType.INVOICE,
            file_path="/path/to/file.pdf",
            file_hash="abc",
            vendor="Company/Name:With*Bad?Chars",
            document_date=date(2024, 1, 1),
            status=DocumentStatus.PROCESSED,
        )

        filename = get_artifact_note_filename(artifact)

        # Should not contain problematic characters
        assert "/" not in filename
        assert ":" not in filename
        assert "*" not in filename
        assert "?" not in filename


class TestArtifactNoteExporter:
    """Tests for artifact note exporter."""

    def test_exporter_requires_vault_path(self) -> None:
        """Test that exporter requires vault path."""
        from alibi.mycelium.notes import ArtifactNoteExporter

        mock_db = MagicMock()

        with patch("alibi.mycelium.notes.get_config") as mock_config:
            mock_config.return_value.vault_path = None

            with pytest.raises(ValueError, match="No vault path configured"):
                ArtifactNoteExporter(db=mock_db)

    def test_exporter_creates_note_directory(self) -> None:
        """Test that exporter creates note directories."""
        from alibi.mycelium.notes import ArtifactNoteExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir)
            mock_db = MagicMock()

            with patch("alibi.mycelium.notes.get_config") as mock_config:
                mock_config.return_value.vault_path = vault_path

                exporter = ArtifactNoteExporter(db=mock_db, vault_path=vault_path)

                artifact = Artifact(
                    id="test-123",
                    space_id="default",
                    type=DocumentType.RECEIPT,
                    file_path="/path/to/receipt.jpg",
                    file_hash="abc",
                    vendor="Test Store",
                    document_date=date(2024, 1, 15),
                    status=DocumentStatus.PROCESSED,
                )

                note_path = exporter.export_artifact(artifact)

                assert note_path.exists()
                assert (
                    note_path.parent == vault_path / "vault" / "documents" / "receipts"
                )
                assert note_path.suffix == ".md"

    def test_exporter_avoids_overwrite(self) -> None:
        """Test that exporter adds suffix to avoid overwriting."""
        from alibi.mycelium.notes import ArtifactNoteExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir)
            mock_db = MagicMock()

            with patch("alibi.mycelium.notes.get_config") as mock_config:
                mock_config.return_value.vault_path = vault_path

                exporter = ArtifactNoteExporter(db=mock_db, vault_path=vault_path)

                artifact = Artifact(
                    id="test-123",
                    space_id="default",
                    type=DocumentType.RECEIPT,
                    file_path="/path/to/receipt.jpg",
                    file_hash="abc",
                    vendor="Test Store",
                    document_date=date(2024, 1, 15),
                    status=DocumentStatus.PROCESSED,
                )

                path1 = exporter.export_artifact(artifact)
                path2 = exporter.export_artifact(artifact, overwrite=False)

                assert path1 != path2
                assert "_1.md" in str(path2)


class TestMyceliumWatcher:
    """Tests for Mycelium watcher."""

    def test_watcher_default_paths(self) -> None:
        """Test watcher uses default paths."""
        from alibi.mycelium.watcher import (
            DEFAULT_INBOX_SUBPATH,
            DEFAULT_VAULT_PATH,
            MyceliumWatcher,
        )

        with patch("alibi.mycelium.watcher.get_config") as mock_config:
            mock_config.return_value.vault_path = None
            mock_config.return_value.get_inbox_path.return_value = None

            watcher = MyceliumWatcher()

            assert watcher.vault_path == DEFAULT_VAULT_PATH
            assert watcher.inbox_subpath == DEFAULT_INBOX_SUBPATH
            assert watcher.inbox_path == DEFAULT_VAULT_PATH / DEFAULT_INBOX_SUBPATH

    def test_watcher_custom_paths(self) -> None:
        """Test watcher with custom paths."""
        from alibi.mycelium.watcher import MyceliumWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir)

            with patch("alibi.mycelium.watcher.get_config") as mock_config:
                mock_config.return_value.vault_path = None

                watcher = MyceliumWatcher(
                    vault_path=vault_path,
                    inbox_subpath="custom/inbox",
                )

                assert watcher.vault_path == vault_path
                assert watcher.inbox_path == vault_path / "custom" / "inbox"

    def test_watcher_status_not_running(self) -> None:
        """Test watcher status when not running."""
        from alibi.mycelium.watcher import MyceliumWatcher

        status = MyceliumWatcher.get_running_status()
        # Should return None when no PID file exists
        assert status is None or not status.running

    def test_watcher_scan_empty_inbox(self) -> None:
        """Test scanning empty inbox."""
        from alibi.mycelium.watcher import MyceliumWatcher

        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir)
            inbox_path = vault_path / "inbox" / "documents"
            inbox_path.mkdir(parents=True)

            with patch("alibi.mycelium.watcher.get_config") as mock_config:
                mock_config.return_value.vault_path = vault_path

                watcher = MyceliumWatcher(vault_path=vault_path)
                results = watcher.scan_inbox()

                assert results == []


class TestMyceliumStatus:
    """Tests for Mycelium status dataclass."""

    def test_status_defaults(self) -> None:
        """Test MyceliumStatus default values."""
        from alibi.mycelium.watcher import MyceliumStatus

        status = MyceliumStatus(running=False)

        assert status.running is False
        assert status.pid is None
        assert status.vault_path is None
        assert status.inbox_path is None
        assert status.pending_files == 0
        assert status.processed_count == 0
        assert status.notes_generated == 0
        assert status.last_activity is None
        assert status.last_sync is None
        assert status.uptime_seconds == 0.0

    def test_status_with_values(self) -> None:
        """Test MyceliumStatus with values."""
        from datetime import datetime
        from alibi.mycelium.watcher import MyceliumStatus

        now = datetime.now()
        status = MyceliumStatus(
            running=True,
            pid=12345,
            vault_path=Path("/test/vault"),
            pending_files=3,
            processed_count=10,
            notes_generated=8,
            last_activity=now,
            uptime_seconds=3600.5,
        )

        assert status.running is True
        assert status.pid == 12345
        assert status.pending_files == 3
        assert status.notes_generated == 8


class TestSyncDetector:
    """Tests for git sync detection."""

    def test_detector_init(self) -> None:
        """Test SyncDetector initialization."""
        from alibi.mycelium.sync import SyncDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir)

            detector = SyncDetector(vault_path=vault_path)

            assert detector.vault_path == vault_path
            assert detector.inbox_path == vault_path / "inbox" / "documents"

    def test_detector_custom_inbox(self) -> None:
        """Test SyncDetector with custom inbox."""
        from alibi.mycelium.sync import SyncDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir)

            detector = SyncDetector(
                vault_path=vault_path,
                inbox_subpath="custom/path",
            )

            assert detector.inbox_path == vault_path / "custom" / "path"

    def test_detector_not_git_repo(self) -> None:
        """Test detector on non-git directory."""
        from alibi.mycelium.sync import SyncDetector

        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir)

            detector = SyncDetector(vault_path=vault_path)
            commit = detector.get_current_commit()

            assert commit is None


class TestSyncStatus:
    """Tests for SyncStatus dataclass."""

    def test_sync_status_success(self) -> None:
        """Test SyncStatus for successful sync."""
        from alibi.mycelium.sync import SyncStatus

        status = SyncStatus(
            success=True,
            new_files=[Path("/test/file1.jpg"), Path("/test/file2.pdf")],
            modified_files=[Path("/test/file3.png")],
            commit_hash="abc123def",
            commit_message="iOS sync",
        )

        assert status.success is True
        assert len(status.new_files) == 2
        assert len(status.modified_files) == 1
        assert status.commit_hash == "abc123def"
        assert status.error is None

    def test_sync_status_failure(self) -> None:
        """Test SyncStatus for failed sync."""
        from alibi.mycelium.sync import SyncStatus

        status = SyncStatus(
            success=False,
            new_files=[],
            modified_files=[],
            error="Not a git repository",
        )

        assert status.success is False
        assert status.error == "Not a git repository"


class TestPostPullHook:
    """Tests for git hook generation."""

    def test_create_hook_content(self) -> None:
        """Test hook content generation."""
        from alibi.mycelium.sync import create_post_pull_hook

        content = create_post_pull_hook()

        assert "#!/bin/bash" in content
        assert "post-merge" in content.lower()
        assert "lt mycelium scan" in content
        assert "LOG_FILE" in content

    def test_install_hook(self) -> None:
        """Test hook installation."""
        from alibi.mycelium.sync import install_post_pull_hook

        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir)
            git_dir = vault_path / ".git" / "hooks"
            git_dir.mkdir(parents=True)

            hook_path = install_post_pull_hook(vault_path)

            assert hook_path.exists()
            assert hook_path.name == "post-merge"
            assert os.access(hook_path, os.X_OK)  # Check executable


class TestMyceliumHandler:
    """Tests for MyceliumHandler."""

    def test_handler_init(self) -> None:
        """Test MyceliumHandler initialization."""
        from alibi.mycelium.watcher import MyceliumHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir)

            handler = MyceliumHandler(
                vault_path=vault_path,
                archive_processed=False,
            )

            assert handler.vault_path == vault_path
            assert handler.archive_processed is False

    def test_handler_archive_enabled(self) -> None:
        """Test handler with archiving enabled."""
        from alibi.mycelium.watcher import MyceliumHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir)

            handler = MyceliumHandler(
                vault_path=vault_path,
                archive_processed=True,
            )

            assert handler.archive_processed is True
