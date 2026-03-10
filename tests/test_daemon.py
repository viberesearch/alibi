"""Tests for the daemon module."""

import os
import signal
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alibi.daemon.handlers import DocumentHandler, on_document_created
from alibi.daemon.watcher_service import (
    DaemonStatus,
    DebouncedEventHandler,
    WatchedFile,
    WatcherDaemon,
)


class TestWatchedFile:
    """Tests for WatchedFile dataclass."""

    def test_create_watched_file(self, tmp_path: Path) -> None:
        """Test creating a WatchedFile."""
        test_file = tmp_path / "test.pdf"
        test_file.touch()

        watched = WatchedFile(
            path=test_file,
            first_seen=time.time(),
            last_modified=time.time(),
            size=0,
        )

        assert watched.path == test_file
        assert watched.size == 0


class TestDaemonStatus:
    """Tests for DaemonStatus dataclass."""

    def test_create_daemon_status(self) -> None:
        """Test creating a DaemonStatus."""
        status = DaemonStatus(
            running=True,
            pid=12345,
            inbox_path=Path("/test/inbox"),
            pending_files=5,
            processed_count=10,
        )

        assert status.running is True
        assert status.pid == 12345
        assert status.pending_files == 5
        assert status.processed_count == 10

    def test_daemon_status_defaults(self) -> None:
        """Test DaemonStatus default values."""
        status = DaemonStatus(running=False)

        assert status.running is False
        assert status.pid is None
        assert status.inbox_path is None
        assert status.pending_files == 0


class TestDebouncedEventHandler:
    """Tests for DebouncedEventHandler."""

    def test_supported_file_check(self) -> None:
        """Test file extension support check."""
        callback = MagicMock()
        handler = DebouncedEventHandler(callback, debounce_seconds=0.1)

        assert handler._is_supported(Path("test.pdf")) is True
        assert handler._is_supported(Path("test.jpg")) is True
        assert handler._is_supported(Path("test.png")) is True
        assert handler._is_supported(Path("test.txt")) is False
        assert handler._is_supported(Path("test.doc")) is False

    def test_add_pending(self, tmp_path: Path) -> None:
        """Test adding file to pending queue."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test content")

        callback = MagicMock()
        handler = DebouncedEventHandler(callback, debounce_seconds=0.1)

        handler._add_pending(test_file)

        assert str(test_file) in handler.pending
        assert handler.pending[str(test_file)].size > 0

    def test_update_pending(self, tmp_path: Path) -> None:
        """Test updating pending file."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test content")

        callback = MagicMock()
        handler = DebouncedEventHandler(callback, debounce_seconds=0.1)

        handler._add_pending(test_file)
        original_time = handler.pending[str(test_file)].last_modified

        time.sleep(0.05)
        handler._update_pending(test_file)

        assert handler.pending[str(test_file)].last_modified >= original_time

    def test_process_pending_stable_file(self, tmp_path: Path) -> None:
        """Test processing a stable file."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test content")

        callback = MagicMock()
        handler = DebouncedEventHandler(callback, debounce_seconds=0.1)

        handler._add_pending(test_file)

        # Wait for debounce
        time.sleep(0.15)

        processed = handler.process_pending()

        assert len(processed) == 1
        assert processed[0] == test_file
        callback.assert_called_once_with(test_file)
        assert handler.processed_count == 1

    def test_process_pending_unstable_file(self, tmp_path: Path) -> None:
        """Test that unstable files are not processed."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test content")

        callback = MagicMock()
        handler = DebouncedEventHandler(callback, debounce_seconds=1.0)

        handler._add_pending(test_file)

        # Don't wait for debounce
        processed = handler.process_pending()

        assert len(processed) == 0
        callback.assert_not_called()

    def test_process_pending_deleted_file(self, tmp_path: Path) -> None:
        """Test handling of deleted files."""
        test_file = tmp_path / "test.pdf"
        test_file.write_text("test content")

        callback = MagicMock()
        handler = DebouncedEventHandler(callback, debounce_seconds=0.1)

        handler._add_pending(test_file)

        # Delete the file
        test_file.unlink()

        time.sleep(0.15)
        processed = handler.process_pending()

        # File should be removed from pending but not processed
        assert len(processed) == 0
        assert str(test_file) not in handler.pending

    def test_pending_count(self, tmp_path: Path) -> None:
        """Test pending count property."""
        callback = MagicMock()
        handler = DebouncedEventHandler(callback, debounce_seconds=1.0)

        file1 = tmp_path / "test1.pdf"
        file2 = tmp_path / "test2.jpg"
        file1.write_text("content1")
        file2.write_text("content2")

        handler._add_pending(file1)
        handler._add_pending(file2)

        assert handler.pending_count == 2


class TestWatcherDaemon:
    """Tests for WatcherDaemon."""

    def test_init_with_defaults(self) -> None:
        """Test daemon initialization with defaults."""
        with patch("alibi.daemon.watcher_service.get_config") as mock_config:
            mock_config.return_value.get_inbox_path.return_value = Path("/test/inbox")
            daemon = WatcherDaemon()

            assert daemon.debounce_seconds == 2.0
            assert daemon.poll_interval == 0.5
            assert daemon.recursive is True

    def test_init_with_custom_path(self, tmp_path: Path) -> None:
        """Test daemon initialization with custom inbox path."""
        daemon = WatcherDaemon(inbox_path=tmp_path)

        assert daemon.inbox_path == tmp_path

    def test_is_running_initially_false(self, tmp_path: Path) -> None:
        """Test that daemon is not running initially."""
        daemon = WatcherDaemon(inbox_path=tmp_path)

        assert daemon.is_running() is False

    def test_get_status_not_running(self, tmp_path: Path) -> None:
        """Test getting status when not running."""
        daemon = WatcherDaemon(inbox_path=tmp_path)
        status = daemon.get_status()

        assert status.running is False
        assert status.pid is None

    @patch("alibi.daemon.watcher_service.get_config")
    def test_start_missing_inbox(self, mock_config: MagicMock) -> None:
        """Test start fails with missing inbox path."""
        mock_config.return_value.get_inbox_path.return_value = None
        daemon = WatcherDaemon(inbox_path=None)

        with pytest.raises(ValueError, match="No inbox path configured"):
            daemon.start(foreground=False)

    def test_start_nonexistent_inbox(self, tmp_path: Path) -> None:
        """Test start fails with nonexistent inbox path."""
        nonexistent = tmp_path / "nonexistent"
        daemon = WatcherDaemon(inbox_path=nonexistent)

        with pytest.raises(ValueError, match="Inbox path does not exist"):
            daemon.start(foreground=False)

    @patch("alibi.daemon.handlers.DocumentHandler")
    def test_start_and_stop(self, mock_handler: MagicMock, tmp_path: Path) -> None:
        """Test starting and stopping the daemon."""
        inbox = tmp_path / "inbox"
        inbox.mkdir()

        daemon = WatcherDaemon(inbox_path=inbox, poll_interval=0.1)

        # Start in a thread since foreground=False doesn't block but we want to test quickly
        def run_daemon():
            daemon.start(foreground=False)
            time.sleep(0.2)

        thread = threading.Thread(target=run_daemon)
        thread.start()

        time.sleep(0.1)

        assert daemon.is_running() is True
        status = daemon.get_status()
        assert status.running is True

        daemon.stop()
        thread.join(timeout=1.0)

        assert daemon.is_running() is False

    def test_get_running_status_no_pid_file(self, tmp_path: Path) -> None:
        """Test get_running_status when no PID file exists."""
        with patch(
            "alibi.daemon.watcher_service.PID_FILE", tmp_path / "nonexistent.pid"
        ):
            status = WatcherDaemon.get_running_status()
            assert status is None

    def test_get_running_status_stale_pid(self, tmp_path: Path) -> None:
        """Test get_running_status with stale PID file."""
        pid_file = tmp_path / "watcher.pid"
        pid_file.write_text("99999999")  # Non-existent process

        with patch("alibi.daemon.watcher_service.PID_FILE", pid_file):
            status = WatcherDaemon.get_running_status()
            assert status is None


class TestDocumentHandler:
    """Tests for DocumentHandler."""

    def test_init(self) -> None:
        """Test DocumentHandler initialization."""
        handler = DocumentHandler()

        assert handler.space_id == "default"
        assert handler.user_id == "daemon"

    def test_init_custom_settings(self) -> None:
        """Test DocumentHandler with custom settings."""
        handler = DocumentHandler(space_id="custom", user_id="test_user")

        assert handler.space_id == "custom"
        assert handler.user_id == "test_user"

    def test_on_document_created_nonexistent(self, tmp_path: Path) -> None:
        """Test handling nonexistent file."""
        handler = DocumentHandler()
        nonexistent = tmp_path / "nonexistent.pdf"

        result = handler.on_document_created(nonexistent)

        assert result.success is False
        assert result.error is not None
        assert "no longer exists" in result.error.lower()

    def test_on_document_created_unsupported(self, tmp_path: Path) -> None:
        """Test handling unsupported file type."""
        handler = DocumentHandler()
        unsupported = tmp_path / "test.txt"
        unsupported.write_text("test")

        result = handler.on_document_created(unsupported)

        assert result.success is False
        assert result.error is not None
        assert "unsupported" in result.error.lower()

    @patch("alibi.services.ingestion.process_file")
    @patch("alibi.daemon.handlers.get_db")
    def test_on_document_created_success(
        self, mock_get_db: MagicMock, mock_svc_process: MagicMock, tmp_path: Path
    ) -> None:
        """Test successful document processing via service layer."""
        from alibi.processing.pipeline import ProcessingResult

        # Setup mocks
        mock_db = MagicMock()
        mock_db.is_initialized.return_value = True
        mock_get_db.return_value = mock_db

        mock_svc_process.return_value = ProcessingResult(
            success=True,
            file_path=tmp_path / "test.pdf",
            document_id="test-artifact-id",
            extracted_data={"vendor": "Test Vendor", "total": "100.00"},
        )

        # Create test file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test content")

        handler = DocumentHandler()
        result = handler.on_document_created(test_file)

        assert result.success is True
        assert result.document_id == "test-artifact-id"
        # Watcher always sets provenance
        call_args = mock_svc_process.call_args
        assert call_args[0] == (mock_db, test_file)
        ctx = call_args[1]["folder_context"]
        assert ctx.source == "watcher"
        assert ctx.user_id == "system"

    @patch("alibi.daemon.handlers.get_db")
    def test_on_document_modified_already_processed(
        self, mock_get_db: MagicMock, tmp_path: Path
    ) -> None:
        """Test that already-processed documents are skipped on modification."""
        # Setup mocks
        mock_db = MagicMock()
        mock_db.is_initialized.return_value = True
        mock_db.fetchone.return_value = ("existing-artifact-id",)
        mock_get_db.return_value = mock_db

        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test content")

        handler = DocumentHandler()
        result = handler.on_document_modified(test_file)

        assert result is None

    def test_close(self) -> None:
        """Test handler cleanup."""
        handler = DocumentHandler()
        handler.close()  # Should not raise


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    @patch("alibi.daemon.handlers._get_handler")
    def test_on_document_created_function(
        self, mock_get_handler: MagicMock, tmp_path: Path
    ) -> None:
        """Test module-level on_document_created function."""
        mock_handler = MagicMock()
        mock_get_handler.return_value = mock_handler

        test_file = tmp_path / "test.pdf"
        on_document_created(test_file)

        mock_handler.on_document_created.assert_called_once_with(test_file)


class TestServiceFiles:
    """Tests for service configuration files."""

    def test_systemd_service_file_exists(self) -> None:
        """Test that systemd service file exists."""
        from pathlib import Path

        service_file = (
            Path(__file__).parent.parent
            / "alibi"
            / "daemon"
            / "systemd"
            / "alibi-watcher.service"
        )
        assert service_file.exists()

    def test_launchd_plist_file_exists(self) -> None:
        """Test that launchd plist file exists."""
        from pathlib import Path

        plist_file = (
            Path(__file__).parent.parent
            / "alibi"
            / "daemon"
            / "launchd"
            / "com.alibi.watcher.plist"
        )
        assert plist_file.exists()

    def test_systemd_service_content(self) -> None:
        """Test systemd service file has required sections."""
        from pathlib import Path

        service_file = (
            Path(__file__).parent.parent
            / "alibi"
            / "daemon"
            / "systemd"
            / "alibi-watcher.service"
        )
        content = service_file.read_text()

        assert "[Unit]" in content
        assert "[Service]" in content
        assert "[Install]" in content
        assert "alibi.daemon.watcher_service" in content

    def test_launchd_plist_content(self) -> None:
        """Test launchd plist file has required elements."""
        from pathlib import Path

        plist_file = (
            Path(__file__).parent.parent
            / "alibi"
            / "daemon"
            / "launchd"
            / "com.alibi.watcher.plist"
        )
        content = plist_file.read_text()

        assert "<key>Label</key>" in content
        assert "com.alibi.watcher" in content
        assert "alibi.daemon.watcher_service" in content


class TestNoteSubscriber:
    """Tests for NoteSubscriber event-driven note generation."""

    def test_subscribe_and_unsubscribe(self) -> None:
        """Test that NoteSubscriber registers and unregisters handlers."""
        from alibi.services.events import EventBus, EventType
        from alibi.services.subscribers.notes import NoteSubscriber

        bus = EventBus()
        subscriber = NoteSubscriber(
            db_factory=MagicMock,
            vault_path=Path("/fake/vault"),
            bus=bus,
        )
        subscriber.start()
        assert len(bus._subscribers[EventType.DOCUMENT_INGESTED]) == 1

        subscriber.stop()
        assert len(bus._subscribers[EventType.DOCUMENT_INGESTED]) == 0

    def test_skips_duplicates(self) -> None:
        """Test that duplicate documents don't trigger note generation."""
        from alibi.services.events import EventBus, EventType
        from alibi.services.subscribers.notes import NoteSubscriber

        bus = EventBus()
        subscriber = NoteSubscriber(
            db_factory=MagicMock,
            vault_path=Path("/fake/vault"),
            bus=bus,
        )
        subscriber.start()

        with patch.object(subscriber, "_generate_notes") as mock_gen:
            bus.emit(
                EventType.DOCUMENT_INGESTED,
                {"document_id": "doc-1", "is_duplicate": True},
            )
            mock_gen.assert_not_called()

    def test_calls_generate_notes_for_new_doc(self) -> None:
        """Test that non-duplicate documents trigger note generation."""
        from alibi.services.events import EventBus, EventType
        from alibi.services.subscribers.notes import NoteSubscriber

        bus = EventBus()
        subscriber = NoteSubscriber(
            db_factory=MagicMock,
            vault_path=Path("/fake/vault"),
            bus=bus,
        )
        subscriber.start()

        with patch.object(subscriber, "_generate_notes") as mock_gen:
            bus.emit(
                EventType.DOCUMENT_INGESTED,
                {"document_id": "doc-1", "is_duplicate": False},
            )
            mock_gen.assert_called_once_with("doc-1")

    def test_skips_missing_document_id(self) -> None:
        """Test that events without document_id are ignored."""
        from alibi.services.events import EventBus, EventType
        from alibi.services.subscribers.notes import NoteSubscriber

        bus = EventBus()
        subscriber = NoteSubscriber(
            db_factory=MagicMock,
            vault_path=Path("/fake/vault"),
            bus=bus,
        )
        subscriber.start()

        with patch.object(subscriber, "_generate_notes") as mock_gen:
            bus.emit(
                EventType.DOCUMENT_INGESTED,
                {"is_duplicate": False},
            )
            mock_gen.assert_not_called()
