"""Mycelium vault watcher for processing documents from iOS sync.

This watcher monitors the Obsidian vault inbox for new documents
and processes them through the Alibi pipeline.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from watchdog.observers import Observer

from alibi.config import get_config
from alibi.daemon.handlers import DocumentHandler
from alibi.daemon.watcher_service import DebouncedEventHandler
from alibi.processing.folder_router import scan_inbox_recursive
from alibi.processing.pipeline import ProcessingResult
from alibi.services.subscribers.analytics import AnalyticsExportSubscriber
from alibi.services.subscribers.enrichment import EnrichmentSubscriber
from alibi.services.subscribers.notes import NoteSubscriber

logger = logging.getLogger(__name__)

# Default mycelium paths
DEFAULT_VAULT_PATH = Path.home() / "Obsidian" / "vault"
DEFAULT_INBOX_SUBPATH = "inbox/documents"

# PID and log files for mycelium watcher
MYCELIUM_PID_FILE = Path.home() / ".alibi" / "mycelium.pid"
MYCELIUM_LOG_FILE = Path.home() / ".alibi" / "mycelium.log"


@dataclass
class MyceliumStatus:
    """Status of the Mycelium watcher."""

    running: bool
    pid: Optional[int] = None
    vault_path: Optional[Path] = None
    inbox_path: Optional[Path] = None
    pending_files: int = 0
    processed_count: int = 0
    notes_generated: int = 0
    items_enriched: int = 0
    last_activity: Optional[datetime] = None
    last_sync: Optional[datetime] = None
    uptime_seconds: float = 0.0


class MyceliumHandler(DocumentHandler):
    """Extended document handler with archiving support."""

    def __init__(
        self,
        vault_path: Path,
        archive_processed: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the Mycelium document handler.

        Args:
            vault_path: Path to the Obsidian vault
            archive_processed: Whether to move processed files to archive
            **kwargs: Arguments passed to DocumentHandler
        """
        super().__init__(**kwargs)
        self.vault_path = vault_path
        self.archive_processed = archive_processed

    def on_document_created(self, path: Path) -> ProcessingResult:
        """Handle new document creation with optional archiving.

        Note generation is handled by NoteSubscriber via the event bus,
        not by this handler.

        Args:
            path: Path to the new document

        Returns:
            ProcessingResult from the pipeline
        """
        result = super().on_document_created(path)

        if result.success and self.archive_processed:
            try:
                self._archive_file(path)
            except Exception as e:
                logger.error(f"Failed to archive {path.name}: {e}")

        return result

    def _archive_file(self, path: Path) -> None:
        """Move processed file to archive folder.

        Args:
            path: Path to file to archive
        """
        archive_dir = self.vault_path / "vault" / "documents" / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)

        dest = archive_dir / path.name
        if dest.exists():
            # Add timestamp to avoid collision
            stem = path.stem
            suffix = path.suffix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = archive_dir / f"{stem}_{timestamp}{suffix}"

        path.rename(dest)
        logger.info(f"Archived: {path.name} -> {dest.name}")


class MyceliumWatcher:
    """Watcher for Mycelium Obsidian vault integration.

    Monitors the vault inbox for new documents from iOS sync
    and processes them through the Alibi pipeline.
    """

    def __init__(
        self,
        vault_path: Optional[Path] = None,
        inbox_subpath: str = DEFAULT_INBOX_SUBPATH,
        generate_notes: bool = True,
        archive_processed: bool = False,
        debounce_seconds: float = 3.0,
        poll_interval: float = 1.0,
    ) -> None:
        """Initialize the Mycelium watcher.

        Args:
            vault_path: Path to Obsidian vault (defaults to ~/Obsidian/vault)
            inbox_subpath: Subpath within vault to watch (default: inbox/documents)
            generate_notes: Whether to generate Obsidian notes for processed docs
            archive_processed: Whether to move processed files to archive
            debounce_seconds: Wait time for file stability (higher for git sync)
            poll_interval: How often to check pending files
        """
        self.config = get_config()
        self.vault_path = vault_path or self.config.vault_path or DEFAULT_VAULT_PATH
        self.inbox_subpath = inbox_subpath
        self.inbox_path = self.vault_path / inbox_subpath
        self.generate_notes = generate_notes
        self.archive_processed = archive_processed
        self.debounce_seconds = debounce_seconds
        self.poll_interval = poll_interval

        self._observer: Any = None  # Observer type from watchdog
        self._handler: Optional[DebouncedEventHandler] = None
        self._mycelium_handler: Optional[MyceliumHandler] = None
        self._note_subscriber: Optional[NoteSubscriber] = None
        self._analytics_subscriber: Optional[AnalyticsExportSubscriber] = None
        self._enrichment_subscriber: Optional[EnrichmentSubscriber] = None
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._shutdown_event = threading.Event()
        self._last_sync: Optional[datetime] = None

    def _setup_logging(self) -> None:
        """Configure logging for daemon mode."""
        MYCELIUM_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler(MYCELIUM_LOG_FILE)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        if threading.current_thread() is not threading.main_thread():
            logger.debug("Not in main thread, skipping signal handler setup")
            return

        def handle_shutdown(signum: int, frame: Any) -> None:
            logger.info(f"Received signal {signum}, shutting down...")
            self._shutdown_event.set()
            self.stop()

        signal.signal(signal.SIGTERM, handle_shutdown)
        signal.signal(signal.SIGINT, handle_shutdown)

    def _write_pid(self) -> None:
        """Write PID file."""
        MYCELIUM_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        MYCELIUM_PID_FILE.write_text(str(os.getpid()))

    def _remove_pid(self) -> None:
        """Remove PID file."""
        if MYCELIUM_PID_FILE.exists():
            MYCELIUM_PID_FILE.unlink()

    def _poll_loop(self) -> None:
        """Background loop to process pending files."""
        while self._running and not self._shutdown_event.is_set():
            try:
                if self._handler:
                    processed = self._handler.process_pending()
                    if processed:
                        self._last_sync = datetime.now()
            except Exception as e:
                logger.error(f"Error in poll loop: {e}")
            self._shutdown_event.wait(timeout=self.poll_interval)

    def start(self, foreground: bool = True) -> None:
        """Start the Mycelium watcher.

        Args:
            foreground: Run in foreground (True) or background (False)
        """
        if self._running:
            logger.warning("Mycelium watcher already running")
            return

        if not self.inbox_path.exists():
            logger.info(f"Creating inbox directory: {self.inbox_path}")
            self.inbox_path.mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        # Create Mycelium-specific handler
        self._mycelium_handler = MyceliumHandler(
            vault_path=self.vault_path,
            archive_processed=self.archive_processed,
            inbox_root=self.inbox_path,
        )

        # Wire note subscriber (generates notes via event bus)
        if self.generate_notes:
            self._note_subscriber = NoteSubscriber(
                db_factory=self._mycelium_handler._get_db,
                vault_path=self.vault_path,
            )
            self._note_subscriber.start()

        # Wire analytics export subscriber (pushes facts to analytics stack)
        config = get_config()
        if config.analytics_export_enabled and config.analytics_stack_url:
            self._analytics_subscriber = AnalyticsExportSubscriber(
                db_factory=self._mycelium_handler._get_db,
                analytics_url=config.analytics_stack_url,
            )
            self._analytics_subscriber.start()
            logger.info(
                "Analytics export subscriber started -> %s", config.analytics_stack_url
            )

        # Wire enrichment subscriber (OFF barcode lookup + name matching)
        self._enrichment_subscriber = EnrichmentSubscriber(
            db_factory=self._mycelium_handler._get_db,
        )
        self._enrichment_subscriber.start()
        logger.info("Enrichment subscriber started")

        # Create debounced event handler
        self._handler = DebouncedEventHandler(
            process_callback=self._mycelium_handler.on_document_created,
            on_error=self._mycelium_handler.on_document_error,
            debounce_seconds=self.debounce_seconds,
        )

        # Create and start observer
        self._observer = Observer()
        self._observer.schedule(
            self._handler,
            str(self.inbox_path),
            recursive=True,
        )
        self._observer.start()
        self._running = True
        self._start_time = time.time()

        # Start polling thread
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

        self._setup_signal_handlers()
        self._write_pid()

        logger.info(f"Mycelium watcher started: {self.inbox_path} (pid={os.getpid()})")

        if foreground:
            try:
                while self._running and not self._shutdown_event.is_set():
                    self._shutdown_event.wait(timeout=1.0)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.stop()

    def stop(self) -> None:
        """Stop the Mycelium watcher."""
        if not self._running:
            return

        self._running = False
        self._shutdown_event.set()

        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None

        if self._note_subscriber:
            self._note_subscriber.stop()
            self._note_subscriber = None

        if self._analytics_subscriber:
            self._analytics_subscriber.stop()
            self._analytics_subscriber = None

        if self._enrichment_subscriber:
            self._enrichment_subscriber.stop()
            self._enrichment_subscriber = None

        if self._mycelium_handler:
            self._mycelium_handler.close()
            self._mycelium_handler = None

        self._remove_pid()
        logger.info("Mycelium watcher stopped")

    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running

    def get_status(self) -> MyceliumStatus:
        """Get current watcher status."""
        uptime = 0.0
        if self._start_time and self._running:
            uptime = time.time() - self._start_time

        notes_generated = 0
        if self._note_subscriber:
            notes_generated = self._note_subscriber.notes_generated

        items_enriched = 0
        if self._enrichment_subscriber:
            items_enriched = self._enrichment_subscriber.enrichment_count

        return MyceliumStatus(
            running=self._running,
            pid=os.getpid() if self._running else None,
            vault_path=self.vault_path,
            inbox_path=self.inbox_path,
            pending_files=self._handler.pending_count if self._handler else 0,
            processed_count=self._handler.processed_count if self._handler else 0,
            notes_generated=notes_generated,
            items_enriched=items_enriched,
            last_activity=self._handler.last_activity if self._handler else None,
            last_sync=self._last_sync,
            uptime_seconds=uptime,
        )

    def scan_inbox(self) -> list[ProcessingResult]:
        """Manually scan and process all files in inbox.

        Uses scan_inbox_recursive() for consistent filtering (skips
        config files, .alibi.yaml caches, etc.).

        Returns:
            List of processing results
        """
        if not self.inbox_path.exists():
            return []

        results: list[ProcessingResult] = []

        if self._mycelium_handler is None:
            self._mycelium_handler = MyceliumHandler(
                vault_path=self.vault_path,
                archive_processed=self.archive_processed,
                inbox_root=self.inbox_path,
            )

        # Wire note subscriber for scan if not already active
        if self.generate_notes and self._note_subscriber is None:
            self._note_subscriber = NoteSubscriber(
                db_factory=self._mycelium_handler._get_db,
                vault_path=self.vault_path,
            )
            self._note_subscriber.start()

        for file_path, _ctx in scan_inbox_recursive(self.inbox_path):
            logger.info(f"Scanning: {file_path.name}")
            result = self._mycelium_handler.on_document_created(file_path)
            results.append(result)

        return results

    @staticmethod
    def get_running_status() -> Optional[MyceliumStatus]:
        """Check if Mycelium watcher is running from PID file.

        Returns:
            MyceliumStatus if running, None if not running
        """
        if not MYCELIUM_PID_FILE.exists():
            return None

        try:
            pid = int(MYCELIUM_PID_FILE.read_text().strip())
            os.kill(pid, 0)  # Check if process exists
            return MyceliumStatus(running=True, pid=pid)
        except (ValueError, OSError, ProcessLookupError):
            return None

    @staticmethod
    def stop_running() -> bool:
        """Stop a running Mycelium watcher by sending SIGTERM.

        Returns:
            True if watcher was stopped, False if not running
        """
        if not MYCELIUM_PID_FILE.exists():
            return False

        try:
            pid = int(MYCELIUM_PID_FILE.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            for _ in range(10):
                time.sleep(0.5)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
            return True
        except (ValueError, OSError, ProcessLookupError):
            return False
