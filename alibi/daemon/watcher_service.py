"""Watcher daemon service for auto-processing documents.

This module provides a daemon service that watches the inbox directory
for new documents and automatically processes them through the pipeline.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from alibi.config import get_config
from alibi.extraction.yaml_cache import _get_yaml_store_root
from alibi.processing.watcher import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

# PID file location
PID_FILE = Path.home() / ".alibi" / "watcher.pid"
LOG_FILE = Path.home() / ".alibi" / "watcher.log"


@dataclass
class WatchedFile:
    """Represents a file being watched for stability before processing."""

    path: Path
    first_seen: float
    last_modified: float
    size: int


@dataclass
class DaemonStatus:
    """Status of the watcher daemon."""

    running: bool
    pid: Optional[int] = None
    inbox_path: Optional[Path] = None
    pending_files: int = 0
    processed_count: int = 0
    last_activity: Optional[datetime] = None
    uptime_seconds: float = 0.0


YAML_SUFFIX = ".alibi.yaml"


class DebouncedEventHandler(FileSystemEventHandler):
    """Handle file system events with debouncing for stable file detection."""

    def __init__(
        self,
        process_callback: Callable[[Path], Any],
        on_error: Optional[Callable[[Path, Exception], None]] = None,
        debounce_seconds: float = 2.0,
        supported_extensions: Optional[set[str]] = None,
        yaml_callback: Optional[Callable[[Path], Any]] = None,
        yaml_delete_callback: Optional[Callable[[Path], Any]] = None,
    ) -> None:
        """Initialize the event handler.

        Args:
            process_callback: Function to call when file is ready
            on_error: Optional error handler
            debounce_seconds: Wait time before processing (file stability)
            supported_extensions: Set of file extensions to process
            yaml_callback: Function to call when .alibi.yaml is modified
            yaml_delete_callback: Function to call when .alibi.yaml is deleted
        """
        super().__init__()
        self.process_callback = process_callback
        self.on_error = on_error
        self.debounce_seconds = debounce_seconds
        self.supported_extensions = supported_extensions or SUPPORTED_EXTENSIONS
        self.yaml_callback = yaml_callback
        self.yaml_delete_callback = yaml_delete_callback
        self.pending: dict[str, WatchedFile] = {}
        self.pending_yaml: dict[str, WatchedFile] = {}
        self._lock = threading.Lock()
        self.processed_count = 0
        self.last_activity: Optional[datetime] = None

    def _is_supported(self, path: Path) -> bool:
        """Check if file format is supported."""
        return path.suffix.lower() in self.supported_extensions

    @staticmethod
    def _is_yaml_file(path: Path) -> bool:
        """Check if this is an .alibi.yaml file."""
        return path.name.endswith(YAML_SUFFIX)

    def on_created(self, event: Any) -> None:
        """Handle file creation events."""
        if event.is_directory:
            return

        path = Path(event.src_path)

        # Ignore .alibi.yaml on creation (created by our pipeline)
        if self._is_yaml_file(path):
            return

        if not self._is_supported(path):
            return

        logger.info(f"New file detected: {path.name}")
        self._add_pending(path)

    def on_modified(self, event: Any) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return

        path = Path(event.src_path)

        # Route .alibi.yaml modifications to YAML pending queue
        if self._is_yaml_file(path) and self.yaml_callback:
            logger.info(f"YAML modified: {path.name}")
            self._add_pending_yaml(path)
            return

        if not self._is_supported(path):
            return

        self._update_pending(path)

    def on_deleted(self, event: Any) -> None:
        """Handle file deletion events."""
        if event.is_directory:
            return

        path = Path(event.src_path)

        # Route .alibi.yaml deletions immediately (no debounce)
        if self._is_yaml_file(path) and self.yaml_delete_callback:
            logger.info(f"YAML deleted: {path.name}")
            try:
                self.yaml_delete_callback(path)
            except Exception as e:
                logger.error(f"Error handling YAML deletion {path.name}: {e}")

    def on_moved(self, event: Any) -> None:
        """Handle file move events (e.g., temp file renamed)."""
        if event.is_directory:
            return

        dest_path = Path(event.dest_path)
        if not self._is_supported(dest_path):
            return

        # Remove old path if tracked
        with self._lock:
            self.pending.pop(event.src_path, None)

        logger.info(f"File moved to: {dest_path.name}")
        self._add_pending(dest_path)

    def _add_pending(self, path: Path) -> None:
        """Add file to pending queue."""
        now = time.time()
        try:
            size = path.stat().st_size
        except OSError:
            size = 0

        with self._lock:
            self.pending[str(path)] = WatchedFile(
                path=path,
                first_seen=now,
                last_modified=now,
                size=size,
            )

    def _add_pending_yaml(self, path: Path) -> None:
        """Add .alibi.yaml file to YAML pending queue."""
        now = time.time()
        try:
            size = path.stat().st_size
        except OSError:
            size = 0

        with self._lock:
            self.pending_yaml[str(path)] = WatchedFile(
                path=path,
                first_seen=now,
                last_modified=now,
                size=size,
            )

    def _update_pending(self, path: Path) -> None:
        """Update pending file's last modified time."""
        now = time.time()
        try:
            size = path.stat().st_size
        except OSError:
            return

        with self._lock:
            key = str(path)
            if key in self.pending:
                self.pending[key].last_modified = now
                self.pending[key].size = size
            else:
                self.pending[key] = WatchedFile(
                    path=path,
                    first_seen=now,
                    last_modified=now,
                    size=size,
                )

    def process_pending(self) -> list[Path]:
        """Process files that have stabilized (no changes for debounce period).

        Processes document files first, then YAML files.

        Returns:
            List of processed file paths
        """
        now = time.time()
        ready: list[Path] = []
        ready_yaml: list[Path] = []

        with self._lock:
            to_remove = []
            for key, watched in self.pending.items():
                # Check if file has stabilized
                if now - watched.last_modified > self.debounce_seconds:
                    # Verify file still exists and size is stable
                    try:
                        current_size = watched.path.stat().st_size
                        if current_size == watched.size and current_size > 0:
                            ready.append(watched.path)
                            to_remove.append(key)
                        else:
                            # Size changed, update and wait more
                            watched.size = current_size
                            watched.last_modified = now
                    except OSError:
                        # File gone, remove from pending
                        to_remove.append(key)

            for key in to_remove:
                del self.pending[key]

            # Process stable YAML files
            yaml_remove = []
            for key, watched in self.pending_yaml.items():
                if now - watched.last_modified > self.debounce_seconds:
                    try:
                        current_size = watched.path.stat().st_size
                        if current_size == watched.size and current_size > 0:
                            ready_yaml.append(watched.path)
                            yaml_remove.append(key)
                        else:
                            watched.size = current_size
                            watched.last_modified = now
                    except OSError:
                        yaml_remove.append(key)

            for key in yaml_remove:
                del self.pending_yaml[key]

        # Process ready document files outside lock
        processed = []
        for path in ready:
            try:
                logger.info(f"Processing stable file: {path.name}")
                self.process_callback(path)
                self.processed_count += 1
                self.last_activity = datetime.now()
                processed.append(path)
            except Exception as e:
                logger.error(f"Error processing {path.name}: {e}")
                if self.on_error:
                    self.on_error(path, e)

        # Process ready YAML files
        if self.yaml_callback:
            for path in ready_yaml:
                try:
                    logger.info(f"Processing YAML correction: {path.name}")
                    self.yaml_callback(path)
                    self.processed_count += 1
                    self.last_activity = datetime.now()
                    processed.append(path)
                except Exception as e:
                    logger.error(f"Error processing YAML {path.name}: {e}")
                    if self.on_error:
                        self.on_error(path, e)

        return processed

    @property
    def pending_count(self) -> int:
        """Get count of pending files."""
        with self._lock:
            return len(self.pending)


class WatcherDaemon:
    """Daemon service for watching and processing documents."""

    def __init__(
        self,
        inbox_path: Optional[Path] = None,
        debounce_seconds: float = 2.0,
        poll_interval: float = 0.5,
        recursive: bool = True,
        log_file: Optional[Path] = None,
    ) -> None:
        """Initialize the watcher daemon.

        Args:
            inbox_path: Path to watch (defaults to config inbox)
            debounce_seconds: Wait time for file stability
            poll_interval: How often to check pending files
            recursive: Watch subdirectories
            log_file: Path to log file
        """
        self.config = get_config()
        self.inbox_path = inbox_path or self.config.get_inbox_path()
        self.debounce_seconds = debounce_seconds
        self.poll_interval = poll_interval
        self.recursive = recursive
        self.log_file = log_file or LOG_FILE

        self._observer: Any = None  # Observer type from watchdog
        self._yaml_store_observer: Any = (
            None  # Second observer for yaml_store directory
        )
        self._handler: Optional[DebouncedEventHandler] = None
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._shutdown_event = threading.Event()
        self._scheduler: Any = None  # EnrichmentScheduler (lazy import)

    def _setup_logging(self) -> None:
        """Configure logging for daemon mode."""
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Configure file handler
        handler = logging.FileHandler(self.log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # Add to root logger and alibi loggers
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown.

        Note: Signal handlers can only be set from the main thread.
        This method silently skips if called from a non-main thread.
        """
        import threading

        # Signal handlers can only be set in the main thread
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
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(os.getpid()))

    def _remove_pid(self) -> None:
        """Remove PID file."""
        if PID_FILE.exists():
            PID_FILE.unlink()

    def _poll_loop(self) -> None:
        """Background loop to process pending files."""
        while self._running and not self._shutdown_event.is_set():
            try:
                if self._handler:
                    self._handler.process_pending()
            except Exception as e:
                logger.error(f"Error in poll loop: {e}")
            self._shutdown_event.wait(timeout=self.poll_interval)

    def start(self, foreground: bool = True) -> None:
        """Start the watcher daemon.

        Args:
            foreground: Run in foreground (True) or background (False)
        """
        if self._running:
            logger.warning("Watcher already running")
            return

        if self.inbox_path is None:
            raise ValueError("No inbox path configured. Set ALIBI_VAULT_PATH.")

        if not self.inbox_path.exists():
            raise ValueError(f"Inbox path does not exist: {self.inbox_path}")

        # Setup logging
        self._setup_logging()

        # Import handler here to avoid circular imports
        from alibi.daemon.handlers import DocumentHandler

        # Create handler with processing callback
        doc_handler = DocumentHandler(inbox_root=self.inbox_path)
        self._handler = DebouncedEventHandler(
            process_callback=doc_handler.on_document_created,
            on_error=doc_handler.on_document_error,
            debounce_seconds=self.debounce_seconds,
            yaml_callback=doc_handler.on_yaml_modified,
            yaml_delete_callback=doc_handler.on_yaml_deleted,
        )

        # Create and start observer for inbox directory
        self._observer = Observer()
        self._observer.schedule(
            self._handler,
            str(self.inbox_path),
            recursive=self.recursive,
        )
        self._observer.start()

        # If yaml_store is configured and exists outside the inbox, watch it too
        yaml_store_root = _get_yaml_store_root()
        if yaml_store_root is not None and yaml_store_root.exists():
            # Only add a separate observer when yaml_store is outside the inbox tree.
            # If it were inside the inbox the existing observer already covers it.
            try:
                yaml_store_root.resolve().relative_to(self.inbox_path.resolve())
                # yaml_store is inside inbox — already covered, no second observer needed
                logger.debug(
                    f"yaml_store {yaml_store_root} is inside inbox, "
                    "no additional observer needed"
                )
            except ValueError:
                # yaml_store is outside inbox — add a dedicated observer
                self._yaml_store_observer = Observer()
                self._yaml_store_observer.schedule(
                    self._handler,
                    str(yaml_store_root),
                    recursive=True,
                )
                self._yaml_store_observer.start()
                logger.info(f"Also watching yaml_store: {yaml_store_root}")

        self._running = True
        self._start_time = time.time()

        # Start polling thread for pending file processing
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

        # Start enrichment scheduler if enabled
        if self.config.enrichment_schedule_enabled:
            from alibi.daemon.scheduler import EnrichmentScheduler
            from alibi.db.connection import DatabaseManager

            def _db_factory() -> DatabaseManager:
                return DatabaseManager(self.config)

            self._scheduler = EnrichmentScheduler(
                db_factory=_db_factory, config=self.config
            )
            self._scheduler.start()

        # Setup signal handlers
        self._setup_signal_handlers()

        # Write PID file
        self._write_pid()

        logger.info(f"Started watching: {self.inbox_path} (pid={os.getpid()})")

        if foreground:
            # Block until shutdown signal
            try:
                while self._running and not self._shutdown_event.is_set():
                    self._shutdown_event.wait(timeout=1.0)
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.stop()

    def stop(self) -> None:
        """Stop the watcher daemon."""
        if not self._running:
            return

        self._running = False
        self._shutdown_event.set()

        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None

        if self._yaml_store_observer is not None:
            self._yaml_store_observer.stop()
            self._yaml_store_observer.join(timeout=5.0)
            self._yaml_store_observer = None

        if self._scheduler is not None:
            self._scheduler.stop()
            self._scheduler = None

        self._remove_pid()
        logger.info("Watcher stopped")

    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running

    def get_status(self) -> DaemonStatus:
        """Get current daemon status."""
        uptime = 0.0
        if self._start_time and self._running:
            uptime = time.time() - self._start_time

        return DaemonStatus(
            running=self._running,
            pid=os.getpid() if self._running else None,
            inbox_path=self.inbox_path,
            pending_files=self._handler.pending_count if self._handler else 0,
            processed_count=self._handler.processed_count if self._handler else 0,
            last_activity=self._handler.last_activity if self._handler else None,
            uptime_seconds=uptime,
        )

    @staticmethod
    def get_running_status() -> Optional[DaemonStatus]:
        """Check if daemon is running from PID file.

        Returns:
            DaemonStatus if running, None if not running
        """
        if not PID_FILE.exists():
            return None

        try:
            pid = int(PID_FILE.read_text().strip())
            # Check if process is running
            os.kill(pid, 0)
            return DaemonStatus(running=True, pid=pid)
        except (ValueError, OSError, ProcessLookupError):
            # PID file exists but process is not running
            return None

    @staticmethod
    def stop_running() -> bool:
        """Stop a running daemon by sending SIGTERM.

        Returns:
            True if daemon was stopped, False if not running
        """
        if not PID_FILE.exists():
            return False

        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            # Wait a bit for process to exit
            for _ in range(10):
                time.sleep(0.5)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
            return True
        except (ValueError, OSError, ProcessLookupError):
            return False


def main() -> None:
    """Main entry point for daemon."""
    daemon = WatcherDaemon()
    daemon.start(foreground=True)


if __name__ == "__main__":
    main()
