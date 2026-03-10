"""File system watcher for document processing."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from alibi.config import get_config

if TYPE_CHECKING:
    from watchdog.observers.api import BaseObserver

logger = logging.getLogger(__name__)


# File extensions to process
SUPPORTED_EXTENSIONS = {
    # Images
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
    # Documents
    ".pdf",
}


def is_supported_file(path: Path) -> bool:
    """Check if file is a supported document type."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


class DocumentEventHandler(FileSystemEventHandler):
    """Handler for file system events in the inbox folder."""

    def __init__(
        self,
        on_new_file: Callable[[Path], None],
        debounce_seconds: float = 1.0,
    ) -> None:
        """Initialize the handler.

        Args:
            on_new_file: Callback function when a new file is detected
            debounce_seconds: Minimum time between processing same file
        """
        self.on_new_file = on_new_file
        self.debounce_seconds = debounce_seconds
        self._last_processed: dict[str, float] = {}

    def _should_process(self, path: Path) -> bool:
        """Check if file should be processed (debounce)."""
        path_str = str(path)
        now = time.time()

        if path_str in self._last_processed:
            if now - self._last_processed[path_str] < self.debounce_seconds:
                return False

        self._last_processed[path_str] = now
        return True

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if event.is_directory:
            return

        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode("utf-8")
        path = Path(src_path)
        if is_supported_file(path) and self._should_process(path):
            logger.info(f"New file detected: {path}")
            try:
                self.on_new_file(path)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return

        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode("utf-8")
        path = Path(src_path)
        if is_supported_file(path) and self._should_process(path):
            logger.debug(f"File modified: {path}")
            # Could trigger re-processing if needed


class InboxWatcher:
    """Watches the inbox folder for new documents."""

    def __init__(
        self,
        inbox_path: Path | None = None,
        on_new_file: Callable[[Path], None] | None = None,
    ) -> None:
        """Initialize the inbox watcher.

        Args:
            inbox_path: Path to watch (defaults to config inbox)
            on_new_file: Callback for new files (defaults to logging)
        """
        config = get_config()
        self.inbox_path = inbox_path or config.get_inbox_path()

        if self.inbox_path is None:
            raise ValueError("No inbox path configured. Set ALIBI_VAULT_PATH.")

        self.on_new_file = on_new_file or self._default_handler
        self._observer: BaseObserver | None = None
        self._running = False

    def _default_handler(self, path: Path) -> None:
        """Default handler just logs the file."""
        logger.info(f"Would process: {path}")

    def start(self) -> None:
        """Start watching the inbox folder."""
        if self._running:
            logger.warning("Watcher already running")
            return

        if self.inbox_path is None or not self.inbox_path.exists():
            raise ValueError(f"Inbox path does not exist: {self.inbox_path}")

        self._observer = Observer()
        handler = DocumentEventHandler(self.on_new_file)

        self._observer.schedule(handler, str(self.inbox_path), recursive=True)
        self._observer.start()
        self._running = True

        logger.info(f"Started watching: {self.inbox_path}")

    def stop(self) -> None:
        """Stop watching."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None

        self._running = False
        logger.info("Stopped watching")

    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running

    def __enter__(self) -> "InboxWatcher":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.stop()


def scan_inbox(inbox_path: Path | None = None) -> list[Path]:
    """Scan inbox for unprocessed files.

    Args:
        inbox_path: Path to scan (defaults to config inbox)

    Returns:
        List of supported files found
    """
    config = get_config()
    path = inbox_path or config.get_inbox_path()

    if path is None:
        raise ValueError("No inbox path configured. Set ALIBI_VAULT_PATH.")

    if not path.exists():
        return []

    files = []
    for item in path.rglob("*"):
        if item.is_file() and is_supported_file(item):
            files.append(item)

    return sorted(files, key=lambda p: p.stat().st_mtime)
