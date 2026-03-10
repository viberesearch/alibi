"""Document arrival handlers for the watcher daemon.

This module contains the handlers that are triggered when new documents
are detected in the inbox directory. Processing is delegated to the
service layer so that events are emitted consistently.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from alibi.config import get_config
from alibi.db.connection import DatabaseManager, get_db
from alibi.processing.folder_router import FolderContext, resolve_folder_context
from alibi.processing.pipeline import ProcessingResult
from alibi.processing.watcher import is_supported_file

logger = logging.getLogger(__name__)


class DocumentHandler:
    """Handler for document processing events.

    Routes documents through the service layer (services.ingestion)
    so that event bus notifications fire for every processed document.
    """

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        space_id: str = "default",
        user_id: str = "daemon",
        inbox_root: Optional[Path] = None,
    ) -> None:
        """Initialize the document handler.

        Args:
            db: Optional database manager (creates one if not provided)
            space_id: Space to store artifacts in
            user_id: User identifier for tracking
            inbox_root: Root of the inbox directory for folder-based routing
        """
        self.config = get_config()
        self._db = db
        self.space_id = space_id
        self.user_id = user_id
        self.inbox_root = inbox_root or self.config.get_inbox_path()

    def _get_db(self) -> DatabaseManager:
        """Get or create database manager."""
        if self._db is None:
            self._db = get_db()
            if not self._db.is_initialized():
                self._db.initialize()
        return self._db

    def _resolve_folder_context(self, path: Path) -> FolderContext | None:
        """Resolve folder routing context for a file path.

        Returns FolderContext if the file is inside the inbox root,
        None otherwise (skips folder-based routing).
        """
        if self.inbox_root is None:
            return None
        try:
            path.resolve().relative_to(self.inbox_root.resolve())
        except ValueError:
            return None
        return resolve_folder_context(path, self.inbox_root)

    def on_document_created(self, path: Path) -> ProcessingResult:
        """Handle new document creation.

        Delegates to the service layer so that event bus notifications
        fire (DOCUMENT_INGESTED, FACT_CREATED).

        Args:
            path: Path to the new document

        Returns:
            ProcessingResult from the pipeline
        """
        if not path.exists():
            logger.warning(f"Document no longer exists: {path}")
            return ProcessingResult(
                success=False,
                file_path=path,
                error="File no longer exists",
            )

        if not is_supported_file(path):
            logger.debug(f"Skipping unsupported file: {path}")
            return ProcessingResult(
                success=False,
                file_path=path,
                error=f"Unsupported file type: {path.suffix}",
            )

        logger.info(f"Processing new document: {path.name}")
        folder_context = self._resolve_folder_context(path)

        # Set provenance: watcher entry point, system user
        if folder_context is None:
            folder_context = FolderContext()
        if folder_context.source is None:
            folder_context.source = "watcher"
        if folder_context.user_id is None:
            folder_context.user_id = "system"

        try:
            from alibi.services.ingestion import process_file as svc_process

            result = svc_process(self._get_db(), path, folder_context=folder_context)

            if result.success:
                if result.is_duplicate:
                    logger.info(
                        f"Duplicate detected: {path.name} "
                        f"(original: {result.duplicate_of})"
                    )
                else:
                    logger.info(
                        f"Successfully processed: {path.name} "
                        f"-> {result.document_id}"
                    )
                    if result.extracted_data:
                        vendor = result.extracted_data.get("vendor", "Unknown")
                        total = result.extracted_data.get("total", "N/A")
                        logger.info(f"  Extracted: vendor={vendor}, total={total}")
            else:
                logger.error(f"Failed to process {path.name}: {result.error}")

            return result

        except Exception as e:
            logger.exception(f"Error processing document {path}: {e}")
            return ProcessingResult(
                success=False,
                file_path=path,
                error=str(e),
            )

    def on_document_modified(self, path: Path) -> Optional[ProcessingResult]:
        """Handle document modification.

        By default, we don't re-process modified documents to avoid
        duplicates. Unrecognized files are treated as new.

        Args:
            path: Path to the modified document

        Returns:
            ProcessingResult if re-processed, None if skipped
        """
        if not path.exists():
            return None

        if not is_supported_file(path):
            return None

        # Check if already processed
        db = self._get_db()
        existing = db.fetchone(
            "SELECT id FROM documents WHERE file_path = ?",
            (str(path),),
        )

        if existing:
            logger.debug(f"Document already processed, skipping: {path.name}")
            return None

        # If not yet processed, treat as new
        return self.on_document_created(path)

    def on_yaml_modified(self, yaml_path: Path) -> Optional[ProcessingResult]:
        """Handle .alibi.yaml modification (admin correction).

        Resolves the source document, checks if YAML actually changed,
        and re-ingests if so.

        Args:
            yaml_path: Path to the modified .alibi.yaml file

        Returns:
            ProcessingResult if re-ingested, None if skipped
        """
        from alibi.extraction.yaml_cache import resolve_source_from_yaml
        from alibi.services.ingestion import check_yaml_changed, reingest_from_yaml

        resolved = resolve_source_from_yaml(yaml_path)
        if resolved is None:
            logger.warning(f"Cannot resolve source for YAML: {yaml_path}")
            return None

        source_path, is_group = resolved
        db = self._get_db()

        if not check_yaml_changed(db, source_path, is_group):
            logger.debug(f"YAML unchanged, skipping: {yaml_path}")
            return None

        logger.info(f"YAML correction detected: {yaml_path} → re-ingesting")
        try:
            result = reingest_from_yaml(db, source_path, is_group=is_group)
            if result.success:
                logger.info(
                    f"Re-ingested from YAML: {source_path.name} "
                    f"→ {result.document_id}"
                )
                # Commit the corrected YAML to git
                try:
                    from alibi.mycelium.yaml_versioning import (
                        get_yaml_versioner,
                    )

                    get_yaml_versioner().commit_single(yaml_path)
                except Exception as ve:
                    logger.debug(f"YAML git commit skipped: {ve}")
            else:
                logger.error(
                    f"Re-ingestion failed for {source_path.name}: " f"{result.error}"
                )
            return result
        except Exception as e:
            logger.exception(f"Error re-ingesting from YAML {yaml_path}: {e}")
            return None

    def on_yaml_deleted(self, yaml_path: Path) -> bool:
        """Handle .alibi.yaml deletion — cascade delete DB records.

        Args:
            yaml_path: Path to the deleted .alibi.yaml file

        Returns:
            True if DB records were cleaned up, False otherwise
        """
        from alibi.extraction.yaml_cache import resolve_source_from_yaml
        from alibi.db import v2_store

        resolved = resolve_source_from_yaml(yaml_path)
        if resolved is None:
            logger.warning(f"Cannot resolve source for deleted YAML: {yaml_path}")
            return False

        source_path, _is_group = resolved
        db = self._get_db()

        doc = v2_store.get_document_by_path(db, str(source_path))
        if doc is None:
            logger.debug(f"No DB record for deleted YAML source: {source_path}")
            return False

        logger.info(
            f"YAML deleted: removing DB records for {source_path.name} "
            f"(doc {doc['id'][:8]})"
        )
        cleanup_result = v2_store.cleanup_document(db, doc["id"])
        return bool(cleanup_result["cleaned"])

    def on_document_error(self, path: Path, error: Exception) -> None:
        """Handle processing error.

        Args:
            path: Path to the document that failed
            error: The exception that occurred
        """
        logger.error(f"Processing error for {path.name}: {error}")

    def close(self) -> None:
        """Close resources."""
        pass


# Module-level convenience functions
_default_handler: Optional[DocumentHandler] = None


def _get_handler() -> DocumentHandler:
    """Get or create the default handler."""
    global _default_handler
    if _default_handler is None:
        _default_handler = DocumentHandler()
    return _default_handler


def on_document_created(path: Path) -> ProcessingResult:
    """Process a newly created document.

    Args:
        path: Path to the document

    Returns:
        ProcessingResult from the pipeline
    """
    return _get_handler().on_document_created(path)


def on_document_modified(path: Path) -> Optional[ProcessingResult]:
    """Handle a modified document.

    Args:
        path: Path to the document

    Returns:
        ProcessingResult if processed, None if skipped
    """
    return _get_handler().on_document_modified(path)
