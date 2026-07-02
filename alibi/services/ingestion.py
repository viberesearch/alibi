"""Document ingestion service.

Wraps ProcessingPipeline behind a service interface for use by
CLI, API, MCP, Telegram, and other consumers.

All entry points that receive raw bytes persist documents to the
inbox before processing, so that YAML caches and source files are
available on disk regardless of the caller.
"""

import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from alibi.config import Config
from alibi.db.connection import DatabaseManager
from alibi.db.models import DocumentType
from alibi.processing.folder_router import FolderContext
from alibi.processing.pipeline import ProcessingPipeline, ProcessingResult
from alibi.services.events import EventType, event_bus

logger = logging.getLogger(__name__)

# DocumentType -> inbox subfolder name
_DOCTYPE_TO_FOLDER: dict[DocumentType, str] = {
    DocumentType.RECEIPT: "receipts",
    DocumentType.INVOICE: "invoices",
    DocumentType.PAYMENT_CONFIRMATION: "payments",
    DocumentType.STATEMENT: "statements",
    DocumentType.WARRANTY: "warranties",
    DocumentType.CONTRACT: "contracts",
}

# Fallback directory when no vault inbox is configured
_FALLBACK_DIR = Path("data/uploads")


def _resolve_upload_dir(doc_type: DocumentType | None) -> Path:
    """Return the directory where an uploaded file should be saved."""
    cfg = Config()
    inbox = cfg.get_inbox_path()

    subfolder = _DOCTYPE_TO_FOLDER.get(doc_type) if doc_type else None  # type: ignore[arg-type]
    base = inbox if inbox is not None else _FALLBACK_DIR
    return base / (subfolder or "unsorted")


def persist_upload(
    data: bytes,
    filename: str,
    folder_context: FolderContext | None = None,
) -> Path:
    """Persist raw bytes to the inbox directory.

    The file is placed in the appropriate type subfolder, named with a
    source-prefixed timestamp to avoid collisions.  Returns the path
    to the written file.
    """
    doc_type = folder_context.doc_type if folder_context else None
    source = (folder_context.source if folder_context else None) or "upload"

    dest_dir = _resolve_upload_dir(doc_type)
    dest_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem
    suffix = Path(filename).suffix or ".jpg"
    ts = int(time.time())
    dest = dest_dir / f"{source}_{ts}_{stem}{suffix}"

    dest.write_bytes(data)
    logger.info("Persisted upload to %s (%d bytes)", dest, len(data))
    return dest


def persist_upload_group(
    pages: Sequence[tuple[bytes, str]],
    folder_context: FolderContext | None = None,
) -> list[Path]:
    """Persist multiple pages as a document group in a subfolder.

    Creates a timestamped subfolder under the type directory and writes
    each page as ``page_NNN{suffix}``.  Returns the ordered list of
    written paths.
    """
    doc_type = folder_context.doc_type if folder_context else None
    source = (folder_context.source if folder_context else None) or "upload"

    parent_dir = _resolve_upload_dir(doc_type)
    ts = int(time.time())
    group_dir = parent_dir / f"{source}_{ts}"
    group_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for i, (data, filename) in enumerate(pages):
        suffix = Path(filename).suffix or ".jpg"
        dest = group_dir / f"page_{i:03d}{suffix}"
        dest.write_bytes(data)
        logger.info("Persisted group page to %s (%d bytes)", dest, len(data))
        saved.append(dest)
    return saved


def _finalize_ingestion(db: DatabaseManager, result: ProcessingResult) -> None:
    """Make a freshly-created fact analytics-ready in the ingesting process.

    In the default deployment the API server (``lt serve``) is the only
    long-running process and it wires no event subscribers, so the post-fact
    steps that otherwise rely on the watcher daemon / enrichment scheduler
    never run for API / Telegram uploads. This performs them synchronously, in
    whatever process ingested the document, so a receipt is complete the moment
    it lands rather than needing a manual ``lt fx backfill`` /
    ``lt enrich categorize`` follow-up:

      1. reconcile item totals to the printed total (the weighed-line
         double-multiply guard) -- runs first because a successful reconcile
         re-collapses the cloud (new fact id) and would otherwise leave the new
         fact unrated;
      2. stamp ``facts.eur_rate`` (EUR -> 1.0; foreign -> historical rate);
      3. assign a local-taxonomy category to this document's uncategorised
         items (document-scoped, so the global backlog is never dragged in);
      4. detect (never auto-merge) whether this upload duplicates an
         already-stored receipt, and log it for review;
      5. refresh the ``item_stars`` mirror so the ``*_eur`` / category columns
         reflect the above.

    Every step is best-effort and idempotent: a failure in one is logged and
    never blocks ingestion, event emission, or the remaining steps.
    """
    if not result.success or result.is_duplicate or not result.document_id:
        return

    from alibi.db import v2_store

    document_id = result.document_id
    short = document_id[:8]

    # 1. Reconcile line totals to the printed total (may re-collapse the cloud).
    try:
        from alibi.services.reconcile import reconcile_fact

        for fact in v2_store.get_facts_for_document(db, document_id):
            fact_id = fact.get("id")
            if fact_id:
                reconcile_fact(db, fact_id, apply=True)
    except Exception:
        logger.exception("Finalize: price reconciliation failed for %s", short)

    # 2. Stamp EUR rate on the (possibly re-collapsed) facts.
    try:
        from alibi.services.fx import stamp_fact_rate

        for fact in v2_store.get_facts_for_document(db, document_id):
            fact_id = fact.get("id")
            if fact_id:
                stamp_fact_rate(db, fact_id)
    except Exception:
        logger.exception("Finalize: EUR rate stamping failed for %s", short)

    # 3. Categorise just this document's items (local taxonomy, no cloud call).
    try:
        from alibi.enrichment.categorize import enrich_pending_categories

        enrich_pending_categories(db, document_id=document_id)
    except Exception:
        logger.exception("Finalize: categorisation failed for %s", short)

    # 4. Detection-only duplicate check: warn if THIS upload duplicates an
    #    existing receipt. Never auto-deletes here -- resolution stays a
    #    deliberate `lt` action so a legitimate re-visit is not silently merged.
    try:
        from alibi.services.dedup import deduplicate_facts

        new_fact_ids = {
            f.get("id") for f in v2_store.get_facts_for_document(db, document_id)
        }
        report = deduplicate_facts(db, apply=False)
        for action in report.resolved + report.review:
            if action.redundant.fact_id in new_fact_ids:
                logger.warning(
                    "Finalize: %s looks like a duplicate of %s (%s) — "
                    "%s %s %s; review with `lt facts dedup`",
                    short,
                    action.keeper.fact_id[:8],
                    action.reason,
                    action.keeper.vendor,
                    action.keeper.event_date,
                    action.keeper.total_amount,
                )
    except Exception:
        logger.exception("Finalize: duplicate check failed for %s", short)

    # 5. Sync the materialised analytics mirror with the writes above.
    try:
        from alibi.services.item_stars import refresh_item_stars_for_document

        refresh_item_stars_for_document(db, document_id)
    except Exception:
        logger.exception("Finalize: item_stars refresh failed for %s", short)


def _emit_ingestion_events(result: ProcessingResult) -> None:
    """Emit events for a successful processing result."""
    if not result.success:
        return
    event_bus.emit(
        EventType.DOCUMENT_INGESTED,
        {
            "document_id": result.document_id,
            "file_path": str(result.file_path),
            "is_duplicate": result.is_duplicate,
            "source": result.source,
            "user_id": result.user_id,
        },
    )
    if result.document_id and not result.is_duplicate:
        event_bus.emit(
            EventType.FACT_CREATED,
            {
                "document_id": result.document_id,
                "vendor": (result.extracted_data or {}).get("vendor"),
                "amount": (result.extracted_data or {}).get("total"),
                "source": result.source,
                "user_id": result.user_id,
            },
        )


def process_file(
    db: DatabaseManager,
    path: Path | str,
    folder_context: FolderContext | None = None,
) -> ProcessingResult:
    """Process a single document file.

    Args:
        db: Database manager to use for storage.
        path: Path to the document. Strings are converted to Path.
        folder_context: Optional folder routing context supplying document
            type, country, and vendor hint derived from the inbox layout.

    Returns:
        ProcessingResult with outcome, document ID, and extracted data.
    """
    result = ProcessingPipeline(db).process_file(
        Path(path),
        folder_context=folder_context,
    )
    _finalize_ingestion(db, result)
    _emit_ingestion_events(result)
    return result


def process_batch(
    db: DatabaseManager,
    paths: list[Path | str],
    folder_context: FolderContext | None = None,
) -> list[ProcessingResult]:
    """Process a list of document files.

    Each file is processed independently. A single shared FolderContext
    is applied to all files; pass None to let the pipeline detect type
    from each file individually.

    Args:
        db: Database manager to use for storage.
        paths: List of document paths. Strings are converted to Path.
        folder_context: Optional folder routing context applied to every
            file in the batch.

    Returns:
        List of ProcessingResult in the same order as paths.
    """
    resolved = [Path(p) for p in paths]
    folder_contexts = [folder_context] * len(resolved) if folder_context else None
    results = ProcessingPipeline(db).process_batch(
        resolved,
        folder_contexts=folder_contexts,
    )
    for result in results:
        _finalize_ingestion(db, result)
        _emit_ingestion_events(result)
    return results


def process_document_group(
    db: DatabaseManager,
    paths: Sequence[Path | str],
    folder_context: FolderContext | None = None,
) -> ProcessingResult:
    """Process multiple files as pages of a single document.

    All pages are sent to the LLM together in one call, producing a single
    unified extraction. Use this for multi-page scans stored as separate
    image files.

    Args:
        db: Database manager to use for storage.
        paths: Ordered list of page file paths. Strings are converted to
            Path. The list must be non-empty.
        folder_context: Optional folder routing context. The folder_path
            passed to the pipeline is the parent of the first page.

    Returns:
        Single ProcessingResult representing the grouped document.
    """
    resolved = [Path(p) for p in paths]
    if not resolved:
        folder_path = Path(".")
        return ProcessingPipeline(db).process_document_group(folder_path, resolved)

    folder_path = resolved[0].parent
    pipeline = ProcessingPipeline(db)
    result = pipeline.process_document_group(
        folder_path, resolved, folder_context=folder_context
    )
    _finalize_ingestion(db, result)
    _emit_ingestion_events(result)
    return result


def patch_yaml_field(
    source_path: Path,
    field: str,
    value: Any,
) -> bool:
    """Patch a single field in a .alibi.yaml file.

    Finds the YAML sidecar for source_path, reads it, updates the field,
    and writes it back. Does NOT reingest - use reingest_from_yaml after.

    Args:
        source_path: Path to the source document (or group folder).
        field: Top-level YAML field name to update.
        value: New value for the field.

    Returns:
        True if the file was modified, False if YAML not found or unchanged.
    """
    import yaml as _yaml

    from alibi.extraction.yaml_cache import get_yaml_path

    is_group = source_path.is_dir()
    yaml_path = get_yaml_path(source_path, is_group=is_group)
    if not yaml_path.exists():
        logger.warning("patch_yaml_field: no YAML found for %s", source_path)
        return False

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = _yaml.safe_load(f)
    except _yaml.YAMLError as e:
        logger.warning("patch_yaml_field: failed to parse %s: %s", yaml_path, e)
        return False

    if not isinstance(data, dict):
        logger.warning("patch_yaml_field: unexpected YAML structure in %s", yaml_path)
        return False

    old_value = data.get(field)
    if old_value == value:
        return False

    data[field] = value

    try:
        with open(yaml_path, "w", encoding="utf-8") as f:
            _yaml.dump(
                data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=120,
            )
    except Exception as e:
        logger.warning("patch_yaml_field: failed to write %s: %s", yaml_path, e)
        return False

    logger.info("patch_yaml_field: set %s=%r in %s", field, value, yaml_path)
    return True


def reingest_from_yaml(
    db: DatabaseManager,
    source_path: Path | str,
    is_group: bool = False,
) -> ProcessingResult:
    """Re-ingest a document from its edited .alibi.yaml.

    Reads YAML, cleans up existing DB records, runs Phase B.
    Source file is optional (file_hash read from YAML _meta).

    Args:
        db: Database manager.
        source_path: Path to the source file or folder.
        is_group: True if source_path is a folder (document group).

    Returns:
        ProcessingResult with outcome.
    """
    result = ProcessingPipeline(db).ingest_from_yaml(
        db, Path(source_path), is_group=is_group
    )
    if result.success and not result.is_duplicate:
        event_bus.emit(
            EventType.CORRECTION_APPLIED,
            {
                "document_id": result.document_id,
                "file_path": str(source_path),
                "source": "yaml_correction",
            },
        )
    return result


def check_yaml_changed(
    db: DatabaseManager,
    source_path: Path | str,
    is_group: bool = False,
) -> bool:
    """Check whether the .alibi.yaml has changed since last ingestion.

    Compares stored yaml_hash in the DB against the current file hash.

    Args:
        db: Database manager.
        source_path: Path to the source file or folder.
        is_group: True if source_path is a folder (document group).

    Returns:
        True if the YAML has changed (or no stored hash exists), False if identical.
    """
    from alibi.extraction.yaml_cache import compute_yaml_hash

    current_hash = compute_yaml_hash(Path(source_path), is_group)
    if current_hash is None:
        return False  # No YAML file at all

    from alibi.db import v2_store

    doc = v2_store.get_document_by_path(db, str(Path(source_path).resolve()))
    if doc is None:
        # Try by file_path as stored (may not be resolved)
        doc = v2_store.get_document_by_path(db, str(source_path))
    if doc is None:
        return True  # No DB record — treat as new

    stored_hash = doc.get("yaml_hash")
    return stored_hash != current_hash


def scan_yaml_corrections(
    db: DatabaseManager,
) -> list[Path]:
    """Find all .alibi.yaml files in the store that changed since ingestion.

    Args:
        db: Database manager.

    Returns:
        List of source paths whose YAMLs have changed.
    """
    from alibi.extraction.yaml_cache import (
        scan_yaml_store,
        resolve_source_from_yaml,
    )

    changed: list[Path] = []

    for yaml_path in scan_yaml_store():
        resolved = resolve_source_from_yaml(yaml_path)
        if resolved is None:
            continue
        source_path, is_group = resolved
        if check_yaml_changed(db, source_path, is_group):
            changed.append(source_path)

    return changed


def scan_low_confidence_yamls(
    threshold: float = 0.5,
) -> list[tuple[Path, dict[str, Any]]]:
    """Find YAML files with low confidence needing manual review.

    Scans the yaml_store for .alibi.yaml files where either
    ``_meta.needs_review`` is True or ``_meta.confidence`` is below
    ``threshold``.

    Args:
        threshold: Confidence cutoff (inclusive lower bound triggers review).

    Returns:
        List of (source_path, meta_dict) tuples sorted by confidence
        ascending (lowest confidence first).
    """
    from alibi.extraction.yaml_cache import (
        scan_yaml_store,
        resolve_source_from_yaml,
    )

    results: list[tuple[Path, dict[str, Any]]] = []

    for yaml_path in scan_yaml_store():
        resolved = resolve_source_from_yaml(yaml_path)
        if resolved is None:
            continue
        source_path, _is_group = resolved

        try:
            import yaml as _yaml

            with open(yaml_path, "r", encoding="utf-8") as f:
                data = _yaml.safe_load(f)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        meta = data.get("_meta", {})
        if not isinstance(meta, dict):
            continue

        flagged = meta.get("needs_review", False)
        confidence = meta.get("confidence")

        if flagged or (confidence is not None and confidence < threshold):
            results.append((source_path, meta))

    results.sort(key=lambda t: (t[1].get("confidence") or 1.0))
    return results


def process_bytes(
    db: DatabaseManager,
    data: bytes,
    filename: str,
    folder_context: FolderContext | None = None,
) -> ProcessingResult:
    """Process a document supplied as raw bytes.

    Persists the file to the inbox directory (so that YAML caches and
    source files remain on disk), then delegates to process_file.

    Args:
        db: Database manager to use for storage.
        data: Raw document bytes.
        filename: Original filename, used to derive the file extension.
        folder_context: Optional folder routing context.

    Returns:
        ProcessingResult with outcome. The file_path in the result points
        to the persisted inbox file.
    """
    saved_path = persist_upload(data, filename, folder_context)
    logger.debug("process_bytes: persisted %d bytes to %s", len(data), saved_path)
    return process_file(db, saved_path, folder_context=folder_context)
