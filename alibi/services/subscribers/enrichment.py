"""Product enrichment subscriber — enriches fact items on FACT_CREATED events."""

from __future__ import annotations

import logging
import threading
from typing import Callable

from alibi.db.connection import DatabaseManager
from alibi.services.events import Event, EventBus, EventType

logger = logging.getLogger(__name__)

# Minimum batch size to trigger immediate Gemini flush (before timer fires).
_GEMINI_BATCH_THRESHOLD = 20

# Seconds to wait before flushing a partial Gemini batch.
_GEMINI_DEBOUNCE_SECONDS = 30.0


class EnrichmentSubscriber:
    """Enriches fact items with external product data on fact creation.

    Subscribes to FACT_CREATED. On each event:
    1. Items WITH barcode: lookup via Open Food Facts (cached)
    2. Items WITHOUT barcode: fuzzy name match against known products
    3. Items still lacking brand+category: queued for Gemini mega-batch

    Runs enrichment in a daemon thread to avoid blocking the pipeline.
    Gemini batches are debounced: flush fires after 30s or when 20+ items
    accumulate, whichever comes first.
    """

    def __init__(
        self,
        db_factory: Callable[[], DatabaseManager],
        bus: EventBus | None = None,
    ) -> None:
        self._db_factory = db_factory
        self._bus = bus
        self.enrichment_count = 0

        # Phase 3: Gemini batch queue (thread-safe)
        self._pending_items: list[dict[str, str]] = []
        self._batch_lock = threading.Lock()
        self._batch_timer: threading.Timer | None = None

    def start(self) -> None:
        """Subscribe to FACT_CREATED events."""
        if self._bus is None:
            from alibi.services.events import event_bus

            self._bus = event_bus
        self._bus.subscribe(EventType.FACT_CREATED, self._handle)

    def stop(self) -> None:
        """Unsubscribe from events and flush any pending Gemini items."""
        if self._bus is not None:
            self._bus.unsubscribe(EventType.FACT_CREATED, self._handle)

        with self._batch_lock:
            if self._batch_timer is not None:
                self._batch_timer.cancel()
                self._batch_timer = None

        self._flush_gemini_batch()

    def _handle(self, event: Event) -> None:
        """Handle FACT_CREATED by enriching items in a background thread."""
        thread = threading.Thread(
            target=self._enrich,
            args=(event,),
            daemon=True,
        )
        thread.start()

    def _enrich(self, event: Event) -> None:
        """Enrich fact items for the document that triggered this event."""
        from alibi.enrichment.service import enrich_item_cascade, enrich_item_by_name

        document_id = event.data.get("document_id")
        if not document_id:
            return

        try:
            db = self._db_factory()
            enriched = 0

            # Phase 1: enrich items WITH barcodes via OFF
            barcode_rows = db.fetchall(
                "SELECT fi.id, fi.barcode FROM fact_items fi "
                "JOIN facts f ON fi.fact_id = f.id "
                "JOIN clouds c ON f.cloud_id = c.id "
                "JOIN cloud_bundles cb ON c.id = cb.cloud_id "
                "JOIN bundles b ON cb.bundle_id = b.id "
                "JOIN bundle_atoms ba ON b.id = ba.bundle_id "
                "JOIN atoms a ON ba.atom_id = a.id "
                "WHERE a.document_id = ? "
                "AND fi.barcode IS NOT NULL AND fi.barcode != '' "
                "AND (fi.brand IS NULL OR fi.brand = '')",
                (document_id,),
            )

            for row in barcode_rows:
                result = enrich_item_cascade(db, row["id"], row["barcode"])
                if result.success:
                    enriched += 1

            # Phase 2: enrich items WITHOUT barcodes via name matching
            name_rows = db.fetchall(
                "SELECT fi.id, fi.name, f.vendor_key "
                "FROM fact_items fi "
                "JOIN facts f ON fi.fact_id = f.id "
                "JOIN clouds c ON f.cloud_id = c.id "
                "JOIN cloud_bundles cb ON c.id = cb.cloud_id "
                "JOIN bundles b ON cb.bundle_id = b.id "
                "JOIN bundle_atoms ba ON b.id = ba.bundle_id "
                "JOIN atoms a ON ba.atom_id = a.id "
                "WHERE a.document_id = ? "
                "AND (fi.barcode IS NULL OR fi.barcode = '') "
                "AND (fi.brand IS NULL OR fi.brand = '') "
                "AND fi.name IS NOT NULL AND fi.name != ''",
                (document_id,),
            )

            for row in name_rows:
                result = enrich_item_by_name(
                    db, row["id"], row["name"], vendor_key=row["vendor_key"]
                )
                if result.success:
                    enriched += 1

            if enriched:
                self.enrichment_count += enriched
                logger.info(
                    "Enriched %d items for document %s",
                    enriched,
                    document_id[:8],
                )

            # Phase 3: queue items still lacking brand+category for Gemini
            self._queue_for_gemini(db, document_id)

        except Exception:
            logger.exception(
                "Enrichment failed for document %s",
                document_id[:8] if document_id else "?",
            )

    def _queue_for_gemini(self, db: DatabaseManager, document_id: str) -> None:
        """Queue items still lacking brand+category after Phase 1+2 for Gemini."""
        from alibi.config import get_config

        cfg = get_config()
        if not cfg.gemini_enrichment_enabled:
            return
        if not cfg.gemini_api_key:
            return

        still_pending = db.fetchall(
            "SELECT fi.id, fi.name, fi.barcode "
            "FROM fact_items fi "
            "JOIN facts f ON fi.fact_id = f.id "
            "JOIN clouds c ON f.cloud_id = c.id "
            "JOIN cloud_bundles cb ON c.id = cb.cloud_id "
            "JOIN bundles b ON cb.bundle_id = b.id "
            "JOIN bundle_atoms ba ON b.id = ba.bundle_id "
            "JOIN atoms a ON ba.atom_id = a.id "
            "WHERE a.document_id = ? "
            "AND (fi.brand IS NULL OR fi.brand = '') "
            "AND (fi.category IS NULL OR fi.category = '') "
            "AND fi.name IS NOT NULL AND fi.name != ''",
            (document_id,),
        )

        if not still_pending:
            return

        items = [
            {
                "id": row["id"],
                "name": row["name"],
                "barcode": row["barcode"] or "",
            }
            for row in still_pending
        ]

        flush_now = False
        with self._batch_lock:
            self._pending_items.extend(items)

            if len(self._pending_items) >= _GEMINI_BATCH_THRESHOLD:
                if self._batch_timer is not None:
                    self._batch_timer.cancel()
                    self._batch_timer = None
                flush_now = True
            elif self._batch_timer is None:
                self._batch_timer = threading.Timer(
                    _GEMINI_DEBOUNCE_SECONDS, self._on_timer_fired
                )
                self._batch_timer.daemon = True
                self._batch_timer.start()

        if flush_now:
            self._flush_gemini_batch()

    def _on_timer_fired(self) -> None:
        """Called by the debounce timer when it expires."""
        with self._batch_lock:
            self._batch_timer = None
        self._flush_gemini_batch()

    def _flush_gemini_batch(self) -> None:
        """Flush pending items to Gemini for enrichment."""
        with self._batch_lock:
            if not self._pending_items:
                return
            items = list(self._pending_items)
            self._pending_items.clear()

        try:
            from alibi.enrichment.gemini_enrichment import enrich_items_by_gemini

            db = self._db_factory()
            results = enrich_items_by_gemini(db, items)
            enriched = sum(1 for r in results if r.success)
            if enriched:
                self.enrichment_count += enriched
                logger.info(
                    "Gemini Phase 3 enriched %d/%d queued items",
                    enriched,
                    len(items),
                )

            # Trigger OFF contribution for barcode items enriched by Gemini
            for r in results:
                if r.success:
                    try:
                        from alibi.enrichment.service import trigger_off_contribution

                        trigger_off_contribution(db, r.item_id)
                    except Exception:
                        pass  # fail-safe
        except Exception:
            logger.exception("Gemini Phase 3 flush failed")
