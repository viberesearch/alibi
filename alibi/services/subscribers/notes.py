"""Obsidian note subscriber — generates notes when facts are created."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from alibi.db.connection import DatabaseManager
from alibi.services.events import Event, EventBus, EventType

logger = logging.getLogger(__name__)


class NoteSubscriber:
    """Generates Obsidian notes in response to ingestion events.

    Subscribes to DOCUMENT_INGESTED and generates a fact note for
    each new (non-duplicate) document that produced a fact.
    """

    def __init__(
        self,
        db_factory: Callable[[], DatabaseManager],
        vault_path: Path,
        bus: EventBus | None = None,
    ) -> None:
        """Initialize the note subscriber.

        Args:
            db_factory: Callable that returns a DatabaseManager instance.
            vault_path: Path to the Obsidian vault root.
            bus: EventBus to subscribe to. Defaults to the global event_bus.
        """
        self._db_factory = db_factory
        self._vault_path = vault_path
        self._bus = bus
        self.notes_generated = 0

    def start(self) -> None:
        """Subscribe to ingestion events on the bus."""
        if self._bus is None:
            from alibi.services.events import event_bus

            self._bus = event_bus
        self._bus.subscribe(EventType.DOCUMENT_INGESTED, self._handle)

    def stop(self) -> None:
        """Unsubscribe from all events."""
        if self._bus is None:
            return
        self._bus.unsubscribe(EventType.DOCUMENT_INGESTED, self._handle)

    def _handle(self, event: Event) -> None:
        """Handle a DOCUMENT_INGESTED event."""
        if event.data.get("is_duplicate"):
            return

        document_id = event.data.get("document_id")
        if not document_id:
            return

        try:
            self._generate_notes(document_id)
        except Exception:
            logger.exception("Note generation failed for document %s", document_id)

    def _generate_notes(self, document_id: str) -> None:
        """Look up facts for a document and generate notes."""
        from alibi.db.v2_store import get_facts_for_document
        from alibi.obsidian.notes import NoteExporter

        db = self._db_factory()
        facts = get_facts_for_document(db, document_id)
        if not facts:
            logger.debug("No facts for document %s, skipping notes", document_id)
            return

        exporter = NoteExporter(db, self._vault_path)
        for fact in facts:
            try:
                path = exporter.export_fact(fact)
                self.notes_generated += 1
                logger.info("Generated note: %s", path.name)
            except Exception:
                logger.exception("Failed to generate note for fact %s", fact.get("id"))
