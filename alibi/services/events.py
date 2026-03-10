"""In-process event bus for the Alibi service layer.

Enables loose coupling between pipeline operations and side effects
(Obsidian note generation, webhooks, notifications). All operations
are synchronous and single-threaded.

Usage::

    from alibi.services.events import event_bus, EventType

    # Subscribe
    def on_fact_created(event):
        print(f"New fact: {event.data['fact_id']}")

    event_bus.subscribe(EventType.FACT_CREATED, on_fact_created)

    # Emit
    event_bus.emit(EventType.FACT_CREATED, {"fact_id": "abc-123"})
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Processing lifecycle event types."""

    DOCUMENT_INGESTED = "document.ingested"
    FACT_CREATED = "fact.created"
    FACT_UPDATED = "fact.updated"
    CORRECTION_APPLIED = "correction.applied"
    ANNOTATION_ADDED = "annotation.added"


@dataclass(frozen=True)
class Event:
    """An event emitted by the service layer."""

    type: EventType
    data: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Subscriber callable type
Subscriber = Callable[[Event], None]


class EventBus:
    """Synchronous in-process event bus.

    Subscribers are called in registration order. Exceptions in
    subscribers are logged but do not propagate (fire-and-forget
    semantics).
    """

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[Subscriber]] = defaultdict(list)

    def subscribe(self, event_type: EventType, handler: Subscriber) -> None:
        """Register a handler for an event type."""
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Subscriber) -> None:
        """Remove a handler. No-op if the handler was not registered."""
        try:
            self._subscribers[event_type].remove(handler)
        except ValueError:
            pass

    def emit(self, event_type: EventType, data: dict[str, Any] | None = None) -> Event:
        """Emit an event and notify all subscribers.

        Args:
            event_type: Type of the event to emit.
            data: Event payload. Defaults to empty dict.

        Returns:
            The emitted Event instance.
        """
        event = Event(type=event_type, data=data or {})
        handlers = self._subscribers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "Event subscriber %s failed for %s",
                    getattr(handler, "__name__", repr(handler)),
                    event_type.value,
                )
        return event

    def clear(self) -> None:
        """Remove all subscribers. Useful for test isolation."""
        self._subscribers.clear()


# Singleton event bus instance used by the service layer
event_bus = EventBus()
