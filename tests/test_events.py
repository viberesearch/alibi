"""Tests for the event bus."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from alibi.services.events import Event, EventBus, EventType, event_bus


class TestEventBus:
    """Core EventBus functionality."""

    def test_subscribe_and_emit(self) -> None:
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(EventType.FACT_CREATED, received.append)

        bus.emit(EventType.FACT_CREATED, {"fact_id": "abc"})

        assert len(received) == 1
        assert received[0].type == EventType.FACT_CREATED
        assert received[0].data == {"fact_id": "abc"}

    def test_multiple_subscribers(self) -> None:
        bus = EventBus()
        calls: list[str] = []
        bus.subscribe(EventType.FACT_CREATED, lambda e: calls.append("a"))
        bus.subscribe(EventType.FACT_CREATED, lambda e: calls.append("b"))

        bus.emit(EventType.FACT_CREATED, {})

        assert calls == ["a", "b"]

    def test_no_cross_event_leaking(self) -> None:
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(EventType.FACT_CREATED, received.append)

        bus.emit(EventType.DOCUMENT_INGESTED, {"doc_id": "x"})

        assert len(received) == 0

    def test_emit_with_no_subscribers(self) -> None:
        bus = EventBus()
        event = bus.emit(EventType.FACT_UPDATED, {"fact_id": "x"})
        assert event.type == EventType.FACT_UPDATED

    def test_emit_returns_event(self) -> None:
        bus = EventBus()
        event = bus.emit(EventType.ANNOTATION_ADDED, {"id": "ann-1"})
        assert isinstance(event, Event)
        assert event.data == {"id": "ann-1"}
        assert event.timestamp is not None

    def test_subscriber_exception_does_not_propagate(self) -> None:
        bus = EventBus()
        good_calls: list[str] = []

        def bad_handler(e: Event) -> None:
            raise RuntimeError("boom")

        bus.subscribe(EventType.FACT_CREATED, bad_handler)
        bus.subscribe(EventType.FACT_CREATED, lambda e: good_calls.append("ok"))

        bus.emit(EventType.FACT_CREATED, {})

        assert good_calls == ["ok"]

    def test_unsubscribe(self) -> None:
        bus = EventBus()
        received: list[Event] = []
        handler = received.append
        bus.subscribe(EventType.FACT_CREATED, handler)
        bus.unsubscribe(EventType.FACT_CREATED, handler)

        bus.emit(EventType.FACT_CREATED, {})

        assert len(received) == 0

    def test_unsubscribe_unknown_handler(self) -> None:
        bus = EventBus()
        bus.unsubscribe(EventType.FACT_CREATED, lambda e: None)  # no-op

    def test_clear(self) -> None:
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(EventType.FACT_CREATED, received.append)
        bus.clear()

        bus.emit(EventType.FACT_CREATED, {})

        assert len(received) == 0

    def test_emit_default_data(self) -> None:
        bus = EventBus()
        event = bus.emit(EventType.FACT_CREATED)
        assert event.data == {}


class TestEventTypes:
    """EventType enum values."""

    def test_all_event_types(self) -> None:
        expected = {
            "document.ingested",
            "fact.created",
            "fact.updated",
            "correction.applied",
            "annotation.added",
        }
        assert {e.value for e in EventType} == expected


class TestGlobalBus:
    """Module-level event_bus singleton."""

    def test_singleton_exists(self) -> None:
        assert isinstance(event_bus, EventBus)

    def test_singleton_clear_for_isolation(self) -> None:
        received: list[Event] = []
        event_bus.subscribe(EventType.FACT_CREATED, received.append)
        event_bus.clear()
        event_bus.emit(EventType.FACT_CREATED, {})
        assert len(received) == 0
