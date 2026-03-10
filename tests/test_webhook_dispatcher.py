"""Tests for the webhook dispatcher subscriber."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from alibi.services.events import EventBus, EventType
from alibi.services.subscribers.webhook import WebhookDispatcher


class _CaptureHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures POST payloads."""

    received: list[dict[str, Any]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self.received.append(json.loads(body))
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass  # suppress noisy logs


@pytest.fixture
def capture_server():
    """Start a local HTTP server that captures POST requests."""
    _CaptureHandler.received = []
    server = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", _CaptureHandler.received
    server.shutdown()


class TestWebhookDispatcher:

    def test_delivers_event(
        self, capture_server: tuple[str, list[dict[str, Any]]]
    ) -> None:
        url, received = capture_server
        bus = EventBus()
        dispatcher = WebhookDispatcher(url, bus=bus, allow_private=True)
        dispatcher.start()

        bus.emit(EventType.FACT_CREATED, {"fact_id": "abc-123"})

        # Wait for async delivery
        time.sleep(0.5)

        assert len(received) == 1
        payload = received[0]
        assert payload["type"] == "fact.created"
        assert payload["data"]["fact_id"] == "abc-123"
        assert "timestamp" in payload

        dispatcher.stop()

    def test_delivers_multiple_events(
        self, capture_server: tuple[str, list[dict[str, Any]]]
    ) -> None:
        url, received = capture_server
        bus = EventBus()
        dispatcher = WebhookDispatcher(url, bus=bus, allow_private=True)
        dispatcher.start()

        bus.emit(EventType.FACT_CREATED, {"fact_id": "1"})
        bus.emit(EventType.CORRECTION_APPLIED, {"fact_id": "2"})

        time.sleep(0.5)

        assert len(received) == 2
        types = {r["type"] for r in received}
        assert "fact.created" in types
        assert "correction.applied" in types

        dispatcher.stop()

    def test_stop_unsubscribes(
        self, capture_server: tuple[str, list[dict[str, Any]]]
    ) -> None:
        url, received = capture_server
        bus = EventBus()
        dispatcher = WebhookDispatcher(url, bus=bus, allow_private=True)
        dispatcher.start()
        dispatcher.stop()

        bus.emit(EventType.FACT_CREATED, {"fact_id": "x"})
        time.sleep(0.3)

        assert len(received) == 0

    def test_delivery_failure_does_not_crash(self) -> None:
        bus = EventBus()
        # Port 1 is not open — delivery will fail
        dispatcher = WebhookDispatcher(
            "http://127.0.0.1:1/nope", bus=bus, timeout=0.5, allow_private=True
        )
        dispatcher.start()

        # Should not raise
        bus.emit(EventType.FACT_CREATED, {"fact_id": "x"})

        time.sleep(1.0)
        dispatcher.stop()
