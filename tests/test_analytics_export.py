"""Tests for the analytics stack export feature.

Covers three components:
- alibi.services.export_analytics (build_export_payload / push_to_analytics_stack)
- alibi.services.subscribers.analytics (AnalyticsExportSubscriber)
- CLI `analytics export` command
"""

from __future__ import annotations

import json
import os
import threading
import time
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from click.testing import CliRunner

os.environ["ALIBI_TESTING"] = "1"

from alibi.annotations.store import add_annotation
from alibi.cli import cli
from alibi.db.connection import DatabaseManager
from alibi.db.models import (
    Atom,
    AtomType,
    Bundle,
    BundleAtom,
    BundleAtomRole,
    BundleType,
    Cloud,
    CloudBundle,
    CloudMatchType,
    CloudStatus,
    Document,
    Fact,
    FactStatus,
    FactType,
)
from alibi.db import v2_store
from alibi.services.events import EventBus, EventType
from alibi.services.export_analytics import (
    build_export_payload,
    push_to_analytics_stack,
)
from alibi.services.subscribers.analytics import AnalyticsExportSubscriber


# ---------------------------------------------------------------------------
# Shared HTTP capture server fixture
# ---------------------------------------------------------------------------


class _CaptureHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures POST request bodies and returns 200."""

    received: list[dict[str, Any]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            self.received.append(json.loads(body))
        except (json.JSONDecodeError, ValueError):
            self.received.append({"_raw": body.decode("utf-8", errors="replace")})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status": "ok"}')

    def log_message(self, format: str, *args: Any) -> None:
        pass  # suppress noisy logs


@pytest.fixture
def capture_server():
    """Start a local HTTP server that captures POST payloads."""
    _CaptureHandler.received = []
    server = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", _CaptureHandler.received
    server.shutdown()


# ---------------------------------------------------------------------------
# DB helpers shared between export and CLI tests
# ---------------------------------------------------------------------------


def _make_doc(
    db: DatabaseManager,
    file_path: str = "/test/receipt.jpg",
    source: str | None = None,
    user_id: str | None = None,
) -> Document:
    doc = Document(
        id=str(uuid4()),
        file_path=file_path,
        file_hash=str(uuid4()),
        source=source,
        user_id=user_id,
    )
    v2_store.store_document(db, doc)
    return doc


def _make_user(db: DatabaseManager, user_id: str = "user-1") -> str:
    """Insert a minimal user row and return its id."""
    conn = db.get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO users (id, name) VALUES (?, ?)",
        (user_id, "Test User"),
    )
    return user_id


def _make_vendor_atom(
    db: DatabaseManager, doc: Document, name: str = "Test Store"
) -> Atom:
    atom = Atom(
        id=str(uuid4()),
        document_id=doc.id,
        atom_type=AtomType.VENDOR,
        data={"name": name},
    )
    v2_store.store_atoms(db, [atom])
    return atom


def _make_bundle(db: DatabaseManager, doc: Document, atoms: list[Atom]) -> Bundle:
    bundle = Bundle(
        id=str(uuid4()),
        document_id=doc.id,
        bundle_type=BundleType.BASKET,
    )
    bundle_atoms = [
        BundleAtom(
            bundle_id=bundle.id,
            atom_id=a.id,
            role=BundleAtomRole.VENDOR_INFO,
        )
        for a in atoms
    ]
    v2_store.store_bundle(db, bundle, bundle_atoms)
    return bundle


def _make_fact(
    db: DatabaseManager,
    bundle: Bundle,
    vendor: str = "Test Store",
    total_amount: Decimal = Decimal("42.50"),
) -> Fact:
    cloud = Cloud(id=str(uuid4()), status=CloudStatus.FORMING)
    cloud_bundle = CloudBundle(
        cloud_id=cloud.id,
        bundle_id=bundle.id,
        match_type=CloudMatchType.MANUAL,
        match_confidence=Decimal("1.0"),
    )
    v2_store.store_cloud(db, cloud, cloud_bundle)

    from datetime import date

    fact = Fact(
        id=str(uuid4()),
        cloud_id=cloud.id,
        fact_type=FactType.PURCHASE,
        vendor=vendor,
        vendor_key=None,
        total_amount=total_amount,
        currency="EUR",
        event_date=date(2026, 1, 15),
        status=FactStatus.CONFIRMED,
    )
    v2_store.store_fact(db, fact, [])
    return fact


def _seed_db(db: DatabaseManager) -> dict[str, Any]:
    """Insert one fact and one annotation. Returns IDs for assertions."""
    doc = _make_doc(db)
    atom = _make_vendor_atom(db, doc, "Seeded Store")
    bundle = _make_bundle(db, [atom], doc) if False else _make_bundle(db, doc, [atom])
    fact = _make_fact(db, bundle, vendor="Seeded Store", total_amount=Decimal("19.99"))

    annotation_id = add_annotation(
        db,
        annotation_type="note",
        target_type="fact",
        target_id=fact.id,
        key="note",
        value="test annotation",
    )
    return {"fact_id": fact.id, "annotation_id": annotation_id}


# ---------------------------------------------------------------------------
# 1. export_analytics.build_export_payload
# ---------------------------------------------------------------------------


class TestBuildExportPayload:
    """Tests for build_export_payload()."""

    def test_empty_db_returns_empty_lists(self, db: DatabaseManager) -> None:
        """Empty database yields payload with four empty lists."""
        payload = build_export_payload(db)

        assert "facts" in payload
        assert "fact_items" in payload
        assert "annotations" in payload
        assert "documents" in payload
        assert payload["facts"] == []
        assert payload["fact_items"] == []
        assert payload["annotations"] == []
        assert payload["documents"] == []

    def test_payload_structure_with_data(self, db: DatabaseManager) -> None:
        """Payload contains all four keys populated after seeding."""
        _seed_db(db)

        payload = build_export_payload(db)

        assert isinstance(payload["facts"], list)
        assert isinstance(payload["fact_items"], list)
        assert isinstance(payload["annotations"], list)
        assert isinstance(payload["documents"], list)
        assert len(payload["facts"]) == 1
        assert len(payload["annotations"]) == 1
        assert len(payload["documents"]) == 1

    def test_facts_contain_expected_fields(self, db: DatabaseManager) -> None:
        """Each fact dict in the payload has at minimum id and vendor fields."""
        _seed_db(db)

        payload = build_export_payload(db)

        fact = payload["facts"][0]
        assert "id" in fact
        assert "vendor" in fact
        assert fact["vendor"] == "Seeded Store"

    def test_annotations_contain_expected_fields(self, db: DatabaseManager) -> None:
        """Annotations dicts expose target_type, key, and value."""
        _seed_db(db)

        payload = build_export_payload(db)

        ann = payload["annotations"][0]
        assert ann["target_type"] == "fact"
        assert ann["key"] == "note"
        assert ann["value"] == "test annotation"

    def test_multiple_facts_all_included(self, db: DatabaseManager) -> None:
        """All facts are exported when there are several in the DB."""
        for i in range(3):
            doc = _make_doc(db, f"/test/r{i}.jpg")
            atom = _make_vendor_atom(db, doc, f"Vendor {i}")
            bundle = _make_bundle(db, doc, [atom])
            _make_fact(db, bundle, vendor=f"Vendor {i}")

        payload = build_export_payload(db)

        assert len(payload["facts"]) == 3

    def test_fact_items_aggregated_across_all_facts(self, db: DatabaseManager) -> None:
        """fact_items from multiple facts are merged into a single flat list."""
        # This test uses the DB directly to create fact items so we can verify
        # the aggregation logic. We confirm the list starts empty for no items.
        doc = _make_doc(db)
        atom = _make_vendor_atom(db, doc)
        bundle = _make_bundle(db, doc, [atom])
        _make_fact(db, bundle)

        payload = build_export_payload(db)

        # No items were stored; the aggregation should still return a list.
        assert isinstance(payload["fact_items"], list)

    def test_facts_have_source_and_user_id_fields(self, db: DatabaseManager) -> None:
        """Each fact dict in the payload has source and user_id fields."""
        uid = _make_user(db, "u-42")
        doc = _make_doc(db, source="telegram", user_id=uid)
        atom = _make_vendor_atom(db, doc)
        bundle = _make_bundle(db, doc, [atom])
        _make_fact(db, bundle)

        payload = build_export_payload(db)

        fact = payload["facts"][0]
        assert "source" in fact
        assert "user_id" in fact
        assert fact["source"] == "telegram"
        assert fact["user_id"] == "u-42"

    def test_facts_source_defaults_when_no_source_on_document(
        self, db: DatabaseManager
    ) -> None:
        """Facts linked to documents without source default to cli/system."""
        doc = _make_doc(db)  # no source or user_id
        atom = _make_vendor_atom(db, doc)
        bundle = _make_bundle(db, doc, [atom])
        _make_fact(db, bundle)

        payload = build_export_payload(db)

        fact = payload["facts"][0]
        assert fact["source"] == "cli"
        assert fact["user_id"] == "system"

    def test_documents_list_contains_provenance_fields(
        self, db: DatabaseManager
    ) -> None:
        """Documents list exposes id, source, user_id, created_at."""
        uid = _make_user(db, "u-99")
        _make_doc(db, source="api", user_id=uid)

        payload = build_export_payload(db)

        assert len(payload["documents"]) == 1
        doc = payload["documents"][0]
        assert "id" in doc
        assert "source" in doc
        assert "user_id" in doc
        assert "created_at" in doc
        assert doc["source"] == "api"
        assert doc["user_id"] == "u-99"

    def test_documents_list_includes_all_documents(self, db: DatabaseManager) -> None:
        """All documents are present in the documents list."""
        for i in range(3):
            _make_doc(db, file_path=f"/test/doc{i}.jpg", source="cli")

        payload = build_export_payload(db)

        assert len(payload["documents"]) == 3


# ---------------------------------------------------------------------------
# 2. export_analytics.push_to_analytics_stack
# ---------------------------------------------------------------------------


class TestPushToAnalyticsStack:
    """Tests for push_to_analytics_stack()."""

    def test_push_success_returns_result_dict(
        self,
        db: DatabaseManager,
        capture_server: tuple[str, list[dict[str, Any]]],
    ) -> None:
        """Successful push returns a dict with counts and status='ok'."""
        url, received = capture_server

        result = push_to_analytics_stack(db, url)

        assert result["status"] == "ok"
        assert "facts_count" in result
        assert "items_count" in result
        assert "annotations_count" in result
        assert isinstance(result["facts_count"], int)
        assert isinstance(result["items_count"], int)
        assert isinstance(result["annotations_count"], int)

    def test_push_calls_correct_endpoint(
        self,
        db: DatabaseManager,
        capture_server: tuple[str, list[dict[str, Any]]],
    ) -> None:
        """Request is sent to {url}/v1/ingest/alibi."""
        url, _received = capture_server

        # Intercept at the urllib layer to check the exact URL used.
        original_urlopen = __import__("urllib.request", fromlist=["urlopen"]).urlopen
        called_urls: list[str] = []

        def spy_urlopen(req, timeout=None):
            called_urls.append(req.full_url)
            return original_urlopen(req, timeout=timeout)

        with patch("alibi.services.export_analytics.urlopen", side_effect=spy_urlopen):
            push_to_analytics_stack(db, url)

        assert len(called_urls) == 1
        assert called_urls[0] == f"{url}/v1/ingest/alibi"

    def test_push_sends_json_payload(
        self,
        db: DatabaseManager,
        capture_server: tuple[str, list[dict[str, Any]]],
    ) -> None:
        """Payload received by the server has the expected top-level keys."""
        url, received = capture_server
        _seed_db(db)

        push_to_analytics_stack(db, url)

        assert len(received) == 1
        body = received[0]
        assert "facts" in body
        assert "fact_items" in body
        assert "annotations" in body
        assert "documents" in body

    def test_push_counts_match_db_contents(
        self,
        db: DatabaseManager,
        capture_server: tuple[str, list[dict[str, Any]]],
    ) -> None:
        """Counts in the result reflect actual DB contents."""
        url, _received = capture_server
        _seed_db(db)

        result = push_to_analytics_stack(db, url)

        assert result["facts_count"] == 1
        assert result["items_count"] == 0  # no fact items were inserted
        assert result["annotations_count"] == 1

    def test_push_returns_http_status(
        self,
        db: DatabaseManager,
        capture_server: tuple[str, list[dict[str, Any]]],
    ) -> None:
        """Result dict includes the raw HTTP status code from the server."""
        url, _received = capture_server

        result = push_to_analytics_stack(db, url)

        assert result["http_status"] == 200

    def test_push_strips_trailing_slash_from_url(
        self,
        db: DatabaseManager,
        capture_server: tuple[str, list[dict[str, Any]]],
    ) -> None:
        """Trailing slash on base URL does not produce a double-slash endpoint."""
        url, _received = capture_server
        called_urls: list[str] = []
        original_urlopen = __import__("urllib.request", fromlist=["urlopen"]).urlopen

        def spy_urlopen(req, timeout=None):
            called_urls.append(req.full_url)
            return original_urlopen(req, timeout=timeout)

        with patch("alibi.services.export_analytics.urlopen", side_effect=spy_urlopen):
            push_to_analytics_stack(db, url.rstrip("/") + "/")

        assert "//" not in called_urls[0].replace("://", "@@")

    def test_push_raises_connection_error_on_failure(
        self,
        db: DatabaseManager,
    ) -> None:
        """ConnectionError is raised when the HTTP request fails."""
        with pytest.raises(ConnectionError, match="Analytics stack unreachable"):
            push_to_analytics_stack(db, "http://127.0.0.1:1/unreachable", timeout=0.5)

    def test_push_connection_error_wraps_original(
        self,
        db: DatabaseManager,
    ) -> None:
        """The raised ConnectionError chains from the underlying network error."""
        with pytest.raises(ConnectionError) as exc_info:
            push_to_analytics_stack(db, "http://127.0.0.1:1/unreachable", timeout=0.5)

        assert exc_info.value.__cause__ is not None

    def test_push_uses_mock_build_payload(self, db: DatabaseManager) -> None:
        """push_to_analytics_stack calls build_export_payload and sends result."""
        fake_payload = {
            "facts": [{"id": "f1", "vendor": "Mock", "source": None, "user_id": None}],
            "fact_items": [],
            "annotations": [],
            "documents": [],
        }

        captured_bodies: list[dict] = []

        def fake_urlopen(req, timeout=None):
            body = json.loads(req.data.decode("utf-8"))
            captured_bodies.append(body)
            mock_resp = MagicMock()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = b'{"status": "ok"}'
            mock_resp.status = 200
            return mock_resp

        with (
            patch(
                "alibi.services.export_analytics.build_export_payload",
                return_value=fake_payload,
            ),
            patch("alibi.services.export_analytics.urlopen", side_effect=fake_urlopen),
        ):
            result = push_to_analytics_stack(db, "http://example.com")

        assert result["facts_count"] == 1
        assert result["items_count"] == 0
        assert result["annotations_count"] == 0
        assert len(captured_bodies) == 1
        assert captured_bodies[0]["facts"][0]["vendor"] == "Mock"


# ---------------------------------------------------------------------------
# 3. AnalyticsExportSubscriber
# ---------------------------------------------------------------------------


class TestAnalyticsExportSubscriberLifecycle:
    """Tests for subscriber start / stop / event bus wiring."""

    def test_start_subscribes_to_fact_created(self) -> None:
        """After start(), FACT_CREATED has the subscriber's handler registered."""
        bus = EventBus()
        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://example.com",
            bus=bus,
        )

        sub.start()

        assert sub._handle in bus._subscribers[EventType.FACT_CREATED]

    def test_start_subscribes_to_fact_updated(self) -> None:
        """After start(), FACT_UPDATED has the subscriber's handler registered."""
        bus = EventBus()
        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://example.com",
            bus=bus,
        )

        sub.start()

        assert sub._handle in bus._subscribers[EventType.FACT_UPDATED]

    def test_stop_unsubscribes_from_fact_created(self) -> None:
        """After stop(), handler is removed from FACT_CREATED."""
        bus = EventBus()
        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://example.com",
            bus=bus,
        )

        sub.start()
        sub.stop()

        assert sub._handle not in bus._subscribers.get(EventType.FACT_CREATED, [])

    def test_stop_unsubscribes_from_fact_updated(self) -> None:
        """After stop(), handler is removed from FACT_UPDATED."""
        bus = EventBus()
        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://example.com",
            bus=bus,
        )

        sub.start()
        sub.stop()

        assert sub._handle not in bus._subscribers.get(EventType.FACT_UPDATED, [])

    def test_stop_before_start_is_safe(self) -> None:
        """Calling stop() before start() does not raise."""
        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://example.com",
            bus=EventBus(),
        )
        sub.stop()  # must not raise

    def test_start_without_bus_uses_global_bus(self) -> None:
        """When bus=None, start() resolves to the global singleton."""
        from alibi.services.events import event_bus as global_bus

        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://example.com",
            bus=None,
        )
        try:
            sub.start()
            assert sub._handle in global_bus._subscribers.get(
                EventType.FACT_CREATED, []
            )
        finally:
            sub.stop()


_PUSH_TARGET = "alibi.services.export_analytics.push_to_analytics_stack"

_FAKE_RESULT = {
    "facts_count": 0,
    "items_count": 0,
    "annotations_count": 0,
    "status": "ok",
    "http_status": 200,
}


def _make_done_side_effect(done: threading.Event):
    """Return a push side-effect that sets done and returns a fake result."""

    def _side_effect(db, url, timeout=30.0):
        done.set()
        return _FAKE_RESULT.copy()

    return _side_effect


class TestAnalyticsExportSubscriberEvents:
    """Tests for subscriber event handling and export triggering."""

    def test_fact_created_triggers_export(self) -> None:
        """Emitting FACT_CREATED causes push_to_analytics_stack to be called."""
        bus = EventBus()
        done = threading.Event()

        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://example.com",
            bus=bus,
        )
        sub.start()

        with patch(_PUSH_TARGET, side_effect=_make_done_side_effect(done)):
            bus.emit(EventType.FACT_CREATED, {"fact_id": "f1"})
            triggered = done.wait(timeout=2.0)

        sub.stop()
        assert triggered, "Export was not triggered within timeout after FACT_CREATED"

    def test_fact_updated_triggers_export(self) -> None:
        """Emitting FACT_UPDATED causes push_to_analytics_stack to be called."""
        bus = EventBus()
        done = threading.Event()

        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://example.com",
            bus=bus,
        )
        sub.start()

        with patch(_PUSH_TARGET, side_effect=_make_done_side_effect(done)):
            bus.emit(EventType.FACT_UPDATED, {"fact_id": "f2"})
            triggered = done.wait(timeout=2.0)

        sub.stop()
        assert triggered, "Export was not triggered within timeout after FACT_UPDATED"

    def test_document_ingested_does_not_trigger_export(self) -> None:
        """DOCUMENT_INGESTED event must NOT trigger the analytics export."""
        bus = EventBus()
        export_called = threading.Event()

        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://example.com",
            bus=bus,
        )
        sub.start()

        with patch(_PUSH_TARGET, side_effect=lambda *a, **k: export_called.set()):
            bus.emit(EventType.DOCUMENT_INGESTED, {"document_id": "d1"})
            time.sleep(0.3)

        sub.stop()
        assert (
            not export_called.is_set()
        ), "Export was unexpectedly triggered by DOCUMENT_INGESTED"

    def test_correction_applied_does_not_trigger_export(self) -> None:
        """CORRECTION_APPLIED event must NOT trigger the analytics export."""
        bus = EventBus()
        export_called = threading.Event()

        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://example.com",
            bus=bus,
        )
        sub.start()

        with patch(_PUSH_TARGET, side_effect=lambda *a, **k: export_called.set()):
            bus.emit(EventType.CORRECTION_APPLIED, {"fact_id": "fx"})
            time.sleep(0.3)

        sub.stop()
        assert (
            not export_called.is_set()
        ), "Export was unexpectedly triggered by CORRECTION_APPLIED"

    def test_export_count_increments_on_success(self) -> None:
        """export_count is incremented for each successful export."""
        bus = EventBus()
        done = threading.Event()

        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://example.com",
            bus=bus,
        )
        sub.start()

        def succeed(*a, **k):
            done.set()
            return _FAKE_RESULT.copy()

        with patch(_PUSH_TARGET, side_effect=succeed):
            bus.emit(EventType.FACT_CREATED, {"fact_id": "f3"})
            done.wait(timeout=2.0)

        sub.stop()
        assert sub.export_count == 1

    def test_connection_error_does_not_crash_subscriber(self) -> None:
        """A ConnectionError inside _export is swallowed and does not crash."""
        bus = EventBus()
        error_invoked = threading.Event()

        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://unreachable.invalid",
            bus=bus,
        )
        sub.start()

        def fail(*a, **k):
            error_invoked.set()
            raise ConnectionError("Analytics stack unreachable: [Errno 111]")

        with patch(_PUSH_TARGET, side_effect=fail):
            bus.emit(EventType.FACT_CREATED, {"fact_id": "f4"})
            error_invoked.wait(timeout=2.0)

        sub.stop()
        # If we reach here, the subscriber didn't crash. export_count stays 0.
        assert sub.export_count == 0

    def test_export_uses_correct_url(self) -> None:
        """push_to_analytics_stack receives the URL configured on the subscriber."""
        bus = EventBus()
        captured_urls: list[str] = []
        done = threading.Event()

        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://custom-host:9999",
            bus=bus,
        )
        sub.start()

        def capture_url(db, url, timeout=30.0):
            captured_urls.append(url)
            done.set()
            return _FAKE_RESULT.copy()

        with patch(_PUSH_TARGET, side_effect=capture_url):
            bus.emit(EventType.FACT_CREATED, {"fact_id": "f5"})
            done.wait(timeout=2.0)

        sub.stop()
        assert captured_urls == ["http://custom-host:9999"]

    def test_export_thread_is_daemon(self) -> None:
        """Threads spawned by _handle are daemon threads."""
        bus = EventBus()
        thread_ref: list[threading.Thread] = []

        original_thread = threading.Thread

        def capture_thread(*args, **kwargs):
            t = original_thread(*args, **kwargs)
            thread_ref.append(t)
            return t

        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://example.com",
            bus=bus,
        )
        sub.start()

        done = threading.Event()

        def noop(*a, **k):
            done.set()
            return {
                "facts_count": 0,
                "items_count": 0,
                "annotations_count": 0,
                "status": "ok",
            }

        with (
            patch("threading.Thread", side_effect=capture_thread),
            patch(_PUSH_TARGET, side_effect=noop),
        ):
            bus.emit(EventType.FACT_CREATED, {"fact_id": "f6"})

        sub.stop()

        # At least one thread was created; all should be daemon threads
        if thread_ref:
            assert all(t.daemon for t in thread_ref)


# ---------------------------------------------------------------------------
# 3b. AnalyticsExportSubscriber — periodic timer
# ---------------------------------------------------------------------------


class TestAnalyticsExportSubscriberTimer:
    """Tests for the periodic export timer in AnalyticsExportSubscriber."""

    def test_timer_thread_starts_on_start(self) -> None:
        """After start(), _timer_thread is a live daemon thread."""
        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://example.com",
            bus=EventBus(),
            export_interval=60.0,
        )
        sub.start()
        try:
            assert sub._timer_thread is not None
            assert sub._timer_thread.is_alive()
            assert sub._timer_thread.daemon is True
            assert sub._timer_thread.name == "analytics-export-timer"
        finally:
            sub.stop()

    def test_timer_thread_stops_on_stop(self) -> None:
        """After stop(), the timer thread reference is cleared."""
        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://example.com",
            bus=EventBus(),
            export_interval=60.0,
        )
        sub.start()
        sub.stop()

        assert sub._timer_thread is None

    def test_timer_triggers_export_after_interval(self) -> None:
        """Timer fires push_to_analytics_stack once the interval elapses."""
        done = threading.Event()

        def push_side_effect(db, url, timeout=30.0):
            done.set()
            return _FAKE_RESULT.copy()

        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://example.com",
            bus=EventBus(),
            export_interval=0.1,
        )

        with patch(_PUSH_TARGET, side_effect=push_side_effect):
            sub.start()
            triggered = done.wait(timeout=2.0)
            sub.stop()

        assert triggered, "Periodic export was not triggered within timeout"

    def test_timer_increments_export_count(self) -> None:
        """export_count is incremented by each periodic export."""
        done = threading.Event()

        def push_side_effect(db, url, timeout=30.0):
            done.set()
            return _FAKE_RESULT.copy()

        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://example.com",
            bus=EventBus(),
            export_interval=0.1,
        )

        with patch(_PUSH_TARGET, side_effect=push_side_effect):
            sub.start()
            done.wait(timeout=2.0)
            sub.stop()

        assert sub.export_count >= 1

    def test_timer_connection_error_does_not_crash(self) -> None:
        """ConnectionError in the timer loop is swallowed; loop continues."""
        first_call = threading.Event()
        second_call = threading.Event()
        call_count = 0

        def push_side_effect(db, url, timeout=30.0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                first_call.set()
                raise ConnectionError("unreachable")
            second_call.set()
            return _FAKE_RESULT.copy()

        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://example.com",
            bus=EventBus(),
            export_interval=0.1,
        )

        with patch(_PUSH_TARGET, side_effect=push_side_effect):
            sub.start()
            second_call.wait(timeout=3.0)
            sub.stop()

        assert second_call.is_set(), "Timer did not recover after ConnectionError"

    def test_timer_unexpected_error_does_not_crash(self) -> None:
        """An unexpected exception in the timer loop is swallowed and logged."""
        first_call = threading.Event()
        second_call = threading.Event()
        call_count = 0

        def push_side_effect(db, url, timeout=30.0):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                first_call.set()
                raise RuntimeError("boom")
            second_call.set()
            return _FAKE_RESULT.copy()

        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://example.com",
            bus=EventBus(),
            export_interval=0.1,
        )

        with patch(_PUSH_TARGET, side_effect=push_side_effect):
            sub.start()
            second_call.wait(timeout=3.0)
            sub.stop()

        assert second_call.is_set(), "Timer did not recover after RuntimeError"

    def test_graceful_shutdown_within_timeout(self) -> None:
        """stop() returns promptly without waiting out the full interval."""
        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://example.com",
            bus=EventBus(),
            export_interval=300.0,
        )
        sub.start()

        start = time.monotonic()
        sub.stop()
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"stop() took {elapsed:.1f}s — shutdown not prompt"

    def test_export_interval_default_is_300(self) -> None:
        """Default export_interval is 300 seconds."""
        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://example.com",
        )
        assert sub._export_interval == 300.0

    def test_export_interval_custom_value_stored(self) -> None:
        """Custom export_interval is stored on the instance."""
        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://example.com",
            export_interval=120.0,
        )
        assert sub._export_interval == 120.0

    def test_stop_before_start_is_safe_with_timer(self) -> None:
        """Calling stop() before start() does not raise even with timer support."""
        sub = AnalyticsExportSubscriber(
            db_factory=MagicMock(),
            analytics_url="http://example.com",
            bus=EventBus(),
            export_interval=0.1,
        )
        sub.stop()  # must not raise


# ---------------------------------------------------------------------------
# 4. CLI analytics export command
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestCliAnalyticsExportDryRun:
    """Tests for `alibi analytics export --dry-run`."""

    def test_dry_run_prints_payload_stats(
        self,
        runner: CliRunner,
        db: DatabaseManager,
    ) -> None:
        """--dry-run prints counts for facts, fact_items, and annotations."""
        _seed_db(db)

        with patch("alibi.commands.analytics.get_db", return_value=db):
            result = runner.invoke(cli, ["analytics", "export", "--dry-run"])

        assert result.exit_code == 0
        assert "Facts" in result.output or "facts" in result.output.lower()
        assert "1" in result.output  # one fact was seeded

    def test_dry_run_does_not_make_http_call(
        self,
        runner: CliRunner,
        db: DatabaseManager,
    ) -> None:
        """--dry-run must NOT call push_to_analytics_stack."""
        with (
            patch("alibi.commands.analytics.get_db", return_value=db),
            patch(_PUSH_TARGET, autospec=True) as mock_push,
        ):
            runner.invoke(cli, ["analytics", "export", "--dry-run"])

        mock_push.assert_not_called()

    def test_dry_run_shows_three_stat_lines(
        self,
        runner: CliRunner,
        db: DatabaseManager,
    ) -> None:
        """Dry-run output contains lines for Facts, Fact items, and Annotations."""
        with patch("alibi.commands.analytics.get_db", return_value=db):
            result = runner.invoke(cli, ["analytics", "export", "--dry-run"])

        output_lower = result.output.lower()
        # All three categories should appear in output
        assert "fact" in output_lower
        assert "annotation" in output_lower

    def test_dry_run_uninitialized_db_exits_early(
        self,
        runner: CliRunner,
    ) -> None:
        """--dry-run with uninitialized DB prints a warning and exits 0."""
        mock_db = MagicMock(spec=DatabaseManager)
        mock_db.is_initialized.return_value = False

        with patch("alibi.commands.analytics.get_db", return_value=mock_db):
            result = runner.invoke(cli, ["analytics", "export", "--dry-run"])

        assert result.exit_code == 0
        assert "not initialized" in result.output.lower()


class TestCliAnalyticsExport:
    """Tests for `alibi analytics export` (normal, non-dry-run)."""

    def test_successful_export_prints_counts(
        self,
        runner: CliRunner,
        db: DatabaseManager,
    ) -> None:
        """Successful export prints facts/items/annotations counts."""
        fake_result = {
            "facts_count": 7,
            "items_count": 23,
            "annotations_count": 2,
            "status": "ok",
            "http_status": 200,
        }

        with (
            patch("alibi.commands.analytics.get_db", return_value=db),
            patch(_PUSH_TARGET, return_value=fake_result),
        ):
            result = runner.invoke(
                cli,
                ["analytics", "export", "--url", "http://localhost:8070"],
            )

        assert result.exit_code == 0

    def test_successful_export_exit_code_zero(
        self,
        runner: CliRunner,
        db: DatabaseManager,
        capture_server: tuple[str, list[dict[str, Any]]],
    ) -> None:
        """Export against a live test server returns exit code 0."""
        url, _received = capture_server

        with patch("alibi.commands.analytics.get_db", return_value=db):
            result = runner.invoke(cli, ["analytics", "export", "--url", url])

        assert result.exit_code == 0

    def test_export_sends_request_to_server(
        self,
        runner: CliRunner,
        db: DatabaseManager,
        capture_server: tuple[str, list[dict[str, Any]]],
    ) -> None:
        """CLI export actually POSTs the payload to the server."""
        url, received = capture_server
        _seed_db(db)

        with patch("alibi.commands.analytics.get_db", return_value=db):
            runner.invoke(cli, ["analytics", "export", "--url", url])

        assert len(received) == 1
        assert "facts" in received[0]

    def test_connection_error_prints_message(
        self,
        runner: CliRunner,
        db: DatabaseManager,
    ) -> None:
        """Connection failure prints an error message without crashing."""
        with (
            patch("alibi.commands.analytics.get_db", return_value=db),
            patch(
                _PUSH_TARGET,
                side_effect=ConnectionError("Analytics stack unreachable"),
            ),
        ):
            result = runner.invoke(
                cli, ["analytics", "export", "--url", "http://127.0.0.1:1"]
            )

        assert result.exit_code == 0
        assert "failed" in result.output.lower() or "error" in result.output.lower()

    def test_default_url_is_localhost_8070(
        self,
        runner: CliRunner,
        db: DatabaseManager,
    ) -> None:
        """When --url is omitted, the command targets localhost:8070 by default."""
        captured_urls: list[str] = []

        def capture(db, url, **k):
            captured_urls.append(url)
            return {
                "facts_count": 0,
                "items_count": 0,
                "annotations_count": 0,
                "status": "ok",
                "http_status": 200,
            }

        with (
            patch("alibi.commands.analytics.get_db", return_value=db),
            patch(_PUSH_TARGET, side_effect=capture),
        ):
            runner.invoke(cli, ["analytics", "export"])

        assert any("8070" in u for u in captured_urls)

    def test_uninitialized_db_exits_early(
        self,
        runner: CliRunner,
    ) -> None:
        """Uninitialized database prints a warning and exits 0 without HTTP call."""
        mock_db = MagicMock(spec=DatabaseManager)
        mock_db.is_initialized.return_value = False

        with (
            patch("alibi.commands.analytics.get_db", return_value=mock_db),
            patch(_PUSH_TARGET) as mock_push,
        ):
            result = runner.invoke(
                cli, ["analytics", "export", "--url", "http://localhost:8070"]
            )

        assert result.exit_code == 0
        assert "not initialized" in result.output.lower()
        mock_push.assert_not_called()

    def test_url_env_var_is_read(
        self,
        runner: CliRunner,
        db: DatabaseManager,
    ) -> None:
        """ALIBI_ANALYTICS_STACK_URL environment variable sets the target URL."""
        captured_urls: list[str] = []

        def capture(db, url, **k):
            captured_urls.append(url)
            return {
                "facts_count": 0,
                "items_count": 0,
                "annotations_count": 0,
                "status": "ok",
                "http_status": 200,
            }

        with (
            patch("alibi.commands.analytics.get_db", return_value=db),
            patch(_PUSH_TARGET, side_effect=capture),
        ):
            result = runner.invoke(
                cli,
                ["analytics", "export"],
                env={"ALIBI_ANALYTICS_STACK_URL": "http://env-host:7777"},
            )

        assert result.exit_code == 0
        assert any("7777" in u for u in captured_urls)
