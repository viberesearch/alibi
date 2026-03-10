"""Tests for EnrichmentSubscriber Phase 3 (Gemini batch queuing)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from alibi.services.events import EventBus, EventType
from alibi.services.subscribers.enrichment import (
    EnrichmentSubscriber,
    _GEMINI_BATCH_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(id: str, name: str, barcode: str = "") -> dict:
    return {"id": id, "name": name, "barcode": barcode}


def _make_db(pending_rows: list[dict]) -> MagicMock:
    db = MagicMock()

    def fetchall_side_effect(sql: str, params: tuple = ()) -> list:
        if "fi.id, fi.name, fi.barcode" in sql:
            return pending_rows
        return []

    db.fetchall.side_effect = fetchall_side_effect
    return db


def _make_subscriber(
    pending_rows: list[dict],
) -> tuple[EnrichmentSubscriber, MagicMock]:
    db = _make_db(pending_rows)
    sub = EnrichmentSubscriber(db_factory=lambda: db)
    return sub, db


def _gemini_enabled_config() -> MagicMock:
    cfg = MagicMock()
    cfg.gemini_enrichment_enabled = True
    cfg.gemini_api_key = "test-key"
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPhase3Queuing:
    def test_items_added_to_pending_when_unenriched(self) -> None:
        rows = [_make_row("item-1", "Milk"), _make_row("item-2", "Bread")]
        sub, _ = _make_subscriber(rows)

        with (
            patch(
                "alibi.services.subscribers.enrichment.EnrichmentSubscriber"
                "._flush_gemini_batch"
            ),
            patch("alibi.config.get_config", return_value=_gemini_enabled_config()),
        ):
            sub._queue_for_gemini(sub._db_factory(), "doc-abc")

        assert len(sub._pending_items) == 2
        assert sub._pending_items[0]["id"] == "item-1"

    def test_no_items_added_when_all_enriched(self) -> None:
        sub, _ = _make_subscriber([])

        with patch("alibi.config.get_config", return_value=_gemini_enabled_config()):
            sub._queue_for_gemini(sub._db_factory(), "doc-xyz")

        assert sub._pending_items == []


class TestConfigGuard:
    def test_disabled_flag_skips_queuing(self) -> None:
        rows = [_make_row("item-1", "Milk")]
        sub, _ = _make_subscriber(rows)

        cfg = MagicMock()
        cfg.gemini_enrichment_enabled = False
        cfg.gemini_api_key = "test-key"

        with patch("alibi.config.get_config", return_value=cfg):
            sub._queue_for_gemini(sub._db_factory(), "doc-abc")

        assert sub._pending_items == []

    def test_missing_api_key_skips_queuing(self) -> None:
        rows = [_make_row("item-1", "Milk")]
        sub, _ = _make_subscriber(rows)

        cfg = MagicMock()
        cfg.gemini_enrichment_enabled = True
        cfg.gemini_api_key = None

        with patch("alibi.config.get_config", return_value=cfg):
            sub._queue_for_gemini(sub._db_factory(), "doc-abc")

        assert sub._pending_items == []


class TestBatchFlush:
    def test_flush_calls_gemini_with_pending_items(self) -> None:
        sub, _ = _make_subscriber([])
        sub._pending_items = [
            {"id": "a", "name": "Apple", "barcode": ""},
            {"id": "b", "name": "Butter", "barcode": "1234"},
        ]

        mock_result = MagicMock(success=True)

        with patch(
            "alibi.enrichment.gemini_enrichment.enrich_items_by_gemini",
            return_value=[mock_result, mock_result],
        ) as mock_gemini:
            sub._flush_gemini_batch()

        mock_gemini.assert_called_once()
        call_items = mock_gemini.call_args[0][1]
        assert len(call_items) == 2

    def test_flush_clears_pending_items(self) -> None:
        sub, _ = _make_subscriber([])
        sub._pending_items = [{"id": "a", "name": "Apple", "barcode": ""}]

        with patch(
            "alibi.enrichment.gemini_enrichment.enrich_items_by_gemini",
            return_value=[],
        ):
            sub._flush_gemini_batch()

        assert sub._pending_items == []

    def test_flush_noop_when_queue_empty(self) -> None:
        sub, _ = _make_subscriber([])

        with patch(
            "alibi.enrichment.gemini_enrichment.enrich_items_by_gemini"
        ) as mock_gemini:
            sub._flush_gemini_batch()

        mock_gemini.assert_not_called()

    def test_flush_updates_enrichment_count(self) -> None:
        sub, _ = _make_subscriber([])
        sub._pending_items = [{"id": "a", "name": "Apple", "barcode": ""}]

        with patch(
            "alibi.enrichment.gemini_enrichment.enrich_items_by_gemini",
            return_value=[MagicMock(success=True), MagicMock(success=False)],
        ):
            sub._flush_gemini_batch()

        assert sub.enrichment_count == 1


class TestTimerDebounce:
    def test_timer_started_when_items_queued(self) -> None:
        rows = [_make_row("item-1", "Milk")]
        sub, _ = _make_subscriber(rows)

        with (
            patch("alibi.config.get_config", return_value=_gemini_enabled_config()),
            patch(
                "alibi.services.subscribers.enrichment.EnrichmentSubscriber"
                "._flush_gemini_batch"
            ),
        ):
            sub._queue_for_gemini(sub._db_factory(), "doc-abc")

        assert sub._batch_timer is not None
        sub._batch_timer.cancel()

    def test_threshold_triggers_immediate_flush(self) -> None:
        rows = [
            _make_row(f"item-{i}", f"Product {i}")
            for i in range(_GEMINI_BATCH_THRESHOLD)
        ]
        sub, _ = _make_subscriber(rows)

        flush_calls: list[int] = []
        original_flush = sub._flush_gemini_batch

        def counting_flush() -> None:
            flush_calls.append(1)
            original_flush()

        sub._flush_gemini_batch = counting_flush  # type: ignore[method-assign]

        with (
            patch("alibi.config.get_config", return_value=_gemini_enabled_config()),
            patch(
                "alibi.enrichment.gemini_enrichment.enrich_items_by_gemini",
                return_value=[],
            ),
        ):
            sub._queue_for_gemini(sub._db_factory(), "doc-large")

        assert flush_calls
        assert sub._batch_timer is None


class TestStopFlushes:
    def test_stop_flushes_pending_items(self) -> None:
        sub, _ = _make_subscriber([])
        sub._pending_items = [{"id": "x", "name": "Item X", "barcode": ""}]

        with patch(
            "alibi.enrichment.gemini_enrichment.enrich_items_by_gemini",
            return_value=[],
        ) as mock_gemini:
            sub.stop()

        mock_gemini.assert_called_once()
        assert sub._pending_items == []

    def test_stop_cancels_timer(self) -> None:
        sub, _ = _make_subscriber([])
        mock_timer = MagicMock()
        sub._batch_timer = mock_timer

        with patch(
            "alibi.enrichment.gemini_enrichment.enrich_items_by_gemini",
            return_value=[],
        ):
            sub.stop()

        mock_timer.cancel.assert_called_once()
        assert sub._batch_timer is None

    def test_stop_noop_when_nothing_pending(self) -> None:
        bus = EventBus()
        sub = EnrichmentSubscriber(db_factory=MagicMock(), bus=bus)
        sub.start()

        with patch(
            "alibi.enrichment.gemini_enrichment.enrich_items_by_gemini"
        ) as mock_gemini:
            sub.stop()

        mock_gemini.assert_not_called()
