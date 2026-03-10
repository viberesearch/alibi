"""Tests for scheduled enrichment runner."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.daemon.scheduler import (
    EnrichmentCycleResult,
    EnrichmentScheduler,
    PhaseResult,
    SchedulerState,
    _load_state,
    _save_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_item(  # type: ignore[no-untyped-def]
    db,
    name,
    barcode=None,
    brand=None,
    category=None,
    vendor_key=None,
    enrichment_source=None,
    enrichment_confidence=None,
) -> str:
    """Create a minimal fact_item with parent records."""
    conn = db.get_connection()
    item_id = f"fi-{uuid.uuid4().hex[:8]}"
    fact_id = f"fact-{uuid.uuid4().hex[:8]}"
    doc_id = f"doc-{uuid.uuid4().hex[:8]}"
    cloud_id = f"cloud-{uuid.uuid4().hex[:8]}"
    atom_id = f"atom-{uuid.uuid4().hex[:8]}"

    conn.execute(
        "INSERT OR IGNORE INTO documents (id, file_path, file_hash) "
        "VALUES (?, ?, ?)",
        (doc_id, f"/tmp/{doc_id}.jpg", f"hash-{doc_id}"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO clouds (id, status) VALUES (?, 'collapsed')",
        (cloud_id,),
    )
    conn.execute(
        "INSERT INTO facts "
        "(id, cloud_id, fact_type, vendor, vendor_key, "
        "total_amount, currency, event_date) "
        "VALUES (?, ?, 'purchase', 'Test', ?, 10.0, 'EUR', '2026-01-01')",
        (fact_id, cloud_id, vendor_key),
    )
    conn.execute(
        "INSERT INTO atoms (id, document_id, atom_type, data) "
        "VALUES (?, ?, 'item', '{}')",
        (atom_id, doc_id),
    )
    conn.execute(
        "INSERT INTO fact_items "
        "(id, fact_id, atom_id, name, barcode, brand, category, "
        "enrichment_source, enrichment_confidence, quantity, total_price) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 5.0)",
        (
            item_id,
            fact_id,
            atom_id,
            name,
            barcode,
            brand,
            category,
            enrichment_source,
            enrichment_confidence,
        ),
    )
    conn.commit()
    return item_id


# ===========================================================================
# TestSchedulerState
# ===========================================================================


class TestSchedulerState:
    def test_default_state(self) -> None:
        state = SchedulerState()
        assert state.last_cycle == 0.0
        assert state.last_gemini == 0.0
        assert state.last_maintenance == 0.0
        assert state.cycle_count == 0

    def test_roundtrip(self, tmp_path: Path) -> None:
        state = SchedulerState(
            last_cycle=1000.0,
            last_gemini=2000.0,
            last_maintenance=3000.0,
            cycle_count=5,
        )
        d = state.to_dict()
        restored = SchedulerState.from_dict(d)
        assert restored.last_cycle == 1000.0
        assert restored.last_gemini == 2000.0
        assert restored.last_maintenance == 3000.0
        assert restored.cycle_count == 5

    def test_from_dict_missing_keys(self) -> None:
        state = SchedulerState.from_dict({})
        assert state.last_cycle == 0.0
        assert state.cycle_count == 0

    def test_save_and_load(self, tmp_path: Path) -> None:
        import alibi.daemon.scheduler as sched

        original_file = sched._STATE_FILE
        try:
            sched._STATE_FILE = tmp_path / "state.json"
            state = SchedulerState(last_cycle=42.0, cycle_count=3)
            _save_state(state)

            loaded = _load_state()
            assert loaded.last_cycle == 42.0
            assert loaded.cycle_count == 3
        finally:
            sched._STATE_FILE = original_file

    def test_load_corrupt_file(self, tmp_path: Path) -> None:
        import alibi.daemon.scheduler as sched

        original_file = sched._STATE_FILE
        try:
            sched._STATE_FILE = tmp_path / "state.json"
            sched._STATE_FILE.write_text("not json")
            state = _load_state()
            assert state.cycle_count == 0
        finally:
            sched._STATE_FILE = original_file

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        import alibi.daemon.scheduler as sched

        original_file = sched._STATE_FILE
        try:
            sched._STATE_FILE = tmp_path / "nonexistent.json"
            state = _load_state()
            assert state.cycle_count == 0
        finally:
            sched._STATE_FILE = original_file


# ===========================================================================
# TestPhaseResult
# ===========================================================================


class TestPhaseResult:
    def test_defaults(self) -> None:
        pr = PhaseResult(name="test")
        assert pr.items_processed == 0
        assert pr.duration_seconds == 0.0
        assert pr.skipped is False
        assert pr.error is None

    def test_with_error(self) -> None:
        pr = PhaseResult(name="test", error="something broke")
        assert pr.error == "something broke"


# ===========================================================================
# TestEnrichmentCycleResult
# ===========================================================================


class TestEnrichmentCycleResult:
    def test_duration(self) -> None:
        r = EnrichmentCycleResult(started_at=100.0, finished_at=105.5)
        assert r.duration_seconds == pytest.approx(5.5)

    def test_total_enriched(self) -> None:
        r = EnrichmentCycleResult(
            phases=[
                PhaseResult(name="a", items_processed=10),
                PhaseResult(name="b", items_processed=5, skipped=True),
                PhaseResult(name="c", items_processed=3),
            ]
        )
        # Skipped phases excluded from total
        assert r.total_enriched == 13


# ===========================================================================
# TestEnrichmentScheduler
# ===========================================================================


class TestEnrichmentScheduler:
    def test_run_cycle_empty_db(self, db: MagicMock, tmp_path: Path) -> None:
        """Cycle on empty DB completes without error."""
        import alibi.daemon.scheduler as sched

        original_file = sched._STATE_FILE
        try:
            sched._STATE_FILE = tmp_path / "state.json"
            scheduler = EnrichmentScheduler(
                db_factory=lambda: db,
            )
            result = scheduler.run_now()
            assert isinstance(result, EnrichmentCycleResult)
            assert len(result.phases) == 5
            assert result.duration_seconds >= 0
            assert scheduler.state.cycle_count == 1
        finally:
            sched._STATE_FILE = original_file

    def test_gemini_skipped_when_disabled(self, db: MagicMock, tmp_path: Path) -> None:
        """Gemini phase skipped when not enabled."""
        import alibi.daemon.scheduler as sched

        original_file = sched._STATE_FILE
        try:
            sched._STATE_FILE = tmp_path / "state.json"
            from alibi.config import Config

            config = Config(
                _env_file=None,
                gemini_enrichment_enabled=False,
                enrichment_schedule_gemini_interval=0,
            )
            scheduler = EnrichmentScheduler(
                db_factory=lambda: db,
                config=config,
            )
            result = scheduler.run_now()
            gemini_phase = next(p for p in result.phases if p.name == "gemini_batch")
            assert gemini_phase.skipped is True
        finally:
            sched._STATE_FILE = original_file

    def test_gemini_runs_when_due(self, db: MagicMock, tmp_path: Path) -> None:
        """Gemini phase runs when enabled and interval elapsed."""
        import alibi.daemon.scheduler as sched

        original_file = sched._STATE_FILE
        try:
            sched._STATE_FILE = tmp_path / "state.json"
            from alibi.config import Config

            config = Config(
                _env_file=None,
                gemini_enrichment_enabled=True,
                gemini_api_key="test-key",
                enrichment_schedule_gemini_interval=0,
            )
            scheduler = EnrichmentScheduler(
                db_factory=lambda: db,
                config=config,
            )

            with patch(
                "alibi.enrichment.gemini_enrichment.enrich_pending_by_gemini",
                return_value=[],
            ):
                result = scheduler.run_now()

            gemini_phase = next(p for p in result.phases if p.name == "gemini_batch")
            assert gemini_phase.skipped is False
        finally:
            sched._STATE_FILE = original_file

    def test_maintenance_skipped_when_not_due(
        self, db: MagicMock, tmp_path: Path
    ) -> None:
        """Maintenance phase skipped when interval not elapsed."""
        import alibi.daemon.scheduler as sched

        original_file = sched._STATE_FILE
        try:
            sched._STATE_FILE = tmp_path / "state.json"
            # Set last maintenance to now so it's not due
            state = SchedulerState(last_maintenance=time.time())
            _save_state(state)

            from alibi.config import Config

            config = Config(
                _env_file=None,
                enrichment_schedule_maintenance_interval=999999,
            )
            scheduler = EnrichmentScheduler(
                db_factory=lambda: db,
                config=config,
            )
            result = scheduler.run_now()
            maint_phase = next(p for p in result.phases if p.name == "maintenance")
            assert maint_phase.skipped is True
        finally:
            sched._STATE_FILE = original_file

    def test_maintenance_runs_when_due(self, db: MagicMock, tmp_path: Path) -> None:
        """Maintenance runs when interval has elapsed."""
        import alibi.daemon.scheduler as sched

        original_file = sched._STATE_FILE
        try:
            sched._STATE_FILE = tmp_path / "state.json"
            from alibi.config import Config

            config = Config(
                _env_file=None,
                enrichment_schedule_maintenance_interval=0,
            )
            scheduler = EnrichmentScheduler(
                db_factory=lambda: db,
                config=config,
            )
            result = scheduler.run_now()
            maint_phase = next(p for p in result.phases if p.name == "maintenance")
            assert maint_phase.skipped is False
        finally:
            sched._STATE_FILE = original_file

    def test_state_persists_across_cycles(self, db: MagicMock, tmp_path: Path) -> None:
        """State is saved after each cycle and loaded on next."""
        import alibi.daemon.scheduler as sched

        original_file = sched._STATE_FILE
        try:
            sched._STATE_FILE = tmp_path / "state.json"
            scheduler = EnrichmentScheduler(db_factory=lambda: db)
            scheduler.run_now()
            assert scheduler.state.cycle_count == 1

            scheduler2 = EnrichmentScheduler(db_factory=lambda: db)
            assert scheduler2.state.cycle_count == 1
            scheduler2.run_now()
            assert scheduler2.state.cycle_count == 2
        finally:
            sched._STATE_FILE = original_file

    def test_start_stop(self, db: MagicMock, tmp_path: Path) -> None:
        """Scheduler starts and stops cleanly."""
        import alibi.daemon.scheduler as sched

        original_file = sched._STATE_FILE
        try:
            sched._STATE_FILE = tmp_path / "state.json"
            from alibi.config import Config

            config = Config(
                _env_file=None,
                enrichment_schedule_interval=3600,
            )
            scheduler = EnrichmentScheduler(
                db_factory=lambda: db,
                config=config,
            )
            scheduler.start()
            assert scheduler.is_running is True
            scheduler.stop()
            assert scheduler.is_running is False
        finally:
            sched._STATE_FILE = original_file

    def test_last_result_stored(self, db: MagicMock, tmp_path: Path) -> None:
        """Last result is accessible after run."""
        import alibi.daemon.scheduler as sched

        original_file = sched._STATE_FILE
        try:
            sched._STATE_FILE = tmp_path / "state.json"
            scheduler = EnrichmentScheduler(db_factory=lambda: db)
            assert scheduler.last_result is None
            scheduler.run_now()
            assert scheduler.last_result is not None
            assert scheduler.last_result.total_enriched >= 0
        finally:
            sched._STATE_FILE = original_file

    def test_phase_error_doesnt_stop_cycle(self, db: MagicMock, tmp_path: Path) -> None:
        """One phase failing doesn't prevent other phases from running."""
        import alibi.daemon.scheduler as sched

        original_file = sched._STATE_FILE
        try:
            sched._STATE_FILE = tmp_path / "state.json"
            scheduler = EnrichmentScheduler(db_factory=lambda: db)

            with patch(
                "alibi.enrichment.service.enrich_pending_items",
                side_effect=RuntimeError("DB error"),
            ):
                result = scheduler.run_now()

            # First phase errored but others still ran
            assert result.phases[0].error is not None
            assert result.phases[0].name == "barcode_cascade"
            # Subsequent phases still executed
            assert len(result.phases) == 5
            assert result.phases[1].error is None
        finally:
            sched._STATE_FILE = original_file


# ===========================================================================
# TestEnrichmentSchedulerWithData
# ===========================================================================


class TestEnrichmentSchedulerWithData:
    def test_barcode_match_enriches_items(self, db: MagicMock, tmp_path: Path) -> None:
        """Barcode match phase finds and enriches items with barcodes."""
        import alibi.daemon.scheduler as sched

        original_file = sched._STATE_FILE
        try:
            sched._STATE_FILE = tmp_path / "state.json"

            # Seed items: one enriched, one unenriched with same barcode
            _seed_item(
                db,
                "Milk Enriched",
                barcode="5000159484695",
                brand="Arla",
                category="Dairy",
                vendor_key="vk-a",
                enrichment_source="openfoodfacts",
                enrichment_confidence=0.95,
            )
            _seed_item(
                db,
                "Milk Plain",
                barcode="5000159484695",
                vendor_key="vk-b",
            )

            # Run barcode match directly (bypass cascade which calls OFF API)
            scheduler = EnrichmentScheduler(db_factory=lambda: db)
            phase = scheduler._run_barcode_match(db, limit=500)
            assert phase.name == "barcode_match"
            assert phase.items_processed >= 1
            assert phase.error is None
        finally:
            sched._STATE_FILE = original_file
