"""Scheduled enrichment runner.

Runs enrichment operations on configurable intervals using threading.Timer.
Designed to run inside the WatcherDaemon but can also be triggered manually
via CLI or API.

Operations are executed in priority order:
1. Barcode cascade (OFF -> UPCitemdb -> GS1)
2. Cross-vendor barcode matching
3. Fuzzy name matching
4. Gemini mega-batch (if enabled, separate longer interval)
5. Maintenance (template reliability, identity dedup — weekly)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from alibi.config import Config, get_config
from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)

_STATE_DIR = Path.home() / ".alibi"
_STATE_FILE = _STATE_DIR / "scheduler_state.json"


@dataclass
class PhaseResult:
    """Result of a single enrichment phase."""

    name: str
    items_processed: int = 0
    duration_seconds: float = 0.0
    skipped: bool = False
    error: str | None = None


@dataclass
class EnrichmentCycleResult:
    """Result of a complete enrichment cycle."""

    started_at: float = 0.0
    finished_at: float = 0.0
    phases: list[PhaseResult] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return self.finished_at - self.started_at

    @property
    def total_enriched(self) -> int:
        return sum(p.items_processed for p in self.phases if not p.skipped)


@dataclass
class SchedulerState:
    """Persisted scheduler state for tracking last-run times."""

    last_cycle: float = 0.0
    last_gemini: float = 0.0
    last_maintenance: float = 0.0
    cycle_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchedulerState:
        return cls(
            last_cycle=data.get("last_cycle", 0.0),
            last_gemini=data.get("last_gemini", 0.0),
            last_maintenance=data.get("last_maintenance", 0.0),
            cycle_count=data.get("cycle_count", 0),
        )


def _load_state() -> SchedulerState:
    """Load scheduler state from disk."""
    if _STATE_FILE.exists():
        try:
            data = json.loads(_STATE_FILE.read_text())
            return SchedulerState.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Corrupt scheduler state file, resetting")
    return SchedulerState()


def _save_state(state: SchedulerState) -> None:
    """Persist scheduler state to disk."""
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state.to_dict(), indent=2))


class EnrichmentScheduler:
    """Runs enrichment operations on a configurable schedule.

    Uses threading.Timer for periodic execution (same pattern as
    AnalyticsExportSubscriber). Integrates with the WatcherDaemon.
    """

    def __init__(
        self,
        db_factory: Callable[[], DatabaseManager],
        config: Config | None = None,
    ) -> None:
        self._db_factory = db_factory
        self._config = config or get_config()
        self._timer: threading.Timer | None = None
        self._running = False
        self._lock = threading.Lock()
        self._state = _load_state()
        self._last_result: EnrichmentCycleResult | None = None

    @property
    def state(self) -> SchedulerState:
        return self._state

    @property
    def last_result(self) -> EnrichmentCycleResult | None:
        return self._last_result

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return
        self._running = True
        self._schedule_next()
        logger.info(
            "Enrichment scheduler started (interval=%ds)",
            self._config.enrichment_schedule_interval,
        )

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        logger.info("Enrichment scheduler stopped")

    def run_now(self) -> EnrichmentCycleResult:
        """Run enrichment cycle immediately (blocking)."""
        return self._run_cycle()

    def _schedule_next(self) -> None:
        """Schedule the next enrichment cycle."""
        if not self._running:
            return
        interval = self._config.enrichment_schedule_interval
        self._timer = threading.Timer(interval, self._on_timer)
        self._timer.daemon = True
        self._timer.start()

    def _on_timer(self) -> None:
        """Timer callback — runs cycle then reschedules."""
        try:
            self._run_cycle()
        except Exception:
            logger.exception("Enrichment cycle failed")
        finally:
            self._schedule_next()

    def _run_cycle(self) -> EnrichmentCycleResult:
        """Execute all enrichment steps in priority order."""
        with self._lock:
            result = EnrichmentCycleResult(started_at=time.time())
            db = self._db_factory()
            limit = self._config.enrichment_schedule_limit
            now = time.time()

            # Phase 1: Barcode cascade (OFF -> UPCitemdb -> GS1)
            result.phases.append(self._run_barcode_cascade(db, limit))

            # Phase 2: Cross-vendor barcode matching
            result.phases.append(self._run_barcode_match(db, limit))

            # Phase 3: Fuzzy name matching
            result.phases.append(self._run_name_matching(db, limit))

            # Phase 4: Gemini mega-batch (separate interval)
            gemini_interval = self._config.enrichment_schedule_gemini_interval
            gemini_due = (now - self._state.last_gemini) >= gemini_interval
            if gemini_due and self._config.gemini_enrichment_enabled:
                result.phases.append(self._run_gemini_batch(db, limit))
                self._state.last_gemini = now
            else:
                result.phases.append(PhaseResult(name="gemini_batch", skipped=True))

            # Phase 5: Maintenance (separate interval)
            maint_interval = self._config.enrichment_schedule_maintenance_interval
            maint_due = (now - self._state.last_maintenance) >= maint_interval
            if maint_due:
                result.phases.append(self._run_maintenance(db))
                self._state.last_maintenance = now
            else:
                result.phases.append(PhaseResult(name="maintenance", skipped=True))

            result.finished_at = time.time()

            # Update state
            self._state.last_cycle = now
            self._state.cycle_count += 1
            _save_state(self._state)

            self._last_result = result

            logger.info(
                "Enrichment cycle #%d complete: %d items in %.1fs",
                self._state.cycle_count,
                result.total_enriched,
                result.duration_seconds,
            )
            return result

    def _run_barcode_cascade(self, db: DatabaseManager, limit: int) -> PhaseResult:
        """Phase 1: Multi-source barcode lookup."""
        start = time.time()
        try:
            from alibi.enrichment.service import enrich_pending_items

            results = enrich_pending_items(db, limit=limit)
            return PhaseResult(
                name="barcode_cascade",
                items_processed=sum(1 for r in results if r.success),
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            logger.exception("Barcode cascade failed")
            return PhaseResult(
                name="barcode_cascade",
                duration_seconds=time.time() - start,
                error=str(e),
            )

    def _run_barcode_match(self, db: DatabaseManager, limit: int) -> PhaseResult:
        """Phase 2: Cross-vendor barcode propagation."""
        start = time.time()
        try:
            from alibi.enrichment.barcode_matcher import match_all_barcodes

            results = match_all_barcodes(db, limit=limit)
            return PhaseResult(
                name="barcode_match",
                items_processed=len(results),
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            logger.exception("Barcode matching failed")
            return PhaseResult(
                name="barcode_match",
                duration_seconds=time.time() - start,
                error=str(e),
            )

    def _run_name_matching(self, db: DatabaseManager, limit: int) -> PhaseResult:
        """Phase 3: Fuzzy name matching."""
        start = time.time()
        try:
            from alibi.enrichment.service import enrich_pending_by_name

            results = enrich_pending_by_name(db, limit=limit)
            return PhaseResult(
                name="name_matching",
                items_processed=sum(1 for r in results if r.success),
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            logger.exception("Name matching failed")
            return PhaseResult(
                name="name_matching",
                duration_seconds=time.time() - start,
                error=str(e),
            )

    def _run_gemini_batch(self, db: DatabaseManager, limit: int) -> PhaseResult:
        """Phase 4: Gemini mega-batch enrichment."""
        start = time.time()
        try:
            from alibi.enrichment.gemini_enrichment import (
                enrich_pending_by_gemini,
            )

            results = enrich_pending_by_gemini(db, limit=limit)
            return PhaseResult(
                name="gemini_batch",
                items_processed=sum(1 for r in results if r.success),
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            logger.exception("Gemini batch failed")
            return PhaseResult(
                name="gemini_batch",
                duration_seconds=time.time() - start,
                error=str(e),
            )

    def _run_maintenance(self, db: DatabaseManager) -> PhaseResult:
        """Phase 5: Template reliability + identity dedup."""
        start = time.time()
        try:
            from alibi.maintenance.learning_aggregation import (
                run_full_maintenance,
            )

            report = run_full_maintenance(db)
            total = (
                report.templates_recalculated
                + report.members_deduplicated
                + report.orphaned_members_removed
            )
            return PhaseResult(
                name="maintenance",
                items_processed=total,
                duration_seconds=time.time() - start,
            )
        except Exception as e:
            logger.exception("Maintenance failed")
            return PhaseResult(
                name="maintenance",
                duration_seconds=time.time() - start,
                error=str(e),
            )
