"""Service layer for duplicate-fact detection and resolution.

Thin wrapper over :mod:`alibi.clouds.dedup` so interfaces (CLI, post-batch hook,
future API/MCP) route through the service boundary rather than importing the
clouds package directly.
"""

from __future__ import annotations

import logging

from alibi.clouds.dedup import DedupReport, dedup_pass
from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)


def deduplicate_facts(db: DatabaseManager, *, apply: bool = False) -> DedupReport:
    """Find duplicate facts and, when ``apply`` is set, resolve the safe ones.

    Safe duplicates (corroborated by item price overlap, perceptual-hash match,
    or a zero-item twin) are resolved by deleting the redundant twin; ambiguous
    pairs are returned under :attr:`DedupReport.review` for a human to judge.
    """
    return dedup_pass(db, apply=apply)


def deduplicate_after_batch(db: DatabaseManager) -> DedupReport:
    """Post-batch dedup step: resolve safe duplicates and log the outcome.

    Intended to run at the end of a batch ingest (gated by
    ``config.dedup_after_batch``). Applies safe merges and logs both what was
    resolved and what needs review, so a newly-ingested duplicate of an
    already-stored transaction is collapsed to one fact instead of lingering.
    """
    report = dedup_pass(db, apply=True)
    if report.resolved_count or report.review_count:
        logger.info(
            "Post-batch dedup: resolved %d duplicate fact(s), %d flagged for review",
            report.resolved_count,
            report.review_count,
        )
        for action in report.review:
            logger.warning(
                "Dedup review: %s ~ %s (%s) — %s/%s/%s",
                action.keeper.fact_id[:8],
                action.redundant.fact_id[:8],
                action.reason,
                action.keeper.vendor,
                action.keeper.event_date,
                action.keeper.total_amount,
            )
    return report
