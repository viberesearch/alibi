"""Data retention policies for Alibi.

Provides cleanup functionality to manage storage by removing old records
based on configurable retention policies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class RetentionPolicy:
    """Configuration for data retention."""

    max_duplicate_age_days: int | None = 90  # Age for duplicate document logs
    max_error_documents_days: int | None = 30  # Age for error documents

    def is_configured(self) -> bool:
        """Check if any retention policy is configured."""
        return (
            self.max_duplicate_age_days is not None
            or self.max_error_documents_days is not None
        )


@dataclass
class CleanupCandidate:
    """A record that may be deleted based on retention policy."""

    table: str
    id: str
    reason: str
    created_at: datetime


@dataclass
class CleanupResult:
    """Result of a cleanup operation."""

    candidates: list[CleanupCandidate] = field(default_factory=list)
    deleted_count: int = 0
    dry_run: bool = True
    errors: list[str] = field(default_factory=list)

    @property
    def total_candidates(self) -> int:
        """Total number of cleanup candidates."""
        return len(self.candidates)


def _parse_datetime(dt_value: Any) -> datetime | None:
    """Parse datetime from various formats."""
    if dt_value is None:
        return None
    if isinstance(dt_value, datetime):
        return dt_value
    if isinstance(dt_value, str):
        try:
            return datetime.fromisoformat(dt_value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def find_cleanup_candidates(
    db: "DatabaseManager",
    policy: RetentionPolicy,
) -> list[CleanupCandidate]:
    """Find records eligible for cleanup based on retention policy.

    Args:
        db: DatabaseManager instance
        policy: RetentionPolicy with configured limits

    Returns:
        List of CleanupCandidate objects sorted by created_at (oldest first)
    """
    if not policy.is_configured():
        return []

    candidates = []
    now = datetime.now()

    # Find old duplicate documents
    if policy.max_duplicate_age_days is not None:
        cutoff = now - timedelta(days=policy.max_duplicate_age_days)
        cutoff_str = cutoff.isoformat()

        # Find documents marked as duplicates that are older than cutoff
        rows = db.fetchall(
            """
            SELECT id, created_at FROM documents
            WHERE status = 'duplicate' AND created_at < ?
            ORDER BY created_at ASC
            """,
            (cutoff_str,),
        )

        for row in rows:
            created_at = _parse_datetime(row[1])
            if created_at:
                candidates.append(
                    CleanupCandidate(
                        table="documents",
                        id=row[0],
                        reason=f"duplicate older than {policy.max_duplicate_age_days} days",
                        created_at=created_at,
                    )
                )

    # Find old error documents
    if policy.max_error_documents_days is not None:
        cutoff = now - timedelta(days=policy.max_error_documents_days)
        cutoff_str = cutoff.isoformat()

        # Find documents with error status that are older than cutoff
        rows = db.fetchall(
            """
            SELECT id, created_at FROM documents
            WHERE status = 'error' AND created_at < ?
            ORDER BY created_at ASC
            """,
            (cutoff_str,),
        )

        for row in rows:
            created_at = _parse_datetime(row[1])
            if created_at:
                candidates.append(
                    CleanupCandidate(
                        table="documents",
                        id=row[0],
                        reason=f"error older than {policy.max_error_documents_days} days",
                        created_at=created_at,
                    )
                )

    # Sort by created_at (oldest first)
    candidates.sort(key=lambda x: x.created_at)

    return candidates


def cleanup_old_data(
    db: "DatabaseManager",
    policy: RetentionPolicy,
    dry_run: bool = True,
) -> CleanupResult:
    """Clean up old data based on retention policy.

    Args:
        db: DatabaseManager instance
        policy: RetentionPolicy with configured limits
        dry_run: If True, only report what would be deleted

    Returns:
        CleanupResult with details of the operation
    """
    result = CleanupResult(dry_run=dry_run)

    # Find candidates
    result.candidates = find_cleanup_candidates(db, policy)

    if not result.candidates:
        logger.info("No records to clean up")
        return result

    if dry_run:
        logger.info(f"Dry run: would delete {len(result.candidates)} records")
        return result

    # Perform deletion
    for candidate in result.candidates:
        try:
            if candidate.table == "documents":
                # Delete document and related records via v2 cascade
                db.execute(
                    "DELETE FROM bundle_atoms WHERE bundle_id IN "
                    "(SELECT id FROM bundles WHERE cloud_id IN "
                    "(SELECT cloud_id FROM facts WHERE id IN "
                    "(SELECT fact_id FROM fact_items WHERE fact_id IN "
                    "(SELECT id FROM facts))))",
                    (),
                )
                db.execute("DELETE FROM documents WHERE id = ?", (candidate.id,))
                result.deleted_count += 1
                logger.info(f"Deleted document {candidate.id}")
        except Exception as e:
            error_msg = f"Failed to delete {candidate.table}/{candidate.id}: {e}"
            result.errors.append(error_msg)
            logger.error(error_msg)

    return result


def get_retention_stats(db: "DatabaseManager") -> dict[str, Any]:
    """Get statistics about current data for retention planning.

    Args:
        db: DatabaseManager instance

    Returns:
        Dictionary with data statistics
    """
    # Count documents by status
    status_rows = db.fetchall(
        """
        SELECT status, COUNT(*) as count
        FROM documents
        GROUP BY status
        """
    )
    status_counts = {row[0]: row[1] for row in status_rows}

    # Get age distribution for documents
    age_buckets: dict[str, int] = {
        "< 1 day": 0,
        "1-7 days": 0,
        "7-30 days": 0,
        "30-90 days": 0,
        "> 90 days": 0,
    }

    doc_rows = db.fetchall(
        "SELECT created_at FROM documents WHERE created_at IS NOT NULL"
    )
    now = datetime.now()

    for row in doc_rows:
        created_at = _parse_datetime(row[0])
        if created_at:
            age = now - created_at
            if age < timedelta(days=1):
                age_buckets["< 1 day"] += 1
            elif age < timedelta(days=7):
                age_buckets["1-7 days"] += 1
            elif age < timedelta(days=30):
                age_buckets["7-30 days"] += 1
            elif age < timedelta(days=90):
                age_buckets["30-90 days"] += 1
            else:
                age_buckets["> 90 days"] += 1

    # Get oldest and newest documents
    oldest_row = db.fetchone(
        "SELECT id, created_at FROM documents ORDER BY created_at ASC LIMIT 1"
    )
    newest_row = db.fetchone(
        "SELECT id, created_at FROM documents ORDER BY created_at DESC LIMIT 1"
    )

    oldest = None
    if oldest_row and oldest_row[1]:
        oldest_dt = _parse_datetime(oldest_row[1])
        if oldest_dt:
            oldest = {
                "id": oldest_row[0],
                "created_at": oldest_dt.isoformat(),
                "age_days": (now - oldest_dt).days,
            }

    newest = None
    if newest_row and newest_row[1]:
        newest_dt = _parse_datetime(newest_row[1])
        if newest_dt:
            newest = {
                "id": newest_row[0],
                "created_at": newest_dt.isoformat(),
                "age_days": (now - newest_dt).days,
            }

    # Get total counts
    total_documents = sum(status_counts.values())
    total_facts_row = db.fetchone("SELECT COUNT(*) FROM facts")
    total_facts = total_facts_row[0] if total_facts_row else 0

    return {
        "total_documents": total_documents,
        "total_facts": total_facts,
        "documents_by_status": status_counts,
        "oldest_document": oldest,
        "newest_document": newest,
        "age_distribution": age_buckets,
    }
