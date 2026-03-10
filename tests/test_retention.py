"""Tests for the retention module."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from alibi.retention import (
    CleanupCandidate,
    CleanupResult,
    RetentionPolicy,
    cleanup_old_data,
    find_cleanup_candidates,
    get_retention_stats,
)


class TestRetentionPolicy:
    """Tests for RetentionPolicy configuration."""

    def test_default_policy(self):
        """Test default retention policy values."""
        policy = RetentionPolicy()
        assert policy.max_duplicate_age_days == 90
        assert policy.max_error_documents_days == 30
        assert policy.is_configured() is True

    def test_unconfigured_policy(self):
        """Test that policy with all None values is unconfigured."""
        policy = RetentionPolicy(
            max_duplicate_age_days=None,
            max_error_documents_days=None,
        )
        assert policy.is_configured() is False

    def test_partially_configured_policy(self):
        """Test that policy with any value set is configured."""
        policy = RetentionPolicy(
            max_duplicate_age_days=60,
            max_error_documents_days=None,
        )
        assert policy.is_configured() is True


class TestFindCleanupCandidates:
    """Tests for find_cleanup_candidates function."""

    def test_find_cleanup_candidates_by_age(self):
        """Test finding old duplicates and errors."""
        db = MagicMock()

        now = datetime.now()
        old_duplicate_date = (now - timedelta(days=100)).isoformat()
        old_error_date = (now - timedelta(days=40)).isoformat()

        def fetchall_side_effect(sql, params=None):
            if "status = 'duplicate'" in sql:
                return [
                    ("dup-1", old_duplicate_date),
                    ("dup-2", old_duplicate_date),
                ]
            elif "status = 'error'" in sql:
                return [
                    ("err-1", old_error_date),
                ]
            return []

        db.fetchall.side_effect = fetchall_side_effect

        policy = RetentionPolicy(
            max_duplicate_age_days=90,
            max_error_documents_days=30,
        )

        candidates = find_cleanup_candidates(db, policy)

        assert len(candidates) == 3

        duplicate_ids = [c.id for c in candidates if c.reason.startswith("duplicate")]
        error_ids = [c.id for c in candidates if c.reason.startswith("error")]

        assert len(duplicate_ids) == 2
        assert "dup-1" in duplicate_ids
        assert "dup-2" in duplicate_ids
        assert len(error_ids) == 1
        assert "err-1" in error_ids

        assert all(c.table == "documents" for c in candidates)

        for c in candidates:
            if c.id.startswith("dup"):
                assert "90 days" in c.reason
            elif c.id.startswith("err"):
                assert "30 days" in c.reason

    def test_find_cleanup_candidates_empty(self):
        """Test no candidates when no old records exist."""
        db = MagicMock()
        db.fetchall.return_value = []

        policy = RetentionPolicy(
            max_duplicate_age_days=90,
            max_error_documents_days=30,
        )

        candidates = find_cleanup_candidates(db, policy)
        assert len(candidates) == 0

    def test_find_cleanup_candidates_unconfigured_policy(self):
        """Test that unconfigured policy returns no candidates."""
        db = MagicMock()

        policy = RetentionPolicy(
            max_duplicate_age_days=None,
            max_error_documents_days=None,
        )

        candidates = find_cleanup_candidates(db, policy)
        assert len(candidates) == 0
        db.fetchall.assert_not_called()

    def test_find_cleanup_candidates_only_duplicates(self):
        """Test finding only old duplicates when error policy is None."""
        db = MagicMock()

        now = datetime.now()
        old_date = (now - timedelta(days=100)).isoformat()

        db.fetchall.return_value = [
            ("dup-1", old_date),
        ]

        policy = RetentionPolicy(
            max_duplicate_age_days=90,
            max_error_documents_days=None,
        )

        candidates = find_cleanup_candidates(db, policy)

        assert len(candidates) == 1
        assert candidates[0].id == "dup-1"
        assert "duplicate" in candidates[0].reason

    def test_find_cleanup_candidates_only_errors(self):
        """Test finding only old errors when duplicate policy is None."""
        db = MagicMock()

        now = datetime.now()
        old_date = (now - timedelta(days=40)).isoformat()

        db.fetchall.return_value = [
            ("err-1", old_date),
        ]

        policy = RetentionPolicy(
            max_duplicate_age_days=None,
            max_error_documents_days=30,
        )

        candidates = find_cleanup_candidates(db, policy)

        assert len(candidates) == 1
        assert candidates[0].id == "err-1"
        assert "error" in candidates[0].reason

    def test_find_cleanup_candidates_sorted_by_age(self):
        """Test that candidates are sorted by creation date (oldest first)."""
        db = MagicMock()

        now = datetime.now()
        very_old = (now - timedelta(days=120)).isoformat()
        old = (now - timedelta(days=100)).isoformat()
        recent = (now - timedelta(days=95)).isoformat()

        def fetchall_side_effect(sql, params=None):
            if "status = 'duplicate'" in sql:
                return [
                    ("dup-recent", recent),
                    ("dup-very-old", very_old),
                    ("dup-old", old),
                ]
            return []

        db.fetchall.side_effect = fetchall_side_effect

        policy = RetentionPolicy(max_duplicate_age_days=90)
        candidates = find_cleanup_candidates(db, policy)

        assert len(candidates) == 3
        assert candidates[0].id == "dup-very-old"
        assert candidates[1].id == "dup-old"
        assert candidates[2].id == "dup-recent"


class TestCleanupOldData:
    """Tests for cleanup_old_data function."""

    def test_cleanup_dry_run(self):
        """Test dry run returns candidates but doesn't delete."""
        db = MagicMock()

        now = datetime.now()
        old_date = (now - timedelta(days=100)).isoformat()

        def fetchall_side_effect(sql, params=None):
            if "status = 'duplicate'" in sql:
                return [
                    ("dup-1", old_date),
                    ("dup-2", old_date),
                ]
            return []

        db.fetchall.side_effect = fetchall_side_effect

        policy = RetentionPolicy(
            max_duplicate_age_days=90,
            max_error_documents_days=None,
        )

        result = cleanup_old_data(db, policy, dry_run=True)

        assert len(result.candidates) == 2
        assert result.total_candidates == 2
        assert result.deleted_count == 0
        assert result.dry_run is True
        db.execute.assert_not_called()

    def test_cleanup_execute(self):
        """Test execute mode actually deletes records."""
        db = MagicMock()

        now = datetime.now()
        old_date = (now - timedelta(days=100)).isoformat()

        def fetchall_side_effect(sql, params=None):
            if "status = 'duplicate'" in sql:
                return [
                    ("dup-1", old_date),
                    ("dup-2", old_date),
                ]
            return []

        db.fetchall.side_effect = fetchall_side_effect

        policy = RetentionPolicy(
            max_duplicate_age_days=90,
            max_error_documents_days=None,
        )

        result = cleanup_old_data(db, policy, dry_run=False)

        assert result.deleted_count == 2
        assert result.dry_run is False
        assert len(result.errors) == 0

        # Each document: 2 calls (cascade + document delete)
        assert db.execute.call_count == 4

        # Verify document deletes were called
        calls = db.execute.call_args_list
        assert calls[1][0][0].startswith("DELETE FROM documents")
        assert calls[1][0][1] == ("dup-1",)
        assert calls[3][0][0].startswith("DELETE FROM documents")
        assert calls[3][0][1] == ("dup-2",)

    def test_cleanup_no_candidates(self):
        """Test cleanup with no candidates."""
        db = MagicMock()
        db.fetchall.return_value = []

        policy = RetentionPolicy(max_duplicate_age_days=90)

        result = cleanup_old_data(db, policy, dry_run=False)

        assert len(result.candidates) == 0
        assert result.deleted_count == 0
        assert len(result.errors) == 0

    def test_cleanup_with_errors(self):
        """Test cleanup handles deletion errors gracefully."""
        db = MagicMock()

        now = datetime.now()
        old_date = (now - timedelta(days=100)).isoformat()

        def fetchall_side_effect(sql, params=None):
            if "status = 'duplicate'" in sql:
                return [
                    ("dup-1", old_date),
                    ("dup-2", old_date),
                ]
            return []

        db.fetchall.side_effect = fetchall_side_effect

        # Make the first deletion fail
        call_count = [0]

        def execute_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:  # First call fails
                raise Exception("Database error")

        db.execute.side_effect = execute_side_effect

        policy = RetentionPolicy(
            max_duplicate_age_days=90,
            max_error_documents_days=None,
        )

        result = cleanup_old_data(db, policy, dry_run=False)

        assert len(result.errors) == 1
        assert result.deleted_count == 1
        assert "Database error" in result.errors[0]

    def test_cleanup_result_total_candidates_property(self):
        """Test CleanupResult.total_candidates property."""
        result = CleanupResult()
        assert result.total_candidates == 0

        now = datetime.now()
        result.candidates = [
            CleanupCandidate("documents", "1", "test", now),
            CleanupCandidate("documents", "2", "test", now),
        ]
        assert result.total_candidates == 2


class TestRetentionStats:
    """Tests for get_retention_stats function."""

    def test_retention_stats(self):
        """Test getting statistics about data age."""
        db = MagicMock()

        now = datetime.now()

        status_rows = [
            ("stored", 100),
            ("duplicate", 50),
            ("error", 10),
        ]

        doc_rows = [
            ((now - timedelta(hours=12)).isoformat(),),
            ((now - timedelta(hours=23)).isoformat(),),
            ((now - timedelta(days=2)).isoformat(),),
            ((now - timedelta(days=5)).isoformat(),),
            ((now - timedelta(days=6, hours=23)).isoformat(),),
            ((now - timedelta(days=10)).isoformat(),),
            ((now - timedelta(days=29)).isoformat(),),
            ((now - timedelta(days=45)).isoformat(),),
            ((now - timedelta(days=89)).isoformat(),),
            ((now - timedelta(days=100)).isoformat(),),
            ((now - timedelta(days=365)).isoformat(),),
        ]

        oldest_date = (now - timedelta(days=365)).isoformat()
        newest_date = (now - timedelta(hours=1)).isoformat()

        def fetchall_side_effect(sql, params=None):
            if "GROUP BY status" in sql:
                return status_rows
            elif "created_at FROM documents" in sql:
                return doc_rows
            return []

        def fetchone_side_effect(sql, params=None):
            if "ORDER BY created_at ASC" in sql:
                return ("oldest-id", oldest_date)
            elif "ORDER BY created_at DESC" in sql:
                return ("newest-id", newest_date)
            elif "COUNT(*) FROM facts" in sql:
                return (250,)
            return None

        db.fetchall.side_effect = fetchall_side_effect
        db.fetchone.side_effect = fetchone_side_effect

        stats = get_retention_stats(db)

        assert stats["total_documents"] == 160  # 100 + 50 + 10
        assert stats["total_facts"] == 250

        assert stats["documents_by_status"]["stored"] == 100
        assert stats["documents_by_status"]["duplicate"] == 50
        assert stats["documents_by_status"]["error"] == 10

        age_dist = stats["age_distribution"]
        assert age_dist["< 1 day"] == 2
        assert age_dist["1-7 days"] == 3
        assert age_dist["7-30 days"] == 2
        assert age_dist["30-90 days"] == 2
        assert age_dist["> 90 days"] == 2

        assert stats["oldest_document"]["id"] == "oldest-id"
        assert stats["oldest_document"]["age_days"] == 365

        assert stats["newest_document"]["id"] == "newest-id"
        assert stats["newest_document"]["age_days"] == 0

    def test_retention_stats_empty_database(self):
        """Test stats with empty database."""
        db = MagicMock()

        db.fetchall.return_value = []
        db.fetchone.side_effect = [None, None, (0,)]

        stats = get_retention_stats(db)

        assert stats["total_documents"] == 0
        assert stats["total_facts"] == 0
        assert stats["documents_by_status"] == {}
        assert stats["oldest_document"] is None
        assert stats["newest_document"] is None

        assert stats["age_distribution"]["< 1 day"] == 0
        assert stats["age_distribution"]["1-7 days"] == 0
        assert stats["age_distribution"]["7-30 days"] == 0
        assert stats["age_distribution"]["30-90 days"] == 0
        assert stats["age_distribution"]["> 90 days"] == 0

    def test_retention_stats_no_timestamps(self):
        """Test stats when documents have no timestamps."""
        db = MagicMock()

        status_rows = [("stored", 5)]

        def fetchall_side_effect(sql, params=None):
            if "GROUP BY status" in sql:
                return status_rows
            elif "created_at FROM documents" in sql:
                return [(None,), (None,)]
            return []

        def fetchone_side_effect(sql, params=None):
            if "ORDER BY created_at" in sql:
                return ("doc-id", None)
            elif "COUNT(*) FROM facts" in sql:
                return (10,)
            return None

        db.fetchall.side_effect = fetchall_side_effect
        db.fetchone.side_effect = fetchone_side_effect

        stats = get_retention_stats(db)

        assert stats["total_documents"] == 5
        assert stats["oldest_document"] is None
        assert stats["newest_document"] is None

        age_dist = stats["age_distribution"]
        assert all(count == 0 for count in age_dist.values())


class TestCleanupCandidate:
    """Tests for CleanupCandidate dataclass."""

    def test_cleanup_candidate_creation(self):
        """Test creating a CleanupCandidate."""
        now = datetime.now()
        candidate = CleanupCandidate(
            table="documents",
            id="test-123",
            reason="too old",
            created_at=now,
        )

        assert candidate.table == "documents"
        assert candidate.id == "test-123"
        assert candidate.reason == "too old"
        assert candidate.created_at == now


class TestCleanupResult:
    """Tests for CleanupResult dataclass."""

    def test_cleanup_result_defaults(self):
        """Test CleanupResult default values."""
        result = CleanupResult()

        assert result.candidates == []
        assert result.deleted_count == 0
        assert result.dry_run is True
        assert result.errors == []
        assert result.total_candidates == 0

    def test_cleanup_result_with_data(self):
        """Test CleanupResult with data."""
        now = datetime.now()
        candidates = [
            CleanupCandidate("documents", "1", "test", now),
            CleanupCandidate("documents", "2", "test", now),
        ]

        result = CleanupResult(
            candidates=candidates,
            deleted_count=2,
            dry_run=False,
            errors=["error 1"],
        )

        assert len(result.candidates) == 2
        assert result.deleted_count == 2
        assert result.dry_run is False
        assert len(result.errors) == 1
        assert result.total_candidates == 2
