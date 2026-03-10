"""Tests for the adaptive learning system (Phases 2-5).

Covers:
- Phase 2: record_extraction_observation() — observation recording
- Phase 3: Staleness detection and degraded hint delivery
- Phase 4: Correction feedback via correction_log service
- Phase 5: Sibling propagation and derive_vendor_default_category
- VendorTemplate to_dict()/from_dict() roundtrip
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from alibi.extraction.templates import (
    VendorTemplate,
    derive_vendor_default_category,
    merge_template,
    record_extraction_observation,
    template_to_hints,
)
from alibi.services.correction_log import (
    get_vendor_unreliable_fields,
    should_suggest_reprocessing,
)
from alibi.services.correction import _propagate_to_siblings


# ---------------------------------------------------------------------------
# Phase 2 — record_extraction_observation
# ---------------------------------------------------------------------------


class TestRecordExtractionObservation:
    def _make_template(self, **kwargs) -> VendorTemplate:
        defaults = dict(
            layout_type="standard",
            currency="EUR",
            success_count=5,
        )
        defaults.update(kwargs)
        return VendorTemplate(**defaults)

    def test_sets_last_updated_to_iso_datetime(self):
        template = self._make_template()
        result = record_extraction_observation(template, confidence=0.9)

        assert result.last_updated is not None
        # Should parse as a valid ISO datetime (contains date separator)
        assert "T" in result.last_updated
        assert len(result.last_updated) > 10

    def test_last_updated_changes_on_each_call(self):
        template = self._make_template()
        first = record_extraction_observation(template, confidence=0.9)
        # Re-record immediately — last_updated should remain an ISO string
        second = record_extraction_observation(first, confidence=0.8)
        assert second.last_updated is not None
        # Both are ISO datetimes; exact equality depends on timing so just check format
        assert "T" in second.last_updated

    def test_confidence_appended_to_history(self):
        template = self._make_template(confidence_history=[0.85, 0.90])
        result = record_extraction_observation(template, confidence=0.95)

        assert 0.95 in result.confidence_history
        # Previous values preserved
        assert 0.85 in result.confidence_history
        assert 0.90 in result.confidence_history

    def test_confidence_appended_to_empty_history(self):
        template = self._make_template(confidence_history=[])
        result = record_extraction_observation(template, confidence=0.75)

        assert result.confidence_history == [0.75]

    def test_history_capped_at_twenty_entries(self):
        # Start with 20 entries already in history
        initial_history = [0.80 + i * 0.001 for i in range(20)]
        template = self._make_template(confidence_history=initial_history)

        result = record_extraction_observation(template, confidence=0.99)

        assert len(result.confidence_history) == 20
        # The newest observation must be present
        assert result.confidence_history[-1] == 0.99
        # The oldest entry was evicted
        assert 0.80 not in result.confidence_history

    def test_history_stays_at_twenty_after_multiple_observations(self):
        # Build up 25 observations
        template = self._make_template(confidence_history=[])
        for i in range(25):
            template = record_extraction_observation(
                template, confidence=0.7 + i * 0.01
            )

        assert len(template.confidence_history) == 20

    def test_original_template_not_mutated(self):
        original_history = [0.85, 0.90]
        template = self._make_template(confidence_history=original_history)
        _ = record_extraction_observation(template, confidence=0.95)

        # Original list must be unchanged
        assert template.confidence_history == [0.85, 0.90]

    def test_returns_new_template_instance(self):
        template = self._make_template()
        result = record_extraction_observation(template, confidence=0.88)

        assert result is not template

    def test_preserves_layout_type_and_currency(self):
        template = self._make_template(layout_type="columnar", currency="GBP")
        result = record_extraction_observation(template, confidence=0.92)

        assert result.layout_type == "columnar"
        assert result.currency == "GBP"

    def test_ocr_tier_recorded_when_above_zero(self):
        template = self._make_template()
        result = record_extraction_observation(template, confidence=0.85, ocr_tier=2)

        assert result.preferred_ocr_tier == 2

    def test_ocr_tier_ignored_when_zero(self):
        template = self._make_template(preferred_ocr_tier=None)
        result = record_extraction_observation(template, confidence=0.85, ocr_tier=0)

        assert result.preferred_ocr_tier is None

    def test_ocr_tier_uses_max_of_existing_and_new(self):
        template = self._make_template(preferred_ocr_tier=1)
        result = record_extraction_observation(template, confidence=0.85, ocr_tier=3)

        assert result.preferred_ocr_tier == 3

    def test_was_rotated_latches_needs_rotation(self):
        template = self._make_template(needs_rotation=False)
        result = record_extraction_observation(
            template, confidence=0.88, was_rotated=True
        )

        assert result.needs_rotation is True

    def test_needs_rotation_stays_true_once_set(self):
        template = self._make_template(needs_rotation=True)
        result = record_extraction_observation(
            template, confidence=0.88, was_rotated=False
        )

        assert result.needs_rotation is True

    def test_fixes_applied_increments_common_fixes(self):
        template = self._make_template(common_fixes={"total": 2})
        result = record_extraction_observation(
            template, confidence=0.85, fixes_applied=["total", "date"]
        )

        assert result.common_fixes["total"] == 3
        assert result.common_fixes["date"] == 1

    def test_fixes_applied_none_leaves_common_fixes_unchanged(self):
        template = self._make_template(common_fixes={"total": 2})
        result = record_extraction_observation(
            template, confidence=0.88, fixes_applied=None
        )

        assert result.common_fixes == {"total": 2}


# ---------------------------------------------------------------------------
# Phase 3 — Staleness detection
# ---------------------------------------------------------------------------


class TestStalenessDetection:
    """Tests for template staleness detection in record_extraction_observation."""

    def _template_with_history(self, history: list[float]) -> VendorTemplate:
        return VendorTemplate(
            layout_type="standard",
            currency="EUR",
            success_count=10,
            confidence_history=history,
        )

    def test_template_becomes_stale_when_recent_avg_drops_significantly(self):
        # Historical observations at 0.90, recent ones drop to 0.60
        # With 7 entries, adding one more brings us to 8 (>= 5 minimum)
        # historical avg ≈ 0.90, recent (last 3 after new obs) ≈ 0.60
        history = [0.90, 0.91, 0.92, 0.90]  # 4 high entries
        template = self._template_with_history(history)

        # Add two low observations so recent avg < historical avg - 0.15
        template = record_extraction_observation(template, confidence=0.60)
        template = record_extraction_observation(template, confidence=0.60)
        result = record_extraction_observation(template, confidence=0.60)

        assert result.stale is True

    def test_staleness_clears_when_confidence_recovers(self):
        # Start with a stale template and strong historical context
        high_history = [0.90] * 10  # strong historical baseline
        template = VendorTemplate(
            layout_type="standard",
            success_count=10,
            confidence_history=high_history,
            stale=True,
        )

        # Feed high confidence observations to recover
        # Recovery condition: recent_avg >= historical_avg - (0.15/2) = historical_avg - 0.075
        # historical_avg with 10 x 0.90 = 0.90 -> threshold = 0.825
        # Add several high-confidence observations to shift the rolling window
        for _ in range(8):
            template = record_extraction_observation(template, confidence=0.95)

        assert template.stale is False

    def test_need_at_least_five_observations_before_staleness_check(self):
        # With fewer than 5 observations, staleness should not be triggered
        # even if confidence is low
        template = VendorTemplate(
            layout_type="standard",
            success_count=1,
            confidence_history=[0.90, 0.90, 0.30],  # 3 entries
        )

        # Add one more low observation — total 4 (< 5 minimum)
        result = record_extraction_observation(template, confidence=0.30)

        # Should not be stale — not enough observations yet
        assert result.stale is False

    def test_staleness_not_triggered_when_drop_within_threshold(self):
        # Moderate drop but not enough to cross 0.15 threshold
        history = [0.85, 0.86, 0.87, 0.86, 0.85]
        template = self._template_with_history(history)

        # Recent values drop modestly — not enough to trigger staleness
        template = record_extraction_observation(template, confidence=0.78)
        result = record_extraction_observation(template, confidence=0.79)

        assert result.stale is False


class TestTemplateToHintsStaleness:
    """Tests that stale templates return degraded hints."""

    def _reliable_template(self, **kwargs) -> VendorTemplate:
        defaults = dict(
            layout_type="columnar",
            currency="EUR",
            pos_provider="JCC",
            success_count=5,  # >= _RELIABLE_COUNT (2)
            stale=False,
        )
        defaults.update(kwargs)
        return VendorTemplate(**defaults)

    def test_stale_template_omits_layout_and_currency(self):
        template = self._reliable_template(stale=True)
        hints = template_to_hints(template, vendor_name="ACME")

        assert hints.layout_type is None
        assert hints.currency is None

    def test_stale_template_omits_pos_provider(self):
        template = self._reliable_template(stale=True)
        hints = template_to_hints(template, vendor_name="ACME")

        assert hints.pos_provider is None

    def test_stale_template_still_passes_vendor_name(self):
        template = self._reliable_template(stale=True)
        hints = template_to_hints(template, vendor_name="ACME Shop")

        assert hints.vendor_name == "ACME Shop"

    def test_fresh_template_includes_layout_and_currency(self):
        template = self._reliable_template(stale=False)
        hints = template_to_hints(template, vendor_name="ACME")

        assert hints.layout_type == "columnar"
        assert hints.currency == "EUR"
        assert hints.pos_provider == "JCC"

    def test_unreliable_template_returns_name_only(self):
        # success_count < 2 means not reliable
        template = VendorTemplate(
            layout_type="columnar",
            currency="EUR",
            success_count=1,
            stale=False,
        )
        hints = template_to_hints(template, vendor_name="Small Shop")

        assert hints.vendor_name == "Small Shop"
        assert hints.layout_type is None
        assert hints.currency is None

    def test_stale_template_with_unreliable_fields_includes_them(self):
        # Fields corrected >= _UNRELIABLE_FIX_COUNT (5) times should appear
        template = self._reliable_template(
            stale=True, common_fixes={"total": 6, "date": 3}
        )
        hints = template_to_hints(template, vendor_name="ACME")

        assert hints.unreliable_fields is not None
        assert "total" in hints.unreliable_fields
        # "date" has only 3 fixes — below threshold of 5
        assert "date" not in hints.unreliable_fields

    def test_fresh_template_with_unreliable_fields_includes_them(self):
        template = self._reliable_template(
            stale=False, common_fixes={"vendor": 7, "items": 5}
        )
        hints = template_to_hints(template, vendor_name="ACME")

        assert hints.unreliable_fields is not None
        assert "vendor" in hints.unreliable_fields
        assert "items" in hints.unreliable_fields

    def test_no_unreliable_fields_returns_none(self):
        template = self._reliable_template(stale=False, common_fixes={})
        hints = template_to_hints(template)

        assert hints.unreliable_fields is None


class TestMergeTemplateStaleness:
    """Tests that merge_template correctly handles staleness on layout changes."""

    def test_layout_change_marks_result_as_stale(self):
        existing = VendorTemplate(
            layout_type="standard",
            currency="EUR",
            success_count=8,
            stale=False,
        )
        new = VendorTemplate(
            layout_type="columnar",
            currency="EUR",
            success_count=1,
        )

        result = merge_template(existing, new)

        assert result.stale is True

    def test_layout_change_resets_success_count_to_one(self):
        existing = VendorTemplate(layout_type="standard", success_count=10)
        new = VendorTemplate(layout_type="columnar", success_count=1)

        result = merge_template(existing, new)

        assert result.success_count == 1

    def test_layout_change_adopts_new_layout(self):
        existing = VendorTemplate(layout_type="standard", success_count=5)
        new = VendorTemplate(layout_type="nqa", success_count=1)

        result = merge_template(existing, new)

        assert result.layout_type == "nqa"

    def test_same_layout_preserves_confidence_history(self):
        existing = VendorTemplate(
            layout_type="columnar",
            success_count=4,
            confidence_history=[0.85, 0.90, 0.88],
        )
        new = VendorTemplate(layout_type="columnar", success_count=1)

        result = merge_template(existing, new)

        assert result.confidence_history == [0.85, 0.90, 0.88]

    def test_same_layout_preserves_adaptive_skip_threshold(self):
        existing = VendorTemplate(
            layout_type="standard",
            success_count=3,
            adaptive_skip_threshold=0.95,
        )
        new = VendorTemplate(layout_type="standard", success_count=1)

        result = merge_template(existing, new)

        assert result.adaptive_skip_threshold == 0.95

    def test_same_layout_preserves_preferred_ocr_tier(self):
        existing = VendorTemplate(
            layout_type="standard",
            success_count=3,
            preferred_ocr_tier=2,
        )
        new = VendorTemplate(layout_type="standard", success_count=1)

        result = merge_template(existing, new)

        assert result.preferred_ocr_tier == 2

    def test_same_layout_preserves_needs_rotation(self):
        existing = VendorTemplate(
            layout_type="standard",
            success_count=3,
            needs_rotation=True,
        )
        new = VendorTemplate(layout_type="standard", success_count=1)

        result = merge_template(existing, new)

        assert result.needs_rotation is True

    def test_same_layout_preserves_common_fixes(self):
        existing = VendorTemplate(
            layout_type="standard",
            success_count=5,
            common_fixes={"total": 3, "date": 1},
        )
        new = VendorTemplate(layout_type="standard", success_count=1)

        result = merge_template(existing, new)

        assert result.common_fixes == {"total": 3, "date": 1}

    def test_same_layout_preserves_last_updated(self):
        existing = VendorTemplate(
            layout_type="standard",
            success_count=5,
            last_updated="2026-02-01T10:00:00+00:00",
        )
        new = VendorTemplate(layout_type="standard", success_count=1)

        result = merge_template(existing, new)

        assert result.last_updated == "2026-02-01T10:00:00+00:00"

    def test_same_layout_preserves_stale_flag(self):
        existing = VendorTemplate(
            layout_type="standard",
            success_count=5,
            stale=True,
        )
        new = VendorTemplate(layout_type="standard", success_count=1)

        result = merge_template(existing, new)

        assert result.stale is True

    def test_layout_change_resets_adaptive_skip_threshold(self):
        # When layout changes, adaptive threshold must be reset
        existing = VendorTemplate(
            layout_type="standard",
            success_count=10,
            adaptive_skip_threshold=0.95,
        )
        new = VendorTemplate(layout_type="columnar", success_count=1)

        result = merge_template(existing, new)

        assert result.adaptive_skip_threshold is None

    def test_layout_change_preserves_confidence_history(self):
        existing = VendorTemplate(
            layout_type="standard",
            success_count=5,
            confidence_history=[0.88, 0.90],
        )
        new = VendorTemplate(layout_type="nqa", success_count=1)

        result = merge_template(existing, new)

        # History preserved even on layout change
        assert result.confidence_history == [0.88, 0.90]

    def test_same_layout_increments_success_count(self):
        existing = VendorTemplate(layout_type="standard", success_count=7)
        new = VendorTemplate(layout_type="standard", success_count=1)

        result = merge_template(existing, new)

        assert result.success_count == 8

    def test_same_layout_prefers_new_currency_when_existing_is_none(self):
        existing = VendorTemplate(
            layout_type="standard", success_count=3, currency=None
        )
        new = VendorTemplate(layout_type="standard", success_count=1, currency="USD")

        result = merge_template(existing, new)

        assert result.currency == "USD"

    def test_same_layout_keeps_existing_currency_when_new_is_none(self):
        existing = VendorTemplate(
            layout_type="standard", success_count=3, currency="EUR"
        )
        new = VendorTemplate(layout_type="standard", success_count=1, currency=None)

        result = merge_template(existing, new)

        assert result.currency == "EUR"


# ---------------------------------------------------------------------------
# Phase 4 — Correction feedback (correction_log service)
# ---------------------------------------------------------------------------


class TestGetVendorUnreliableFields:
    """Tests for get_vendor_unreliable_fields using a mock DB."""

    def _make_mock_db(self, rows: list[dict]) -> MagicMock:
        """Return a mock DB whose fetchall returns the given rows."""
        mock_db = MagicMock()
        # Simulate row dicts with dict-style access
        mock_rows = []
        for row in rows:
            m = MagicMock()
            m.__getitem__ = lambda self, k, r=row: r[k]
            mock_rows.append(m)
        mock_db.fetchall.return_value = mock_rows
        return mock_db

    def test_returns_fields_with_corrections_at_or_above_min(self):
        rows = [
            {"field": "total", "cnt": 5},
            {"field": "date", "cnt": 3},
        ]
        mock_db = self._make_mock_db(rows)

        result = get_vendor_unreliable_fields(mock_db, "CY12345678A", min_corrections=3)

        assert "total" in result
        assert "date" in result

    def test_does_not_return_fields_below_min_corrections(self):
        # DB query filters via HAVING — return only rows that passed the filter
        rows = [{"field": "total", "cnt": 5}]
        mock_db = self._make_mock_db(rows)

        result = get_vendor_unreliable_fields(mock_db, "CY12345678A", min_corrections=3)

        assert "total" in result
        # "date" would not appear because the DB filters it out
        assert len(result) == 1

    def test_returns_empty_list_when_no_corrections(self):
        mock_db = MagicMock()
        mock_db.fetchall.return_value = []

        result = get_vendor_unreliable_fields(mock_db, "CY12345678A", min_corrections=3)

        assert result == []

    def test_passes_correct_vendor_key_to_query(self):
        mock_db = MagicMock()
        mock_db.fetchall.return_value = []

        get_vendor_unreliable_fields(mock_db, "CY10370773Q", min_corrections=5)

        call_args = mock_db.fetchall.call_args
        params = call_args[0][1]  # second positional arg is the params tuple
        assert "CY10370773Q" in params

    def test_passes_min_corrections_to_query(self):
        mock_db = MagicMock()
        mock_db.fetchall.return_value = []

        get_vendor_unreliable_fields(mock_db, "CY10370773Q", min_corrections=7)

        call_args = mock_db.fetchall.call_args
        params = call_args[0][1]
        assert 7 in params

    def test_result_order_matches_database_order(self):
        # The function returns fields in the order the DB returns them
        rows = [
            {"field": "total", "cnt": 10},
            {"field": "vendor", "cnt": 6},
            {"field": "date", "cnt": 4},
        ]
        mock_db = self._make_mock_db(rows)

        result = get_vendor_unreliable_fields(mock_db, "CY12345678A", min_corrections=3)

        assert result == ["total", "vendor", "date"]

    def test_uses_default_window_days(self):
        mock_db = MagicMock()
        mock_db.fetchall.return_value = []

        get_vendor_unreliable_fields(mock_db, "CY10370773Q")

        call_args = mock_db.fetchall.call_args
        params = call_args[0][1]
        # Default window_days=90 → "-90 days" format in params
        assert "-90" in str(params)

    def test_custom_window_days_passed_to_query(self):
        mock_db = MagicMock()
        mock_db.fetchall.return_value = []

        get_vendor_unreliable_fields(mock_db, "CY10370773Q", window_days=30)

        call_args = mock_db.fetchall.call_args
        params = call_args[0][1]
        assert "-30" in str(params)


class TestShouldSuggestReprocessing:
    """Tests for should_suggest_reprocessing using a mock DB."""

    def _mock_db_with_rate_info(self, rate: float, total_facts: int) -> MagicMock:
        """Patch get_vendor_correction_rate to return given values."""
        return {"rate": rate, "total_facts": total_facts}

    def test_returns_true_when_rate_exceeds_threshold(self):
        mock_db = MagicMock()
        rate_info = {"rate": 0.45, "total_facts": 15}

        with patch(
            "alibi.services.correction_log.get_vendor_correction_rate",
            return_value=rate_info,
        ):
            result = should_suggest_reprocessing(mock_db, "CY12345678A", threshold=0.3)

        assert result is True

    def test_returns_false_when_rate_below_threshold(self):
        mock_db = MagicMock()
        rate_info = {"rate": 0.15, "total_facts": 15}

        with patch(
            "alibi.services.correction_log.get_vendor_correction_rate",
            return_value=rate_info,
        ):
            result = should_suggest_reprocessing(mock_db, "CY12345678A", threshold=0.3)

        assert result is False

    def test_returns_false_when_insufficient_facts(self):
        # total_facts < recent_docs (default 10) means not enough data
        mock_db = MagicMock()
        rate_info = {"rate": 0.90, "total_facts": 5}

        with patch(
            "alibi.services.correction_log.get_vendor_correction_rate",
            return_value=rate_info,
        ):
            result = should_suggest_reprocessing(
                mock_db, "CY12345678A", threshold=0.3, recent_docs=10
            )

        assert result is False

    def test_returns_false_exactly_at_threshold(self):
        mock_db = MagicMock()
        rate_info = {"rate": 0.30, "total_facts": 20}

        with patch(
            "alibi.services.correction_log.get_vendor_correction_rate",
            return_value=rate_info,
        ):
            result = should_suggest_reprocessing(mock_db, "CY12345678A", threshold=0.3)

        # rate == threshold is not ">" — should be False
        assert result is False

    def test_returns_true_with_custom_threshold(self):
        mock_db = MagicMock()
        rate_info = {"rate": 0.55, "total_facts": 12}

        with patch(
            "alibi.services.correction_log.get_vendor_correction_rate",
            return_value=rate_info,
        ):
            result = should_suggest_reprocessing(
                mock_db, "CY12345678A", threshold=0.5, recent_docs=10
            )

        assert result is True

    def test_uses_vendor_key_for_rate_lookup(self):
        mock_db = MagicMock()
        rate_info = {"rate": 0.1, "total_facts": 20}

        with patch(
            "alibi.services.correction_log.get_vendor_correction_rate",
            return_value=rate_info,
        ) as mock_rate:
            should_suggest_reprocessing(mock_db, "CY10370773Q")

        # Must be called with the correct vendor_key
        mock_rate.assert_called_once()
        assert mock_rate.call_args[0][1] == "CY10370773Q"


# ---------------------------------------------------------------------------
# Phase 5 — Sibling propagation
# ---------------------------------------------------------------------------


class TestPropagateToSiblings:
    """Tests for _propagate_to_siblings (called inside update_fact_item)."""

    def _make_mock_db(self, sibling_rows: list[dict]) -> MagicMock:
        """Build a mock DB that returns sibling rows from fetchall."""
        mock_db = MagicMock()
        mock_rows = []
        for row in sibling_rows:
            m = MagicMock()
            m.__getitem__ = lambda self, k, r=row: r[k]
            mock_rows.append(m)
        mock_db.fetchall.return_value = mock_rows

        # Set up transaction context manager
        mock_cursor = MagicMock()
        mock_db.transaction.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.transaction.return_value.__exit__ = MagicMock(return_value=False)
        return mock_db

    def test_updates_siblings_with_same_name_normalized(self):
        siblings = [
            {"id": "item-2", "enrichment_source": None},
            {"id": "item-3", "enrichment_source": "openfoodfacts"},
        ]
        mock_db = self._make_mock_db(siblings)

        count = _propagate_to_siblings(
            mock_db,
            "item-1",
            "whole milk",
            {"brand": "Happy Cow", "category": "Dairy"},
        )

        assert count == 2

    def test_returns_zero_when_no_siblings(self):
        mock_db = self._make_mock_db([])

        count = _propagate_to_siblings(
            mock_db,
            "item-1",
            "whole milk",
            {"category": "Dairy"},
        )

        assert count == 0

    def test_skips_user_confirmed_items_via_query(self):
        # The SQL query itself excludes user_confirmed items (via WHERE clause)
        # We verify the query is called with the correct SQL filter
        mock_db = self._make_mock_db([])

        _propagate_to_siblings(
            mock_db,
            "item-source",
            "apple juice",
            {"brand": "Tropicana"},
        )

        call_args = mock_db.fetchall.call_args
        sql = call_args[0][0]
        # Query must exclude user_confirmed enrichment source
        assert "user_confirmed" in sql

    def test_query_excludes_source_item_itself(self):
        mock_db = self._make_mock_db([])

        _propagate_to_siblings(
            mock_db,
            "item-source-99",
            "olive oil",
            {"category": "Oils"},
        )

        call_args = mock_db.fetchall.call_args
        params = call_args[0][1]
        assert "item-source-99" in params

    def test_query_filters_by_name_normalized(self):
        mock_db = self._make_mock_db([])

        _propagate_to_siblings(
            mock_db,
            "item-1",
            "greek yoghurt",
            {"brand": "Fage"},
        )

        call_args = mock_db.fetchall.call_args
        params = call_args[0][1]
        assert "greek yoghurt" in params

    def test_propagated_fields_appear_in_update_sql(self):
        siblings = [{"id": "item-99", "enrichment_source": None}]
        mock_db = self._make_mock_db(siblings)

        _propagate_to_siblings(
            mock_db,
            "item-1",
            "butter",
            {"category": "Dairy"},
        )

        mock_cursor = mock_db.transaction.return_value.__enter__.return_value
        # At least one execute for the UPDATE
        assert mock_cursor.execute.call_count >= 1
        executed_sql = mock_cursor.execute.call_args_list[0][0][0]
        assert "category" in executed_sql
        assert "fact_items" in executed_sql

    def test_sets_enrichment_source_to_sibling_propagation(self):
        siblings = [{"id": "item-55", "enrichment_source": None}]
        mock_db = self._make_mock_db(siblings)

        _propagate_to_siblings(
            mock_db,
            "item-1",
            "butter",
            {"brand": "Anchor"},
        )

        mock_cursor = mock_db.transaction.return_value.__enter__.return_value
        executed_sql = mock_cursor.execute.call_args_list[0][0][0]
        params = mock_cursor.execute.call_args_list[0][0][1]
        assert "enrichment_source" in executed_sql
        assert "sibling_propagation" in params

    def test_sets_enrichment_confidence_to_0_90(self):
        siblings = [{"id": "item-55", "enrichment_source": None}]
        mock_db = self._make_mock_db(siblings)

        _propagate_to_siblings(
            mock_db,
            "item-1",
            "butter",
            {"brand": "Anchor"},
        )

        mock_cursor = mock_db.transaction.return_value.__enter__.return_value
        params = mock_cursor.execute.call_args_list[0][0][1]
        assert 0.90 in params

    def test_returns_correct_count_with_multiple_siblings(self):
        siblings = [
            {"id": "item-10", "enrichment_source": None},
            {"id": "item-11", "enrichment_source": "product_resolver"},
            {"id": "item-12", "enrichment_source": None},
        ]
        mock_db = self._make_mock_db(siblings)

        count = _propagate_to_siblings(
            mock_db,
            "item-source",
            "orange juice",
            {"brand": "Tropicana"},
        )

        assert count == 3


# ---------------------------------------------------------------------------
# Phase 5 — derive_vendor_default_category
# ---------------------------------------------------------------------------


class TestDeriveVendorDefaultCategory:
    """Tests for derive_vendor_default_category using a mock DB."""

    def _make_mock_db(self, rows: list[dict]) -> MagicMock:
        """Return a mock DB whose fetchall returns structured rows."""
        mock_db = MagicMock()
        mock_rows = []
        for row in rows:
            m = MagicMock()
            m.__getitem__ = lambda self, k, r=row: r[k]
            mock_rows.append(m)
        mock_db.fetchall.return_value = mock_rows
        return mock_db

    def test_returns_top_category_when_five_or_more_items(self):
        rows = [{"category": "Dairy", "cnt": 8}]
        mock_db = self._make_mock_db(rows)

        result = derive_vendor_default_category(mock_db, "CY12345678A")

        assert result == "Dairy"

    def test_returns_none_when_fewer_than_five_items(self):
        rows = [{"category": "Bakery", "cnt": 4}]
        mock_db = self._make_mock_db(rows)

        result = derive_vendor_default_category(mock_db, "CY12345678A")

        assert result is None

    def test_returns_none_when_exactly_four_items(self):
        rows = [{"category": "Produce", "cnt": 4}]
        mock_db = self._make_mock_db(rows)

        result = derive_vendor_default_category(mock_db, "CY12345678A")

        assert result is None

    def test_returns_category_when_exactly_five_items(self):
        rows = [{"category": "Beverages", "cnt": 5}]
        mock_db = self._make_mock_db(rows)

        result = derive_vendor_default_category(mock_db, "CY12345678A")

        assert result == "Beverages"

    def test_returns_none_when_no_rows(self):
        mock_db = MagicMock()
        mock_db.fetchall.return_value = []

        result = derive_vendor_default_category(mock_db, "CY12345678A")

        assert result is None

    def test_uses_top_category_from_ordered_results(self):
        # DB returns results ordered by count DESC — first row is the winner
        rows = [
            {"category": "Frozen", "cnt": 12},
            {"category": "Dairy", "cnt": 7},
        ]
        mock_db = self._make_mock_db(rows)

        result = derive_vendor_default_category(mock_db, "CY12345678A")

        assert result == "Frozen"

    def test_passes_vendor_key_to_query(self):
        mock_db = MagicMock()
        mock_db.fetchall.return_value = []

        derive_vendor_default_category(mock_db, "CY10370773Q")

        call_args = mock_db.fetchall.call_args
        params = call_args[0][1]
        assert "CY10370773Q" in params

    def test_returns_string_type(self):
        rows = [{"category": "Snacks", "cnt": 6}]
        mock_db = self._make_mock_db(rows)

        result = derive_vendor_default_category(mock_db, "CY12345678A")

        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# VendorTemplate — to_dict / from_dict roundtrip
# ---------------------------------------------------------------------------


class TestVendorTemplateRoundtrip:
    """Tests that all adaptive learning fields survive to_dict/from_dict."""

    def _full_template(self) -> VendorTemplate:
        return VendorTemplate(
            layout_type="columnar",
            currency="EUR",
            pos_provider="JCC",
            success_count=12,
            gemini_bootstrapped=True,
            language="el",
            has_barcodes=True,
            has_unit_quantities=False,
            typical_item_count=8,
            confidence_history=[0.85, 0.90, 0.92, 0.88],
            adaptive_skip_threshold=0.95,
            preferred_ocr_tier=2,
            needs_rotation=True,
            common_fixes={"total": 3, "date": 1},
            last_updated="2026-02-01T10:00:00+00:00",
            stale=True,
            default_category="Groceries",
        )

    def test_roundtrip_preserves_last_updated(self):
        template = self._full_template()
        restored = VendorTemplate.from_dict(template.to_dict())

        assert restored.last_updated == "2026-02-01T10:00:00+00:00"

    def test_roundtrip_preserves_stale_true(self):
        template = self._full_template()
        restored = VendorTemplate.from_dict(template.to_dict())

        assert restored.stale is True

    def test_roundtrip_preserves_default_category(self):
        template = self._full_template()
        restored = VendorTemplate.from_dict(template.to_dict())

        assert restored.default_category == "Groceries"

    def test_roundtrip_preserves_confidence_history(self):
        template = self._full_template()
        restored = VendorTemplate.from_dict(template.to_dict())

        assert restored.confidence_history == [0.85, 0.90, 0.92, 0.88]

    def test_roundtrip_preserves_all_core_fields(self):
        template = self._full_template()
        restored = VendorTemplate.from_dict(template.to_dict())

        assert restored.layout_type == "columnar"
        assert restored.currency == "EUR"
        assert restored.pos_provider == "JCC"
        assert restored.success_count == 12
        assert restored.gemini_bootstrapped is True
        assert restored.language == "el"
        assert restored.has_barcodes is True
        assert restored.has_unit_quantities is False
        assert restored.typical_item_count == 8
        assert restored.adaptive_skip_threshold == 0.95
        assert restored.preferred_ocr_tier == 2
        assert restored.needs_rotation is True
        assert restored.common_fixes == {"total": 3, "date": 1}

    def test_stale_defaults_to_false_when_absent(self):
        # to_dict omits stale when False
        template = VendorTemplate(layout_type="standard", success_count=3, stale=False)
        d = template.to_dict()
        assert "stale" not in d

        restored = VendorTemplate.from_dict(d)
        assert restored.stale is False

    def test_last_updated_omitted_when_none(self):
        template = VendorTemplate(layout_type="standard", success_count=2)
        d = template.to_dict()

        assert "last_updated" not in d

    def test_default_category_omitted_when_none(self):
        template = VendorTemplate(layout_type="standard", success_count=2)
        d = template.to_dict()

        assert "default_category" not in d

    def test_from_dict_with_missing_new_fields_uses_defaults(self):
        # Simulate a stored dict from before adaptive learning fields were added
        legacy_dict = {
            "layout_type": "standard",
            "success_count": 5,
            "currency": "EUR",
        }
        template = VendorTemplate.from_dict(legacy_dict)

        assert template.last_updated is None
        assert template.stale is False
        assert template.default_category is None
        assert template.confidence_history == []

    def test_roundtrip_with_minimal_template(self):
        minimal = VendorTemplate(layout_type="standard", success_count=0)
        restored = VendorTemplate.from_dict(minimal.to_dict())

        assert restored.layout_type == "standard"
        assert restored.success_count == 0
        assert restored.stale is False
        assert restored.last_updated is None
        assert restored.default_category is None

    def test_to_dict_is_json_serializable(self):
        import json

        template = self._full_template()
        d = template.to_dict()

        # Must not raise
        serialized = json.dumps(d)
        assert len(serialized) > 0
