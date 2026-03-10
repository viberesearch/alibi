"""Tests for Telegram bot map URL / location handlers."""

from __future__ import annotations

import time

import pytest

from alibi.telegram.handlers.upload import (
    _PENDING_TTL,
    _pending_location,
    _pop_pending_location,
)


class TestPendingLocationState:
    """Test the pending location state machine."""

    def setup_method(self):
        _pending_location.clear()

    def teardown_method(self):
        _pending_location.clear()

    def test_pop_returns_fact_id(self):
        _pending_location[123] = ("fact-abc", time.time())
        result = _pop_pending_location(123)
        assert result == "fact-abc"
        assert 123 not in _pending_location

    def test_pop_returns_none_when_empty(self):
        assert _pop_pending_location(123) is None

    def test_pop_returns_none_when_expired(self):
        _pending_location[123] = ("fact-abc", time.time() - _PENDING_TTL - 1)
        assert _pop_pending_location(123) is None
        assert 123 not in _pending_location

    def test_pop_removes_entry(self):
        _pending_location[123] = ("fact-abc", time.time())
        _pop_pending_location(123)
        assert _pop_pending_location(123) is None

    def test_different_chat_ids_independent(self):
        _pending_location[100] = ("fact-1", time.time())
        _pending_location[200] = ("fact-2", time.time())
        assert _pop_pending_location(100) == "fact-1"
        assert _pop_pending_location(200) == "fact-2"


class TestFormatResultWithLocationHint:
    """Test that format_result includes location prompt."""

    def test_includes_location_prompt(self):
        from alibi.telegram.handlers.upload import _format_result

        class MockResult:
            success = True
            is_duplicate = False
            document_id = "fact-123"
            extracted_data = {"vendor": "Shop", "total": 10.0}
            line_items = []
            record_type = None
            error = None
            duplicate_of = None

        reply = _format_result(MockResult())
        assert "Google Maps URL" in reply
        assert "/skip" in reply


class TestMapUrlDetection:
    """Test is_map_url for Telegram message filtering."""

    def test_detects_google_maps_url(self):
        from alibi.utils.map_url import is_map_url

        assert is_map_url("https://maps.app.goo.gl/abc123")
        assert is_map_url("https://www.google.com/maps/place/Shop/@34.77,32.42")

    def test_rejects_non_maps_url(self):
        from alibi.utils.map_url import is_map_url

        assert not is_map_url("hello world")
        assert not is_map_url("https://example.com")
