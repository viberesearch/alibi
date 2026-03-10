"""Tests for multi-language display support."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from alibi.api.app import create_app
from alibi.api.deps import get_database
from alibi.config import Config, reset_config
from alibi.db.connection import DatabaseManager
from alibi.i18n import format_line_item_name, get_display_name, get_supported_languages


# ---------------------------------------------------------------------------
# Unit tests for get_display_name
# ---------------------------------------------------------------------------


class TestGetDisplayName:
    """Tests for get_display_name with various input combinations."""

    def test_original_lang_returns_original(self, monkeypatch: pytest.MonkeyPatch):
        """When lang=original, the original name is returned."""
        monkeypatch.setenv("ALIBI_TESTING", "1")
        reset_config()
        result = get_display_name("Gala", "Milk", lang="original")
        assert result == "Gala"

    def test_original_lang_falls_back_to_normalized(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """When lang=original and original is None, falls back to normalized."""
        monkeypatch.setenv("ALIBI_TESTING", "1")
        reset_config()
        result = get_display_name(None, "Milk", lang="original")
        assert result == "Milk"

    def test_en_lang_returns_normalized(self, monkeypatch: pytest.MonkeyPatch):
        """When lang=en, the normalized (English) name is returned."""
        monkeypatch.setenv("ALIBI_TESTING", "1")
        reset_config()
        result = get_display_name("Gala", "Milk", lang="en")
        assert result == "Milk"

    def test_en_lang_falls_back_to_original(self, monkeypatch: pytest.MonkeyPatch):
        """When lang=en and normalized is None, falls back to original."""
        monkeypatch.setenv("ALIBI_TESTING", "1")
        reset_config()
        result = get_display_name("Gala", None, lang="en")
        assert result == "Gala"

    def test_normalized_alias_returns_normalized(self, monkeypatch: pytest.MonkeyPatch):
        """The 'normalized' alias behaves like 'en'."""
        monkeypatch.setenv("ALIBI_TESTING", "1")
        reset_config()
        result = get_display_name("Gala", "Milk", lang="normalized")
        assert result == "Milk"

    def test_both_none_returns_empty(self, monkeypatch: pytest.MonkeyPatch):
        """When both original and normalized are None, returns empty string."""
        monkeypatch.setenv("ALIBI_TESTING", "1")
        reset_config()
        result = get_display_name(None, None, lang="original")
        assert result == ""

    def test_both_none_en_returns_empty(self, monkeypatch: pytest.MonkeyPatch):
        """When both are None with lang=en, returns empty string."""
        monkeypatch.setenv("ALIBI_TESTING", "1")
        reset_config()
        result = get_display_name(None, None, lang="en")
        assert result == ""

    def test_specific_language_prefers_original(self, monkeypatch: pytest.MonkeyPatch):
        """A specific language code (e.g., 'el') prefers original."""
        monkeypatch.setenv("ALIBI_TESTING", "1")
        reset_config()
        result = get_display_name("Gala", "Milk", lang="el")
        assert result == "Gala"

    def test_specific_language_falls_back_to_normalized(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A specific language falls back to normalized when original is None."""
        monkeypatch.setenv("ALIBI_TESTING", "1")
        reset_config()
        result = get_display_name(None, "Milk", lang="el")
        assert result == "Milk"

    def test_uses_config_default_when_no_lang(self, monkeypatch: pytest.MonkeyPatch):
        """When lang is None, uses config.display_language."""
        monkeypatch.setenv("ALIBI_TESTING", "1")
        monkeypatch.setenv("ALIBI_DISPLAY_LANGUAGE", "en")
        reset_config()
        result = get_display_name("Gala", "Milk")
        assert result == "Milk"

    def test_config_default_is_original(self, monkeypatch: pytest.MonkeyPatch):
        """Default config display_language is 'original'."""
        monkeypatch.setenv("ALIBI_TESTING", "1")
        reset_config()
        result = get_display_name("Gala", "Milk")
        assert result == "Gala"

    def test_empty_string_original_treated_as_falsy(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """An empty string original falls through to normalized."""
        monkeypatch.setenv("ALIBI_TESTING", "1")
        reset_config()
        result = get_display_name("", "Milk", lang="original")
        assert result == "Milk"

    def test_empty_string_normalized_treated_as_falsy(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """An empty string normalized falls through to original."""
        monkeypatch.setenv("ALIBI_TESTING", "1")
        reset_config()
        result = get_display_name("Gala", "", lang="en")
        assert result == "Gala"


# ---------------------------------------------------------------------------
# Unit tests for format_line_item_name
# ---------------------------------------------------------------------------


class TestFormatLineItemName:
    """Tests for format_line_item_name helper."""

    def test_formats_with_both_names(self, monkeypatch: pytest.MonkeyPatch):
        """Formats line item dict with both name and name_normalized."""
        monkeypatch.setenv("ALIBI_TESTING", "1")
        reset_config()
        item = {"name": "Gala", "name_normalized": "Milk"}
        result = format_line_item_name(item, lang="en")
        assert result == "Milk"

    def test_formats_with_only_name(self, monkeypatch: pytest.MonkeyPatch):
        """Formats line item dict with only name (no normalized)."""
        monkeypatch.setenv("ALIBI_TESTING", "1")
        reset_config()
        item = {"name": "Bread"}
        result = format_line_item_name(item, lang="en")
        assert result == "Bread"

    def test_formats_with_only_normalized(self, monkeypatch: pytest.MonkeyPatch):
        """Formats line item dict with only name_normalized."""
        monkeypatch.setenv("ALIBI_TESTING", "1")
        reset_config()
        item = {"name": None, "name_normalized": "Cheese"}
        result = format_line_item_name(item, lang="original")
        assert result == "Cheese"

    def test_formats_empty_dict(self, monkeypatch: pytest.MonkeyPatch):
        """Handles dict with no name fields gracefully."""
        monkeypatch.setenv("ALIBI_TESTING", "1")
        reset_config()
        item: dict[str, str | None] = {}
        result = format_line_item_name(item, lang="en")
        assert result == ""


# ---------------------------------------------------------------------------
# Unit tests for get_supported_languages
# ---------------------------------------------------------------------------


class TestGetSupportedLanguages:
    """Tests for get_supported_languages."""

    def test_returns_list(self):
        """Returns a non-empty list."""
        langs = get_supported_languages()
        assert isinstance(langs, list)
        assert len(langs) > 0

    def test_contains_original(self):
        """The 'original' option is present."""
        langs = get_supported_languages()
        codes = [lang["code"] for lang in langs]
        assert "original" in codes

    def test_contains_english(self):
        """English is a supported language."""
        langs = get_supported_languages()
        codes = [lang["code"] for lang in langs]
        assert "en" in codes

    def test_each_entry_has_code_and_name(self):
        """Every entry has 'code' and 'name' keys."""
        for lang in get_supported_languages():
            assert "code" in lang
            assert "name" in lang

    def test_expected_languages(self):
        """All expected language codes are present."""
        langs = get_supported_languages()
        codes = {lang["code"] for lang in langs}
        expected = {"original", "en", "de", "el", "ru", "ar"}
        assert expected == codes


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def i18n_db(tmp_path: Path) -> Generator[DatabaseManager, None, None]:
    """Create a database with line items that have name_normalized."""
    os.environ["ALIBI_TESTING"] = "1"
    config = Config(db_path=tmp_path / "test_i18n.db")
    manager = DatabaseManager(config)
    manager.initialize()

    conn = manager.get_connection()

    # Create required user and space
    conn.execute(
        "INSERT INTO users (id, name) VALUES (?, ?)",
        ("user-1", "Test User"),
    )
    conn.execute(
        "INSERT INTO spaces (id, name, type, owner_id) VALUES (?, ?, ?, ?)",
        ("space-1", "Default", "private", "user-1"),
    )

    # Create a document
    conn.execute(
        "INSERT INTO documents (id, file_path, file_hash) VALUES (?, ?, ?)",
        ("doc-0", "/receipts/greek_receipt.jpg", "hash0"),
    )

    # Create cloud and fact
    conn.execute(
        "INSERT INTO clouds (id, status) VALUES (?, 'collapsed')",
        ("cloud-0",),
    )
    conn.execute(
        """INSERT INTO facts (id, cloud_id, fact_type, vendor, total_amount,
                               currency, event_date, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "fact-0",
            "cloud-0",
            "purchase",
            "Greek Shop",
            "12.50",
            "EUR",
            "2025-06-15",
            "confirmed",
        ),
    )

    # Create atoms for fact_items FK
    for atom_id in ("atom-1", "atom-2", "atom-3"):
        conn.execute(
            "INSERT INTO atoms (id, document_id, atom_type, data) VALUES (?, ?, ?, ?)",
            (atom_id, "doc-0", "item", "{}"),
        )

    # Fact items with Greek names (fact_items stores name only, no name_normalized)
    conn.execute(
        """INSERT INTO fact_items (id, fact_id, atom_id, name, quantity,
                                   unit_price, total_price, category)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "fi-greek-1",
            "fact-0",
            "atom-1",
            "\u0393\u03ac\u03bb\u03b1",
            "1",
            "2.50",
            "2.50",
            "dairy",
        ),
    )
    conn.execute(
        """INSERT INTO fact_items (id, fact_id, atom_id, name, quantity,
                                   unit_price, total_price, category)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "fi-greek-2",
            "fact-0",
            "atom-2",
            "\u03a8\u03c9\u03bc\u03af",
            "2",
            "1.80",
            "3.60",
            "bakery",
        ),
    )
    conn.execute(
        """INSERT INTO fact_items (id, fact_id, atom_id, name, quantity,
                                   unit_price, total_price, category)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ("fi-plain", "fact-0", "atom-3", "Water", "1", "0.80", "0.80", "beverages"),
    )

    conn.commit()
    yield manager
    manager.close()


@pytest.fixture
def i18n_client(i18n_db: DatabaseManager) -> Generator[TestClient, None, None]:
    """Create a test client for i18n API tests."""
    app = create_app()

    def override_get_database() -> DatabaseManager:
        return i18n_db

    app.dependency_overrides[get_database] = override_get_database
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestLineItemsLangParam:
    """API integration tests for fact items (line items) endpoint."""

    def test_list_returns_names(self, i18n_client: TestClient):
        """Fact items return names as stored."""
        resp = i18n_client.get("/api/v1/line-items")
        assert resp.status_code == 200
        data = resp.json()
        names = [item["name"] for item in data["fact_items"]]
        assert "\u0393\u03ac\u03bb\u03b1" in names
        assert "\u03a8\u03c9\u03bc\u03af" in names
        assert "Water" in names

    def test_get_single_fact_item(self, i18n_client: TestClient):
        """GET single fact item returns stored name."""
        resp = i18n_client.get("/api/v1/line-items/fi-greek-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "\u0393\u03ac\u03bb\u03b1"

    def test_get_single_plain_name(self, i18n_client: TestClient):
        """GET fact item with plain name returns it directly."""
        resp = i18n_client.get("/api/v1/line-items/fi-plain")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Water"
