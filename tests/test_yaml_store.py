"""Tests for YAML store feature — decoupled yaml_store path mode.

Tests the yaml_store placement mode where .alibi.yaml files are written to
a central directory tree rather than alongside source documents (sidecar).
"""

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

os.environ["ALIBI_TESTING"] = "1"

from alibi.extraction.yaml_cache import (
    YAML_SUFFIX,
    YAML_VERSION,
    compute_yaml_hash,
    get_yaml_path,
    read_yaml_cache,
    read_yaml_with_meta,
    reset_yaml_store,
    resolve_source_from_yaml,
    scan_yaml_store,
    set_yaml_store_root,
    write_yaml_cache,
)

_EXTRACTED = {"vendor": "Test Vendor", "date": "2024-01-15", "total": 10.0}


@pytest.fixture(autouse=True)
def reset_store():
    """Reset yaml store to None (sidecar mode) before and after each test."""
    set_yaml_store_root(None)
    yield
    set_yaml_store_root(None)
    reset_yaml_store()


def _make_valid_yaml(path: Path, extra_meta: dict[str, Any] | None = None) -> None:
    """Write a minimal valid .alibi.yaml at path."""
    meta = {"version": YAML_VERSION, "extracted_at": "2024-01-15T10:00:00"}
    if extra_meta:
        meta.update(extra_meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "_meta": meta,
                "document_type": "receipt",
                "vendor": "Test Vendor",
                "total": 10.0,
            },
            f,
        )


# ---------------------------------------------------------------------------
# TestYamlStorePath
# ---------------------------------------------------------------------------


class TestYamlStorePath:
    """Path resolution for both sidecar and store modes."""

    def test_get_yaml_path_store_single(self, tmp_path: Path) -> None:
        store = tmp_path / "yaml_store"
        set_yaml_store_root(store)
        source = tmp_path / "docs" / "receipt.jpg"
        result = get_yaml_path(source)
        assert result == store / "system" / "unsorted" / "receipt.alibi.yaml"

    def test_get_yaml_path_store_with_user_doc_type(self, tmp_path: Path) -> None:
        store = tmp_path / "yaml_store"
        set_yaml_store_root(store)
        source = tmp_path / "docs" / "invoice.pdf"
        result = get_yaml_path(source, user_id="alice", doc_type="receipt")
        assert result == store / "alice" / "receipt" / "invoice.alibi.yaml"

    def test_get_yaml_path_store_group(self, tmp_path: Path) -> None:
        store = tmp_path / "yaml_store"
        set_yaml_store_root(store)
        folder = tmp_path / "docs" / "myfolder"
        result = get_yaml_path(folder, is_group=True)
        assert result == store / "system" / "unsorted" / "myfolder.alibi.yaml"


# ---------------------------------------------------------------------------
# TestWriteReadRoundtrip
# ---------------------------------------------------------------------------


class TestWriteReadRoundtrip:
    """write_yaml_cache / read_yaml_cache round-trip in store mode."""

    def test_write_read_store_mode(self, tmp_path: Path) -> None:
        store = tmp_path / "yaml_store"
        set_yaml_store_root(store)
        source = tmp_path / "inbox" / "receipt.jpg"
        source.parent.mkdir(parents=True)
        source.touch()

        written = write_yaml_cache(source, _EXTRACTED, "receipt")
        assert written is not None
        assert written == store / "system" / "receipt" / "receipt.alibi.yaml"

        loaded = read_yaml_cache(source, doc_type="receipt")
        assert loaded is not None
        assert loaded["vendor"] == "Test Vendor"
        assert loaded["total"] == 10.0

    def test_write_stores_source_path_in_meta(self, tmp_path: Path) -> None:
        store = tmp_path / "yaml_store"
        set_yaml_store_root(store)
        source = tmp_path / "inbox" / "invoice.pdf"
        source.parent.mkdir(parents=True)
        source.touch()

        written = write_yaml_cache(source, _EXTRACTED, "invoice")
        assert written is not None

        with open(written, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        meta = raw["_meta"]
        assert "source_path" in meta
        assert Path(meta["source_path"]) == source.resolve()
        assert meta["is_group"] is False

    def test_write_creates_directories(self, tmp_path: Path) -> None:
        store = tmp_path / "yaml_store" / "deep" / "nested"
        set_yaml_store_root(store)
        source = tmp_path / "receipt.jpg"
        source.touch()
        # Directories do not exist yet
        assert not store.exists()
        written = write_yaml_cache(source, _EXTRACTED, "receipt")
        assert written is not None
        assert written.exists()


# ---------------------------------------------------------------------------
# TestComputeYamlHash
# ---------------------------------------------------------------------------


class TestComputeYamlHash:
    """compute_yaml_hash reads from store or sidecar."""

    def test_hash_store_mode(self, tmp_path: Path) -> None:
        store = tmp_path / "yaml_store"
        set_yaml_store_root(store)
        source = tmp_path / "receipt.jpg"
        source.touch()

        written = write_yaml_cache(source, _EXTRACTED, "receipt")
        assert written is not None

        h = compute_yaml_hash(source, doc_type="receipt")
        assert h is not None
        assert len(h) == 64  # SHA-256 hex digest

    def test_hash_no_yaml(self, tmp_path: Path) -> None:
        store = tmp_path / "yaml_store"
        store.mkdir()
        set_yaml_store_root(store)
        source = tmp_path / "receipt.jpg"

        h = compute_yaml_hash(source, doc_type="receipt")
        assert h is None


# ---------------------------------------------------------------------------
# TestResolveSourceFromYaml
# ---------------------------------------------------------------------------


class TestResolveSourceFromYaml:
    """resolve_source_from_yaml: reverse-maps YAML path to source."""

    def test_resolve_from_meta_source_path(self, tmp_path: Path) -> None:
        source = tmp_path / "docs" / "receipt.jpg"
        source.parent.mkdir(parents=True)
        source.touch()
        yaml_path = tmp_path / "receipt.alibi.yaml"
        _make_valid_yaml(
            yaml_path,
            extra_meta={
                "source_path": str(source.resolve()),
                "is_group": False,
            },
        )

        result = resolve_source_from_yaml(yaml_path)
        assert result is not None
        resolved_source, is_group = result
        assert resolved_source == source.resolve()
        assert is_group is False

    def test_resolve_from_meta_is_group(self, tmp_path: Path) -> None:
        folder = tmp_path / "invoice-acme"
        folder.mkdir()
        yaml_path = folder / ".alibi.yaml"
        _make_valid_yaml(
            yaml_path,
            extra_meta={
                "source_path": str(folder.resolve()),
                "is_group": True,
            },
        )

        result = resolve_source_from_yaml(yaml_path)
        assert result is not None
        resolved_source, is_group = result
        assert resolved_source == folder.resolve()
        assert is_group is True


# ---------------------------------------------------------------------------
# TestScanYamlStore
# ---------------------------------------------------------------------------


class TestScanYamlStore:
    """scan_yaml_store discovers YAMLs in the yaml_store tree."""

    def test_scan_store(self, tmp_path: Path) -> None:
        store = tmp_path / "yaml_store"
        set_yaml_store_root(store)
        a = store / "system" / "receipt" / f"a{YAML_SUFFIX}"
        b = store / "system" / "invoice" / f"b{YAML_SUFFIX}"
        _make_valid_yaml(a)
        _make_valid_yaml(b)

        results = scan_yaml_store()
        paths = {r.resolve() for r in results}
        assert a.resolve() in paths
        assert b.resolve() in paths
        assert len(results) == 2

    def test_scan_empty_store(self, tmp_path: Path) -> None:
        store = tmp_path / "yaml_store"
        store.mkdir()
        set_yaml_store_root(store)

        results = scan_yaml_store()
        assert results == []

    def test_scan_no_store(self, tmp_path: Path) -> None:
        set_yaml_store_root(None)
        results = scan_yaml_store()
        assert results == []


# ---------------------------------------------------------------------------
# TestMigration025
# ---------------------------------------------------------------------------


class TestMigration025:
    """DB migration 025: yaml_path column on documents table."""

    def test_yaml_path_column_exists(self, db_manager) -> None:
        conn = db_manager.get_connection()
        cursor = conn.execute("PRAGMA table_info(documents)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "yaml_path" in columns

    def test_store_document_with_yaml_path(self, db_manager) -> None:
        import uuid

        from alibi.db.models import Document
        from alibi.db.v2_store import store_document

        doc_id = str(uuid.uuid4())
        yaml_path_str = "/tmp/yaml_store/system/receipt/mydoc.alibi.yaml"
        doc = Document(
            id=doc_id,
            file_path="/tmp/inbox/mydoc.jpg",
            file_hash="a" * 64,
            yaml_path=yaml_path_str,
        )
        store_document(db_manager, doc)

        row = db_manager.fetchone(
            "SELECT yaml_path FROM documents WHERE id = ?", (doc_id,)
        )
        assert row is not None
        assert row[0] == yaml_path_str
