"""Tests for YAML-first pipeline architecture.

Covers: yaml_cache new functions, v2_store additions, pipeline Phase A/B split,
service layer reingest, and watcher YAML event routing.
"""

import hashlib
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from alibi.extraction.yaml_cache import (
    YAML_SUFFIX,
    YAML_VERSION,
    compute_yaml_hash,
    get_yaml_path,
    read_yaml_with_meta,
    resolve_source_from_yaml,
    write_yaml_cache,
)


# ---------------------------------------------------------------------------
# yaml_cache: read_yaml_with_meta
# ---------------------------------------------------------------------------


class TestReadYamlWithMeta:
    """Tests for read_yaml_with_meta."""

    def test_returns_data_and_meta(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()
        write_yaml_cache(
            source,
            {"vendor": "ACME", "total": 10.0},
            "receipt",
            file_hash="abc123",
            perceptual_hash="def456",
        )

        result = read_yaml_with_meta(source, doc_type="receipt")
        assert result is not None
        extracted, meta = result
        assert extracted["vendor"] == "ACME"
        assert meta["version"] == YAML_VERSION
        assert meta["file_hash"] == "abc123"
        assert meta["perceptual_hash"] == "def456"

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        source = tmp_path / "missing.jpg"
        assert read_yaml_with_meta(source) is None

    def test_returns_none_on_version_mismatch(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()
        yaml_path = get_yaml_path(source, doc_type="receipt")
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(yaml.dump({"_meta": {"version": 999}, "vendor": "X"}))
        assert read_yaml_with_meta(source, doc_type="receipt") is None

    def test_returns_none_on_invalid_yaml(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()
        yaml_path = get_yaml_path(source, doc_type="receipt")
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(": invalid: yaml: [")
        assert read_yaml_with_meta(source, doc_type="receipt") is None

    def test_document_group(self, tmp_path: Path) -> None:
        folder = tmp_path / "invoice-acme"
        folder.mkdir()
        write_yaml_cache(
            folder,
            {"issuer": "ACME Corp", "amount": 100.0},
            "invoice",
            is_group=True,
            file_hash="group_hash",
        )

        result = read_yaml_with_meta(folder, is_group=True, doc_type="invoice")
        assert result is not None
        extracted, meta = result
        assert extracted["issuer"] == "ACME Corp"
        assert meta["file_hash"] == "group_hash"


# ---------------------------------------------------------------------------
# yaml_cache: compute_yaml_hash
# ---------------------------------------------------------------------------


class TestComputeYamlHash:
    """Tests for compute_yaml_hash."""

    def test_returns_sha256(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()
        write_yaml_cache(source, {"vendor": "X"}, "receipt")

        result = compute_yaml_hash(source, doc_type="receipt")
        assert result is not None
        assert len(result) == 64  # SHA-256 hex digest

        # Verify it matches
        yaml_path = get_yaml_path(source, doc_type="receipt")
        expected = hashlib.sha256(yaml_path.read_bytes()).hexdigest()
        assert result == expected

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        source = tmp_path / "missing.jpg"
        assert compute_yaml_hash(source) is None

    def test_changes_when_yaml_edited(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()
        write_yaml_cache(source, {"vendor": "X"}, "receipt")
        hash1 = compute_yaml_hash(source, doc_type="receipt")

        # Edit the YAML
        yaml_path = get_yaml_path(source, doc_type="receipt")
        data = yaml.safe_load(yaml_path.read_text())
        data["vendor"] = "Y"
        yaml_path.write_text(yaml.dump(data))
        hash2 = compute_yaml_hash(source, doc_type="receipt")

        assert hash1 != hash2


# ---------------------------------------------------------------------------
# yaml_cache: resolve_source_from_yaml
# ---------------------------------------------------------------------------


class TestResolveSourceFromYaml:
    """Tests for resolve_source_from_yaml."""

    def test_single_file(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()
        write_yaml_cache(source, {"vendor": "X"}, "receipt")
        yaml_path = get_yaml_path(source, doc_type="receipt")

        result = resolve_source_from_yaml(yaml_path)
        assert result is not None
        path, is_group = result
        assert path == source
        assert is_group is False

    def test_document_group(self, tmp_path: Path) -> None:
        folder = tmp_path / "invoice-acme"
        folder.mkdir()
        write_yaml_cache(folder, {"issuer": "ACME"}, "invoice", is_group=True)
        yaml_path = get_yaml_path(folder, is_group=True, doc_type="invoice")

        result = resolve_source_from_yaml(yaml_path)
        assert result is not None
        path, is_group = result
        assert path == folder
        assert is_group is True

    def test_source_deleted(self, tmp_path: Path) -> None:
        """Source file deleted but YAML remains — return best guess."""
        source = tmp_path / "receipt.jpg"
        source.touch()
        write_yaml_cache(source, {"vendor": "X"}, "receipt")
        yaml_path = get_yaml_path(source, doc_type="receipt")
        source.unlink()  # delete source

        result = resolve_source_from_yaml(yaml_path)
        assert result is not None
        path, is_group = result
        assert path == source  # source_path from _meta
        assert is_group is False


# ---------------------------------------------------------------------------
# yaml_cache: write_yaml_cache with file_hash / perceptual_hash
# ---------------------------------------------------------------------------


class TestWriteYamlCacheV5:
    """Tests for write_yaml_cache v5 with hash fields."""

    def test_stores_file_hash_in_meta(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()
        write_yaml_cache(
            source,
            {"vendor": "ACME"},
            "receipt",
            file_hash="sha256_abc",
            perceptual_hash="dhash_def",
        )

        yaml_path = get_yaml_path(source, doc_type="receipt")
        data = yaml.safe_load(yaml_path.read_text())
        assert data["_meta"]["version"] == 5
        assert data["_meta"]["file_hash"] == "sha256_abc"
        assert data["_meta"]["perceptual_hash"] == "dhash_def"

    def test_omits_hashes_when_none(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()
        write_yaml_cache(source, {"vendor": "ACME"}, "receipt")

        yaml_path = get_yaml_path(source, doc_type="receipt")
        data = yaml.safe_load(yaml_path.read_text())
        assert "file_hash" not in data["_meta"]
        assert "perceptual_hash" not in data["_meta"]


# ---------------------------------------------------------------------------
# v2_store: new functions
# ---------------------------------------------------------------------------


class TestV2StoreNewFunctions:
    """Tests for get_document_by_path and update_yaml_hash."""

    def test_get_document_by_path(self, db: Any) -> None:
        from alibi.db import v2_store
        from alibi.db.models import Document

        doc = Document(
            id="test-doc-1",
            file_path="/tmp/receipt.jpg",
            file_hash="abc123",
        )
        v2_store.store_document(db, doc)

        result = v2_store.get_document_by_path(db, "/tmp/receipt.jpg")
        assert result is not None
        assert result["id"] == "test-doc-1"

    def test_get_document_by_path_not_found(self, db: Any) -> None:
        from alibi.db import v2_store

        result = v2_store.get_document_by_path(db, "/nonexistent")
        assert result is None

    def test_update_yaml_hash(self, db: Any) -> None:
        from alibi.db import v2_store
        from alibi.db.models import Document

        doc = Document(
            id="test-doc-2",
            file_path="/tmp/receipt.jpg",
            file_hash="abc123",
        )
        v2_store.store_document(db, doc)

        v2_store.update_yaml_hash(db, "test-doc-2", "yaml_hash_xyz")

        result = v2_store.get_document_by_hash(db, "abc123")
        assert result is not None
        assert result["yaml_hash"] == "yaml_hash_xyz"

    def test_store_document_with_yaml_hash(self, db: Any) -> None:
        from alibi.db import v2_store
        from alibi.db.models import Document

        doc = Document(
            id="test-doc-3",
            file_path="/tmp/receipt.jpg",
            file_hash="def456",
            yaml_hash="initial_hash",
        )
        v2_store.store_document(db, doc)

        result = v2_store.get_document_by_hash(db, "def456")
        assert result is not None
        assert result["yaml_hash"] == "initial_hash"


# ---------------------------------------------------------------------------
# Pipeline: ingest_from_yaml
# ---------------------------------------------------------------------------


class TestIngestFromYaml:
    """Tests for the public ingest_from_yaml method."""

    def test_returns_error_when_no_yaml(self, db: Any, tmp_path: Path) -> None:
        from alibi.processing.pipeline import ProcessingPipeline

        source = tmp_path / "missing.jpg"
        source.touch()

        pipeline = ProcessingPipeline(db)
        result = pipeline.ingest_from_yaml(db, source)
        assert not result.success
        assert "No valid .alibi.yaml" in (result.error or "")

    def test_returns_error_when_no_file_hash(self, db: Any, tmp_path: Path) -> None:
        from alibi.processing.pipeline import ProcessingPipeline

        source = tmp_path / "receipt.jpg"
        source.touch()
        # Write YAML without file_hash
        write_yaml_cache(source, {"vendor": "X"}, "receipt")

        # Delete the source so there's no way to compute hash
        source.unlink()

        pipeline = ProcessingPipeline(db)
        result = pipeline.ingest_from_yaml(db, source)
        assert not result.success
        assert "file_hash" in (result.error or "")


# ---------------------------------------------------------------------------
# Watcher: YAML event routing
# ---------------------------------------------------------------------------


class TestWatcherYamlRouting:
    """Tests for DebouncedEventHandler YAML routing."""

    def test_yaml_modified_routes_to_yaml_callback(self) -> None:
        from alibi.daemon.watcher_service import DebouncedEventHandler

        doc_cb = MagicMock()
        yaml_cb = MagicMock()
        handler = DebouncedEventHandler(
            process_callback=doc_cb,
            yaml_callback=yaml_cb,
        )

        event = MagicMock()
        event.is_directory = False
        event.src_path = "/inbox/receipt.alibi.yaml"

        handler.on_modified(event)

        # Should be in yaml pending, not document pending
        assert "/inbox/receipt.alibi.yaml" in handler.pending_yaml
        assert "/inbox/receipt.alibi.yaml" not in handler.pending

    def test_yaml_deleted_calls_delete_callback(self) -> None:
        from alibi.daemon.watcher_service import DebouncedEventHandler

        doc_cb = MagicMock()
        delete_cb = MagicMock()
        handler = DebouncedEventHandler(
            process_callback=doc_cb,
            yaml_delete_callback=delete_cb,
        )

        event = MagicMock()
        event.is_directory = False
        event.src_path = "/inbox/receipt.alibi.yaml"

        handler.on_deleted(event)

        delete_cb.assert_called_once()

    def test_yaml_created_ignored(self) -> None:
        from alibi.daemon.watcher_service import DebouncedEventHandler

        doc_cb = MagicMock()
        yaml_cb = MagicMock()
        handler = DebouncedEventHandler(
            process_callback=doc_cb,
            yaml_callback=yaml_cb,
        )

        event = MagicMock()
        event.is_directory = False
        event.src_path = "/inbox/receipt.alibi.yaml"

        handler.on_created(event)

        # YAML creation should be ignored (our pipeline creates it)
        assert len(handler.pending) == 0
        assert len(handler.pending_yaml) == 0

    def test_regular_file_not_routed_to_yaml(self) -> None:
        from alibi.daemon.watcher_service import DebouncedEventHandler

        doc_cb = MagicMock()
        yaml_cb = MagicMock()
        handler = DebouncedEventHandler(
            process_callback=doc_cb,
            yaml_callback=yaml_cb,
        )

        event = MagicMock()
        event.is_directory = False
        event.src_path = "/inbox/receipt.jpg"

        handler.on_created(event)

        assert "/inbox/receipt.jpg" in handler.pending
        assert len(handler.pending_yaml) == 0
