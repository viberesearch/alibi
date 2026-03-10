"""Tests for YAML intermediary file support."""

import tempfile
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from alibi.extraction.yaml_cache import (
    YAML_SUFFIX,
    YAML_VERSION,
    find_yaml_in_store,
    get_yaml_path,
    read_yaml_cache,
    set_yaml_store_root,
    write_yaml_cache,
)


class TestGetYamlPath:
    """Tests for YAML path resolution."""

    def test_single_file_jpg(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        store = tmp_path / "yaml_store"
        assert (
            get_yaml_path(source)
            == store / "system" / "unsorted" / "receipt.alibi.yaml"
        )

    def test_single_file_png(self, tmp_path: Path) -> None:
        source = tmp_path / "scan.png"
        store = tmp_path / "yaml_store"
        assert (
            get_yaml_path(source) == store / "system" / "unsorted" / "scan.alibi.yaml"
        )

    def test_single_file_pdf(self, tmp_path: Path) -> None:
        source = tmp_path / "invoice.pdf"
        store = tmp_path / "yaml_store"
        assert (
            get_yaml_path(source)
            == store / "system" / "unsorted" / "invoice.alibi.yaml"
        )

    def test_document_group(self, tmp_path: Path) -> None:
        folder = tmp_path / "invoice-acme"
        folder.mkdir()
        store = tmp_path / "yaml_store"
        assert (
            get_yaml_path(folder, is_group=True)
            == store / "system" / "unsorted" / "invoice-acme.alibi.yaml"
        )

    def test_nested_path(self, tmp_path: Path) -> None:
        deep = tmp_path / "inbox" / "documents"
        deep.mkdir(parents=True)
        source = deep / "receipt.jpg"
        store = tmp_path / "yaml_store"
        # Store path doesn't depend on source directory location
        assert (
            get_yaml_path(source)
            == store / "system" / "unsorted" / "receipt.alibi.yaml"
        )


class TestWriteYamlCache:
    """Tests for writing YAML cache files."""

    def test_write_basic(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()
        extracted = {
            "vendor": "FRESKO",
            "date": "2025-12-15",
            "total": 25.99,
            "currency": "EUR",
            "line_items": [{"name": "Milk", "quantity": 1, "total_price": 1.29}],
        }

        result = write_yaml_cache(source, extracted, "receipt")

        store = tmp_path / "yaml_store"
        assert result is not None
        assert result.exists()
        assert result == store / "system" / "receipt" / "receipt.alibi.yaml"

        # Verify content
        with open(result) as f:
            data = yaml.safe_load(f)
        assert data["_meta"]["version"] == YAML_VERSION
        assert data["document_type"] == "receipt"
        assert data["vendor"] == "FRESKO"
        assert data["total"] == 25.99
        assert len(data["line_items"]) == 1

    def test_write_document_group(self, tmp_path: Path) -> None:
        folder = tmp_path / "invoice-acme"
        folder.mkdir()
        extracted = {"issuer": "ACME Corp", "amount": 100.0, "currency": "EUR"}

        result = write_yaml_cache(folder, extracted, "invoice", is_group=True)

        store = tmp_path / "yaml_store"
        assert result is not None
        assert result == store / "system" / "invoice" / "invoice-acme.alibi.yaml"
        assert result.exists()

    def test_write_handles_decimal(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()
        extracted = {"total": Decimal("25.99"), "currency": "EUR"}

        result = write_yaml_cache(source, extracted, "receipt")

        assert result is not None
        with open(result) as f:
            data = yaml.safe_load(f)
        assert data["total"] == 25.99

    def test_write_handles_date(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()
        extracted = {"date": date(2025, 12, 15), "currency": "EUR"}

        result = write_yaml_cache(source, extracted, "receipt")

        assert result is not None
        with open(result) as f:
            data = yaml.safe_load(f)
        assert data["date"] == "2025-12-15"

    def test_write_handles_datetime(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()
        extracted = {"extracted_at": datetime(2025, 12, 15, 14, 30, 0)}

        result = write_yaml_cache(source, extracted, "receipt")

        assert result is not None
        with open(result) as f:
            data = yaml.safe_load(f)
        assert "2025-12-15" in data["extracted_at"]

    def test_write_unicode(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()
        extracted = {
            "vendor": "Арабская Лавка",
            "line_items": [{"name": "Молоко 3.5%", "name_en": "Milk 3.5%"}],
        }

        result = write_yaml_cache(source, extracted, "receipt")

        assert result is not None
        with open(result, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["vendor"] == "Арабская Лавка"
        assert data["line_items"][0]["name"] == "Молоко 3.5%"

    def test_write_skips_document_type_in_extracted(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()
        extracted = {"document_type": "receipt", "vendor": "Test"}

        result = write_yaml_cache(source, extracted, "receipt")

        assert result is not None
        with open(result) as f:
            data = yaml.safe_load(f)
        # document_type comes from the parameter, not duplicated
        assert data["document_type"] == "receipt"
        assert data["vendor"] == "Test"

    def test_write_readonly_dir_returns_none(self, tmp_path: Path) -> None:
        # Point yaml_store_root to a non-writable path so write fails
        set_yaml_store_root(Path("/nonexistent/readonly/store"))
        source = Path("/nonexistent/dir/receipt.jpg")
        extracted = {"vendor": "Test"}

        result = write_yaml_cache(source, extracted, "receipt")

        assert result is None


class TestReadYamlCache:
    """Tests for reading YAML cache files."""

    def test_read_no_cache(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()

        result = read_yaml_cache(source)

        assert result is None

    def test_read_basic(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()

        # Write a cache file to the store path
        yaml_path = get_yaml_path(source)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "_meta": {"version": YAML_VERSION, "extracted_at": "2025-12-15T14:30:00"},
            "document_type": "receipt",
            "vendor": "FRESKO",
            "total": 25.99,
            "currency": "EUR",
        }
        with open(yaml_path, "w") as f:
            yaml.dump(cache_data, f)

        result = read_yaml_cache(source)

        assert result is not None
        assert result["document_type"] == "receipt"
        assert result["vendor"] == "FRESKO"
        assert result["total"] == 25.99
        assert "_meta" not in result  # Metadata stripped

    def test_read_document_group(self, tmp_path: Path) -> None:
        folder = tmp_path / "invoice-acme"
        folder.mkdir()

        yaml_path = get_yaml_path(folder, is_group=True)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "_meta": {"version": YAML_VERSION},
            "document_type": "invoice",
            "issuer": "ACME Corp",
        }
        with open(yaml_path, "w") as f:
            yaml.dump(cache_data, f)

        result = read_yaml_cache(folder, is_group=True)

        assert result is not None
        assert result["document_type"] == "invoice"
        assert result["issuer"] == "ACME Corp"

    def test_read_invalid_yaml(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()

        yaml_path = get_yaml_path(source)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text("{{invalid yaml: [unmatched", encoding="utf-8")

        result = read_yaml_cache(source)

        assert result is None

    def test_read_non_dict_yaml(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()

        yaml_path = get_yaml_path(source)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text("- just\n- a\n- list\n", encoding="utf-8")

        result = read_yaml_cache(source)

        assert result is None

    def test_read_future_version_rejected(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()

        yaml_path = get_yaml_path(source)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "_meta": {"version": 999},
            "vendor": "Test",
        }
        with open(yaml_path, "w") as f:
            yaml.dump(cache_data, f)

        result = read_yaml_cache(source)

        assert result is None

    def test_read_no_version_rejected(self, tmp_path: Path) -> None:
        """YAML without _meta.version is rejected — stale cache."""
        source = tmp_path / "receipt.jpg"
        source.touch()

        yaml_path = get_yaml_path(source)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "_meta": {},
            "document_type": "receipt",
            "vendor": "Test",
        }
        with open(yaml_path, "w") as f:
            yaml.dump(cache_data, f)

        result = read_yaml_cache(source)

        assert result is None

    def test_read_no_meta_rejected(self, tmp_path: Path) -> None:
        """YAML without _meta at all is rejected — stale cache."""
        source = tmp_path / "receipt.jpg"
        source.touch()

        yaml_path = get_yaml_path(source)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {"document_type": "receipt", "vendor": "Test", "total": 10.0}
        with open(yaml_path, "w") as f:
            yaml.dump(cache_data, f)

        result = read_yaml_cache(source)

        assert result is None

    def test_read_old_version_rejected(self, tmp_path: Path) -> None:
        """YAML with old version is rejected — re-extract needed."""
        source = tmp_path / "receipt.jpg"
        source.touch()

        yaml_path = get_yaml_path(source)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "_meta": {"version": 1},
            "document_type": "receipt",
            "vendor": "Test",
        }
        with open(yaml_path, "w") as f:
            yaml.dump(cache_data, f)

        result = read_yaml_cache(source)

        assert result is None


class TestRoundtrip:
    """Test write -> read roundtrip."""

    def test_roundtrip_preserves_data(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()

        original = {
            "vendor": "FRESKO",
            "vendor_address": "Kirchstr. 5, Berlin",
            "date": "2025-12-15",
            "time": "14:30",
            "total": 25.99,
            "currency": "EUR",
            "payment_method": "card",
            "card_last4": "1234",
            "language": "de",
            "line_items": [
                {
                    "name": "Vollmilch 3.5%",
                    "name_en": "Whole milk 3.5%",
                    "quantity": 2,
                    "unit_raw": "pcs",
                    "unit_price": 1.29,
                    "total_price": 2.58,
                    "tax_rate": 7,
                    "tax_type": "vat",
                    "category": "dairy",
                },
            ],
            "raw_text": "FRESKO\nKirchstr. 5\n...",
        }

        write_yaml_cache(source, original, "receipt")
        loaded = read_yaml_cache(source, doc_type="receipt")

        assert loaded is not None
        assert loaded["vendor"] == "FRESKO"
        assert loaded["vendor_address"] == "Kirchstr. 5, Berlin"
        assert loaded["total"] == 25.99
        assert loaded["currency"] == "EUR"
        assert loaded["document_type"] == "receipt"
        assert len(loaded["line_items"]) == 1
        assert loaded["line_items"][0]["name"] == "Vollmilch 3.5%"
        assert loaded["line_items"][0]["quantity"] == 2
        assert loaded["line_items"][0]["unit_price"] == 1.29

    def test_roundtrip_document_group(self, tmp_path: Path) -> None:
        folder = tmp_path / "invoice-acme"
        folder.mkdir()

        original = {
            "issuer": "ACME Corp",
            "issue_date": "2025-11-01",
            "amount": 500.0,
            "currency": "EUR",
            "line_items": [
                {"name": "Consulting", "quantity": 10, "unit_raw": "hr"},
            ],
        }

        write_yaml_cache(folder, original, "invoice", is_group=True)
        loaded = read_yaml_cache(folder, is_group=True, doc_type="invoice")

        assert loaded is not None
        assert loaded["issuer"] == "ACME Corp"
        assert loaded["amount"] == 500.0


class TestPipelineIntegration:
    """Test YAML cache integration with the processing pipeline."""

    @pytest.fixture(autouse=True)
    def mock_vision_detect(self) -> Any:
        """Mock vision detection to avoid LLM calls in tests."""
        with patch(
            "alibi.processing.pipeline.vision_detect_document_type",
            return_value="receipt",
        ):
            yield

    @pytest.fixture
    def pipeline(self, tmp_path: Path) -> Any:
        """Create a pipeline with in-memory database."""
        import os

        os.environ["ALIBI_TESTING"] = "1"

        from alibi.config import Config
        from alibi.db.connection import DatabaseManager
        from alibi.processing.pipeline import ProcessingPipeline

        config = Config(db_path=tmp_path / "test.db", _env_file=None)
        db = DatabaseManager(config)
        db.initialize()
        return ProcessingPipeline(db=db)

    def test_process_file_uses_yaml_cache(self, tmp_path: Path, pipeline: Any) -> None:
        """When .alibi.yaml exists, pipeline skips LLM and uses cached data."""
        # Create source image
        source = tmp_path / "receipt.jpg"
        source.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # Fake JPEG header

        # Create YAML cache in the store
        yaml_path = get_yaml_path(source, doc_type="receipt")
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "_meta": {"version": YAML_VERSION},
            "document_type": "receipt",
            "vendor": "Cached Vendor",
            "date": "2025-12-15",
            "total": 42.0,
            "currency": "EUR",
            "line_items": [{"name": "Cached Item", "total_price": 42.0}],
        }
        with open(yaml_path, "w") as f:
            yaml.dump(cache_data, f)

        # Process — should NOT call LLM
        with patch("alibi.processing.pipeline.extract_from_image") as mock_extract:
            result = pipeline.process_file(source)

        # LLM should not have been called
        mock_extract.assert_not_called()

        # Result should reflect cached data
        assert result.success
        assert not result.is_duplicate
        assert result.extracted_data is not None
        assert result.extracted_data["vendor"] == "Cached Vendor"

    def test_process_file_writes_yaml_after_extraction(
        self, tmp_path: Path, pipeline: Any
    ) -> None:
        """After fresh LLM extraction, pipeline writes .alibi.yaml."""
        source = tmp_path / "receipt.jpg"
        source.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        mock_extraction = {
            "vendor": "Fresh Vendor",
            "date": "2025-12-15",
            "total": 10.0,
            "currency": "EUR",
            "line_items": [],
        }

        with patch(
            "alibi.processing.pipeline.extract_from_image",
            return_value=mock_extraction,
        ):
            result = pipeline.process_file(source)

        assert result.success
        # Pipeline writes to the store; doc_type is "receipt"
        yaml_path = get_yaml_path(source, doc_type="receipt")
        assert yaml_path.exists()

        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        assert data["vendor"] == "Fresh Vendor"
        assert data["document_type"] == "receipt"

    def test_process_file_cache_overrides_doc_type(
        self, tmp_path: Path, pipeline: Any
    ) -> None:
        """YAML cache with document_type overrides file-extension-based detection."""
        source = tmp_path / "slip.jpg"  # Extension suggests receipt
        source.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        # Write to the store under payment_confirmation doc_type
        yaml_path = get_yaml_path(source, doc_type="payment_confirmation")
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "_meta": {"version": YAML_VERSION},
            "document_type": "payment_confirmation",  # Override to payment
            "vendor": "Test",
            "date": "2025-12-15",
            "total": 10.0,
            "currency": "EUR",
            "line_items": [],
        }
        with open(yaml_path, "w") as f:
            yaml.dump(cache_data, f)

        with patch("alibi.processing.pipeline.extract_from_image") as mock_extract:
            result = pipeline.process_file(source)

        mock_extract.assert_not_called()
        assert result.success

    def test_process_group_uses_yaml_cache(self, tmp_path: Path, pipeline: Any) -> None:
        """Document group uses .alibi.yaml from inside folder."""
        folder = tmp_path / "invoice-acme"
        folder.mkdir()
        page1 = folder / "page1.jpg"
        page1.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        page2 = folder / "page2.jpg"
        page2.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 200)

        yaml_path = get_yaml_path(folder, is_group=True, doc_type="invoice")
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "_meta": {"version": YAML_VERSION},
            "document_type": "invoice",
            "issuer": "ACME Corp",
            "amount": 500.0,
            "currency": "EUR",
            "line_items": [{"name": "Service", "total_price": 500.0}],
        }
        with open(yaml_path, "w") as f:
            yaml.dump(cache_data, f)

        with patch("alibi.processing.pipeline.extract_from_images") as mock_extract:
            result = pipeline.process_document_group(folder, [page1, page2])

        mock_extract.assert_not_called()
        assert result.success
        assert result.extracted_data is not None
        assert result.extracted_data["issuer"] == "ACME Corp"

    def test_process_group_writes_yaml_after_extraction(
        self, tmp_path: Path, pipeline: Any
    ) -> None:
        """After fresh extraction, document group writes .alibi.yaml."""
        folder = tmp_path / "invoice-acme"
        folder.mkdir()
        page1 = folder / "page1.jpg"
        page1.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        mock_extraction = {
            "issuer": "Fresh Corp",
            "amount": 200.0,
            "currency": "EUR",
            "line_items": [],
        }

        with patch(
            "alibi.processing.pipeline.extract_from_images",
            return_value=mock_extraction,
        ):
            result = pipeline.process_document_group(folder, [page1])

        assert result.success
        # Pipeline writes group YAML to the store (doc_type from vision detection)
        yaml_path = find_yaml_in_store(folder, is_group=True)
        assert yaml_path is not None
        assert yaml_path.exists()

        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        assert data["issuer"] == "Fresh Corp"

    def test_corrupted_yaml_falls_through_to_llm(
        self, tmp_path: Path, pipeline: Any
    ) -> None:
        """Corrupted .alibi.yaml doesn't block processing — falls through to LLM."""
        source = tmp_path / "receipt.jpg"
        source.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        # Write corrupted YAML to the store path
        yaml_path = get_yaml_path(source, doc_type="receipt")
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text("{{broken yaml: [", encoding="utf-8")

        mock_extraction = {
            "vendor": "LLM Vendor",
            "total": 5.0,
            "currency": "EUR",
            "line_items": [],
        }

        with patch(
            "alibi.processing.pipeline.extract_from_image",
            return_value=mock_extraction,
        ):
            result = pipeline.process_file(source)

        assert result.success
        assert result.extracted_data is not None
        assert result.extracted_data["vendor"] == "LLM Vendor"
