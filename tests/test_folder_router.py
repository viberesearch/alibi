"""Tests for folder-based document routing."""

import os
from pathlib import Path

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.db.models import DocumentType
from alibi.processing.folder_router import (
    SKIP_FILES,
    FolderContext,
    _FOLDER_TYPE_MAP,
    is_country_code,
    resolve_folder_context,
    scan_inbox_recursive,
)


class TestIsCountryCode:
    """Tests for ISO 3166-1 alpha-2 detection."""

    def test_valid_two_letter_uppercase(self) -> None:
        assert is_country_code("CY") is True

    def test_valid_gr(self) -> None:
        assert is_country_code("GR") is True

    def test_valid_de(self) -> None:
        assert is_country_code("DE") is True

    def test_valid_us(self) -> None:
        assert is_country_code("US") is True

    def test_lowercase_rejected(self) -> None:
        assert is_country_code("cy") is False

    def test_three_letter_rejected(self) -> None:
        assert is_country_code("USA") is False

    def test_single_letter_rejected(self) -> None:
        assert is_country_code("A") is False

    def test_alphanumeric_rejected(self) -> None:
        assert is_country_code("A1") is False

    def test_numeric_rejected(self) -> None:
        assert is_country_code("12") is False

    def test_mixed_case_rejected(self) -> None:
        assert is_country_code("Cy") is False

    def test_empty_string_rejected(self) -> None:
        assert is_country_code("") is False


class TestFolderTypeMap:
    """Verify the folder name to DocumentType mapping."""

    def test_receipts(self) -> None:
        assert _FOLDER_TYPE_MAP["receipts"] is DocumentType.RECEIPT

    def test_invoices(self) -> None:
        assert _FOLDER_TYPE_MAP["invoices"] is DocumentType.INVOICE

    def test_payments(self) -> None:
        assert _FOLDER_TYPE_MAP["payments"] is DocumentType.PAYMENT_CONFIRMATION

    def test_statements(self) -> None:
        assert _FOLDER_TYPE_MAP["statements"] is DocumentType.STATEMENT

    def test_warranties(self) -> None:
        assert _FOLDER_TYPE_MAP["warranties"] is DocumentType.WARRANTY

    def test_contracts(self) -> None:
        assert _FOLDER_TYPE_MAP["contracts"] is DocumentType.CONTRACT

    def test_unsorted_not_mapped(self) -> None:
        assert "unsorted" not in _FOLDER_TYPE_MAP


class TestResolveFolderContext:
    """Tests for resolve_folder_context()."""

    def _make_file(self, tmp_path: Path, *parts: str) -> Path:
        """Create a file at inbox/parts... and return its path."""
        p = tmp_path
        for part in parts:
            p = p / part
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"test")
        return p

    def test_receipts_folder(self, tmp_path: Path) -> None:
        f = self._make_file(tmp_path, "receipts", "scan.jpg")
        ctx = resolve_folder_context(f, tmp_path)
        assert ctx.doc_type is DocumentType.RECEIPT

    def test_invoices_folder(self, tmp_path: Path) -> None:
        f = self._make_file(tmp_path, "invoices", "inv.pdf")
        ctx = resolve_folder_context(f, tmp_path)
        assert ctx.doc_type is DocumentType.INVOICE

    def test_payments_folder(self, tmp_path: Path) -> None:
        f = self._make_file(tmp_path, "payments", "pay.jpg")
        ctx = resolve_folder_context(f, tmp_path)
        assert ctx.doc_type is DocumentType.PAYMENT_CONFIRMATION

    def test_statements_folder(self, tmp_path: Path) -> None:
        f = self._make_file(tmp_path, "statements", "stmt.pdf")
        ctx = resolve_folder_context(f, tmp_path)
        assert ctx.doc_type is DocumentType.STATEMENT

    def test_warranties_folder(self, tmp_path: Path) -> None:
        f = self._make_file(tmp_path, "warranties", "warranty.pdf")
        ctx = resolve_folder_context(f, tmp_path)
        assert ctx.doc_type is DocumentType.WARRANTY

    def test_contracts_folder(self, tmp_path: Path) -> None:
        f = self._make_file(tmp_path, "contracts", "contract.pdf")
        ctx = resolve_folder_context(f, tmp_path)
        assert ctx.doc_type is DocumentType.CONTRACT

    def test_unsorted_folder(self, tmp_path: Path) -> None:
        f = self._make_file(tmp_path, "unsorted", "unknown.jpg")
        ctx = resolve_folder_context(f, tmp_path)
        assert ctx.doc_type is None

    def test_file_directly_in_inbox_root(self, tmp_path: Path) -> None:
        f = self._make_file(tmp_path, "loose.jpg")
        ctx = resolve_folder_context(f, tmp_path)
        assert ctx.doc_type is None
        assert ctx.vendor_hint is None
        # Country falls back to InboxConfig default ("CY") when no config file
        assert ctx.country == "CY"

    def test_vendor_hint_under_type_folder(self, tmp_path: Path) -> None:
        f = self._make_file(tmp_path, "receipts", "fresko", "scan.jpg")
        ctx = resolve_folder_context(f, tmp_path)
        assert ctx.doc_type is DocumentType.RECEIPT
        assert ctx.vendor_hint == "fresko"

    def test_country_under_type_folder(self, tmp_path: Path) -> None:
        f = self._make_file(tmp_path, "receipts", "GR", "scan.jpg")
        ctx = resolve_folder_context(f, tmp_path)
        assert ctx.doc_type is DocumentType.RECEIPT
        assert ctx.country == "GR"
        assert ctx.vendor_hint is None  # GR is a country, not a vendor

    def test_country_and_vendor(self, tmp_path: Path) -> None:
        f = self._make_file(tmp_path, "receipts", "GR", "taverna", "scan.jpg")
        ctx = resolve_folder_context(f, tmp_path)
        assert ctx.doc_type is DocumentType.RECEIPT
        assert ctx.country == "GR"
        assert ctx.vendor_hint == "taverna"

    def test_file_outside_inbox_root(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        outside = tmp_path / "other" / "scan.jpg"
        outside.parent.mkdir(parents=True)
        outside.write_bytes(b"test")
        ctx = resolve_folder_context(outside, inbox)
        assert ctx.doc_type is None
        assert ctx.country is None
        assert ctx.vendor_hint is None
        assert ctx.inbox_config is None
        assert ctx.vendor_config is None

    def test_vendor_hint_not_set_for_type_folder(self, tmp_path: Path) -> None:
        """The immediate parent 'receipts' should not become a vendor hint."""
        f = self._make_file(tmp_path, "receipts", "scan.jpg")
        ctx = resolve_folder_context(f, tmp_path)
        assert ctx.vendor_hint is None

    def test_unsorted_subfolder_with_vendor(self, tmp_path: Path) -> None:
        """File under unsorted/vendor should have no doc_type but has vendor hint."""
        f = self._make_file(tmp_path, "unsorted", "acme", "doc.jpg")
        ctx = resolve_folder_context(f, tmp_path)
        assert ctx.doc_type is None
        assert ctx.vendor_hint == "acme"

    def test_inbox_config_loaded(self, tmp_path: Path) -> None:
        """Config file at inbox root is loaded into context."""
        import yaml

        config = {"default_country": "DE", "default_currency": "EUR"}
        (tmp_path / "_config.yaml").write_text(yaml.dump(config))
        f = self._make_file(tmp_path, "receipts", "scan.jpg")
        ctx = resolve_folder_context(f, tmp_path)
        assert ctx.inbox_config is not None
        assert ctx.inbox_config.default_country == "DE"

    def test_vendor_config_loaded(self, tmp_path: Path) -> None:
        """Vendor config file is loaded when file is in a vendor folder."""
        import yaml

        vendor_dir = tmp_path / "receipts" / "fresko"
        vendor_dir.mkdir(parents=True)
        vendor_cfg = {"trade_name": "FreSko", "vat_number": "HE123456"}
        (vendor_dir / "_vendor.yaml").write_text(yaml.dump(vendor_cfg))
        f = vendor_dir / "scan.jpg"
        f.write_bytes(b"test")
        ctx = resolve_folder_context(f, tmp_path)
        assert ctx.vendor_config is not None
        assert ctx.vendor_config.trade_name == "FreSko"
        assert ctx.vendor_config.vat_number == "HE123456"

    def test_no_vendor_config_when_no_vendor_folder(self, tmp_path: Path) -> None:
        f = self._make_file(tmp_path, "receipts", "scan.jpg")
        ctx = resolve_folder_context(f, tmp_path)
        assert ctx.vendor_config is None

    def test_default_country_from_config(self, tmp_path: Path) -> None:
        """When no country folder exists, default_country comes from config."""
        import yaml

        config = {"default_country": "CY"}
        (tmp_path / "_config.yaml").write_text(yaml.dump(config))
        f = self._make_file(tmp_path, "receipts", "scan.jpg")
        ctx = resolve_folder_context(f, tmp_path)
        assert ctx.country == "CY"

    def test_country_folder_overrides_config_default(self, tmp_path: Path) -> None:
        """Explicit country folder takes precedence over config default."""
        import yaml

        config = {"default_country": "CY"}
        (tmp_path / "_config.yaml").write_text(yaml.dump(config))
        f = self._make_file(tmp_path, "receipts", "DE", "scan.jpg")
        ctx = resolve_folder_context(f, tmp_path)
        assert ctx.country == "DE"


class TestScanInboxRecursive:
    """Tests for scan_inbox_recursive()."""

    def _populate_inbox(self, inbox: Path) -> None:
        """Create a realistic inbox structure."""
        # Supported files
        (inbox / "receipts").mkdir(parents=True)
        (inbox / "receipts" / "scan1.jpg").write_bytes(b"img")
        (inbox / "receipts" / "scan2.pdf").write_bytes(b"pdf")

        (inbox / "invoices").mkdir()
        (inbox / "invoices" / "inv.pdf").write_bytes(b"pdf")

        (inbox / "receipts" / "fresko").mkdir()
        (inbox / "receipts" / "fresko" / "receipt.png").write_bytes(b"img")

    def test_finds_supported_files(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        self._populate_inbox(inbox)
        results = scan_inbox_recursive(inbox)
        paths = [r[0] for r in results]
        assert len(paths) == 4
        extensions = {p.suffix for p in paths}
        assert extensions <= {".jpg", ".pdf", ".png"}

    def test_skips_config_files(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        (inbox / "receipts").mkdir()
        (inbox / "receipts" / "scan.jpg").write_bytes(b"img")
        # Create files that should be skipped
        (inbox / "_config.yaml").write_text("default_country: CY")
        (inbox / "receipts" / "_vendor.yaml").write_text("trade_name: Test")
        (inbox / "receipts" / "scan.alibi.yaml").write_text("version: 3")
        (inbox / ".DS_Store").write_bytes(b"junk")
        (inbox / ".gitkeep").write_bytes(b"")

        results = scan_inbox_recursive(inbox)
        filenames = {r[0].name for r in results}
        assert "_config.yaml" not in filenames
        assert "_vendor.yaml" not in filenames
        assert "scan.alibi.yaml" not in filenames
        assert ".DS_Store" not in filenames
        assert ".gitkeep" not in filenames

    def test_skips_unsupported_extensions(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        (inbox / "receipts").mkdir()
        (inbox / "receipts" / "notes.txt").write_text("some notes")
        (inbox / "receipts" / "data.csv").write_text("a,b,c")
        (inbox / "receipts" / "scan.jpg").write_bytes(b"img")

        results = scan_inbox_recursive(inbox)
        filenames = {r[0].name for r in results}
        assert "notes.txt" not in filenames
        assert "data.csv" not in filenames
        assert "scan.jpg" in filenames

    def test_results_include_correct_context(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox"
        (inbox / "receipts" / "fresko").mkdir(parents=True)
        (inbox / "invoices").mkdir()
        (inbox / "receipts" / "fresko" / "scan.jpg").write_bytes(b"img")
        (inbox / "invoices" / "inv.pdf").write_bytes(b"pdf")

        results = scan_inbox_recursive(inbox)
        ctx_map = {r[0].name: r[1] for r in results}

        assert ctx_map["scan.jpg"].doc_type is DocumentType.RECEIPT
        assert ctx_map["scan.jpg"].vendor_hint == "fresko"
        assert ctx_map["inv.pdf"].doc_type is DocumentType.INVOICE

    def test_empty_inbox(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        results = scan_inbox_recursive(inbox)
        assert results == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        results = scan_inbox_recursive(tmp_path / "nonexistent")
        assert results == []

    def test_results_sorted_by_path(self, tmp_path: Path) -> None:
        inbox = tmp_path / "inbox"
        (inbox / "receipts").mkdir(parents=True)
        (inbox / "invoices").mkdir()
        (inbox / "receipts" / "b.jpg").write_bytes(b"img")
        (inbox / "receipts" / "a.jpg").write_bytes(b"img")
        (inbox / "invoices" / "c.pdf").write_bytes(b"pdf")

        results = scan_inbox_recursive(inbox)
        paths = [r[0] for r in results]
        assert paths == sorted(paths)


class TestSkipFiles:
    """Verify the SKIP_FILES set."""

    def test_contains_config_yaml(self) -> None:
        assert "_config.yaml" in SKIP_FILES

    def test_contains_vendor_yaml(self) -> None:
        assert "_vendor.yaml" in SKIP_FILES

    def test_contains_ds_store(self) -> None:
        assert ".DS_Store" in SKIP_FILES

    def test_contains_gitkeep(self) -> None:
        assert ".gitkeep" in SKIP_FILES
