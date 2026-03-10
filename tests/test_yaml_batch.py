"""Tests for batch YAML CLI commands: lt yaml list / set-field / rename-vendor."""

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from alibi.cli import cli
from alibi.extraction.yaml_cache import YAML_SUFFIX, YAML_VERSION, get_yaml_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, data: dict) -> Path:
    """Write a minimal valid .alibi.yaml in the yaml store for *path*.

    *path* is the source document path (e.g. ``receipt.jpg``).  The YAML
    is placed in the yaml_store tree via ``get_yaml_path``.
    """
    path.touch()
    yaml_path = get_yaml_path(path)
    # Build payload: custom _meta or default
    if "_meta" in data:
        meta = data.pop("_meta")
    else:
        meta = {
            "version": YAML_VERSION,
            "extracted_at": "2024-01-01T00:00:00",
            "source": str(path),
        }
    meta.setdefault("source_path", str(path.resolve()))
    meta.setdefault("is_group", False)
    payload = {"_meta": meta}
    payload.update(data)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        yaml.dump(payload, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _read_yaml(source: Path) -> dict:
    yaml_path = get_yaml_path(source)
    with open(yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# lt yaml list
# ---------------------------------------------------------------------------


class TestYamlList:
    """Tests for `lt yaml list`."""

    def test_lists_yaml_files(self, runner: CliRunner, tmp_path: Path) -> None:
        _write_yaml(
            tmp_path / "receipt.jpg",
            {
                "document_type": "receipt",
                "vendor": "ACME Corp",
                "total": 42.0,
                "currency": "EUR",
            },
        )
        result = runner.invoke(cli, ["yaml", "list", "--inbox", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "ACME Corp" in result.output
        assert "receipt" in result.output

    def test_shows_confidence(self, runner: CliRunner, tmp_path: Path) -> None:
        _write_yaml(
            tmp_path / "receipt.jpg",
            {
                "document_type": "receipt",
                "vendor": "Shop",
                "_meta": {
                    "version": YAML_VERSION,
                    "extracted_at": "2024-01-01T00:00:00",
                    "source": str(tmp_path / "receipt.jpg"),
                    "confidence": 0.85,
                },
            },
        )
        _write_yaml(
            tmp_path / "receipt2.jpg",
            {
                "document_type": "receipt",
                "vendor": "HighConf Shop",
                "total": 10.0,
                "currency": "EUR",
                "_meta": {
                    "version": YAML_VERSION,
                    "extracted_at": "2024-01-01T00:00:00",
                    "source": str(tmp_path / "receipt2.jpg"),
                    "confidence": 0.85,
                    "needs_review": True,
                },
            },
        )

        result = runner.invoke(cli, ["yaml", "list", "--inbox", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "0.85" in result.output
        assert "yes" in result.output  # needs_review

    def test_filter_by_type(self, runner: CliRunner, tmp_path: Path) -> None:
        _write_yaml(
            tmp_path / "receipt.jpg",
            {"document_type": "receipt", "vendor": "Shop A", "total": 10.0},
        )
        _write_yaml(
            tmp_path / "invoice.pdf",
            {"document_type": "invoice", "issuer": "Vendor B", "amount": 99.0},
        )
        result = runner.invoke(
            cli, ["yaml", "list", "--inbox", str(tmp_path), "--type", "receipt"]
        )
        assert result.exit_code == 0, result.output
        assert "Shop A" in result.output
        assert "Vendor B" not in result.output

    def test_filter_by_vendor(self, runner: CliRunner, tmp_path: Path) -> None:
        _write_yaml(
            tmp_path / "receipt1.jpg",
            {"document_type": "receipt", "vendor": "ACME Corp", "total": 10.0},
        )
        _write_yaml(
            tmp_path / "receipt2.jpg",
            {"document_type": "receipt", "vendor": "Other Shop", "total": 5.0},
        )
        result = runner.invoke(
            cli, ["yaml", "list", "--inbox", str(tmp_path), "--vendor", "acme"]
        )
        assert result.exit_code == 0, result.output
        assert "ACME Corp" in result.output
        assert "Other Shop" not in result.output

    def test_empty_inbox(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(cli, ["yaml", "list", "--inbox", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "No .alibi.yaml" in result.output

    def test_invoice_uses_issuer_field(self, runner: CliRunner, tmp_path: Path) -> None:
        _write_yaml(
            tmp_path / "invoice.pdf",
            {"document_type": "invoice", "issuer": "BigCo Ltd", "amount": 500.0},
        )
        result = runner.invoke(cli, ["yaml", "list", "--inbox", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "BigCo Ltd" in result.output

    def test_no_inbox_configured(self, runner: CliRunner) -> None:
        """No inbox configured and no --inbox flag should abort."""
        from unittest.mock import MagicMock, patch

        mock_config = MagicMock()
        mock_config.get_inbox_path.return_value = None

        with patch("alibi.commands.yaml_ops.get_config", return_value=mock_config):
            result = runner.invoke(cli, ["yaml", "list"])

        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# lt yaml set-field
# ---------------------------------------------------------------------------


class TestYamlSetField:
    """Tests for `lt yaml set-field`."""

    def test_sets_string_field(self, runner: CliRunner, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        _write_yaml(
            source, {"document_type": "receipt", "vendor": "OldName", "total": 10.0}
        )

        result = runner.invoke(
            cli, ["yaml", "set-field", str(source), "vendor", "NewName"]
        )
        assert result.exit_code == 0, result.output
        assert "Updated" in result.output

        data = _read_yaml(source)
        assert data["vendor"] == "NewName"

    def test_sets_numeric_field_as_float(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        source = tmp_path / "receipt.jpg"
        _write_yaml(
            source, {"document_type": "receipt", "vendor": "Shop", "total": 10.0}
        )

        result = runner.invoke(
            cli, ["yaml", "set-field", str(source), "total", "99.99"]
        )
        assert result.exit_code == 0, result.output

        data = _read_yaml(source)
        assert abs(float(data["total"]) - 99.99) < 0.001

    def test_no_change_when_same_value(self, runner: CliRunner, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        _write_yaml(
            source, {"document_type": "receipt", "vendor": "Same", "total": 10.0}
        )

        result = runner.invoke(
            cli, ["yaml", "set-field", str(source), "vendor", "Same"]
        )
        assert result.exit_code == 0, result.output
        assert "No change" in result.output

    def test_missing_yaml_aborts(self, runner: CliRunner, tmp_path: Path) -> None:
        source = tmp_path / "nonexistent.jpg"
        # Do NOT create source or yaml — no YAML sidecar
        result = runner.invoke(cli, ["yaml", "set-field", str(source), "vendor", "X"])
        assert result.exit_code != 0

    def test_sets_integer_field(self, runner: CliRunner, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        _write_yaml(
            source, {"document_type": "receipt", "vendor": "Shop", "total": 10.0}
        )

        result = runner.invoke(cli, ["yaml", "set-field", str(source), "total", "42"])
        assert result.exit_code == 0, result.output

        data = _read_yaml(source)
        assert data["total"] == 42


# ---------------------------------------------------------------------------
# lt yaml rename-vendor
# ---------------------------------------------------------------------------


class TestYamlRenameVendor:
    """Tests for `lt yaml rename-vendor`."""

    def test_renames_matching_vendors(self, runner: CliRunner, tmp_path: Path) -> None:
        _write_yaml(
            tmp_path / "r1.jpg",
            {"document_type": "receipt", "vendor": "acme market", "total": 10.0},
        )
        _write_yaml(
            tmp_path / "r2.jpg",
            {"document_type": "receipt", "vendor": "Other Shop", "total": 5.0},
        )

        result = runner.invoke(
            cli,
            ["yaml", "rename-vendor", str(tmp_path), "acme", "ACME Corp"],
        )
        assert result.exit_code == 0, result.output
        assert "1" in result.output  # "Renamed 1 file(s)"

        data1 = _read_yaml(tmp_path / "r1.jpg")
        data2 = _read_yaml(tmp_path / "r2.jpg")
        assert data1["vendor"] == "ACME Corp"
        assert data2["vendor"] == "Other Shop"  # unchanged

    def test_case_insensitive_match(self, runner: CliRunner, tmp_path: Path) -> None:
        _write_yaml(
            tmp_path / "r1.jpg",
            {"document_type": "receipt", "vendor": "FRESKO HYPERMARKET", "total": 30.0},
        )

        result = runner.invoke(
            cli,
            ["yaml", "rename-vendor", str(tmp_path), "fresko", "Fresko"],
        )
        assert result.exit_code == 0, result.output
        data = _read_yaml(tmp_path / "r1.jpg")
        assert data["vendor"] == "Fresko"

    def test_dry_run_shows_matches_but_no_change(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        _write_yaml(
            tmp_path / "r1.jpg",
            {"document_type": "receipt", "vendor": "old name", "total": 10.0},
        )

        result = runner.invoke(
            cli,
            [
                "yaml",
                "rename-vendor",
                str(tmp_path),
                "old name",
                "New Name",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Dry run" in result.output
        assert "New Name" in result.output

        # No change applied
        data = _read_yaml(tmp_path / "r1.jpg")
        assert data["vendor"] == "old name"

    def test_no_matches_shows_message(self, runner: CliRunner, tmp_path: Path) -> None:
        _write_yaml(
            tmp_path / "r1.jpg",
            {"document_type": "receipt", "vendor": "Totally Different", "total": 10.0},
        )

        result = runner.invoke(
            cli,
            ["yaml", "rename-vendor", str(tmp_path), "nonexistent", "X"],
        )
        assert result.exit_code == 0, result.output
        assert "No YAML files found" in result.output

    def test_invoice_uses_issuer_field(self, runner: CliRunner, tmp_path: Path) -> None:
        _write_yaml(
            tmp_path / "inv1.pdf",
            {"document_type": "invoice", "issuer": "Old Issuer Ltd", "amount": 100.0},
        )

        result = runner.invoke(
            cli,
            ["yaml", "rename-vendor", str(tmp_path), "old issuer", "New Issuer Ltd"],
        )
        assert result.exit_code == 0, result.output
        data = _read_yaml(tmp_path / "inv1.pdf")
        assert data["issuer"] == "New Issuer Ltd"

    def test_renames_multiple_files(self, runner: CliRunner, tmp_path: Path) -> None:
        for i in range(3):
            _write_yaml(
                tmp_path / f"r{i}.jpg",
                {"document_type": "receipt", "vendor": "ACME store", "total": float(i)},
            )

        result = runner.invoke(
            cli,
            ["yaml", "rename-vendor", str(tmp_path), "acme", "ACME Corp"],
        )
        assert result.exit_code == 0, result.output
        assert "3" in result.output

        for i in range(3):
            data = _read_yaml(tmp_path / f"r{i}.jpg")
            assert data["vendor"] == "ACME Corp"

    def test_rename_in_subdirectories(self, runner: CliRunner, tmp_path: Path) -> None:
        subdir = tmp_path / "receipts"
        subdir.mkdir()
        _write_yaml(
            subdir / "r1.jpg",
            {"document_type": "receipt", "vendor": "nested vendor", "total": 5.0},
        )

        result = runner.invoke(
            cli,
            ["yaml", "rename-vendor", str(tmp_path), "nested vendor", "Renamed Vendor"],
        )
        assert result.exit_code == 0, result.output
        data = _read_yaml(subdir / "r1.jpg")
        assert data["vendor"] == "Renamed Vendor"
