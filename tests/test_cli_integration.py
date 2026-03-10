"""CLI integration tests using click.testing.CliRunner."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from alibi.cli import cli
from alibi.db.connection import DatabaseManager


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestDbInfo:
    """Tests for `lt db info`."""

    def test_lt_db_info(self, runner: CliRunner, db_manager: DatabaseManager) -> None:
        """lt db info returns schema version and row counts."""
        with patch("alibi.commands.maintenance.get_db", return_value=db_manager):
            result = runner.invoke(cli, ["db", "info"])

        assert result.exit_code == 0
        assert "Schema version" in result.output
        assert "documents" in result.output
        assert "facts" in result.output

    def test_lt_db_info_uninitialized(self, runner: CliRunner) -> None:
        """lt db info warns when DB is not initialized."""
        mock_db = MagicMock(spec=DatabaseManager)
        mock_db.is_initialized.return_value = False

        with patch("alibi.commands.maintenance.get_db", return_value=mock_db):
            result = runner.invoke(cli, ["db", "info"])

        assert result.exit_code == 0
        assert "not initialized" in result.output.lower()


class TestTransactionsList:
    """Tests for `lt transactions`."""

    def test_lt_transactions_list(
        self, runner: CliRunner, db_manager: DatabaseManager
    ) -> None:
        """lt transactions lists v1 transactions (empty = no error)."""
        with patch("alibi.commands.misc.get_db", return_value=db_manager):
            result = runner.invoke(cli, ["transactions"])

        assert result.exit_code == 0


class TestFactsList:
    """Tests for `lt facts list`."""

    def test_lt_facts_list(
        self, runner: CliRunner, db_manager: DatabaseManager
    ) -> None:
        """lt facts list returns v2 facts (empty = no error)."""
        with patch("alibi.commands.facts.get_db", return_value=db_manager):
            result = runner.invoke(cli, ["facts", "list"])

        assert result.exit_code == 0
        # Either shows facts or "No facts found"
        assert "facts" in result.output.lower() or "No facts" in result.output


class TestProcessDryRun:
    """Tests for `lt process -n` (dry run)."""

    def test_lt_process_dry_run(
        self, runner: CliRunner, db_manager: DatabaseManager, tmp_path: Path
    ) -> None:
        """lt process -n runs without modifying DB."""
        # Create a dummy inbox directory
        inbox = tmp_path / "inbox"
        inbox.mkdir()

        with (
            patch("alibi.commands.process.get_db", return_value=db_manager),
            patch("alibi.commands.process.get_config") as mock_config,
        ):
            mock_config.return_value.inbox_path = str(inbox)
            result = runner.invoke(cli, ["process", "-n"])

        # Should complete without error
        assert result.exit_code == 0
