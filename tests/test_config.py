"""Tests for configuration management."""

import os
from pathlib import Path

import pytest

from alibi.config import Config, get_config, reset_config, get_project_root


class TestConfigDefaults:
    """Tests for default configuration values."""

    def test_config_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that config has correct default values."""
        # Clear any env vars that would override defaults
        for key in list(os.environ):
            if key.startswith("ALIBI_") or key == "TELEGRAM_BOT_TOKEN":
                monkeypatch.delenv(key, raising=False)

        config = Config(_env_file=None)  # type: ignore[call-arg]

        # Database
        assert config.db_path == Path("data/alibi.db")

        # Obsidian vault
        assert config.vault_path is None
        assert config.inbox_folder == "inbox/documents"

        # Ollama settings
        assert config.ollama_url == "http://127.0.0.1:11434"
        assert config.ollama_model == "qwen3-vl:30b"

        # Processing
        assert config.auto_process is False

        # Default currency
        assert config.default_currency == "EUR"

        # Telegram bot
        assert config.telegram_token == ""

        # LanceDB
        assert config.lance_path is None


class TestConfigEnvironmentLoading:
    """Tests for loading configuration from environment variables."""

    def test_config_loads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that config loads from ALIBI_ prefixed environment variables."""
        # Set testing mode to prevent .env file loading
        monkeypatch.setenv("ALIBI_TESTING", "1")

        # Set various config values via env vars
        monkeypatch.setenv("ALIBI_DB_PATH", "/tmp/test.db")
        monkeypatch.setenv("ALIBI_VAULT_PATH", "/tmp/vault")
        monkeypatch.setenv("ALIBI_INBOX_FOLDER", "custom/inbox")
        monkeypatch.setenv("ALIBI_OLLAMA_URL", "http://localhost:11434")
        monkeypatch.setenv("ALIBI_OLLAMA_MODEL", "llama2")
        monkeypatch.setenv("ALIBI_AUTO_PROCESS", "true")
        monkeypatch.setenv("ALIBI_DEFAULT_CURRENCY", "USD")
        monkeypatch.setenv("ALIBI_LANCE_PATH", "/tmp/lance")

        config = Config()

        assert config.db_path == Path("/tmp/test.db")
        assert config.vault_path == Path("/tmp/vault")
        assert config.inbox_folder == "custom/inbox"
        assert config.ollama_url == "http://localhost:11434"
        assert config.ollama_model == "llama2"
        assert config.auto_process is True
        assert config.default_currency == "USD"
        assert config.lance_path == Path("/tmp/lance")

    def test_config_ignores_unknown_env_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that config ignores unknown environment variables (extra='ignore')."""
        monkeypatch.setenv("ALIBI_TESTING", "1")
        monkeypatch.setenv("ALIBI_UNKNOWN_SETTING", "value")
        monkeypatch.setenv("ALIBI_ANOTHER_UNKNOWN", "123")

        # Should not raise an error
        config = Config()
        assert config is not None

        # Unknown fields should not be set
        assert not hasattr(config, "unknown_setting")
        assert not hasattr(config, "another_unknown")

    def test_config_telegram_token_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that TELEGRAM_BOT_TOKEN works without ALIBI_ prefix."""
        monkeypatch.setenv("ALIBI_TESTING", "1")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-12345")

        config = Config()
        assert config.telegram_token == "test-token-12345"


class TestConfigSingleton:
    """Tests for config singleton pattern."""

    def test_get_config_returns_singleton(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that get_config returns the same instance."""
        monkeypatch.setenv("ALIBI_TESTING", "1")

        # Reset to ensure clean state
        reset_config()

        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_config_reset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that reset_config clears the singleton."""
        monkeypatch.setenv("ALIBI_TESTING", "1")

        # Get initial config
        config1 = get_config()

        # Reset
        reset_config()

        # Get new config
        config2 = get_config()

        # Should be different instances
        assert config1 is not config2

    def test_config_reset_reloads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that reset_config allows reloading from changed environment."""
        monkeypatch.setenv("ALIBI_TESTING", "1")

        # Get initial config with default currency
        reset_config()
        config1 = get_config()
        assert config1.default_currency == "EUR"

        # Change environment and reset
        monkeypatch.setenv("ALIBI_DEFAULT_CURRENCY", "USD")
        reset_config()

        # New config should have new value
        config2 = get_config()
        assert config2.default_currency == "USD"


class TestConfigPathResolution:
    """Tests for path resolution methods."""

    def test_get_absolute_db_path_with_relative_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that relative db_path is resolved relative to project root."""
        monkeypatch.setenv("ALIBI_TESTING", "1")
        monkeypatch.setenv("ALIBI_DB_PATH", "data/test.db")

        config = Config()
        absolute_path = config.get_absolute_db_path()

        project_root = get_project_root()
        expected_path = project_root / "data" / "test.db"

        assert absolute_path == expected_path
        assert absolute_path.is_absolute()

    def test_get_absolute_db_path_with_absolute_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test that absolute db_path is returned unchanged."""
        monkeypatch.setenv("ALIBI_TESTING", "1")

        absolute_db = tmp_path / "absolute.db"
        monkeypatch.setenv("ALIBI_DB_PATH", str(absolute_db))

        config = Config()
        result = config.get_absolute_db_path()

        assert result == absolute_db
        assert result.is_absolute()

    def test_get_inbox_path_with_vault_set(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test that inbox path is resolved relative to vault."""
        monkeypatch.setenv("ALIBI_TESTING", "1")

        vault = tmp_path / "vault"
        vault.mkdir()

        monkeypatch.setenv("ALIBI_VAULT_PATH", str(vault))
        monkeypatch.setenv("ALIBI_INBOX_FOLDER", "custom/inbox")

        config = Config()
        inbox_path = config.get_inbox_path()

        assert inbox_path == vault / "custom" / "inbox"

    def test_get_inbox_path_with_no_vault(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that inbox path is None when vault is not set."""
        monkeypatch.setenv("ALIBI_TESTING", "1")

        config = Config()
        inbox_path = config.get_inbox_path()

        assert inbox_path is None

    def test_get_lance_path_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test default lance path resolution."""
        monkeypatch.setenv("ALIBI_TESTING", "1")

        config = Config()
        lance_path = config.get_lance_path()

        project_root = get_project_root()
        expected_path = project_root / "data" / "lancedb"

        assert lance_path == expected_path
        assert lance_path.is_absolute()

    def test_get_lance_path_with_relative_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test lance path resolution with relative path."""
        monkeypatch.setenv("ALIBI_TESTING", "1")
        monkeypatch.setenv("ALIBI_LANCE_PATH", "custom/lance")

        config = Config()
        lance_path = config.get_lance_path()

        project_root = get_project_root()
        expected_path = project_root / "custom" / "lance"

        assert lance_path == expected_path
        assert lance_path.is_absolute()

    def test_get_lance_path_with_absolute_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test lance path with absolute path."""
        monkeypatch.setenv("ALIBI_TESTING", "1")

        absolute_lance = tmp_path / "lancedb"
        monkeypatch.setenv("ALIBI_LANCE_PATH", str(absolute_lance))

        config = Config()
        result = config.get_lance_path()

        assert result == absolute_lance
        assert result.is_absolute()


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_validate_paths_with_valid_vault(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test that validate_paths returns no errors for valid vault."""
        monkeypatch.setenv("ALIBI_TESTING", "1")

        vault = tmp_path / "vault"
        vault.mkdir()

        monkeypatch.setenv("ALIBI_VAULT_PATH", str(vault))

        config = Config()
        errors = config.validate_paths()

        assert errors == []

    def test_validate_paths_with_missing_vault(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test that validate_paths returns error for missing vault."""
        monkeypatch.setenv("ALIBI_TESTING", "1")

        vault = tmp_path / "nonexistent_vault"
        monkeypatch.setenv("ALIBI_VAULT_PATH", str(vault))

        config = Config()
        errors = config.validate_paths()

        assert len(errors) == 1
        assert "Vault path does not exist" in errors[0]
        assert str(vault) in errors[0]

    def test_validate_paths_with_no_vault(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that validate_paths returns no errors when vault is not set."""
        monkeypatch.setenv("ALIBI_TESTING", "1")

        config = Config()
        errors = config.validate_paths()

        assert errors == []


class TestProjectRoot:
    """Tests for project root detection."""

    def test_get_project_root_is_absolute(self) -> None:
        """Test that project root is an absolute path."""
        root = get_project_root()
        assert root.is_absolute()

    def test_get_project_root_exists(self) -> None:
        """Test that project root exists."""
        root = get_project_root()
        assert root.exists()
        assert root.is_dir()

    def test_get_project_root_contains_alibi_package(self) -> None:
        """Test that project root contains the alibi package."""
        root = get_project_root()
        alibi_dir = root / "alibi"
        assert alibi_dir.exists()
        assert alibi_dir.is_dir()
