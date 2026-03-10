"""Tests for health check functionality."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from alibi.config import Config
from alibi.health import (
    HealthStatus,
    check_health,
    check_ollama_available,
    check_ollama_model,
    get_available_models,
)


class TestHealthStatus:
    """Tests for HealthStatus dataclass."""

    def test_health_status_defaults(self) -> None:
        """Test that HealthStatus has correct default values."""
        status = HealthStatus()

        assert status.ollama_available is False
        assert status.ollama_model_loaded is False
        assert status.ollama_url == ""
        assert status.ollama_model == ""
        assert status.database_accessible is False
        assert status.database_path == ""
        assert status.vault_exists is False
        assert status.vault_path == ""
        assert status.errors == []

    def test_health_status_healthy_property_all_good(self) -> None:
        """Test healthy property returns True when all critical services are up."""
        status = HealthStatus(
            ollama_available=True,
            ollama_model_loaded=True,
            database_accessible=True,
        )

        assert status.healthy is True

    def test_health_status_healthy_property_ollama_unavailable(self) -> None:
        """Test healthy property returns False when Ollama is unavailable."""
        status = HealthStatus(
            ollama_available=False,
            ollama_model_loaded=True,
            database_accessible=True,
        )

        assert status.healthy is False

    def test_health_status_healthy_property_model_not_loaded(self) -> None:
        """Test healthy property returns False when model is not loaded."""
        status = HealthStatus(
            ollama_available=True,
            ollama_model_loaded=False,
            database_accessible=True,
        )

        assert status.healthy is False

    def test_health_status_healthy_property_database_inaccessible(self) -> None:
        """Test healthy property returns False when database is inaccessible."""
        status = HealthStatus(
            ollama_available=True,
            ollama_model_loaded=True,
            database_accessible=False,
        )

        assert status.healthy is False

    def test_health_status_warnings_vault_missing(self) -> None:
        """Test warnings property includes vault path warning when vault doesn't exist."""
        status = HealthStatus(
            vault_exists=False,
            vault_path="/path/to/vault",
        )

        warnings = status.warnings
        assert len(warnings) == 1
        assert "Vault path not found: /path/to/vault" in warnings

    def test_health_status_warnings_vault_exists(self) -> None:
        """Test warnings property is empty when vault exists."""
        status = HealthStatus(
            vault_exists=True,
            vault_path="/path/to/vault",
        )

        warnings = status.warnings
        assert len(warnings) == 0


class TestCheckOllamaAvailable:
    """Tests for check_ollama_available function."""

    @patch("alibi.health.httpx.Client")
    def test_check_ollama_available_success(self, mock_client_class: Mock) -> None:
        """Test check_ollama_available returns True when Ollama responds."""
        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 200

        # Mock the client context manager
        mock_client = MagicMock()
        mock_client.__enter__.return_value.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        result = check_ollama_available("http://localhost:11434")

        assert result is True
        mock_client.__enter__.return_value.get.assert_called_once_with(
            "http://localhost:11434/api/tags"
        )

    @patch("alibi.health.httpx.Client")
    def test_check_ollama_available_connect_error(
        self, mock_client_class: Mock
    ) -> None:
        """Test check_ollama_available returns False on connection error."""
        # Mock the client to raise ConnectError
        mock_client = MagicMock()
        mock_client.__enter__.return_value.get.side_effect = httpx.ConnectError(
            "Connection refused"
        )
        mock_client_class.return_value = mock_client

        result = check_ollama_available("http://localhost:11434")

        assert result is False

    @patch("alibi.health.httpx.Client")
    def test_check_ollama_available_timeout(self, mock_client_class: Mock) -> None:
        """Test check_ollama_available returns False on timeout."""
        # Mock the client to raise TimeoutException
        mock_client = MagicMock()
        mock_client.__enter__.return_value.get.side_effect = httpx.TimeoutException(
            "Request timeout"
        )
        mock_client_class.return_value = mock_client

        result = check_ollama_available("http://localhost:11434")

        assert result is False

    @patch("alibi.health.httpx.Client")
    def test_check_ollama_available_non_200_status(
        self, mock_client_class: Mock
    ) -> None:
        """Test check_ollama_available returns False on non-200 status code."""
        # Mock the response with error status
        mock_response = Mock()
        mock_response.status_code = 500

        mock_client = MagicMock()
        mock_client.__enter__.return_value.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        result = check_ollama_available("http://localhost:11434")

        assert result is False


class TestCheckOllamaModel:
    """Tests for check_ollama_model function."""

    @patch("alibi.health.httpx.Client")
    def test_check_ollama_model_exact_match(self, mock_client_class: Mock) -> None:
        """Test check_ollama_model returns True when exact model name matches."""
        # Mock the response with model list
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2:7b"},
                {"name": "qwen3-vl:30b"},
                {"name": "mistral:latest"},
            ]
        }

        mock_client = MagicMock()
        mock_client.__enter__.return_value.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        result = check_ollama_model("http://localhost:11434", "qwen3-vl:30b")

        assert result is True

    @patch("alibi.health.httpx.Client")
    def test_check_ollama_model_partial_match(self, mock_client_class: Mock) -> None:
        """Test check_ollama_model returns True when model name matches base."""
        # Mock the response with model list
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2:7b"},
                {"name": "qwen3-vl:latest"},
            ]
        }

        mock_client = MagicMock()
        mock_client.__enter__.return_value.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Search for base model name
        result = check_ollama_model("http://localhost:11434", "qwen3-vl")

        assert result is True

    @patch("alibi.health.httpx.Client")
    def test_check_ollama_model_not_found(self, mock_client_class: Mock) -> None:
        """Test check_ollama_model returns False when model is not in list."""
        # Mock the response with model list
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2:7b"},
                {"name": "mistral:latest"},
            ]
        }

        mock_client = MagicMock()
        mock_client.__enter__.return_value.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        result = check_ollama_model("http://localhost:11434", "qwen3-vl:30b")

        assert result is False

    @patch("alibi.health.httpx.Client")
    def test_check_ollama_model_connect_error(self, mock_client_class: Mock) -> None:
        """Test check_ollama_model returns False on connection error."""
        mock_client = MagicMock()
        mock_client.__enter__.return_value.get.side_effect = httpx.ConnectError(
            "Connection refused"
        )
        mock_client_class.return_value = mock_client

        result = check_ollama_model("http://localhost:11434", "qwen3-vl:30b")

        assert result is False

    @patch("alibi.health.httpx.Client")
    def test_check_ollama_model_non_200_status(self, mock_client_class: Mock) -> None:
        """Test check_ollama_model returns False on non-200 status code."""
        mock_response = Mock()
        mock_response.status_code = 500

        mock_client = MagicMock()
        mock_client.__enter__.return_value.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        result = check_ollama_model("http://localhost:11434", "qwen3-vl:30b")

        assert result is False


class TestGetAvailableModels:
    """Tests for get_available_models function."""

    @patch("alibi.health.httpx.Client")
    def test_get_available_models_success(self, mock_client_class: Mock) -> None:
        """Test get_available_models returns list of model names."""
        # Mock the response with model list
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2:7b"},
                {"name": "qwen3-vl:30b"},
                {"name": "mistral:latest"},
            ]
        }

        mock_client = MagicMock()
        mock_client.__enter__.return_value.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        result = get_available_models("http://localhost:11434")

        assert result == ["llama2:7b", "qwen3-vl:30b", "mistral:latest"]

    @patch("alibi.health.httpx.Client")
    def test_get_available_models_empty_list(self, mock_client_class: Mock) -> None:
        """Test get_available_models returns empty list when no models."""
        # Mock the response with empty model list
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": []}

        mock_client = MagicMock()
        mock_client.__enter__.return_value.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        result = get_available_models("http://localhost:11434")

        assert result == []

    @patch("alibi.health.httpx.Client")
    def test_get_available_models_connect_error(self, mock_client_class: Mock) -> None:
        """Test get_available_models returns empty list on connection error."""
        mock_client = MagicMock()
        mock_client.__enter__.return_value.get.side_effect = httpx.ConnectError(
            "Connection refused"
        )
        mock_client_class.return_value = mock_client

        result = get_available_models("http://localhost:11434")

        assert result == []

    @patch("alibi.health.httpx.Client")
    def test_get_available_models_non_200_status(self, mock_client_class: Mock) -> None:
        """Test get_available_models returns empty list on non-200 status."""
        mock_response = Mock()
        mock_response.status_code = 500

        mock_client = MagicMock()
        mock_client.__enter__.return_value.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        result = get_available_models("http://localhost:11434")

        assert result == []

    @patch("alibi.health.get_config")
    @patch("alibi.health.httpx.Client")
    def test_get_available_models_uses_config_url(
        self, mock_client_class: Mock, mock_get_config: Mock
    ) -> None:
        """Test get_available_models uses config URL when no URL provided."""
        # Mock config
        mock_config = Mock()
        mock_config.ollama_url = "http://configured-url:11434"
        mock_get_config.return_value = mock_config

        # Mock the response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "test:latest"}]}

        mock_client = MagicMock()
        mock_client.__enter__.return_value.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        result = get_available_models()

        assert result == ["test:latest"]
        mock_client.__enter__.return_value.get.assert_called_once_with(
            "http://configured-url:11434/api/tags"
        )


class TestCheckHealth:
    """Tests for check_health function."""

    @patch("alibi.health.get_db")
    @patch("alibi.health.check_ollama_model")
    @patch("alibi.health.check_ollama_available")
    @patch("alibi.health.get_config")
    def test_health_all_services_available(
        self,
        mock_get_config: Mock,
        mock_check_ollama_available: Mock,
        mock_check_ollama_model: Mock,
        mock_get_db: Mock,
        tmp_path: Path,
    ) -> None:
        """Test check_health when all services are available and healthy."""
        # Mock config
        vault_path = tmp_path / "vault"
        vault_path.mkdir()

        mock_config = Config(
            db_path=tmp_path / "test.db",
            vault_path=vault_path,
            ollama_url="http://localhost:11434",
            ollama_model="qwen3-vl:30b",
        )
        mock_get_config.return_value = mock_config

        # Mock Ollama checks
        mock_check_ollama_available.return_value = True
        mock_check_ollama_model.return_value = True

        # Mock database
        mock_db = Mock()
        mock_db.db_path = tmp_path / "test.db"
        mock_db.is_initialized.return_value = True
        mock_get_db.return_value = mock_db

        # Run health check
        status = check_health()

        # Verify status
        assert status.ollama_available is True
        assert status.ollama_model_loaded is True
        assert status.ollama_url == "http://localhost:11434"
        assert status.ollama_model == "qwen3-vl:30b"
        assert status.database_accessible is True
        assert status.database_path == str(tmp_path / "test.db")
        assert status.vault_exists is True
        assert status.vault_path == str(vault_path)
        assert status.errors == []
        assert status.healthy is True

    @patch("alibi.health.get_db")
    @patch("alibi.health.check_ollama_available")
    @patch("alibi.health.get_config")
    def test_health_ollama_unavailable(
        self,
        mock_get_config: Mock,
        mock_check_ollama_available: Mock,
        mock_get_db: Mock,
        tmp_path: Path,
    ) -> None:
        """Test check_health when Ollama is unavailable."""
        # Mock config
        mock_config = Config(
            db_path=tmp_path / "test.db",
            ollama_url="http://localhost:11434",
            ollama_model="qwen3-vl:30b",
        )
        mock_get_config.return_value = mock_config

        # Mock Ollama as unavailable
        mock_check_ollama_available.return_value = False

        # Mock database
        mock_db = Mock()
        mock_db.db_path = tmp_path / "test.db"
        mock_db.is_initialized.return_value = True
        mock_get_db.return_value = mock_db

        # Run health check
        status = check_health()

        # Verify status
        assert status.ollama_available is False
        assert status.ollama_model_loaded is False
        assert status.database_accessible is True
        assert len(status.errors) == 1
        assert "Cannot connect to Ollama" in status.errors[0]
        assert "ollama serve" in status.errors[0]
        assert status.healthy is False

    @patch("alibi.health.get_db")
    @patch("alibi.health.check_ollama_model")
    @patch("alibi.health.check_ollama_available")
    @patch("alibi.health.get_config")
    def test_health_model_not_loaded(
        self,
        mock_get_config: Mock,
        mock_check_ollama_available: Mock,
        mock_check_ollama_model: Mock,
        mock_get_db: Mock,
        tmp_path: Path,
    ) -> None:
        """Test check_health when Ollama model is not loaded."""
        # Mock config
        mock_config = Config(
            db_path=tmp_path / "test.db",
            ollama_url="http://localhost:11434",
            ollama_model="qwen3-vl:30b",
        )
        mock_get_config.return_value = mock_config

        # Mock Ollama available but model not loaded
        mock_check_ollama_available.return_value = True
        mock_check_ollama_model.return_value = False

        # Mock database
        mock_db = Mock()
        mock_db.db_path = tmp_path / "test.db"
        mock_db.is_initialized.return_value = True
        mock_get_db.return_value = mock_db

        # Run health check
        status = check_health()

        # Verify status
        assert status.ollama_available is True
        assert status.ollama_model_loaded is False
        assert status.database_accessible is True
        assert len(status.errors) == 1
        assert "model 'qwen3-vl:30b' not found" in status.errors[0]
        assert "ollama pull qwen3-vl:30b" in status.errors[0]
        assert status.healthy is False

    @patch("alibi.health.get_db")
    @patch("alibi.health.check_ollama_model")
    @patch("alibi.health.check_ollama_available")
    @patch("alibi.health.get_config")
    def test_health_database_missing(
        self,
        mock_get_config: Mock,
        mock_check_ollama_available: Mock,
        mock_check_ollama_model: Mock,
        mock_get_db: Mock,
        tmp_path: Path,
    ) -> None:
        """Test check_health when database is not initialized."""
        # Mock config
        mock_config = Config(
            db_path=tmp_path / "test.db",
            ollama_url="http://localhost:11434",
            ollama_model="qwen3-vl:30b",
        )
        mock_get_config.return_value = mock_config

        # Mock Ollama checks
        mock_check_ollama_available.return_value = True
        mock_check_ollama_model.return_value = True

        # Mock database as not initialized
        mock_db = Mock()
        mock_db.db_path = tmp_path / "test.db"
        mock_db.is_initialized.return_value = False
        mock_get_db.return_value = mock_db

        # Run health check
        status = check_health()

        # Verify status
        assert status.ollama_available is True
        assert status.ollama_model_loaded is True
        assert status.database_accessible is False
        assert len(status.errors) == 1
        assert "Database not initialized" in status.errors[0]
        assert "lt init" in status.errors[0]
        assert status.healthy is False

    @patch("alibi.health.get_db")
    @patch("alibi.health.check_ollama_available")
    @patch("alibi.health.get_config")
    def test_health_check_model_false(
        self,
        mock_get_config: Mock,
        mock_check_ollama_available: Mock,
        mock_get_db: Mock,
        tmp_path: Path,
    ) -> None:
        """Test check_health with check_model=False skips model check."""
        # Mock config
        mock_config = Config(
            db_path=tmp_path / "test.db",
            ollama_url="http://localhost:11434",
            ollama_model="qwen3-vl:30b",
        )
        mock_get_config.return_value = mock_config

        # Mock Ollama available
        mock_check_ollama_available.return_value = True

        # Mock database
        mock_db = Mock()
        mock_db.db_path = tmp_path / "test.db"
        mock_db.is_initialized.return_value = True
        mock_get_db.return_value = mock_db

        # Run health check with check_model=False
        status = check_health(check_model=False)

        # Verify status - model should be considered loaded
        assert status.ollama_available is True
        assert status.ollama_model_loaded is True
        assert status.database_accessible is True
        assert status.errors == []
        assert status.healthy is True

    @patch("alibi.health.get_db")
    @patch("alibi.health.check_ollama_available")
    @patch("alibi.health.get_config")
    def test_health_database_exception(
        self,
        mock_get_config: Mock,
        mock_check_ollama_available: Mock,
        mock_get_db: Mock,
        tmp_path: Path,
    ) -> None:
        """Test check_health when database raises an exception."""
        # Mock config
        mock_config = Config(
            db_path=tmp_path / "test.db",
            ollama_url="http://localhost:11434",
            ollama_model="qwen3-vl:30b",
        )
        mock_get_config.return_value = mock_config

        # Mock Ollama available
        mock_check_ollama_available.return_value = True

        # Mock database to raise exception
        mock_get_db.side_effect = Exception("Database connection failed")

        # Run health check
        status = check_health(check_model=False)

        # Verify status
        assert status.database_accessible is False
        assert len(status.errors) >= 1
        assert "Database error: Database connection failed" in status.errors

    @patch("alibi.health.get_db")
    @patch("alibi.health.check_ollama_available")
    @patch("alibi.health.get_config")
    def test_health_vault_not_configured(
        self,
        mock_get_config: Mock,
        mock_check_ollama_available: Mock,
        mock_get_db: Mock,
        tmp_path: Path,
    ) -> None:
        """Test check_health when vault path is not configured."""
        # Mock config without vault
        mock_config = Config(
            db_path=tmp_path / "test.db",
            vault_path=None,
            ollama_url="http://localhost:11434",
            ollama_model="qwen3-vl:30b",
        )
        mock_get_config.return_value = mock_config

        # Mock Ollama available
        mock_check_ollama_available.return_value = True

        # Mock database
        mock_db = Mock()
        mock_db.db_path = tmp_path / "test.db"
        mock_db.is_initialized.return_value = True
        mock_get_db.return_value = mock_db

        # Run health check
        status = check_health(check_model=False)

        # Verify vault status
        assert status.vault_exists is False
        assert status.vault_path == "Not configured"

    @patch("alibi.health.get_db")
    @patch("alibi.health.check_ollama_available")
    @patch("alibi.health.get_config")
    def test_health_vault_missing_warning(
        self,
        mock_get_config: Mock,
        mock_check_ollama_available: Mock,
        mock_get_db: Mock,
        tmp_path: Path,
    ) -> None:
        """Test check_health when vault path doesn't exist (warning, not error)."""
        # Mock config with non-existent vault
        vault_path = tmp_path / "nonexistent_vault"
        mock_config = Config(
            db_path=tmp_path / "test.db",
            vault_path=vault_path,
            ollama_url="http://localhost:11434",
            ollama_model="qwen3-vl:30b",
        )
        mock_get_config.return_value = mock_config

        # Mock Ollama available
        mock_check_ollama_available.return_value = True

        # Mock database
        mock_db = Mock()
        mock_db.db_path = tmp_path / "test.db"
        mock_db.is_initialized.return_value = True
        mock_get_db.return_value = mock_db

        # Run health check
        status = check_health(check_model=False)

        # Verify vault status - should be warning, not error
        assert status.vault_exists is False
        assert status.vault_path == str(vault_path)
        assert len(status.warnings) == 1
        assert "Vault path not found" in status.warnings[0]
        # Should still be healthy as vault is not critical
        assert status.healthy is True
