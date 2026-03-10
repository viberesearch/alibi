"""Health check module for Alibi services."""

import logging
from dataclasses import dataclass, field

import httpx

from alibi.config import get_config
from alibi.db.connection import get_db

logger = logging.getLogger(__name__)

HEALTH_CHECK_TIMEOUT = 5.0


@dataclass
class HealthStatus:
    """Health status of all Alibi services."""

    ollama_available: bool = False
    ollama_model_loaded: bool = False
    ollama_url: str = ""
    ollama_model: str = ""
    database_accessible: bool = False
    database_path: str = ""
    vault_exists: bool = False
    vault_path: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        """Return True if all critical services are healthy."""
        return (
            self.ollama_available
            and self.ollama_model_loaded
            and self.database_accessible
        )

    @property
    def warnings(self) -> list[str]:
        """Return list of warnings (non-critical issues)."""
        warnings = []
        if not self.vault_exists:
            warnings.append(f"Vault path not found: {self.vault_path}")
        return warnings


def check_ollama_available(url: str, timeout: float = HEALTH_CHECK_TIMEOUT) -> bool:
    """Check if Ollama API is available."""
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{url}/api/tags")
            return response.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError) as e:
        logger.debug(f"Ollama connection check failed: {e}")
        return False


def check_ollama_model(
    url: str, model: str, timeout: float = HEALTH_CHECK_TIMEOUT
) -> bool:
    """Check if specific Ollama model is available."""
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{url}/api/tags")
            if response.status_code != 200:
                return False

            data = response.json()
            models = data.get("models", [])
            model_names = [m.get("name", "") for m in models]

            # Check for exact match or partial match (model:tag)
            for name in model_names:
                if name == model or name.startswith(f"{model}:"):
                    return True

            # Also check without tag
            model_base = model.split(":")[0]
            for name in model_names:
                if name.split(":")[0] == model_base:
                    return True

            return False
    except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError) as e:
        logger.debug(f"Ollama model check failed: {e}")
        return False


def check_health(check_model: bool = True) -> HealthStatus:
    """Check health of all Alibi services.

    Args:
        check_model: Whether to check if the configured model is loaded

    Returns:
        HealthStatus with status of all services
    """
    config = get_config()
    status = HealthStatus(
        ollama_url=config.ollama_url,
        ollama_model=config.ollama_model,
        vault_path=str(config.vault_path) if config.vault_path else "",
    )

    # Check Ollama availability
    status.ollama_available = check_ollama_available(config.ollama_url)
    if not status.ollama_available:
        status.errors.append(
            f"Cannot connect to Ollama at {config.ollama_url}. "
            "Run 'ollama serve' to start it."
        )
    elif check_model:
        status.ollama_model_loaded = check_ollama_model(
            config.ollama_url, config.ollama_model
        )
        if not status.ollama_model_loaded:
            status.errors.append(
                f"Ollama model '{config.ollama_model}' not found. "
                f"Run 'ollama pull {config.ollama_model}' to download it."
            )
    else:
        # If not checking model, consider it loaded if Ollama is available
        status.ollama_model_loaded = True

    # Check database
    try:
        db = get_db()
        status.database_path = str(db.db_path)
        status.database_accessible = db.is_initialized()
        if not status.database_accessible:
            status.errors.append(
                f"Database not initialized at {db.db_path}. "
                "Run 'lt init' to initialize."
            )
    except Exception as e:
        status.errors.append(f"Database error: {e}")

    # Check vault path
    if config.vault_path:
        status.vault_exists = config.vault_path.exists()
        if not status.vault_exists:
            # This is a warning, not an error
            pass
    else:
        status.vault_path = "Not configured"

    return status


def get_available_models(url: str | None = None) -> list[str]:
    """Get list of available Ollama models.

    Args:
        url: Ollama API URL (uses config default if not specified)

    Returns:
        List of model names
    """
    if url is None:
        config = get_config()
        url = config.ollama_url

    try:
        with httpx.Client(timeout=HEALTH_CHECK_TIMEOUT) as client:
            response = client.get(f"{url}/api/tags")
            if response.status_code != 200:
                return []

            data = response.json()
            models = data.get("models", [])
            return [m.get("name", "") for m in models if m.get("name")]
    except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError):
        return []
