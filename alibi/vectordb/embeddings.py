"""Embedding generation using Ollama nomic-embed-text."""

import logging
from typing import Any, Callable, cast

import httpx

from alibi.config import get_config
from alibi.utils.retry import with_retry

logger = logging.getLogger(__name__)

# Retry configuration for Ollama API calls
OLLAMA_RETRY_EXCEPTIONS = (httpx.TimeoutException, httpx.ConnectError)

# nomic-embed-text dimensions
EMBEDDING_DIM = 768
EMBEDDING_MODEL = "nomic-embed-text"


class EmbeddingError(Exception):
    """Error during embedding generation."""

    pass


def get_embedding(
    text: str,
    ollama_url: str | None = None,
    model: str = EMBEDDING_MODEL,
    timeout: float = 30.0,
) -> list[float]:
    """Get embedding vector from Ollama.

    Args:
        text: Text to embed
        ollama_url: Ollama API URL (defaults to config)
        model: Embedding model (default: nomic-embed-text)
        timeout: Request timeout in seconds

    Returns:
        Embedding vector as list of floats (768 dimensions)

    Raises:
        EmbeddingError: If embedding generation fails
    """
    config = get_config()
    ollama_url = ollama_url or config.ollama_url

    # Truncate text to avoid token limits (nomic-embed-text: 8192 tokens).
    # JSON/structured text tokenizes at ~2.5 chars/token, so cap at 6000 chars
    # to stay safely within the context window for all content types.
    truncated_text = text[:6000]

    result = _call_ollama_embedding(ollama_url, model, truncated_text, timeout)

    if "error" in result:
        raise EmbeddingError(f"Ollama error: {result['error']}")

    if "embedding" not in result:
        raise EmbeddingError(f"No embedding in response: {result}")

    embedding: list[float] = result["embedding"]

    if len(embedding) != EMBEDDING_DIM:
        logger.warning(
            f"Unexpected embedding dimension: {len(embedding)} (expected {EMBEDDING_DIM})"
        )

    return embedding


@with_retry(max_attempts=3, base_delay=2.0, exceptions=OLLAMA_RETRY_EXCEPTIONS)
def _call_ollama_embedding(
    ollama_url: str,
    model: str,
    text: str,
    timeout: float,
) -> dict[str, Any]:
    """Make HTTP request to Ollama embeddings API with retry support."""
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{ollama_url}/api/embeddings",
                json={
                    "model": model,
                    "prompt": text,
                },
            )
            response.raise_for_status()
        return cast(dict[str, Any], response.json())
    except httpx.HTTPStatusError as e:
        raise EmbeddingError(f"HTTP error: {e.response.status_code}") from e
    except httpx.RequestError as e:
        if isinstance(e, (httpx.TimeoutException, httpx.ConnectError)):
            raise  # Re-raise for retry
        raise EmbeddingError(f"Request failed: {e}") from e


async def get_embedding_async(
    text: str,
    ollama_url: str | None = None,
    model: str = EMBEDDING_MODEL,
    timeout: float = 30.0,
) -> list[float]:
    """Async version of get_embedding.

    Args:
        text: Text to embed
        ollama_url: Ollama API URL (defaults to config)
        model: Embedding model (default: nomic-embed-text)
        timeout: Request timeout in seconds

    Returns:
        Embedding vector as list of floats (768 dimensions)

    Raises:
        EmbeddingError: If embedding generation fails
    """
    config = get_config()
    ollama_url = ollama_url or config.ollama_url

    truncated_text = text[:6000]

    result = await _call_ollama_embedding_async(
        ollama_url, model, truncated_text, timeout
    )

    if "error" in result:
        raise EmbeddingError(f"Ollama error: {result['error']}")

    if "embedding" not in result:
        raise EmbeddingError(f"No embedding in response: {result}")

    embedding: list[float] = result["embedding"]

    return embedding


@with_retry(max_attempts=3, base_delay=2.0, exceptions=OLLAMA_RETRY_EXCEPTIONS)
async def _call_ollama_embedding_async(
    ollama_url: str,
    model: str,
    text: str,
    timeout: float,
) -> dict[str, Any]:
    """Async HTTP request to Ollama embeddings API with retry support."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{ollama_url}/api/embeddings",
                json={
                    "model": model,
                    "prompt": text,
                },
            )
            response.raise_for_status()
        return cast(dict[str, Any], response.json())
    except httpx.HTTPStatusError as e:
        raise EmbeddingError(f"HTTP error: {e.response.status_code}") from e
    except httpx.RequestError as e:
        if isinstance(e, (httpx.TimeoutException, httpx.ConnectError)):
            raise  # Re-raise for retry
        raise EmbeddingError(f"Request failed: {e}") from e


def create_mock_embedding_fn(
    dim: int = EMBEDDING_DIM,
) -> Callable[[str], list[float]]:
    """Create a mock embedding function for testing.

    Returns a deterministic embedding based on text hash.

    Args:
        dim: Embedding dimension

    Returns:
        Mock embedding function
    """
    import hashlib

    def mock_fn(text: str) -> list[float]:
        # Create deterministic embedding from text hash
        h = hashlib.sha256(text.encode()).hexdigest()
        # Convert hex to floats
        embedding: list[float] = []
        for i in range(0, min(len(h) * 2, dim * 2), 2):
            if len(embedding) >= dim:
                break
            # Take 2 hex chars and convert to float in range [-1, 1]
            val = int(h[i % len(h) : i % len(h) + 2], 16) / 255.0 * 2 - 1
            embedding.append(val)

        # Pad if necessary
        while len(embedding) < dim:
            embedding.append(0.0)

        return embedding[:dim]

    return mock_fn
