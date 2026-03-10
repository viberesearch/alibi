"""Stage 2: Structure raw OCR text into JSON using a text-only model.

Takes the raw text from Stage 1 (OCR) and sends it to a text-only LLM
(e.g. qwen3:30b) via the Ollama /api/generate endpoint — no images field.
The model parses the OCR text and returns structured extraction JSON.
"""

import logging
import re
from typing import Any, cast

import httpx

from alibi.config import get_config
from alibi.extraction.prompts import get_text_extraction_prompt
from alibi.extraction.vision import VisionExtractionError, extract_json_from_response
from alibi.utils.retry import with_retry

logger = logging.getLogger(__name__)

_RETRY_EXCEPTIONS = (httpx.TimeoutException, httpx.ConnectError)


@with_retry(max_attempts=3, base_delay=2.0, exceptions=_RETRY_EXCEPTIONS)
def _call_ollama_text(
    ollama_url: str,
    model: str,
    prompt: str,
    timeout: float,
) -> dict[str, Any]:
    """Call Ollama /api/generate with text-only prompt (no images)."""
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
            )
            response.raise_for_status()
        return cast(dict[str, Any], response.json())
    except httpx.HTTPStatusError as e:
        raise VisionExtractionError(
            f"Structure HTTP error: {e.response.status_code}"
        ) from e
    except httpx.RequestError as e:
        if isinstance(e, (httpx.TimeoutException, httpx.ConnectError)):
            raise  # Re-raise for retry
        raise VisionExtractionError(f"Structure request failed: {e}") from e


def structure_ocr_text(
    raw_text: str,
    doc_type: str = "receipt",
    model: str | None = None,
    ollama_url: str | None = None,
    timeout: float = 120.0,
    prompt_mode: str | None = None,
    emphasis_prompt: str | None = None,
) -> dict[str, Any]:
    """Send OCR text to a text-only model for structured extraction.

    Routes to Gemini when ALIBI_GEMINI_EXTRACTION_ENABLED=true and no
    emphasis_prompt is provided (Gemini uses its own system prompt).
    Falls back to Ollama on Gemini failure.

    Args:
        raw_text: Raw OCR text from Stage 1.
        doc_type: Document type (receipt, invoice, statement, …).
        model: Model name (defaults to config.ollama_structure_model).
        ollama_url: Ollama API URL (defaults to config).
        timeout: Request timeout in seconds.
        prompt_mode: 'specialized' or 'universal' (defaults to config).
        emphasis_prompt: If provided, used instead of the default prompt
            (for retry with emphasis on failed checks).

    Returns:
        Structured extraction dict.

    Raises:
        VisionExtractionError: If structuring fails.
    """
    config = get_config()

    # Route to Gemini when enabled and not doing emphasis retry
    # (emphasis prompts are Ollama-specific correction prompts)
    if config.gemini_extraction_enabled and not emphasis_prompt:
        try:
            from alibi.extraction.gemini_structurer import (
                GeminiExtractionError,
                structure_ocr_text_gemini,
            )

            return structure_ocr_text_gemini(raw_text, doc_type=doc_type)
        except GeminiExtractionError as e:
            logger.warning("Gemini extraction failed, falling back to Ollama: %s", e)
        except Exception as e:
            logger.warning(
                "Gemini extraction unexpected error, falling back to Ollama: %s", e
            )

    # Ollama path (default or fallback)
    model = model or config.ollama_structure_model
    ollama_url = ollama_url or config.ollama_url
    prompt_mode = prompt_mode or config.prompt_mode

    if emphasis_prompt:
        prompt = emphasis_prompt
    else:
        prompt = get_text_extraction_prompt(
            raw_text, doc_type, version=2, mode=prompt_mode
        )

    # qwen3 models use a thinking mode by default that generates thousands
    # of reasoning tokens before the actual output. For structured extraction
    # this wastes time and causes timeouts. Prefix with /no_think to disable.
    if "qwen3" in model:
        prompt = "/no_think\n" + prompt

    result = _call_ollama_text(ollama_url, model, prompt, timeout)

    if "error" in result:
        raise VisionExtractionError(f"Structure error: {result['error']}")

    if "response" not in result:
        raise VisionExtractionError(f"Unexpected structure response: {result}")

    response_text = result["response"]

    # Strip any leaked <think>...</think> blocks before JSON extraction
    response_text = re.sub(r"<think>[\s\S]*?</think>", "", response_text).strip()

    return extract_json_from_response(response_text)
