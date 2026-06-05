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


# Models routed through /api/chat because /api/generate mishandles their
# reasoning stream (consumes the num_predict budget, returns empty response).
_CHAT_ENDPOINT_MODELS = ("gemma4", "gemma3", "exaone", "sarvam")

# Models with a hybrid reasoning mode that must be disabled at the API level
# (think=false) for structured extraction. On Ollama >= 0.30 the in-prompt
# "/no_think" directive is ignored: qwen3.5 burns the ENTIRE num_predict
# budget on reasoning tokens and returns an empty response. The think=false
# request field is the only reliable switch.
_THINKING_CAPABLE_MODELS = ("qwen3", "gemma4", "gemma3", "exaone", "sarvam")


def get_extraction_json_schema(doc_type: str) -> dict[str, Any]:
    """Return the JSON schema for a document type's extraction output.

    Reuses the Pydantic extraction models that already back the Gemini
    structured-output path, so the local Ollama model is constrained to the
    exact same contract. The schema is the universal normalization target:
    whatever the document's country or tax vocabulary (MwSt, TVA, IVA, sales
    tax, GST), the model must collapse it into these canonical fields.
    """
    from alibi.extraction.gemini_structurer import _get_extraction_model

    return _get_extraction_model(doc_type).model_json_schema()


@with_retry(max_attempts=3, base_delay=2.0, exceptions=_RETRY_EXCEPTIONS)
def _call_ollama_text(
    ollama_url: str,
    model: str,
    prompt: str,
    timeout: float,
    response_format: dict[str, Any] | None = None,
    num_predict: int | None = None,
    num_ctx: int | None = None,
) -> dict[str, Any]:
    """Call Ollama with text-only prompt.

    Uses /api/chat for models whose reasoning stream /api/generate mishandles
    (Gemma4 etc.). For hybrid reasoning models the request sets ``think=false``
    so the model spends its token budget on the answer, not on reasoning —
    structured extraction needs no chain-of-thought, and on Ollama >= 0.30 the
    in-prompt "/no_think" directive is ignored.

    When ``response_format`` (a JSON schema) is supplied it is passed as
    Ollama's ``format`` field, which constrains decoding to schema-conforming
    output — the enforcement layer that turns the prose "return JSON" prompt
    into a contract.
    """
    config = get_config()
    use_chat_endpoint = any(t in model for t in _CHAT_ENDPOINT_MODELS)
    can_think = any(t in model for t in _THINKING_CAPABLE_MODELS)
    options = {
        "temperature": 0.1,
        "num_predict": num_predict or config.ollama_num_predict,
        "num_ctx": num_ctx or config.ollama_num_ctx,
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            if use_chat_endpoint:
                body: dict[str, Any] = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "keep_alive": config.ollama_keep_alive,
                    "options": options,
                }
                if can_think:
                    body["think"] = False
                if response_format is not None:
                    body["format"] = response_format
                response = client.post(f"{ollama_url}/api/chat", json=body)
                response.raise_for_status()
                data = response.json()
                # Normalize chat response to generate-style format
                return {
                    "response": data.get("message", {}).get("content", ""),
                    **{k: v for k, v in data.items() if k != "message"},
                }
            else:
                body = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": config.ollama_keep_alive,
                    "options": options,
                }
                if can_think:
                    body["think"] = False
                if response_format is not None:
                    body["format"] = response_format
                response = client.post(f"{ollama_url}/api/generate", json=body)
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

    # qwen3 and other hybrid models default to a reasoning mode that spends
    # thousands of tokens before emitting output. For structured extraction
    # this wastes the budget and causes timeouts/empty responses. It is
    # disabled via the think=false request field in _call_ollama_text (the
    # in-prompt "/no_think" directive is silently ignored on Ollama >= 0.30).

    # Constrain decoding to the extraction schema when enabled. This is the
    # WHAT-prompt enforcement layer: the local model can only emit
    # schema-conforming JSON, matching the Gemini response_schema path.
    # Skipped for emphasis/correction retries, whose free-form correction
    # prompts ask for shapes the strict schema would reject.
    response_format: dict[str, Any] | None = None
    if config.ollama_structured_output and not emphasis_prompt:
        try:
            response_format = get_extraction_json_schema(doc_type)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Could not build extraction schema for %s: %s", doc_type, e)

    result = _call_ollama_text(
        ollama_url, model, prompt, timeout, response_format=response_format
    )

    # Per-document escalation: a large receipt can exhaust the output budget
    # and return truncated, unparseable JSON. Ollama reports done_reason=="length"
    # when generation was cut off at num_predict. Retry once with larger
    # num_predict AND num_ctx (prompt + output must fit the context) and more
    # time. Applies to the emphasis/correction path too.
    if (
        result.get("done_reason") == "length"
        and config.ollama_num_predict_escalated > config.ollama_num_predict
    ):
        logger.warning(
            "Structuring output truncated (done_reason=length, eval=%s); "
            "retrying with num_predict=%d num_ctx=%d",
            result.get("eval_count"),
            config.ollama_num_predict_escalated,
            config.ollama_num_ctx_escalated,
        )
        result = _call_ollama_text(
            ollama_url,
            model,
            prompt,
            max(timeout, 300.0),
            response_format=response_format,
            num_predict=config.ollama_num_predict_escalated,
            num_ctx=config.ollama_num_ctx_escalated,
        )

    if "error" in result:
        raise VisionExtractionError(f"Structure error: {result['error']}")

    if "response" not in result:
        raise VisionExtractionError(f"Unexpected structure response: {result}")

    response_text = result["response"]

    # Strip any leaked <think>...</think> blocks before JSON extraction
    response_text = re.sub(r"<think>[\s\S]*?</think>", "", response_text).strip()

    return extract_json_from_response(response_text)
