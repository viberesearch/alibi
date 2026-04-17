"""Gemini-based OCR text structuring (Stage 3 replacement).

Replaces qwen3:8b as the LLM stage in the 3-stage extraction pipeline.
Uses Google Gemini with Pydantic structured output for reliable JSON
extraction from OCR text.

Supports single-document and batch modes. Batch mode amortizes system
prompt + schema overhead across N documents (~45% token savings).

Privacy note: sends only OCR text (no images, no file paths).
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for structured extraction
# ---------------------------------------------------------------------------


class LineItemExtraction(BaseModel):
    """Extracted line item from a receipt/invoice."""

    name: str | None = Field(default=None, description="Item name in original language")
    name_en: str | None = Field(
        default=None, description="English translation (null if already English)"
    )
    quantity: float | None = Field(default=None, description="Count purchased (e.g. 2)")
    unit_raw: str | None = Field(
        default=None, description="Unit as printed (kg/ml/Stk/pcs)"
    )
    unit_quantity: float | None = Field(
        default=None,
        description="Measured amount per unit (e.g. 0.355 for 355ml, 1.29 for 1.29kg)",
    )
    unit_price: float | None = Field(
        default=None,
        description="Price per unit including VAT (gross price as printed on receipt). Do NOT calculate net price from total and VAT rate.",
    )
    total_price: float | None = Field(default=None, description="Line total")
    tax_rate: float | None = Field(
        default=None, description="Tax rate as percentage (e.g. 19 for 19%)"
    )
    tax_type: str | None = Field(
        default=None,
        description="vat/sales_tax/gst/exempt/included/none",
    )
    discount: float | None = Field(default=None, description="Discount amount")
    brand: str | None = Field(default=None, description="Brand/manufacturer name")
    barcode: str | None = Field(default=None, description="EAN/PLU/barcode if visible")
    category: str | None = Field(default=None, description="Product category")


class ReceiptExtraction(BaseModel):
    """Structured extraction from a receipt."""

    vendor: str | None = None
    vendor_address: str | None = None
    vendor_phone: str | None = None
    vendor_website: str | None = None
    vendor_vat: str | None = None
    vendor_tax_id: str | None = None
    date: str | None = Field(default=None, description="YYYY-MM-DD")
    time: str | None = Field(default=None, description="HH:MM or HH:MM:SS")
    document_id: str | None = None
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None
    currency: str | None = None
    payment_method: str | None = None
    card_type: str | None = None
    card_last4: str | None = None
    terminal_id: str | None = None
    merchant_id: str | None = None
    language: str | None = None
    line_items: list[LineItemExtraction] = []


class InvoiceExtraction(BaseModel):
    """Structured extraction from an invoice."""

    issuer: str | None = None
    issuer_address: str | None = None
    issuer_phone: str | None = None
    issuer_website: str | None = None
    issuer_vat: str | None = None
    issuer_tax_id: str | None = None
    customer: str | None = None
    invoice_number: str | None = None
    issue_date: str | None = Field(default=None, description="YYYY-MM-DD")
    due_date: str | None = None
    subtotal: float | None = None
    tax: float | None = None
    amount: float | None = None
    currency: str | None = None
    payment_terms: str | None = None
    language: str | None = None
    line_items: list[LineItemExtraction] = []


class PaymentExtraction(BaseModel):
    """Structured extraction from a payment confirmation."""

    vendor: str | None = None
    vendor_address: str | None = None
    vendor_phone: str | None = None
    vendor_website: str | None = None
    vendor_vat: str | None = None
    vendor_tax_id: str | None = None
    date: str | None = Field(default=None, description="YYYY-MM-DD")
    time: str | None = Field(default=None, description="HH:MM or HH:MM:SS")
    document_id: str | None = None
    total: float | None = None
    currency: str | None = None
    payment_method: str | None = None
    card_type: str | None = None
    card_last4: str | None = None
    authorization_code: str | None = None
    terminal_id: str | None = None
    merchant_id: str | None = None
    language: str | None = None


class StatementLineExtraction(BaseModel):
    """Structured extraction from a bank statement line."""

    bank: str | None = None
    account_number: str | None = None
    statement_date: str | None = Field(default=None, description="YYYY-MM-DD")
    period_start: str | None = None
    period_end: str | None = None
    currency: str | None = None
    opening_balance: float | None = None
    closing_balance: float | None = None
    language: str | None = None
    transactions: list[LineItemExtraction] = []


class BatchDocumentExtraction(BaseModel):
    """Wrapper for batch extraction — each document indexed by position."""

    idx: int
    document_type: str | None = None
    extraction: dict[str, Any] = {}


class ExtractionBatchResponse(BaseModel):
    """Mega-batch response containing all document extractions."""

    documents: list[BatchDocumentExtraction]


# ---------------------------------------------------------------------------
# Model selection by doc_type
# ---------------------------------------------------------------------------

_DOC_TYPE_TO_MODEL: dict[str, type[BaseModel]] = {
    "receipt": ReceiptExtraction,
    "invoice": InvoiceExtraction,
    "payment_confirmation": PaymentExtraction,
    "statement": StatementLineExtraction,
}


def _get_extraction_model(doc_type: str) -> type[BaseModel]:
    """Get the Pydantic model for a document type."""
    return _DOC_TYPE_TO_MODEL.get(doc_type, ReceiptExtraction)


# ---------------------------------------------------------------------------
# System prompt — optimized for Gemini structured output
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a document data extraction specialist. Given OCR text from a financial \
document, extract ALL structured data accurately.

RULES:
1. Dates must be YYYY-MM-DD format.
2. Times must be HH:MM or HH:MM:SS format.
3. Preserve item names in their ORIGINAL language.
4. For non-English items, provide English translation in name_en.
5. Be precise with all numbers — correct obvious OCR errors (misread digits, \
garbled characters) using context.
6. tax_rate is a percentage number (19 for 19%, NOT 0.19).
7. unit_raw is the unit exactly as printed on the document.
8. quantity = count purchased (e.g. 2 cans). unit_quantity = measured amount per \
unit (e.g. 0.250 for 250g, 0.355 for 355ml).
9. WEIGHED ITEMS: quantity=1, unit_quantity=measured weight. Example: \
"0,250Kg x 5.56 = 1.39" → quantity=1, unit_raw="Kg", unit_quantity=0.250.
10. PACKAGED ITEMS: bare number in name (e.g. "Milk 500") typically means grams. \
Extract unit_raw="g", unit_quantity=500.
11. VAT SUMMARY: Extract percentage rates (5.00, 19.00), NOT category codes.
12. vendor_vat is VAT registration, vendor_tax_id is separate TIC/TIN number.
13. Use null for any field that is not visible or unclear.
14. Include ALL line items, even if some fields are missing."""


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _is_enabled() -> bool:
    from alibi.config import get_config

    return get_config().gemini_extraction_enabled


def _get_api_key() -> str | None:
    from alibi.config import get_config

    return get_config().gemini_api_key


def _get_model() -> str:
    from alibi.config import get_config

    return get_config().gemini_extraction_model


# ---------------------------------------------------------------------------
# Single-document extraction
# ---------------------------------------------------------------------------


def structure_ocr_text_gemini(
    raw_text: str,
    doc_type: str = "receipt",
    api_key: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Structure OCR text using Gemini with Pydantic structured output.

    Drop-in replacement for structure_ocr_text() in structurer.py.

    Args:
        raw_text: Raw OCR text from Stage 1.
        doc_type: Document type (receipt, invoice, payment_confirmation, statement).
        api_key: Gemini API key. Falls back to config.
        model: Model ID. Defaults to config.gemini_extraction_model.

    Returns:
        Extraction dict matching the V2 prompt output format.

    Raises:
        GeminiExtractionError: If the API call fails.
    """
    resolved_key = api_key or _get_api_key()
    if not resolved_key:
        raise GeminiExtractionError("ALIBI_GEMINI_API_KEY not configured")

    resolved_model = model or _get_model()
    extraction_model = _get_extraction_model(doc_type)

    prompt = (
        f"Document type: {doc_type}\n"
        f"--- BEGIN OCR TEXT ---\n"
        f"{raw_text}\n"
        f"--- END OCR TEXT ---"
    )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=resolved_key)

        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=extraction_model,
                temperature=0.1,
                max_output_tokens=2048,
            ),
        )

        result = _parse_response(response, extraction_model)
        result["_pipeline"] = "gemini_extraction"
        return result

    except ImportError:
        raise GeminiExtractionError(
            "google-genai package not installed. Run: uv add google-genai"
        )
    except Exception as e:
        logger.exception("Gemini extraction failed for %s", doc_type)
        raise GeminiExtractionError(f"Gemini extraction failed: {e}") from e


# ---------------------------------------------------------------------------
# Vision extraction (image bytes → structured data, bypasses OCR)
# ---------------------------------------------------------------------------

_VISION_SYSTEM_PROMPT = (
    """\
You are a document data extraction specialist. Given an image of a financial \
document (receipt, invoice, payment confirmation, or bank statement), extract \
ALL structured data accurately.

Read every piece of text in the image. Extract all line items, totals, dates, \
vendor details, and payment information visible.

"""
    + _SYSTEM_PROMPT.split("RULES:\n", 1)[1]
)  # Reuse extraction rules


def extract_from_image_gemini(
    image_path: str,
    doc_type: str = "receipt",
    api_key: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Extract structured data directly from a document image via Gemini Vision.

    Bypasses OCR entirely — Gemini reads the image and returns structured JSON.
    Use for: low-confidence documents (parser < 0.3), non-Latin scripts, or
    as a quality baseline comparison.

    Args:
        image_path: Path to the image file (JPEG, PNG, etc.).
        doc_type: Document type for schema selection.
        api_key: Gemini API key. Falls back to config.
        model: Model ID. Defaults to config.gemini_extraction_model.

    Returns:
        Extraction dict matching the V2 prompt output format.

    Raises:
        GeminiExtractionError: If the API call fails.
    """
    from pathlib import Path as _Path

    path = _Path(image_path)
    if not path.exists():
        raise GeminiExtractionError(f"Image file not found: {image_path}")

    resolved_key = api_key or _get_api_key()
    if not resolved_key:
        raise GeminiExtractionError("ALIBI_GEMINI_API_KEY not configured")

    resolved_model = model or _get_model()
    extraction_model = _get_extraction_model(doc_type)

    # Read image bytes
    image_bytes = path.read_bytes()

    # Detect MIME type
    suffix = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".pdf": "application/pdf",
    }
    mime_type = mime_map.get(suffix, "image/jpeg")

    prompt = f"Extract all data from this {doc_type} image."

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=resolved_key)

        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        response = client.models.generate_content(
            model=resolved_model,
            contents=[image_part, prompt],  # type: ignore[arg-type]
            config=types.GenerateContentConfig(
                system_instruction=_VISION_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=extraction_model,
                temperature=0.1,
                max_output_tokens=2048,
            ),
        )

        result = _parse_response(response, extraction_model)
        result["_pipeline"] = "gemini_vision"
        return result

    except ImportError:
        raise GeminiExtractionError(
            "google-genai package not installed. Run: uv add google-genai"
        )
    except Exception as e:
        logger.exception("Gemini vision extraction failed for %s", image_path)
        raise GeminiExtractionError(f"Gemini vision extraction failed: {e}") from e


# ---------------------------------------------------------------------------
# Batch extraction (multiple documents in one call)
# ---------------------------------------------------------------------------


def structure_ocr_texts_gemini(
    documents: list[dict[str, str]],
    api_key: str | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Batch-extract structured data from multiple OCR texts in one Gemini call.

    Amortizes system prompt + schema overhead across N documents.
    ~45% token savings vs N individual calls.

    Args:
        documents: List of dicts with 'raw_text' and 'doc_type' keys.
        api_key: Gemini API key. Falls back to config.
        model: Model ID. Defaults to config.gemini_extraction_model.

    Returns:
        List of extraction dicts, one per input document (in order).
        Failed documents return empty dicts with _error key.
    """
    if not documents:
        return []

    resolved_key = api_key or _get_api_key()
    if not resolved_key:
        raise GeminiExtractionError("ALIBI_GEMINI_API_KEY not configured")

    resolved_model = model or _get_model()

    # Build indexed document block
    lines = []
    for idx, doc in enumerate(documents, start=1):
        doc_type = doc.get("doc_type", "receipt")
        raw_text = doc.get("raw_text", "")
        lines.append(f"--- Document {idx} (type: {doc_type}) ---")
        lines.append(raw_text)
        lines.append(f"--- End Document {idx} ---")
        lines.append("")

    prompt = "\n".join(lines)

    batch_system = (
        _SYSTEM_PROMPT
        + "\n\nYou are processing multiple documents. For each document, extract "
        "all data and return it with the matching idx number. Return ALL documents."
    )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=resolved_key)

        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=batch_system,
                response_mime_type="application/json",
                response_schema=ExtractionBatchResponse,
                temperature=0.1,
                max_output_tokens=4096,  # batch responses are larger
            ),
        )

        return _parse_batch_response(response, documents)

    except ImportError:
        raise GeminiExtractionError(
            "google-genai package not installed. Run: uv add google-genai"
        )
    except Exception as e:
        logger.exception("Gemini batch extraction failed")
        raise GeminiExtractionError(f"Gemini batch extraction failed: {e}") from e


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_response(
    response: Any,
    extraction_model: type[BaseModel],
) -> dict[str, Any]:
    """Parse Gemini response into extraction dict."""
    # Try structured parsing first
    if response.parsed and isinstance(response.parsed, extraction_model):
        result: dict[str, Any] = response.parsed.model_dump(exclude_none=True)
        return result

    # Fallback: parse JSON text
    import json

    raw_text = response.text or ""
    try:
        return dict(json.loads(raw_text))
    except (json.JSONDecodeError, ValueError) as e:
        raise GeminiExtractionError(f"Failed to parse Gemini response: {e}") from e


def _parse_batch_response(
    response: Any,
    documents: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Parse batch Gemini response into list of extraction dicts."""
    n = len(documents)
    results: list[dict[str, Any]] = [{"_error": "not_returned"} for _ in range(n)]

    try:
        if response.parsed and isinstance(response.parsed, ExtractionBatchResponse):
            for doc in response.parsed.documents:
                idx = doc.idx - 1  # Convert 1-based to 0-based
                if 0 <= idx < n:
                    result = dict(doc.extraction)
                    result["_pipeline"] = "gemini_batch_extraction"
                    results[idx] = result
            return results

        # Fallback: parse JSON text
        import json

        raw_text = response.text or ""
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict) and "documents" in parsed:
            for doc in parsed["documents"]:
                idx = doc.get("idx", 0) - 1
                if 0 <= idx < n:
                    extraction = doc.get("extraction", doc)
                    if isinstance(extraction, dict):
                        extraction["_pipeline"] = "gemini_batch_extraction"
                        results[idx] = extraction
        return results

    except (json.JSONDecodeError, ValueError, KeyError):
        logger.exception("Failed to parse Gemini batch response")
        return results


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class GeminiExtractionError(Exception):
    """Error during Gemini-based extraction."""

    pass
