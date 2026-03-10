"""Gemini batch verification: cross-validate extracted receipts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# --- Config helpers (same pattern as gemini_enrichment.py) ---


def _get_api_key(api_key: str | None = None) -> str | None:
    if api_key:
        return api_key
    from alibi.config import get_config

    return get_config().gemini_api_key


def _get_model() -> str:
    from alibi.config import get_config

    return get_config().gemini_extraction_model


# --- Pydantic models for structured output ---


class ItemVerification(BaseModel):
    """Verification result for a single line item."""

    idx: int = Field(description="0-based index of item in the receipt")
    name_ok: bool = Field(default=True, description="Item name looks correct")
    amount_ok: bool = Field(default=True, description="Price/amount looks correct")
    quantity_ok: bool = Field(default=True, description="Quantity looks correct")
    suggested_name: str | None = Field(
        default=None, description="Corrected name if name_ok=False"
    )
    suggested_amount: str | None = Field(
        default=None, description="Corrected amount if amount_ok=False"
    )
    note: str | None = Field(default=None, description="Brief explanation of issue")


class ReceiptVerification(BaseModel):
    """Verification result for a single receipt."""

    doc_idx: int = Field(description="0-based index of receipt in the batch")
    vendor_ok: bool = Field(default=True, description="Vendor name correct")
    total_ok: bool = Field(default=True, description="Total matches sum of items")
    date_ok: bool = Field(default=True, description="Date looks valid")
    currency_ok: bool = Field(default=True, description="Currency consistent")
    items_sum_matches: bool = Field(
        default=True, description="Sum of item amounts matches total"
    )
    suggested_vendor: str | None = Field(default=None)
    suggested_total: str | None = Field(default=None)
    note: str | None = Field(default=None, description="Overall receipt issues")
    items: list[ItemVerification] = Field(
        default_factory=list, description="Per-item issues (only items with problems)"
    )


class VerificationBatchResponse(BaseModel):
    """Response for batch verification."""

    receipts: list[ReceiptVerification]


@dataclass
class VerificationResult:
    """Result for a single document verification."""

    doc_id: str
    issues: list[dict[str, Any]] = field(default_factory=list)
    all_ok: bool = True
    raw: dict[str, Any] | None = None


# --- System prompt ---

_SYSTEM_PROMPT = """\
You are a receipt verification expert. You will receive N extracted receipts \
(OCR text + extracted structured data). For each receipt, verify:

1. Does the vendor name match what appears in the text?
2. Does the total match the sum of line item amounts (accounting for tax/discounts)?
3. Are line item names reasonable (not garbled OCR)?
4. Are quantities and amounts plausible?
5. Is the date valid and consistent with any date patterns in the text?
6. Is the currency consistent across the receipt?

Only flag genuine issues. Minor formatting differences are OK. \
Focus on data accuracy, not cosmetic differences.\
"""


def _build_batch_prompt(documents: list[dict[str, Any]]) -> str:
    """Build verification prompt for N documents."""
    parts = []
    for i, doc in enumerate(documents):
        part = f"--- Receipt {i} ---\n"
        if doc.get("ocr_text"):
            part += f"OCR Text:\n{doc['ocr_text'][:3000]}\n\n"
        part += "Extracted Data:\n"
        extracted = {
            k: v
            for k, v in doc.items()
            if k not in ("ocr_text", "doc_id", "yaml_path") and v is not None
        }
        part += json.dumps(extracted, indent=2, default=str)
        parts.append(part)
    return "\n\n".join(parts)


def verify_batch(
    documents: list[dict[str, Any]],
    api_key: str | None = None,
    model: str | None = None,
) -> list[ReceiptVerification]:
    """Verify a batch of extracted receipts via Gemini.

    Args:
        documents: List of dicts with vendor, total, items, and optionally
            ocr_text for cross-validation.
        api_key: Override API key.
        model: Override model name.

    Returns:
        List of ReceiptVerification results.
    """
    resolved_key = _get_api_key(api_key)
    if not resolved_key:
        logger.warning("Gemini verification: no API key configured")
        return []

    resolved_model = model or _get_model()

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.error("google-genai package not installed")
        return []

    prompt = _build_batch_prompt(documents)
    client = genai.Client(api_key=resolved_key)

    try:
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=VerificationBatchResponse,
                temperature=0.1,
            ),
        )

        # Primary: structured parsing
        if response.parsed and isinstance(response.parsed, VerificationBatchResponse):
            return response.parsed.receipts

        # Fallback: JSON text parsing
        raw_text = response.text or ""
        parsed = json.loads(raw_text)
        receipts_data = parsed.get("receipts", [])
        return [ReceiptVerification(**r) for r in receipts_data]

    except Exception:
        logger.exception("Gemini verification: API call failed")
        return []


def verify_documents(
    db: Any,
    doc_ids: list[str] | None = None,
    limit: int = 20,
    api_key: str | None = None,
) -> list[VerificationResult]:
    """Load documents from DB and verify via Gemini.

    Args:
        db: DatabaseManager instance.
        doc_ids: Specific document IDs to verify. If None, picks recent.
        limit: Max documents to verify in one batch.
        api_key: Override API key.

    Returns:
        List of VerificationResult with issues found.
    """
    import yaml as _yaml

    if doc_ids:
        rows = []
        for did in doc_ids[:limit]:
            row = db.fetchone(
                "SELECT id, yaml_path FROM documents WHERE id = ?", (did,)
            )
            if row:
                rows.append(row)
    else:
        rows = db.fetchall(
            "SELECT id, yaml_path FROM documents " "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )

    if not rows:
        return []

    # Load YAML data for each document
    documents: list[dict[str, Any]] = []
    doc_id_map: dict[int, str] = {}  # idx -> doc_id
    for row in rows:
        doc_id = row[0]
        yaml_path = row[1]
        if not yaml_path or not Path(yaml_path).exists():
            continue

        try:
            with open(yaml_path) as f:
                data = _yaml.safe_load(f) or {}
        except Exception:
            logger.warning("Failed to load YAML for doc %s", doc_id)
            continue

        ocr_text = data.get("_meta", {}).get("ocr_text", "")
        doc_data = {k: v for k, v in data.items() if not k.startswith("_")}
        doc_data["ocr_text"] = ocr_text
        doc_data["doc_id"] = doc_id
        doc_id_map[len(documents)] = doc_id
        documents.append(doc_data)

    if not documents:
        return []

    verifications = verify_batch(documents, api_key=api_key)

    results = []
    for v in verifications:
        doc_id = doc_id_map.get(v.doc_idx, f"unknown-{v.doc_idx}")
        issues: list[dict[str, Any]] = []

        if not v.vendor_ok:
            issues.append(
                {"field": "vendor", "suggested": v.suggested_vendor, "note": v.note}
            )
        if not v.total_ok:
            issues.append(
                {"field": "total", "suggested": v.suggested_total, "note": v.note}
            )
        if not v.date_ok:
            issues.append({"field": "date", "note": v.note})
        if not v.currency_ok:
            issues.append({"field": "currency", "note": v.note})
        if not v.items_sum_matches:
            issues.append(
                {"field": "items_sum", "note": "Sum of items doesn't match total"}
            )

        for item in v.items:
            item_issues: dict[str, Any] = {}
            if not item.name_ok:
                item_issues["name"] = item.suggested_name
            if not item.amount_ok:
                item_issues["amount"] = item.suggested_amount
            if not item.quantity_ok:
                item_issues["quantity"] = True
            if item_issues:
                item_issues["item_idx"] = item.idx
                item_issues["note"] = item.note
                issues.append({"field": "item", **item_issues})

        results.append(
            VerificationResult(
                doc_id=doc_id,
                issues=issues,
                all_ok=len(issues) == 0,
                raw=v.model_dump(),
            )
        )

    return results
