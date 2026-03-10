"""Gemini-assisted template bootstrapping for new vendor documents.

When a document from a new/unknown vendor is processed, sends it to Gemini
for comprehensive extraction to:

1. Fill gaps in parser output (vendor contacts, VAT, tax ID, address, etc.)
2. Learn document schema (layout type, field patterns, item structure)
3. Store vendor details on identity for future documents

This enables "learn once, apply forever" — the first document from a new
vendor triggers a Gemini call, and all subsequent documents benefit from
the cached vendor details and enhanced template.

Trigger: parser-only extraction (confidence >= threshold) + no existing
vendor template (or template not yet Gemini-bootstrapped).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Document-type-specific vendor field mappings
# Maps extraction dict keys to normalized vendor_details keys
_VENDOR_FIELD_MAP: dict[str, dict[str, str]] = {
    "receipt": {
        "vendor_address": "address",
        "vendor_phone": "phone",
        "vendor_website": "website",
        "vendor_legal_name": "legal_name",
        "vendor_vat": "vat",
        "vendor_tax_id": "tax_id",
    },
    "payment_confirmation": {
        "vendor_address": "address",
        "vendor_phone": "phone",
        "vendor_website": "website",
        "vendor_legal_name": "legal_name",
        "vendor_vat": "vat",
        "vendor_tax_id": "tax_id",
    },
    "invoice": {
        "issuer_address": "address",
        "issuer_phone": "phone",
        "issuer_website": "website",
        "issuer_legal_name": "legal_name",
        "issuer_vat": "vat",
        "issuer_tax_id": "tax_id",
    },
}

# Reverse map: vendor_details key -> extraction field per doc_type
_DETAILS_TO_EXTRACTION: dict[str, dict[str, str]] = {
    "receipt": {
        "address": "vendor_address",
        "phone": "vendor_phone",
        "website": "vendor_website",
        "legal_name": "vendor_legal_name",
        "vat": "vendor_vat",
        "tax_id": "vendor_tax_id",
    },
    "payment_confirmation": {
        "address": "vendor_address",
        "phone": "vendor_phone",
        "website": "vendor_website",
        "legal_name": "vendor_legal_name",
        "vat": "vendor_vat",
        "tax_id": "vendor_tax_id",
    },
    "invoice": {
        "address": "issuer_address",
        "phone": "issuer_phone",
        "website": "issuer_website",
        "legal_name": "issuer_legal_name",
        "vat": "issuer_vat",
        "tax_id": "issuer_tax_id",
    },
}


def needs_bootstrapping(
    existing_template: Any,
    gemini_enabled: bool,
) -> bool:
    """Check if this vendor needs Gemini template bootstrapping.

    Returns True when:
    - Gemini extraction is enabled
    - No existing template OR template not yet Gemini-bootstrapped
    """
    if not gemini_enabled:
        return False
    if existing_template is None:
        return True
    return not existing_template.gemini_bootstrapped


def bootstrap_with_gemini(
    ocr_text: str,
    doc_type: str,
    api_key: str | None = None,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Run Gemini extraction for comprehensive field capture.

    Uses the existing Gemini structurer to extract all fields from OCR text.
    Returns extraction dict or None on failure.
    """
    if not ocr_text or not ocr_text.strip():
        return None

    try:
        from alibi.extraction.gemini_structurer import (
            GeminiExtractionError,
            structure_ocr_text_gemini,
        )

        result = structure_ocr_text_gemini(
            ocr_text,
            doc_type=doc_type,
            api_key=api_key,
            model=model,
        )
        if result:
            result["_pipeline"] = "gemini_bootstrap"
            logger.info(
                f"Gemini bootstrap extraction: "
                f"{len(result.get('line_items', []))} items, "
                f"vendor={result.get('vendor') or result.get('issuer')}"
            )
        return result
    except Exception as e:
        logger.warning(f"Gemini bootstrap failed: {e}")
        return None


def merge_extraction(
    parser_data: dict[str, Any],
    gemini_data: dict[str, Any],
    doc_type: str,
) -> dict[str, Any]:
    """Merge Gemini fields into parser result, filling gaps.

    Strategy: parser data is primary (it's deterministic and fast).
    Gemini fills fields the parser missed (vendor details, contacts,
    additional item fields). Does NOT overwrite existing parser values.

    For line_items: if Gemini found more items or richer item data,
    merge per-item fields (brand, barcode, category, unit_quantity).
    """
    if not gemini_data:
        return parser_data

    merged = dict(parser_data)

    # Merge top-level scalar fields (fill gaps only)
    _SKIP_KEYS = {
        "line_items",
        "transactions",
        "vat_analysis",
        "_pipeline",
        "_parser_confidence",
        "_layout_type",
        "raw_text",
    }
    for key, value in gemini_data.items():
        if key in _SKIP_KEYS:
            continue
        if key.startswith("_"):
            continue
        existing = merged.get(key)
        if not existing and value:
            merged[key] = value

    # Merge line items (enrich parser items with Gemini fields)
    parser_items = merged.get("line_items", [])
    gemini_items = gemini_data.get("line_items", [])

    if parser_items and gemini_items:
        merged["line_items"] = _merge_line_items(parser_items, gemini_items)
    elif not parser_items and gemini_items:
        # Parser found no items, Gemini did — use Gemini's
        merged["line_items"] = gemini_items

    # Mark as bootstrapped
    merged["_gemini_bootstrapped"] = True

    return merged


def _merge_line_items(
    parser_items: list[dict[str, Any]],
    gemini_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge per-item fields from Gemini into parser items.

    Matches items by position (index alignment) — both parse the same
    document so item order should match. For each matched pair, fills
    missing fields from Gemini (brand, barcode, category, unit_quantity,
    name_en).
    """
    result = []
    enrichable_fields = {
        "brand",
        "barcode",
        "category",
        "unit_quantity",
        "unit_raw",
        "name_en",
        "tax_rate",
        "tax_type",
    }

    for i, p_item in enumerate(parser_items):
        merged_item = dict(p_item)
        if i < len(gemini_items):
            g_item = gemini_items[i]
            for field in enrichable_fields:
                existing = merged_item.get(field)
                gemini_val = g_item.get(field)
                if not existing and gemini_val:
                    merged_item[field] = gemini_val
        result.append(merged_item)

    return result


def extract_vendor_details(
    extraction_data: dict[str, Any],
    doc_type: str,
) -> dict[str, str]:
    """Extract vendor contact details from extraction dict.

    Maps doc-type-specific fields (vendor_address, issuer_address, etc.)
    to normalized keys (address, phone, website, legal_name, vat, tax_id).
    """
    field_map = _VENDOR_FIELD_MAP.get(doc_type, _VENDOR_FIELD_MAP["receipt"])
    details: dict[str, str] = {}

    for extract_key, detail_key in field_map.items():
        value = extraction_data.get(extract_key)
        if value and isinstance(value, str) and value.strip():
            details[detail_key] = value.strip()

    return details


def apply_vendor_details(
    extraction_data: dict[str, Any],
    vendor_details: dict[str, str],
    doc_type: str,
) -> dict[str, Any]:
    """Apply cached vendor details to extraction, filling gaps.

    Used for subsequent documents from a known vendor — pre-fills
    vendor contact fields the parser may have missed.
    """
    if not vendor_details:
        return extraction_data

    detail_map = _DETAILS_TO_EXTRACTION.get(
        doc_type, _DETAILS_TO_EXTRACTION.get("receipt", {})
    )
    updated = dict(extraction_data)

    for detail_key, extract_key in detail_map.items():
        cached = vendor_details.get(detail_key)
        existing = updated.get(extract_key)
        if cached and not existing:
            updated[extract_key] = cached

    return updated


def build_enhanced_template(
    extraction_data: dict[str, Any],
    ocr_text: str,
    doc_type: str,
) -> Any:
    """Build an enhanced template from Gemini extraction results.

    Captures document schema insights beyond basic layout:
    - Language of the document
    - Whether items have barcodes, unit quantities
    - Typical number of line items
    """
    from alibi.extraction.templates import (
        VendorTemplate,
        detect_layout_type,
        detect_pos_provider,
    )

    layout = detect_layout_type(extraction_data)
    currency = extraction_data.get("currency")
    pos = detect_pos_provider(ocr_text)
    language = extraction_data.get("language")

    items = extraction_data.get("line_items", [])
    has_barcodes = any(i.get("barcode") for i in items) if items else None
    has_uq = any(i.get("unit_quantity") for i in items) if items else None
    typical_count = len(items) if items is not None else None

    return VendorTemplate(
        layout_type=layout,
        currency=currency,
        pos_provider=pos,
        success_count=1,
        gemini_bootstrapped=True,
        language=language,
        has_barcodes=has_barcodes,
        has_unit_quantities=has_uq,
        typical_item_count=typical_count,
    )
