"""Micro-prompt extraction for targeted LLM field repair.

Instead of sending the entire document to the LLM for correction,
generates small focused prompts for specific uncertain fields scoped
to their relevant OCR text region. Each micro-prompt asks about one
field (or a small group), reducing input tokens and improving accuracy.

Usage flow:
  1. Heuristic parser returns ParseResult with field_confidence + regions
  2. build_micro_prompts() identifies what needs LLM help
  3. Each prompt is sent independently to the structure model
  4. Responses are merged back into the parsed data

Replaces the monolithic correction prompt for confidence 0.3-0.9 cases.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

from alibi.extraction.text_parser import ParseResult, TextRegions

logger = logging.getLogger(__name__)

# Fields the LLM always provides (no OCR text needed)
SEMANTIC_FIELDS = {"name_en", "category", "brand", "language"}

# Map fields to their OCR text region
HEADER_FIELDS = {
    "vendor",
    "vendor_address",
    "vendor_phone",
    "vendor_website",
    "vendor_vat",
    "vendor_tax_id",
    "issuer",
    "issuer_address",
    "issuer_vat",
    "issuer_tax_id",
    "date",
    "time",
    "issue_date",
    "due_date",
    "invoice_number",
    "document_id",
    "institution",
    "account_type",
    "account_last4",
    "customer",
}

FOOTER_FIELDS = {
    "total",
    "amount",
    "subtotal",
    "tax",
    "discount",
    "payment_method",
    "card_type",
    "card_last4",
    "authorization_code",
    "terminal_id",
    "merchant_id",
    "currency",
}

BODY_FIELDS = {"line_items", "transactions"}

# Fields that can be grouped into a single header micro-prompt
HEADER_GROUPABLE = {
    "vendor",
    "vendor_address",
    "vendor_phone",
    "vendor_website",
    "vendor_vat",
    "vendor_tax_id",
    "date",
    "time",
    "document_id",
}

# Fields that can be grouped into a single footer micro-prompt
FOOTER_GROUPABLE = {
    "total",
    "subtotal",
    "tax",
    "discount",
    "currency",
    "payment_method",
    "card_type",
    "card_last4",
}

# Minimum uncertain fields to justify micro-prompts over correction prompt.
# Below this, the overhead of multiple LLM calls exceeds the benefit.
MIN_FIELDS_FOR_MICRO = 1

# Maximum micro-prompt calls before we fall back to monolithic correction.
# Prevents excessive API calls for very low-confidence documents.
MAX_MICRO_CALLS = 3


@dataclass
class MicroPrompt:
    """A targeted prompt for specific fields."""

    fields: list[str]
    region: str  # "header", "body", "footer"
    prompt: str
    ocr_text: str  # scoped OCR text for this prompt


def _get_region_text(regions: TextRegions | None, region: str) -> str | None:
    """Get OCR text for a specific region."""
    if regions is None:
        return None
    if region == "header":
        return regions.header or None
    if region == "body":
        return regions.body or None
    if region == "footer":
        return regions.footer or None
    return None


def _field_to_region(field_name: str) -> str:
    """Map a field name to its OCR text region."""
    if field_name in HEADER_FIELDS:
        return "header"
    if field_name in FOOTER_FIELDS:
        return "footer"
    if field_name in BODY_FIELDS:
        return "body"
    return "header"  # default


def _build_header_prompt(
    fields: list[str],
    ocr_text: str,
    doc_type: str,
    parsed_data: dict[str, Any],
) -> str:
    """Build a micro-prompt for header fields (vendor, date, etc.)."""
    field_list = ", ".join(fields)
    existing = {k: v for k, v in parsed_data.items() if k in fields and v is not None}
    existing_note = ""
    if existing:
        existing_note = (
            "\n\nAlready extracted (verify and correct if wrong):\n"
            + json.dumps(existing, indent=2, ensure_ascii=False)
        )

    return (
        f"Extract the following fields from this {doc_type} header text.\n"
        f"Fields needed: {field_list}\n"
        f"{existing_note}\n\n"
        f"--- OCR TEXT ---\n{ocr_text}\n--- END ---\n\n"
        "Return ONLY a JSON object with the requested fields. "
        "Use null for fields not found. Use YYYY-MM-DD for dates, HH:MM for times."
    )


def _build_footer_prompt(
    fields: list[str],
    ocr_text: str,
    doc_type: str,
    parsed_data: dict[str, Any],
) -> str:
    """Build a micro-prompt for footer fields (totals, payment, etc.)."""
    field_list = ", ".join(fields)
    existing = {k: v for k, v in parsed_data.items() if k in fields and v is not None}
    existing_note = ""
    if existing:
        existing_note = (
            "\n\nAlready extracted (verify and correct if wrong):\n"
            + json.dumps(existing, indent=2, ensure_ascii=False)
        )

    return (
        f"Extract the following fields from this {doc_type} footer/totals text.\n"
        f"Fields needed: {field_list}\n"
        f"{existing_note}\n\n"
        f"--- OCR TEXT ---\n{ocr_text}\n--- END ---\n\n"
        "Return ONLY a JSON object with the requested fields. "
        "Use null for fields not found. Amounts should be numbers (not strings)."
    )


def _build_line_items_prompt(
    ocr_text: str,
    doc_type: str,
    parsed_items: list[dict[str, Any]],
) -> str:
    """Build a micro-prompt for line item correction/enrichment."""
    items_json = json.dumps(parsed_items, indent=2, ensure_ascii=False)

    return (
        f"The following line items were pre-parsed from a {doc_type}. "
        "Some may have OCR errors or missing fields.\n\n"
        f"--- PRE-PARSED ITEMS ---\n{items_json}\n--- END ---\n\n"
        f"--- OCR TEXT (item section) ---\n{ocr_text}\n--- END ---\n\n"
        "Tasks:\n"
        "1. Fix OCR errors in item names, quantities, and prices.\n"
        "2. Verify math: quantity * unit_price = total_price.\n"
        "3. Measured items: if a number has 3 decimal places (e.g. 0.765, 1.535) "
        "it is a measured quantity — set quantity to that number, "
        "unit_raw to 'kg' for produce/meat/deli or 'l' for beverages/liquids, "
        "and the adjacent 2-decimal number is unit_price. "
        "The '#' symbol means the same as '@' (price separator).\n"
        "4. Add name_en (English translation) for non-English items.\n"
        "5. Add category (dairy/produce/bakery/beverages/meat/household/etc).\n"
        "6. Add brand if identifiable.\n\n"
        'Return ONLY a JSON object: {"line_items": [...]}\n'
        "Do NOT skip any items. Use null for unknown fields."
    )


def _build_enrichment_prompt(
    parsed_data: dict[str, Any],
    doc_type: str,
) -> str:
    """Build a prompt for semantic enrichment (categories, translations, language).

    Used when all structural fields are high-confidence but semantic
    fields (category, name_en, brand, language) need LLM help.
    """
    items = parsed_data.get("line_items", [])
    items_json = json.dumps(items, indent=2, ensure_ascii=False)
    vendor = parsed_data.get("vendor") or parsed_data.get("issuer") or "unknown"

    return (
        f"The following line items are from a {doc_type} by '{vendor}'.\n"
        "Add semantic fields for each item:\n"
        "- name_en: English translation (null if already English)\n"
        "- category: product category (dairy/produce/bakery/beverages/meat/"
        "household/frozen/snacks/personal_care/cleaning/etc)\n"
        "- brand: brand name if identifiable from the item name\n\n"
        f"Also determine the document language (ISO 639-1 code).\n\n"
        f"--- ITEMS ---\n{items_json}\n--- END ---\n\n"
        'Return a JSON object: {"language": "xx", "line_items": [...]}\n'
        "Each item must have: name, name_en, category, brand (plus existing fields)."
    )


def build_micro_prompts(
    parse_result: ParseResult,
    ocr_text: str,
    doc_type: str,
) -> list[MicroPrompt] | None:
    """Determine which micro-prompts to generate for uncertain fields.

    Returns a list of MicroPrompt objects, or None if micro-prompts
    aren't suitable (too many uncertain fields → use monolithic correction).

    Args:
        parse_result: Heuristic parser output with field_confidence.
        ocr_text: Full OCR text.
        doc_type: Document type string.

    Returns:
        List of MicroPrompt objects, or None to fall back to correction prompt.
    """
    fc = parse_result.field_confidence
    regions = parse_result.regions

    # Identify uncertain structural fields (not semantic)
    uncertain = {k: v for k, v in fc.items() if v < 1.0 and k not in SEMANTIC_FIELDS}

    if not uncertain:
        # Only semantic enrichment needed
        if parse_result.data.get("line_items"):
            enrichment_prompt = _build_enrichment_prompt(parse_result.data, doc_type)
            return [
                MicroPrompt(
                    fields=["language", "line_items"],
                    region="body",
                    prompt=enrichment_prompt,
                    ocr_text="",
                )
            ]
        return None

    # Group uncertain fields by region
    header_uncertain = [f for f in uncertain if f in HEADER_FIELDS]
    footer_uncertain = [f for f in uncertain if f in FOOTER_FIELDS]
    body_uncertain = [f for f in uncertain if f in BODY_FIELDS]

    prompts: list[MicroPrompt] = []

    # Header micro-prompt (group related header fields)
    if header_uncertain:
        region_text = _get_region_text(regions, "header")
        text = region_text if region_text else _first_n_lines(ocr_text, 15)
        prompts.append(
            MicroPrompt(
                fields=header_uncertain,
                region="header",
                prompt=_build_header_prompt(
                    header_uncertain, text, doc_type, parse_result.data
                ),
                ocr_text=text,
            )
        )

    # Footer micro-prompt (group related footer fields)
    if footer_uncertain:
        region_text = _get_region_text(regions, "footer")
        text = region_text if region_text else _last_n_lines(ocr_text, 15)
        prompts.append(
            MicroPrompt(
                fields=footer_uncertain,
                region="footer",
                prompt=_build_footer_prompt(
                    footer_uncertain, text, doc_type, parse_result.data
                ),
                ocr_text=text,
            )
        )

    # Body micro-prompt (line items or transactions)
    if body_uncertain:
        region_text = _get_region_text(regions, "body")
        text = region_text if region_text else ocr_text
        items = parse_result.data.get("line_items", [])
        if items:
            prompts.append(
                MicroPrompt(
                    fields=["line_items"],
                    region="body",
                    prompt=_build_line_items_prompt(text, doc_type, items),
                    ocr_text=text,
                )
            )

    # If too many prompts, fall back to monolithic correction
    if len(prompts) > MAX_MICRO_CALLS:
        logger.debug(
            f"Too many micro-prompts ({len(prompts)}), "
            "falling back to correction prompt"
        )
        return None

    # Add enrichment prompt if there are line items and no body micro-prompt
    has_body_prompt = any(p.region == "body" for p in prompts)
    if not has_body_prompt and parse_result.data.get("line_items"):
        enrichment_prompt = _build_enrichment_prompt(parse_result.data, doc_type)
        prompts.append(
            MicroPrompt(
                fields=["language", "line_items"],
                region="body",
                prompt=enrichment_prompt,
                ocr_text="",
            )
        )

    if not prompts:
        return None

    return prompts


def merge_micro_responses(
    base_data: dict[str, Any],
    micro_prompts: list[MicroPrompt],
    responses: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge micro-prompt responses back into the base parsed data.

    LLM responses for each micro-prompt are merged field by field.
    Non-null LLM values override parser values; null LLM values are
    skipped (parser's value kept).

    Args:
        base_data: Original parsed data from heuristic parser.
        micro_prompts: The micro-prompts that were sent.
        responses: LLM response dicts, one per micro-prompt.

    Returns:
        Merged data dict.
    """
    result = dict(base_data)

    for mp, response in zip(micro_prompts, responses):
        if not response:
            continue

        for field_name in mp.fields:
            if field_name == "line_items":
                # Merge line items: use LLM items if they have more data
                llm_items = response.get("line_items", [])
                if llm_items:
                    result["line_items"] = _merge_line_items(
                        result.get("line_items", []), llm_items
                    )
            elif field_name == "language":
                lang = response.get("language")
                if lang:
                    result["language"] = lang
            else:
                val = response.get(field_name)
                if val is not None:
                    result[field_name] = val

    return result


def _merge_line_items(
    parser_items: list[dict[str, Any]],
    llm_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge parser and LLM line items.

    Strategy: if LLM returned the same number of items, merge field-by-field
    (LLM non-null overrides parser). If counts differ, prefer the longer list
    and patch from the shorter where names match.
    """
    if not parser_items:
        return llm_items
    if not llm_items:
        return parser_items

    # Same count: 1:1 merge
    if len(parser_items) == len(llm_items):
        merged = []
        for p_item, l_item in zip(parser_items, llm_items):
            item = dict(p_item)
            for k, v in l_item.items():
                if v is not None:
                    item[k] = v
            merged.append(item)
        return merged

    # Different counts: keep longer list, enrich from shorter by name match
    if len(llm_items) >= len(parser_items):
        base, donor = llm_items, parser_items
    else:
        base, donor = parser_items, llm_items

    # Build name→item lookup from donor
    donor_map: dict[str, dict[str, Any]] = {}
    for item in donor:
        name = (item.get("name") or "").strip().lower()
        if name:
            donor_map[name] = item

    merged = []
    for item in base:
        result_item = dict(item)
        name = (item.get("name") or "").strip().lower()
        if name and name in donor_map:
            for k, v in donor_map[name].items():
                if v is not None and result_item.get(k) is None:
                    result_item[k] = v
        merged.append(result_item)

    return merged


def run_micro_prompts(
    parse_result: ParseResult,
    ocr_text: str,
    doc_type: str,
    ollama_url: str | None = None,
    model: str | None = None,
    timeout: float = 120.0,
) -> dict[str, Any] | None:
    """Build and execute micro-prompts, returning merged extraction data.

    Returns None if micro-prompts are not suitable (caller should fall
    back to monolithic correction prompt).

    Args:
        parse_result: Heuristic parser output.
        ocr_text: Full OCR text.
        doc_type: Document type string.
        ollama_url: Ollama API URL.
        model: Structure model name.
        timeout: Per-request timeout.

    Returns:
        Merged extraction dict with _pipeline="micro_prompts", or None.
    """
    prompts = build_micro_prompts(parse_result, ocr_text, doc_type)
    if not prompts:
        return None

    from alibi.extraction.structurer import structure_ocr_text

    responses: list[dict[str, Any]] = []
    for mp in prompts:
        try:
            response = structure_ocr_text(
                ocr_text,
                doc_type=doc_type,
                ollama_url=ollama_url,
                model=model,
                timeout=timeout,
                emphasis_prompt=mp.prompt,
            )
            responses.append(response)
            logger.debug(
                f"Micro-prompt [{mp.region}] for {mp.fields}: "
                f"{len(response)} keys returned"
            )
        except Exception as e:
            logger.warning(f"Micro-prompt [{mp.region}] for {mp.fields} failed: {e}")
            responses.append({})

    merged = merge_micro_responses(parse_result.data, prompts, responses)
    merged["_pipeline"] = "micro_prompts"
    merged["_parser_confidence"] = parse_result.confidence
    merged["_micro_prompt_count"] = len(prompts)
    return merged


def _first_n_lines(text: str, n: int) -> str:
    """Get first n lines of text."""
    lines = text.split("\n")
    return "\n".join(lines[:n])


def _last_n_lines(text: str, n: int) -> str:
    """Get last n lines of text."""
    lines = text.split("\n")
    return "\n".join(lines[-n:])
