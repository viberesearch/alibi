"""Verification layer for extraction results.

Runs pure-arithmetic checks on structured extraction output to score
confidence and recommend re-runs. All checks are fast and require no
LLM calls.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from alibi.extraction.prompts import get_text_extraction_prompt

logger = logging.getLogger(__name__)

# Weights for each check (must sum to 1.0)
_WEIGHTS = {
    "amount_sum": 0.35,
    "line_item_math": 0.25,
    "item_count": 0.15,
    "required_fields": 0.15,
    "date_valid": 0.10,
}

# Tolerance for amount comparisons (absolute, in currency units)
_AMOUNT_TOLERANCE = Decimal("0.05")

# Confidence threshold below which we recommend a re-run
RERUN_THRESHOLD = 0.5

# Confidence threshold below which we fall back to legacy vision
FALLBACK_THRESHOLD = 0.3


@dataclass
class VerificationResult:
    """Result of extraction verification."""

    confidence: float
    passed: bool
    flags: list[str] = field(default_factory=list)
    rerun_recommended: bool = False
    check_scores: dict[str, float] = field(default_factory=dict)


def _to_decimal(value: Any) -> Decimal | None:
    """Safely convert a value to Decimal."""
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _check_amount_sum(extracted: dict[str, Any]) -> tuple[float, list[str]]:
    """Check if line item totals sum to the document total.

    Tries both tax-inclusive and tax-exclusive comparisons.

    Returns (score 0-1, list of flag strings).
    """
    total = _to_decimal(extracted.get("total") or extracted.get("amount"))
    if total is None:
        return 0.5, ["no_total"]

    line_items = extracted.get("line_items") or []
    if not line_items:
        # No line items to check — can't verify, neutral score
        return 0.5, []

    item_sum = Decimal("0")
    for item in line_items:
        tp = _to_decimal(item.get("total_price"))
        if tp is not None:
            item_sum += tp

    if item_sum == Decimal("0"):
        return 0.5, ["no_item_prices"]

    # Try tax-inclusive match
    if abs(item_sum - total) <= _AMOUNT_TOLERANCE:
        return 1.0, []

    # Try matching against subtotal (items might be pre-tax)
    subtotal = _to_decimal(extracted.get("subtotal"))
    if subtotal is not None and abs(item_sum - subtotal) <= _AMOUNT_TOLERANCE:
        return 1.0, []

    # Try adding tax to item sum
    tax = _to_decimal(extracted.get("tax"))
    if tax is not None:
        if abs(item_sum + tax - total) <= _AMOUNT_TOLERANCE:
            return 0.9, []

    # Compute how far off we are as a proportion
    diff = abs(item_sum - total)
    if total > 0:
        pct_off = float(diff / total)
        if pct_off <= 0.01:
            return 0.9, []
        elif pct_off <= 0.05:
            return 0.7, [f"sum_off_{pct_off:.1%}"]
        elif pct_off <= 0.15:
            return 0.4, [f"sum_off_{pct_off:.1%}"]
        else:
            return 0.1, [f"sum_off_{pct_off:.1%}"]

    return 0.3, ["sum_mismatch"]


def _check_line_item_math(extracted: dict[str, Any]) -> tuple[float, list[str]]:
    """Check qty * unit_price ≈ total_price for each line item.

    Returns (score 0-1, flags).
    """
    line_items = extracted.get("line_items") or []
    if not line_items:
        return 0.5, []

    correct = 0
    checkable = 0
    flags: list[str] = []

    for i, item in enumerate(line_items):
        qty = _to_decimal(item.get("quantity"))
        up = _to_decimal(item.get("unit_price"))
        tp = _to_decimal(item.get("total_price"))

        if qty is not None and up is not None and tp is not None and qty > 0:
            checkable += 1
            expected = qty * up
            if abs(expected - tp) <= _AMOUNT_TOLERANCE:
                correct += 1
            else:
                flags.append(f"item_{i}_math")

    if checkable == 0:
        return 0.5, []

    return correct / checkable, flags


def _check_item_count(
    extracted: dict[str, Any], ocr_text: str | None
) -> tuple[float, list[str]]:
    """Check if OCR text mentions an item count matching line_items length.

    Looks for patterns like "23 Items", "15 Artikel", "Total: 8 items".
    """
    if not ocr_text:
        return 0.5, []

    line_items = extracted.get("line_items") or []
    if not line_items:
        return 0.5, []

    # Match patterns like "23 Items", "15 Artikel", "8 articles"
    patterns = [
        r"(\d+)\s*(?:items?|artikel|articles?|prod(?:ucts?|uits?)|pozycj[ie])",
    ]
    found_counts: list[int] = []
    for pattern in patterns:
        for match in re.finditer(pattern, ocr_text, re.IGNORECASE):
            found_counts.append(int(match.group(1)))

    if not found_counts:
        return 0.5, []

    actual_count = len(line_items)
    # Check if any found count matches
    for count in found_counts:
        if count == actual_count:
            return 1.0, []
        # Close match (within 2)
        if abs(count - actual_count) <= 2:
            return 0.7, [f"item_count_close_{count}_vs_{actual_count}"]

    # Significant mismatch
    return 0.2, [f"item_count_mismatch_{found_counts[0]}_vs_{actual_count}"]


def _check_required_fields(extracted: dict[str, Any]) -> tuple[float, list[str]]:
    """Check that essential fields are present.

    Required: vendor, date, total/amount, currency.
    """
    flags: list[str] = []
    present = 0
    total_fields = 4

    if extracted.get("vendor"):
        present += 1
    else:
        flags.append("missing_vendor")

    if extracted.get("date") or extracted.get("document_date"):
        present += 1
    else:
        flags.append("missing_date")

    if extracted.get("total") or extracted.get("amount"):
        present += 1
    else:
        flags.append("missing_total")

    if extracted.get("currency"):
        present += 1
    else:
        flags.append("missing_currency")

    return present / total_fields, flags


def _check_date_valid(extracted: dict[str, Any]) -> tuple[float, list[str]]:
    """Check that the date parses and is within a 2-year range."""
    date_str = extracted.get("date") or extracted.get("document_date")
    if not date_str:
        return 0.5, ["no_date"]

    parsed: date | None = None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y"):
        try:
            parsed = datetime.strptime(str(date_str), fmt).date()
            break
        except ValueError:
            continue

    if parsed is None:
        return 0.0, ["date_unparseable"]

    today = date.today()
    age_days = abs((today - parsed).days)
    if age_days <= 365 * 2:
        return 1.0, []

    return 0.2, [f"date_out_of_range_{age_days}d"]


def verify_extraction(
    extracted: dict[str, Any],
    ocr_text: str | None = None,
) -> VerificationResult:
    """Run all verification checks on an extraction result.

    Args:
        extracted: Structured extraction dict from Stage 2.
        ocr_text: Raw OCR text from Stage 1 (used for item count check).

    Returns:
        VerificationResult with confidence score and flags.
    """
    if not extracted:
        return VerificationResult(
            confidence=0.0,
            passed=False,
            flags=["empty_extraction"],
            rerun_recommended=True,
        )

    all_flags: list[str] = []
    weighted_score = 0.0
    check_scores: dict[str, float] = {}

    # Run each check
    score, flags = _check_amount_sum(extracted)
    check_scores["amount_sum"] = score
    weighted_score += score * _WEIGHTS["amount_sum"]
    all_flags.extend(flags)

    score, flags = _check_line_item_math(extracted)
    check_scores["line_item_math"] = score
    weighted_score += score * _WEIGHTS["line_item_math"]
    all_flags.extend(flags)

    score, flags = _check_item_count(extracted, ocr_text)
    check_scores["item_count"] = score
    weighted_score += score * _WEIGHTS["item_count"]
    all_flags.extend(flags)

    score, flags = _check_required_fields(extracted)
    check_scores["required_fields"] = score
    weighted_score += score * _WEIGHTS["required_fields"]
    all_flags.extend(flags)

    score, flags = _check_date_valid(extracted)
    check_scores["date_valid"] = score
    weighted_score += score * _WEIGHTS["date_valid"]
    all_flags.extend(flags)

    passed = weighted_score >= RERUN_THRESHOLD
    rerun = not passed

    return VerificationResult(
        confidence=round(weighted_score, 3),
        passed=passed,
        flags=all_flags,
        rerun_recommended=rerun,
        check_scores=check_scores,
    )


def build_emphasis_prompt(
    raw_text: str,
    doc_type: str,
    failed_checks: dict[str, float],
    prompt_mode: str = "specialized",
) -> str:
    """Build a re-structuring prompt that emphasizes failed verification areas.

    Args:
        raw_text: OCR text.
        doc_type: Document type.
        failed_checks: Dict of check_name -> score for checks that scored low.
        prompt_mode: 'specialized' or 'universal'.

    Returns:
        Augmented prompt string.
    """
    base = get_text_extraction_prompt(raw_text, doc_type, version=2, mode=prompt_mode)

    emphasis_parts: list[str] = []

    if failed_checks.get("amount_sum", 1.0) < 0.7:
        emphasis_parts.append(
            "PAY SPECIAL ATTENTION to the total amount. Ensure the total matches "
            "the sum of line item prices. Double-check every digit."
        )

    if failed_checks.get("line_item_math", 1.0) < 0.7:
        emphasis_parts.append(
            "For EACH line item, verify that quantity * unit_price = total_price. "
            "Fix any arithmetic errors."
        )

    if failed_checks.get("item_count", 1.0) < 0.7:
        emphasis_parts.append(
            "Make sure you extract ALL line items. Do not skip or merge items."
        )

    if failed_checks.get("required_fields", 1.0) < 0.7:
        emphasis_parts.append(
            "Ensure vendor name, date, total, and currency are all present."
        )

    if not emphasis_parts:
        return base

    emphasis = (
        "\n\n⚠️ IMPORTANT CORRECTIONS NEEDED:\n"
        + "\n".join(f"- {p}" for p in emphasis_parts)
        + "\n\n"
    )

    return base + emphasis
