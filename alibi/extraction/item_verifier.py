"""3-layer item verification and correction.

Layer 1 (deterministic): fixes embedded weights, leading quantities,
math mismatches, header/garbage rows — zero LLM cost.

Layer 2 (targeted LLM): sends only flagged items to the LLM for
correction — minimal cost, called only when Layer 1 flags exist.

Layer 3 (cross-validation): checks sum(item totals) vs receipt total,
item count vs declared count, barcode-item cross-validation with auto-correction.

Injection point: _extract_to_yaml() in pipeline.py, between vendor/locale
gap fill and YAML write.
"""

import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ItemCorrection:
    """A correction applied to a line item."""

    item_index: int
    field: str
    old_value: Any
    new_value: Any
    reason: str


@dataclass
class ItemFlag:
    """An item flagged for LLM review."""

    item_index: int
    issue: str
    context: str


@dataclass
class ItemVerificationResult:
    """Aggregate result from Layer 1 verification."""

    corrections: list[ItemCorrection] = field(default_factory=list)
    flags: list[ItemFlag] = field(default_factory=list)
    removed_indices: list[int] = field(default_factory=list)
    items_verified: int = 0
    items_corrected: int = 0
    items_flagged: int = 0
    items_removed: int = 0


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# Pattern 1: "NAME 0.765 3.50" — 3-decimal measured qty + 2-decimal price at end
_EMBEDDED_WEIGHT_P1 = re.compile(r"^(.+?)\s+(\d+[.,]\d{3})\s+(\d+[.,]\d{2})\s*$")

# Pattern 2: "NAME 0.765 # 3.50 each" — already matched by parser after
# Phase 1 fix, but kept as safety net
_EMBEDDED_WEIGHT_P2 = re.compile(
    r"^(.+?)\s+(\d+[.,]\d{3})\s*[#@]\s*(\d+[.,]\d{2})\s*(?:each)?\s*$",
    re.IGNORECASE,
)

# Pattern 3: "0.765 12345 NAME 3.50" — weight + product code prefix
_EMBEDDED_WEIGHT_P3 = re.compile(
    r"^(\d+[.,]\d{3})\s+(\d{3,})\s+(.+?)\s+(\d+[.,]\d{2})\s*$"
)

# Leading quantity: "4 Red Bull 250ml 3.96"
_LEADING_QTY = re.compile(r"^(\d{1,2})\s+(.+)$")

# Deposit/discount/refund — exempt from zero-price flagging
_EXEMPT_NAMES = re.compile(
    r"deposit|pfand|rabatt|discount|refund|coupon|gutschein|εκπτωση|возврат",
    re.IGNORECASE,
)

# Header/garbage patterns
_HEADER_PATTERNS = [
    re.compile(
        r"^(QTY|QUANTITY|DESCRIPTION|PRICE|AMOUNT|UNIT|ITEM|CODE|VAT|TAX)"
        r"(\s+(QTY|QUANTITY|DESCRIPTION|PRICE|AMOUNT|UNIT|ITEM|CODE|VAT|TAX))+\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"^VAT\s+\d+[.,]\d{2}\s*%", re.IGNORECASE),
    re.compile(r"^\s*[-=]{3,}\s*$"),
    # Receipt ticket/transaction metadata leaked as item names
    re.compile(r"^TICKET\s*:", re.IGNORECASE),
]

_MATH_TOLERANCE = Decimal("0.02")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_decimal(value: Any) -> Decimal | None:
    """Safely convert a value to Decimal."""
    if value is None:
        return None
    try:
        s = str(value).replace(",", ".")
        return Decimal(s)
    except (InvalidOperation, ValueError):
        return None


def _parse_embedded(s: str) -> Decimal | None:
    """Parse a number string from regex match into Decimal."""
    return _to_decimal(s.replace(",", "."))


# ---------------------------------------------------------------------------
# Layer 1 sub-functions
# ---------------------------------------------------------------------------


def _is_header_or_garbage(item: dict[str, Any]) -> bool:
    """Detect column headers, VAT lines, short/all-digit names."""
    name = str(item.get("name") or "").strip()

    # Too short
    if len(name) < 2:
        return True

    # All digits (product codes leaked as items)
    if name.replace(" ", "").isdigit():
        return True

    # Multi-word header combos
    for pat in _HEADER_PATTERNS:
        if pat.match(name):
            return True

    return False


def _fix_embedded_weight(
    item: dict[str, Any], index: int, result: ItemVerificationResult
) -> bool:
    """Fix weight embedded in item name. Returns True if fixed."""
    name = str(item.get("name") or "")
    qty = _to_decimal(item.get("quantity"))
    unit = str(item.get("unit_raw") or item.get("unit") or "").lower()

    # Only fix qty=1 items without a weight/volume unit already set
    if qty is not None and qty != Decimal("1"):
        return False
    if unit in ("kg", "g", "gr", "lb", "lbs", "ml", "l", "lt", "ltr", "cl", "oz"):
        return False

    # Pattern 1: "NAME 0.765 3.50"
    m = _EMBEDDED_WEIGHT_P1.match(name)
    if m:
        weight = _parse_embedded(m.group(2))
        price = _parse_embedded(m.group(3))
        if weight and price and Decimal("0.001") <= weight <= Decimal("50"):
            clean_name = m.group(1).strip()
            old_name = name
            item["name"] = clean_name
            item["quantity"] = str(weight)
            # Don't assume unit — could be kg or L. Leave for
            # atom parser / LLM to resolve from item name context.
            item.pop("unit_raw", None)
            item.pop("unit", None)
            item["unit_price"] = str(price)
            # Compute total_price
            total = (weight * price).quantize(Decimal("0.01"))
            item["total_price"] = str(total)
            result.corrections.append(
                ItemCorrection(
                    index,
                    "name",
                    old_name,
                    clean_name,
                    "embedded weight extracted from name",
                )
            )
            result.items_corrected += 1
            return True

    # Pattern 2: "NAME 0.765 # 3.50 each"
    m = _EMBEDDED_WEIGHT_P2.match(name)
    if m:
        weight = _parse_embedded(m.group(2))
        price = _parse_embedded(m.group(3))
        if weight and price and Decimal("0.001") <= weight <= Decimal("50"):
            clean_name = m.group(1).strip()
            old_name = name
            item["name"] = clean_name
            item["quantity"] = str(weight)
            # Don't assume unit — could be kg or L. Leave for
            # atom parser / LLM to resolve from item name context.
            item.pop("unit_raw", None)
            item.pop("unit", None)
            item["unit_price"] = str(price)
            total = (weight * price).quantize(Decimal("0.01"))
            item["total_price"] = str(total)
            result.corrections.append(
                ItemCorrection(
                    index,
                    "name",
                    old_name,
                    clean_name,
                    "embedded weight (# separator) extracted",
                )
            )
            result.items_corrected += 1
            return True

    # Pattern 3: "0.765 12345 NAME 3.50"
    m = _EMBEDDED_WEIGHT_P3.match(name)
    if m:
        weight = _parse_embedded(m.group(1))
        price = _parse_embedded(m.group(4))
        if weight and price and Decimal("0.001") <= weight <= Decimal("50"):
            clean_name = f"{m.group(2)} {m.group(3)}".strip()
            old_name = name
            item["name"] = clean_name
            item["quantity"] = str(weight)
            # Don't assume unit — could be kg or L. Leave for
            # atom parser / LLM to resolve from item name context.
            item.pop("unit_raw", None)
            item.pop("unit", None)
            item["unit_price"] = str(price)
            total = (weight * price).quantize(Decimal("0.01"))
            item["total_price"] = str(total)
            result.corrections.append(
                ItemCorrection(
                    index,
                    "name",
                    old_name,
                    clean_name,
                    "weight+code prefix extracted from name",
                )
            )
            result.items_corrected += 1
            return True

    return False


def _fix_leading_quantity(
    item: dict[str, Any], index: int, result: ItemVerificationResult
) -> bool:
    """Fix leading integer quantity like '4 Red Bull 250ml'. Returns True if fixed."""
    name = str(item.get("name") or "")
    unit_price = _to_decimal(item.get("unit_price"))
    total_price = _to_decimal(item.get("total_price"))

    if not unit_price or not total_price:
        return False

    m = _LEADING_QTY.match(name)
    if not m:
        return False

    candidate_qty = int(m.group(1))
    if candidate_qty < 2 or candidate_qty > 99:
        return False

    expected = (Decimal(str(candidate_qty)) * unit_price).quantize(Decimal("0.01"))
    if abs(expected - total_price) <= _MATH_TOLERANCE:
        clean_name = m.group(2).strip()
        old_qty = item.get("quantity")
        item["name"] = clean_name
        item["quantity"] = str(candidate_qty)
        result.corrections.append(
            ItemCorrection(
                index,
                "quantity",
                old_qty,
                str(candidate_qty),
                f"leading quantity {candidate_qty} extracted from name",
            )
        )
        result.items_corrected += 1
        return True

    return False


def _validate_item_math(
    item: dict[str, Any], index: int, result: ItemVerificationResult
) -> None:
    """Check qty * unit_price == total_price, try to fix from name numbers."""
    qty = _to_decimal(item.get("quantity"))
    unit_price = _to_decimal(item.get("unit_price"))
    total_price = _to_decimal(item.get("total_price"))

    if not qty or not unit_price or not total_price:
        return

    expected = (qty * unit_price).quantize(Decimal("0.01"))
    if abs(expected - total_price) <= _MATH_TOLERANCE:
        return  # Math checks out

    # Detect net/gross confusion: unit_price is net, total_price is gross
    # This happens when LLM back-calculates net from total/VAT rate
    if qty == Decimal("1") and unit_price and total_price and unit_price < total_price:
        tax_rate_val = item.get("tax_rate")
        if tax_rate_val is not None:
            try:
                tax_rate = Decimal(str(tax_rate_val))
                if tax_rate > 0:
                    expected_gross = (unit_price * (1 + tax_rate / 100)).quantize(
                        Decimal("0.01")
                    )
                    if abs(expected_gross - total_price) <= Decimal("0.02"):
                        item["unit_price"] = float(total_price)
                        result.corrections.append(
                            ItemCorrection(
                                index,
                                "unit_price",
                                float(unit_price),
                                float(total_price),
                                f"unit_price {unit_price} was net "
                                f"(ex-VAT), corrected to gross {total_price}",
                            )
                        )
                        result.items_corrected += 1
                        return
            except (InvalidOperation, ValueError):
                pass

    # Try numbers in the name as candidate quantity
    name = str(item.get("name") or "")
    candidates = re.findall(r"(\d+[.,]\d+)", name)
    for c_str in candidates:
        c = _to_decimal(c_str)
        if c and c > Decimal("0"):
            c_expected = (c * unit_price).quantize(Decimal("0.01"))
            if abs(c_expected - total_price) <= _MATH_TOLERANCE:
                old_qty = item.get("quantity")
                # Clean the number from the name
                cleaned = name.replace(c_str, "", 1).strip()
                cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
                if cleaned:
                    item["name"] = cleaned
                item["quantity"] = str(c)
                # 3-decimal number means measured quantity (kg or L);
                # clear unit so atom parser resolves from name context
                if re.match(r"\d+[.,]\d{3}$", c_str):
                    item.pop("unit_raw", None)
                    item.pop("unit", None)
                result.corrections.append(
                    ItemCorrection(
                        index,
                        "quantity",
                        old_qty,
                        str(c),
                        f"math fix: {c} * {unit_price} = {total_price}",
                    )
                )
                result.items_corrected += 1
                return

    # Unfixable — flag for LLM
    result.flags.append(
        ItemFlag(
            index,
            "math_mismatch",
            f"qty={qty} * unit_price={unit_price} = {expected}, "
            f"expected total_price={total_price}",
        )
    )
    result.items_flagged += 1


def _check_unreasonable_price(
    item: dict[str, Any], index: int, result: ItemVerificationResult
) -> None:
    """Flag unreasonable prices likely from OCR digit merge.

    OCR sometimes merges adjacent numbers (e.g., "5.50 11.00" -> "5011.0").
    Detect and flag items with total_price > 500 as likely OCR artifacts.
    """
    total_price = _to_decimal(item.get("total_price"))
    if total_price is None:
        return

    # Threshold: individual grocery item > 500 is almost certainly OCR error
    if total_price > Decimal("500"):
        name = str(item.get("name") or "")
        result.flags.append(
            ItemFlag(
                index,
                "unreasonable_price",
                f"item '{name}' total_price={total_price} exceeds 500 "
                f"(likely OCR digit merge)",
            )
        )
        result.items_flagged += 1


def _check_zero_price(
    item: dict[str, Any], index: int, result: ItemVerificationResult
) -> None:
    """Flag zero/null unit_price unless deposit/discount/refund."""
    unit_price = _to_decimal(item.get("unit_price"))
    total_price = _to_decimal(item.get("total_price"))
    name = str(item.get("name") or "")

    if unit_price is not None and unit_price != Decimal("0"):
        return
    if total_price is not None and total_price != Decimal("0"):
        return

    # Exempt categories
    if _EXEMPT_NAMES.search(name):
        return

    result.flags.append(
        ItemFlag(index, "zero_price", f"item '{name}' has zero/null price")
    )
    result.items_flagged += 1


def _detect_price_swap(
    items: list[dict[str, Any]], result: ItemVerificationResult
) -> None:
    """Detect and fix price/total swapping between adjacent items.

    When item[i].qty * item[i].unit_price != item[i].total_price,
    check if swapping total_price with item[i+1] fixes both items' math.
    """
    for i in range(len(items) - 1):
        a = items[i]
        b = items[i + 1]

        a_qty = _to_decimal(a.get("quantity"))
        a_up = _to_decimal(a.get("unit_price"))
        a_tp = _to_decimal(a.get("total_price"))
        b_qty = _to_decimal(b.get("quantity"))
        b_up = _to_decimal(b.get("unit_price"))
        b_tp = _to_decimal(b.get("total_price"))

        if not all([a_qty, a_up, a_tp, b_qty, b_up, b_tp]):
            continue

        # Narrowing for mypy (all() guard ensures non-None)
        assert a_qty is not None and a_up is not None and a_tp is not None
        assert b_qty is not None and b_up is not None and b_tp is not None

        a_expected = (a_qty * a_up).quantize(Decimal("0.01"))
        b_expected = (b_qty * b_up).quantize(Decimal("0.01"))
        a_ok = abs(a_expected - a_tp) <= _MATH_TOLERANCE
        b_ok = abs(b_expected - b_tp) <= _MATH_TOLERANCE

        if a_ok and b_ok:
            continue  # Both are fine

        # Try swap: does a's math work with b's total, and vice versa?
        if (
            abs(a_expected - b_tp) <= _MATH_TOLERANCE
            and abs(b_expected - a_tp) <= _MATH_TOLERANCE
        ):
            old_a_tp = str(a_tp)
            old_b_tp = str(b_tp)
            a["total_price"], b["total_price"] = b["total_price"], a["total_price"]
            result.corrections.append(
                ItemCorrection(
                    i,
                    "total_price",
                    old_a_tp,
                    str(b_tp),
                    f"price swap with adjacent item {i + 1}",
                )
            )
            result.corrections.append(
                ItemCorrection(
                    i + 1,
                    "total_price",
                    old_b_tp,
                    str(a_tp),
                    f"price swap with adjacent item {i}",
                )
            )
            result.items_corrected += 2


def _merge_continuation_items(
    items: list[dict[str, Any]], result: ItemVerificationResult
) -> list[int]:
    """Detect and merge continuation line items (no prices) into adjacent items.

    When LLM splits multi-line names (e.g. "EGGS JONIS" + "FARM (1X30)")
    into separate items where one has no prices, merge them.

    Returns indices to remove.
    """
    remove: list[int] = []
    i = 0
    while i < len(items):
        item = items[i]
        name = str(item.get("name") or "").strip()
        has_price = (
            _to_decimal(item.get("unit_price")) is not None
            or _to_decimal(item.get("total_price")) is not None
        )

        if has_price or not name or len(name) > 40:
            i += 1
            continue

        # This item has no prices — candidate for merge
        # Prefer merging into previous item
        if i > 0 and i - 1 not in remove:
            prev = items[i - 1]
            prev_has_price = (
                _to_decimal(prev.get("unit_price")) is not None
                or _to_decimal(prev.get("total_price")) is not None
            )
            if prev_has_price:
                old_name = str(prev.get("name") or "")
                prev["name"] = old_name + " " + name
                result.corrections.append(
                    ItemCorrection(
                        i - 1,
                        "name",
                        old_name,
                        prev["name"],
                        f"merged continuation item {i}: '{name}'",
                    )
                )
                result.items_corrected += 1
                remove.append(i)
                i += 1
                continue

        # Try next item
        if i + 1 < len(items):
            nxt = items[i + 1]
            nxt_has_price = (
                _to_decimal(nxt.get("unit_price")) is not None
                or _to_decimal(nxt.get("total_price")) is not None
            )
            if nxt_has_price:
                old_name = str(nxt.get("name") or "")
                nxt["name"] = name + " " + old_name
                result.corrections.append(
                    ItemCorrection(
                        i + 1,
                        "name",
                        old_name,
                        nxt["name"],
                        f"merged continuation item {i}: '{name}'",
                    )
                )
                result.items_corrected += 1
                remove.append(i)

        i += 1

    return remove


# ---------------------------------------------------------------------------
# Layer 1 entry point
# ---------------------------------------------------------------------------


def verify_items(extracted: dict[str, Any]) -> ItemVerificationResult:
    """Layer 1: deterministic item verification and correction.

    Mutates extracted["line_items"] in-place. Returns verification result.
    """
    result = ItemVerificationResult()
    items = extracted.get("line_items")
    if not items or not isinstance(items, list):
        return result

    # Phase 0: Merge continuation items (priceless lines → adjacent items)
    merge_remove = _merge_continuation_items(items, result)

    remove_indices: list[int] = list(merge_remove)

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if i in remove_indices:
            continue  # Skip merged items
        result.items_verified += 1

        # 1. Header/garbage removal
        if _is_header_or_garbage(item):
            remove_indices.append(i)
            result.items_removed += 1
            continue

        # 2. Embedded weight fix
        if _fix_embedded_weight(item, i, result):
            continue  # Already corrected, skip other fixes

        # 3. Leading quantity fix
        if _fix_leading_quantity(item, i, result):
            continue

        # 4. Math validation (may fix or flag)
        _validate_item_math(item, i, result)

        # 5. Unreasonable price check (OCR digit merge detection)
        _check_unreasonable_price(item, i, result)

        # 6. Zero price check
        _check_zero_price(item, i, result)

    # Phase 2: Cross-item price swap detection (after per-item checks)
    _detect_price_swap(items, result)

    # Remove merged/garbage items (reverse order to preserve indices)
    for i in sorted(set(remove_indices), reverse=True):
        removed = items.pop(i)
        result.removed_indices.append(i)
        logger.debug("Removed item %d: %s", i, removed.get("name", "?"))

    if result.items_corrected or result.items_removed:
        logger.info(
            "Item verification: %d verified, %d corrected, %d removed, %d flagged",
            result.items_verified,
            result.items_corrected,
            result.items_removed,
            result.items_flagged,
        )

    return result


# ---------------------------------------------------------------------------
# Layer 2: targeted LLM verification
# ---------------------------------------------------------------------------


def verify_items_llm(
    extracted: dict[str, Any],
    flags: list[ItemFlag],
    ocr_text: str,
    config: Any | None = None,
) -> list[ItemCorrection]:
    """Layer 2: send flagged items to LLM for targeted correction.

    Only called when Layer 1 produces flags. Builds a focused prompt
    with just the flagged items + receipt total + relevant OCR snippet.

    Returns list of corrections applied.
    """
    import json

    from alibi.config import get_config
    from alibi.extraction.structurer import structure_ocr_text

    config = config or get_config()
    corrections: list[ItemCorrection] = []

    items = extracted.get("line_items", [])
    if not items or not flags:
        return corrections

    # Build focused context
    flagged_indices = {f.item_index for f in flags}
    flagged_items = []
    for f in flags:
        if 0 <= f.item_index < len(items):
            flagged_items.append(
                {
                    "index": f.item_index,
                    "issue": f.issue,
                    "context": f.context,
                    "item": items[f.item_index],
                }
            )

    if not flagged_items:
        return corrections

    receipt_total = extracted.get("total") or extracted.get("total_amount")
    items_json = json.dumps(flagged_items, indent=2, ensure_ascii=False)

    # Truncate OCR text to relevant portion (body region)
    ocr_snippet = ocr_text[:2000] if len(ocr_text) > 2000 else ocr_text

    prompt = (
        "The following line items from a receipt have issues. "
        "Fix them based on the OCR text.\n\n"
        f"Receipt total: {receipt_total}\n\n"
        f"--- FLAGGED ITEMS ---\n{items_json}\n--- END ---\n\n"
        f"--- OCR TEXT ---\n{ocr_snippet}\n--- END ---\n\n"
        "For each flagged item, return the corrected version.\n"
        'Return ONLY a JSON object: {"corrections": [{"index": N, '
        '"name": "...", "quantity": N, "unit_raw": "...", '
        '"unit_price": N, "total_price": N}]}\n'
        "Use null for fields you cannot determine."
    )

    try:
        result = structure_ocr_text(
            raw_text=ocr_text,
            doc_type="receipt",
            emphasis_prompt=prompt,
            timeout=60.0,
        )

        llm_corrections = result.get("corrections", [])
        for corr in llm_corrections:
            idx = corr.get("index")
            if idx is None or idx < 0 or idx >= len(items):
                continue
            if idx not in flagged_indices:
                continue

            item = items[idx]
            for fld in ("name", "quantity", "unit_raw", "unit_price", "total_price"):
                new_val = corr.get(fld)
                if new_val is not None and str(new_val) != str(item.get(fld)):
                    old_val = item.get(fld)
                    item[fld] = new_val
                    corrections.append(
                        ItemCorrection(idx, fld, old_val, new_val, "LLM correction")
                    )

        if corrections:
            logger.info(
                "Layer 2 LLM verification: %d corrections applied",
                len(corrections),
            )

    except Exception as e:
        logger.warning("Layer 2 LLM verification failed: %s", e)

    return corrections


# ---------------------------------------------------------------------------
# Barcode-item cross-validation (product_cache lookup)
# ---------------------------------------------------------------------------


def validate_barcode_items(
    extracted: dict[str, Any],
    db: Any = None,
) -> list[ItemFlag]:
    """Cross-validate barcode→item assignment using product_cache.

    For each item with a barcode, looks up the cached product name
    (from OFF/UPCitemdb) and compares it against the extracted item name.
    Returns flags for mismatches where the product name is significantly
    different from the item name.

    Args:
        extracted: Extraction dict with line_items.
        db: DatabaseManager instance. If None, skips validation.

    Returns:
        List of ItemFlag for barcode-item mismatches.
    """
    flags: list[ItemFlag] = []
    items = extracted.get("line_items")
    if not items or db is None:
        return flags

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        barcode = str(item.get("barcode") or "").strip()
        if not barcode or len(barcode) < 8:
            continue  # Only validate real barcodes (EAN-8+)

        item_name = str(item.get("name") or "").strip()
        if not item_name:
            continue

        # Look up product in cache (no API call — cache only)
        try:
            row = db.fetchone(
                "SELECT data FROM product_cache WHERE barcode = ?",
                (barcode,),
            )
        except Exception:
            continue

        if not row:
            continue

        try:
            product = json.loads(row["data"])
        except (json.JSONDecodeError, TypeError):
            continue

        if product.get("_not_found"):
            continue

        product_name = str(product.get("product_name") or "").strip()
        if not product_name:
            continue

        # Fuzzy compare item name vs product name
        ratio = difflib.SequenceMatcher(
            None,
            item_name.lower(),
            product_name.lower(),
        ).ratio()

        if ratio < 0.40:
            # Auto-correct when barcode is a valid EAN (high trust)
            from alibi.extraction.barcode_detector import _is_valid_ean

            if _is_valid_ean(barcode):
                old_name = item_name
                items[i]["name"] = product_name
                flags.append(
                    ItemFlag(
                        i,
                        "barcode_item_corrected",
                        f"barcode {barcode}: corrected '{old_name}' → "
                        f"'{product_name}' (similarity: {ratio:.2f})",
                    )
                )
                logger.info(
                    "Barcode-item auto-corrected: barcode=%s, "
                    "'%s' → '%s' (similarity=%.2f)",
                    barcode,
                    old_name,
                    product_name,
                    ratio,
                )
            else:
                # Non-EAN barcode (product code) — flag only, don't correct
                flags.append(
                    ItemFlag(
                        i,
                        "barcode_item_mismatch",
                        f"barcode {barcode} maps to OFF product "
                        f"'{product_name}' but item name is "
                        f"'{item_name}' (similarity: {ratio:.2f})",
                    )
                )
                logger.info(
                    "Barcode-item mismatch: barcode=%s, item='%s', "
                    "product='%s', similarity=%.2f",
                    barcode,
                    item_name,
                    product_name,
                    ratio,
                )

    return flags


# ---------------------------------------------------------------------------
# Layer 3: cross-validation
# ---------------------------------------------------------------------------


@dataclass
class CrossValidationResult:
    """Result from cross-validation of item totals vs receipt total."""

    warnings: list[str] = field(default_factory=list)
    needs_review: bool = False
    mismatch_pct: Decimal | None = None
    item_count_mismatch: int = 0  # declared - actual (positive = items missing)


def cross_validate_receipt(extracted: dict[str, Any]) -> CrossValidationResult:
    """Layer 3: check sum(item.total_price) vs receipt total.

    Returns CrossValidationResult with warnings and needs_review flag.
    When mismatch > 50%, sets needs_review=True.
    """
    result = CrossValidationResult()

    items = extracted.get("line_items")
    if not items:
        return result

    receipt_total = _to_decimal(extracted.get("total") or extracted.get("total_amount"))
    if not receipt_total or receipt_total == Decimal("0"):
        return result

    items_sum = Decimal("0")
    for item in items:
        tp = _to_decimal(item.get("total_price"))
        if tp:
            items_sum += tp

    if items_sum == Decimal("0"):
        return result

    diff = abs(items_sum - receipt_total)
    pct = (diff / receipt_total * 100) if receipt_total else Decimal("0")
    result.mismatch_pct = pct

    if pct > Decimal("50"):
        result.needs_review = True
        result.warnings.append(
            f"Item sum ({items_sum}) differs from receipt total "
            f"({receipt_total}) by {pct:.1f}% — flagged needs_review"
        )
        logger.warning(
            "Cross-validation CRITICAL: items_sum=%s, receipt_total=%s, "
            "diff=%s%% — fact flagged needs_review",
            items_sum,
            receipt_total,
            pct,
        )
    elif pct > Decimal("10"):
        result.warnings.append(
            f"Item sum ({items_sum}) differs from receipt total "
            f"({receipt_total}) by {pct:.1f}%"
        )
        logger.warning(
            "Cross-validation mismatch: items_sum=%s, receipt_total=%s, diff=%s%%",
            items_sum,
            receipt_total,
            pct,
        )

    # Item count validation: compare extracted count vs declared count
    declared = extracted.get("declared_item_count")
    if declared is not None and isinstance(declared, int) and declared > 0:
        actual_count = len(items)
        if actual_count != declared:
            result.item_count_mismatch = declared - actual_count
            result.warnings.append(
                f"Extracted {actual_count} items but receipt declares "
                f"{declared} items"
            )
            logger.info(
                "Item count mismatch: extracted=%d, declared=%d",
                actual_count,
                declared,
            )
            # Investigate missing items when declared > actual
            if declared > actual_count:
                missed = investigate_missing_items(extracted, declared)
                if missed:
                    result.warnings.append(f"Possible missed items from OCR: {missed}")

    return result


# ---------------------------------------------------------------------------
# Item count mismatch investigation
# ---------------------------------------------------------------------------

_PRICE_LINE_RE = re.compile(r"\d+[.,]\d{2}")
_TOTAL_KEYWORDS = re.compile(
    r"\b(?:total|subtotal|tax|vat|change|cash|card|balance|"
    r"σύνολο|φπα|gesamt|mwst|итого|ндс)\b",
    re.IGNORECASE,
)


def investigate_missing_items(
    extracted: dict[str, Any],
    declared_count: int,
) -> list[str]:
    """Scan OCR text for potential missed items when count mismatch detected.

    Returns list of OCR lines that look like items but weren't extracted.
    """
    ocr_text = extracted.get("_ocr_text", "")
    if not ocr_text:
        return []

    items = extracted.get("line_items") or []
    actual_count = len(items)
    max_missed = declared_count - actual_count
    if max_missed <= 0:
        return []

    # Build set of extracted item names (lowered) for matching
    extracted_names = {
        (item.get("name") or "").strip().lower() for item in items if item.get("name")
    }

    candidates: list[str] = []
    for line in ocr_text.splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) < 4:
            continue
        # Must contain a price-like pattern
        if not _PRICE_LINE_RE.search(stripped):
            continue
        # Skip total/tax/footer lines
        if _TOTAL_KEYWORDS.search(stripped):
            continue
        # Check if this line matches any extracted item
        low = stripped.lower()
        matched = any(
            difflib.SequenceMatcher(None, low, name).ratio() > 0.6
            for name in extracted_names
            if name
        )
        if not matched:
            candidates.append(stripped)
        if len(candidates) >= max_missed:
            break

    return candidates
