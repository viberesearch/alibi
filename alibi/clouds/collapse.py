"""Fact collapse — convert clouds into confirmed facts.

Rules:
- Single-bundle clouds (standalone receipt) collapse immediately.
- Multi-bundle clouds need corroborating evidence (matching amounts,
  vendor, dates from different documents).
- Split payments: multiple payment atoms summing to invoice total.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import uuid4

from alibi.db.models import (
    AtomType,
    BundleType,
    Cloud,
    CloudStatus,
    Fact,
    FactItem,
    FactStatus,
    FactType,
    TaxType,
    UnitType,
)
from alibi.extraction.historical import make_vendor_key


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum confidence for a cloud to collapse into a fact
_SINGLE_BUNDLE_THRESHOLD = Decimal("0")  # Always collapses
_MULTI_BUNDLE_THRESHOLD = Decimal("0.5")

# Sum-of-parts tolerance (for split payments)
_SUM_TOLERANCE = Decimal("0.02")  # 2 cents

# Item name cleanup patterns
_TRAILING_EA_PATTERN = re.compile(r"\s+ea$", re.IGNORECASE)
_BARCODE_SUFFIX_PATTERN = re.compile(r"\s*Barcode:\s*(\S+)\s*$", re.IGNORECASE)
# Leading SKU: 3-6 digits + "ea" + whitespace, lookahead for non-digit
_LEADING_SKU_PATTERN = re.compile(r"^\d{3,6}\s+ea\s+(?=\D)", re.IGNORECASE)

# Non-product line patterns — discount/annotation lines that are not real items
_NON_PRODUCT_PATTERNS = [
    re.compile(r"^FROM\s+[\d.,]+\s+TO\s+[\d.,]+", re.IGNORECASE),  # price change
    re.compile(r"\d+%\s*OFF\b", re.IGNORECASE),  # percentage discount
    re.compile(r"^\d+\s+ea\s+[\d.,]+$", re.IGNORECASE),  # qty-price metadata
    re.compile(r"^VAT\d?\s+\d+", re.IGNORECASE),  # VAT summary line
    re.compile(r"^Subtotal$", re.IGNORECASE),  # subtotal line
]
_NON_PRODUCT_KEYWORDS = {"discount", "coupon"}
# Exact-match totals (Greek + English OCR variants)
_TOTAL_LINE_NAMES = {"total", "σynoao", "σynοδο", "συνολο", "σύνολο"}


# ---------------------------------------------------------------------------
# Collapse result
# ---------------------------------------------------------------------------


@dataclass
class CollapseResult:
    """Result of attempting to collapse a cloud into a fact."""

    collapsed: bool = False
    fact: Fact | None = None
    items: list[FactItem] = field(default_factory=list)
    cloud_status: CloudStatus = CloudStatus.FORMING


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def try_collapse(
    cloud: Cloud,
    bundles: list[dict[str, Any]],
) -> CollapseResult:
    """Attempt to collapse a cloud into a confirmed fact.

    Args:
        cloud: The cloud to evaluate.
        bundles: List of bundle dicts, each containing:
            - bundle_id: str
            - bundle_type: str (BundleType value)
            - atoms: list of atom data dicts

    Returns:
        CollapseResult with fact and items if collapsed.
    """
    if not bundles:
        return CollapseResult()

    # Single-bundle cloud: immediate collapse
    if len(bundles) == 1:
        return _collapse_single(cloud, bundles[0])

    # Multi-bundle cloud: check corroborating evidence
    return _collapse_multi(cloud, bundles)


def infer_fact_type(bundles: list[dict[str, Any]]) -> FactType:
    """Infer fact type from bundle composition.

    Mapping:
    - BASKET or INVOICE bundles → PURCHASE
    - PAYMENT_RECORD alone (no basket/invoice) → PURCHASE (card-only receipt)
    - STATEMENT_LINE alone → PURCHASE (bank transaction)
    - Mixed basket+payment → PURCHASE (corroborated)

    REFUND and SUBSCRIPTION_PAYMENT require explicit signals (negative amounts
    or subscription detection) and are set downstream, not inferred here.
    """
    types = set()
    for b in bundles:
        try:
            types.add(BundleType(b.get("bundle_type", "")))
        except ValueError:
            pass

    # Default: purchase covers baskets, invoices, payment records, statements
    return FactType.PURCHASE


# ---------------------------------------------------------------------------
# Internal collapse logic
# ---------------------------------------------------------------------------


def _collapse_single(cloud: Cloud, bundle: dict[str, Any]) -> CollapseResult:
    """Collapse a single-bundle cloud immediately."""
    atoms = bundle.get("atoms", [])

    vendor = _extract_vendor(atoms)
    total = _extract_total(atoms)
    currency = _extract_currency(atoms)
    event_date = _extract_date(atoms)
    payments = _extract_payments(atoms)
    items = _extract_items(atoms, currency)

    registration = _extract_vendor_registration(atoms)
    vendor_key = make_vendor_key(registration, vendor)

    fact = Fact(
        id=str(uuid4()),
        cloud_id=cloud.id,
        fact_type=FactType.PURCHASE,
        vendor=vendor,
        vendor_key=vendor_key,
        total_amount=total,
        currency=currency,
        event_date=event_date,
        payments=payments,
        status=FactStatus.CONFIRMED,
    )

    return CollapseResult(
        collapsed=True,
        fact=fact,
        items=items,
        cloud_status=CloudStatus.COLLAPSED,
    )


def _collapse_multi(cloud: Cloud, bundles: list[dict[str, Any]]) -> CollapseResult:
    """Attempt to collapse a multi-bundle cloud."""
    # Collect all atoms across bundles
    all_atoms: list[dict[str, Any]] = []
    for b in bundles:
        all_atoms.extend(b.get("atoms", []))

    # Check: do we have corroborating evidence?
    vendor = _extract_vendor(all_atoms)
    totals = _extract_all_totals(all_atoms)
    payment_amounts = _extract_payment_amounts(all_atoms)
    event_date = _extract_date(all_atoms)
    currency = _extract_currency(all_atoms)
    payments = _extract_payments(all_atoms)

    # Corroboration checks
    confidence = Decimal("0")

    # Same vendor across bundles
    vendors = set()
    for b in bundles:
        v = _extract_vendor(b.get("atoms", []))
        if v:
            vendors.add(v.lower())
    if len(vendors) <= 1 and vendors:
        confidence += Decimal("0.3")

    # Amounts match or sum correctly
    if totals:
        max_total = max(totals)
        if len(set(totals)) == 1:
            # All bundles agree on total
            confidence += Decimal("0.4")
        elif payment_amounts and _amounts_sum_to(payment_amounts, max_total):
            confidence += Decimal("0.3")

    # Multiple document types = stronger evidence
    bundle_types = {b.get("bundle_type") for b in bundles}
    if len(bundle_types) > 1:
        confidence += Decimal("0.2")

    if confidence < _MULTI_BUNDLE_THRESHOLD:
        return CollapseResult(cloud_status=CloudStatus.FORMING)

    # Collapse
    total = max(totals) if totals else None
    items = _extract_items(all_atoms, currency)
    fact_type = infer_fact_type(bundles)
    registration = _extract_vendor_registration(all_atoms)
    vendor_key = make_vendor_key(registration, vendor)

    fact = Fact(
        id=str(uuid4()),
        cloud_id=cloud.id,
        fact_type=fact_type,
        vendor=vendor,
        vendor_key=vendor_key,
        total_amount=total,
        currency=currency,
        event_date=event_date,
        payments=payments,
        status=(
            FactStatus.CONFIRMED if confidence >= Decimal("0.8") else FactStatus.PARTIAL
        ),
    )

    return CollapseResult(
        collapsed=True,
        fact=fact,
        items=items,
        cloud_status=CloudStatus.COLLAPSED,
    )


# ---------------------------------------------------------------------------
# Atom extraction helpers
# ---------------------------------------------------------------------------


def _extract_vendor(atoms: list[dict[str, Any]]) -> str | None:
    """Extract vendor name from atom list."""
    for atom in atoms:
        atype = atom.get("atom_type", "")
        if atype == AtomType.VENDOR.value or atype == AtomType.VENDOR:
            name: str | None = atom.get("data", {}).get("name")
            return name
    return None


def _extract_vendor_registration(atoms: list[dict[str, Any]]) -> str | None:
    """Extract vendor VAT number (primary) or tax_id from atom list."""
    for atom in atoms:
        atype = atom.get("atom_type", "")
        if atype == AtomType.VENDOR.value or atype == AtomType.VENDOR:
            data = atom.get("data", {})
            reg: str | None = data.get("vat_number") or data.get("tax_id")
            if reg and reg.strip():
                return reg.strip()
    return None


def _extract_total(atoms: list[dict[str, Any]]) -> Decimal | None:
    """Extract total amount from atom list."""
    for atom in atoms:
        atype = atom.get("atom_type", "")
        if atype == AtomType.AMOUNT.value or atype == AtomType.AMOUNT:
            data = atom.get("data", {})
            if data.get("semantic_type") == "total":
                try:
                    return Decimal(str(data["value"]))
                except (InvalidOperation, ValueError, KeyError):
                    pass
    return None


def _extract_all_totals(atoms: list[dict[str, Any]]) -> list[Decimal]:
    """Extract all total amounts across atoms."""
    totals = []
    for atom in atoms:
        atype = atom.get("atom_type", "")
        if atype == AtomType.AMOUNT.value or atype == AtomType.AMOUNT:
            data = atom.get("data", {})
            if data.get("semantic_type") == "total":
                try:
                    totals.append(Decimal(str(data["value"])))
                except (InvalidOperation, ValueError, KeyError):
                    pass
    return totals


def _extract_payment_amounts(atoms: list[dict[str, Any]]) -> list[Decimal]:
    """Extract amounts from payment atoms."""
    amounts = []
    for atom in atoms:
        atype = atom.get("atom_type", "")
        if atype == AtomType.PAYMENT.value or atype == AtomType.PAYMENT:
            data = atom.get("data", {})
            try:
                amounts.append(Decimal(str(data["amount"])))
            except (InvalidOperation, ValueError, KeyError):
                pass
    return amounts


def _extract_currency(atoms: list[dict[str, Any]]) -> str:
    """Extract currency from atom list."""
    for atom in atoms:
        data = atom.get("data", {})
        if "currency" in data:
            return str(data["currency"])
    return "EUR"


def _extract_date(atoms: list[dict[str, Any]]) -> date | None:
    """Extract event date from atom list."""
    for atom in atoms:
        atype = atom.get("atom_type", "")
        if atype == AtomType.DATETIME.value or atype == AtomType.DATETIME:
            date_str = atom.get("data", {}).get("value", "")
            if date_str and len(date_str) >= 10:
                try:
                    return date.fromisoformat(date_str[:10])
                except ValueError:
                    pass
    return None


def _extract_payments(atoms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract payment details from atom list."""
    payments = []
    for atom in atoms:
        atype = atom.get("atom_type", "")
        if atype == AtomType.PAYMENT.value or atype == AtomType.PAYMENT:
            data = atom.get("data", {})
            payment = {}
            if "method" in data:
                payment["method"] = data["method"]
            if "card_last4" in data:
                payment["card_last4"] = data["card_last4"]
            if "amount" in data:
                payment["amount"] = data["amount"]
            if "auth_code" in data:
                payment["auth_code"] = data["auth_code"]
            if "amount_tendered" in data:
                payment["amount_tendered"] = data["amount_tendered"]
            if "change_due" in data:
                payment["change_due"] = data["change_due"]
            if payment:
                payments.append(payment)
    return payments


def _clean_item_name(name: str, item: FactItem) -> str:
    """Clean OCR artifacts from an item name.

    1. Strip leading SKU/article code (e.g. "10163 ea TILDEN BASMAT")
    2. Strip trailing "Barcode: NNNNN" suffix, move barcode to item if empty
    3. Strip trailing " ea" suffix
    """
    name = _LEADING_SKU_PATTERN.sub("", name)

    barcode_match = _BARCODE_SUFFIX_PATTERN.search(name)
    if barcode_match:
        extracted_barcode = barcode_match.group(1)
        name = name[: barcode_match.start()]
        if not item.barcode:
            item.barcode = extracted_barcode

    name = _TRAILING_EA_PATTERN.sub("", name)
    return name.strip()


def _extract_items(atoms: list[dict[str, Any]], currency: str) -> list[FactItem]:
    """Extract and denormalize item atoms into FactItem objects."""
    items = []
    for atom in atoms:
        atype = atom.get("atom_type", "")
        if atype != AtomType.ITEM.value and atype != AtomType.ITEM:
            continue

        data = atom.get("data", {})
        atom_id = atom.get("id", str(uuid4()))
        name = data.get("name")
        if not name:
            continue

        item = FactItem(
            id=str(uuid4()),
            fact_id="",  # Set after fact creation
            atom_id=atom_id,
            name=name,
            name_normalized=data.get("name_normalized") or name,
            comparable_name=data.get("comparable_name"),
        )

        # Quantity
        try:
            item.quantity = Decimal(str(data.get("quantity", "1")))
        except (InvalidOperation, ValueError):
            pass

        # Unit quantity (weight/volume per unit)
        if "unit_quantity" in data:
            try:
                item.unit_quantity = Decimal(str(data["unit_quantity"]))
            except (InvalidOperation, ValueError):
                pass

        # Unit
        unit_str = data.get("unit", "pcs")
        try:
            item.unit = UnitType(unit_str)
        except ValueError:
            item.unit = UnitType.PIECE

        # Prices
        if "unit_price" in data:
            try:
                item.unit_price = Decimal(str(data["unit_price"]))
            except (InvalidOperation, ValueError):
                pass
        if "total_price" in data:
            try:
                item.total_price = Decimal(str(data["total_price"]))
            except (InvalidOperation, ValueError):
                pass

        # Metadata
        item.brand = data.get("brand")
        item.category = data.get("category")
        item.barcode = data.get("barcode")

        # Comparable price
        if "comparable_unit_price" in data:
            try:
                item.comparable_unit_price = Decimal(str(data["comparable_unit_price"]))
            except (InvalidOperation, ValueError):
                pass
        if "comparable_unit" in data:
            try:
                item.comparable_unit = UnitType(data["comparable_unit"])
            except ValueError:
                pass

        # Tax
        if "tax_rate" in data:
            try:
                item.tax_rate = Decimal(str(data["tax_rate"]))
            except (InvalidOperation, ValueError):
                pass
        if "tax_type" in data:
            try:
                item.tax_type = TaxType(data["tax_type"])
            except ValueError:
                pass

        # Clean OCR artifacts from item name
        original_name = item.name
        cleaned_name = _clean_item_name(item.name, item)
        item.name = cleaned_name
        if item.name_normalized == original_name:
            item.name_normalized = cleaned_name

        items.append(item)

    # Filter non-product lines (discounts, annotations, metadata)
    items = _filter_non_product_items(items)

    # Deduplicate items with identical name+price (from overlapping document groups)
    items = _deduplicate_items(items)

    return items


def _is_non_product_item(item: FactItem) -> bool:
    """Check if an item name is a discount/annotation, not a real product."""
    name = (item.name or "").strip()
    if not name:
        return True
    for pat in _NON_PRODUCT_PATTERNS:
        if pat.search(name):
            return True
    name_lower = name.lower()
    if name_lower in _TOTAL_LINE_NAMES:
        return True
    for kw in _NON_PRODUCT_KEYWORDS:
        if kw in name_lower:
            price = item.total_price or Decimal("0")
            if price <= 0:
                return True
    return False


def _filter_non_product_items(items: list[FactItem]) -> list[FactItem]:
    """Remove non-product line items (discounts, annotations)."""
    return [i for i in items if not _is_non_product_item(i)]


def _item_metadata_score(item: FactItem) -> int:
    """Count populated metadata fields for dedup tie-breaking."""
    score = 0
    if item.barcode:
        score += 1
    if item.brand:
        score += 1
    if item.category:
        score += 1
    if item.unit_quantity:
        score += 1
    if item.comparable_unit_price:
        score += 1
    return score


def _deduplicate_items(items: list[FactItem]) -> list[FactItem]:
    """Remove duplicate items with identical name+price+quantity.

    When two bundles carry the same receipt content (different file hashes,
    same content), cloud formation correctly matches them but collapse
    creates duplicate fact_items. Keep the one with most metadata.
    """
    groups: dict[tuple[str, Decimal | None, Decimal | None], list[FactItem]] = {}
    for item in items:
        key = (
            (item.name or "").lower().strip(),
            item.total_price,
            item.quantity,
        )
        groups.setdefault(key, []).append(item)

    result = []
    for group in groups.values():
        if len(group) == 1:
            result.append(group[0])
        else:
            # Keep the item with the most metadata
            best = max(group, key=_item_metadata_score)
            result.append(best)
    return result


def _amounts_sum_to(amounts: list[Decimal], target: Decimal) -> bool:
    """Check if amounts sum to target within tolerance."""
    if not amounts:
        return False
    total = sum(amounts)
    return abs(total - target) <= _SUM_TOLERANCE
