"""Parse universal extraction output into typed atoms and bundles.

Converts raw LLM extraction JSON (from UNIVERSAL_PROMPT_V2 or any extraction)
into the Atom-Cloud-Fact observation model:

  raw_extraction dict -> atoms + bundle + bundle_atoms

Absorbs normalization logic from PurchaseRefiner (unit extraction, price
parsing, tax resolution, comparable price) so that refiners become unnecessary
in the v2 pipeline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

from alibi.db.models import (
    Atom,
    AtomType,
    Bundle,
    BundleAtom,
    BundleAtomRole,
    BundleType,
    TaxType,
    UnitType,
)
from alibi.extraction.text_parser import _clean_invoice_item_name
from alibi.normalizers.units import normalize_unit
from alibi.refiners.base import _normalize_amount, _parse_quantity_unit

# ---------------------------------------------------------------------------
# Constants (moved from PurchaseRefiner)
# ---------------------------------------------------------------------------

_NAME_UNIT_PATTERN = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s*(kg|g|gr|ml|l|lt|ltr|cl|oz|lb|lbs)\b",
    re.IGNORECASE,
)
_TRAILING_UNIT_PATTERN = re.compile(
    r"\s+(kg|g|gr|ml|l|lt|ltr|cl|oz|lb|lbs)\s*$",
    re.IGNORECASE,
)
_BARE_NUMBER_PATTERN = re.compile(r"\s(\d{2,4})\s*$")
_COMMON_GRAM_VALUES = frozenset(
    {50, 75, 100, 125, 150, 175, 200, 250, 300, 330, 350, 400, 450, 500, 750, 800, 1000}
)
# Bare 3-decimal weight (0.001-50.0) without unit — assume kg
_BARE_WEIGHT_PATTERN = re.compile(r"\s(\d{1,2}[.,]\d{3})\s*$")
_WEIGHT_VOLUME_UNITS = frozenset(
    {
        UnitType.GRAM,
        UnitType.KILOGRAM,
        UnitType.POUND,
        UnitType.OUNCE,
        UnitType.MILLILITER,
        UnitType.LITER,
        UnitType.GALLON,
    }
)
_CONVERSION_FACTOR_SUSPECTS: dict[UnitType, frozenset[Decimal]] = {
    UnitType.KILOGRAM: frozenset({Decimal("1000")}),
    UnitType.LITER: frozenset({Decimal("1000")}),
}
_CONTINUOUS_UNITS = frozenset({UnitType.KILOGRAM, UnitType.LITER, UnitType.POUND})
_VAT_ANALYSIS_PATTERN = re.compile(r"\b(\d{2,3})\s+(\d+(?:\.\d+)?)\s*%")
_MAX_REASONABLE_TAX_RATE = Decimal("50")

# Comparable price conversion factors
_WEIGHT_CONVERSIONS: dict[UnitType, tuple[Decimal, str]] = {
    UnitType.GRAM: (Decimal("1000"), "kg"),
    UnitType.KILOGRAM: (Decimal("1"), "kg"),
    UnitType.POUND: (Decimal("2.20462"), "kg"),
    UnitType.OUNCE: (Decimal("35.274"), "kg"),
}
_VOLUME_CONVERSIONS: dict[UnitType, tuple[Decimal, str]] = {
    UnitType.MILLILITER: (Decimal("1000"), "l"),
    UnitType.LITER: (Decimal("1"), "l"),
    UnitType.GALLON: (Decimal("0.264172"), "l"),
}

# ---------------------------------------------------------------------------
# Non-item name filter (defense-in-depth, mirrors collapse.py patterns)
# ---------------------------------------------------------------------------

_NON_ITEM_PATTERNS = [
    re.compile(r"^VAT\d?\s+\d+", re.IGNORECASE),
    re.compile(r"^Subtotal$", re.IGNORECASE),
    re.compile(r"^FROM\s+[\d.,]+\s+TO\s+[\d.,]+", re.IGNORECASE),
    re.compile(r"^\d+\s+ea\s+[\d.,]+$", re.IGNORECASE),
    # Greek VAT summary lines: structural pattern XX% XX + WORD = WORD
    # Handles OCR mixing of Greek/Latin characters (e.g. "ΦΠA% ΦΠA + KABAPD = MEIKTO")
    re.compile(r"^.{2,4}%\s+.{2,4}\s*\+\s*\w+\s*=\s*\w+", re.IGNORECASE),
    # Store loyalty/tracking codes (e.g. "LIM TRAK9")
    re.compile(r"\bLIM\s+TRAK\d*\b", re.IGNORECASE),
]
_TOTAL_LINE_NAMES = {"total", "σynoao", "σynοδο", "συνολο", "σύνολο"}


def _is_non_item_name(name: str) -> bool:
    """Return True if *name* looks like a VAT summary, total, or annotation."""
    stripped = name.strip()
    if not stripped:
        return True
    if stripped.lower() in _TOTAL_LINE_NAMES:
        return True
    for pat in _NON_ITEM_PATTERNS:
        if pat.search(stripped):
            return True
    return False


# ---------------------------------------------------------------------------
# Parse result container
# ---------------------------------------------------------------------------


@dataclass
class BundleResult:
    """A single bundle with its atom links."""

    bundle: Bundle
    bundle_atoms: list[BundleAtom] = field(default_factory=list)


@dataclass
class AtomParseResult:
    """Result of parsing an extraction into atoms and bundles.

    For most documents, there is one bundle. For statements, there is one
    STATEMENT_LINE bundle per transaction line.
    """

    atoms: list[Atom] = field(default_factory=list)
    bundles: list[BundleResult] = field(default_factory=list)

    @property
    def bundle(self) -> Bundle | None:
        """First bundle (convenience for single-bundle documents)."""
        return self.bundles[0].bundle if self.bundles else None

    @property
    def bundle_atoms(self) -> list[BundleAtom]:
        """All bundle-atom links across all bundles."""
        result: list[BundleAtom] = []
        for br in self.bundles:
            result.extend(br.bundle_atoms)
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_extraction(document_id: str, raw: dict[str, Any]) -> AtomParseResult:
    """Parse universal extraction output into atoms + bundles.

    For most document types, produces one bundle. For bank statements,
    produces one STATEMENT_LINE bundle per transaction line.

    Args:
        document_id: ID of the source document.
        raw: Raw extraction dict (from LLM or YAML cache).

    Returns:
        AtomParseResult with atoms and bundles (one or many).
    """
    doc_type = (raw.get("document_type") or "").lower()
    transactions = raw.get("transactions") or []

    # Statements → one bundle per transaction line (may be empty)
    if doc_type == "statement":
        if transactions:
            return _parse_statement(document_id, raw, transactions)
        return AtomParseResult()  # Empty statement

    # All other document types → single bundle
    return _parse_single_bundle(document_id, raw)


def _parse_single_bundle(document_id: str, raw: dict[str, Any]) -> AtomParseResult:
    """Parse a single-bundle document (receipt, invoice, payment, etc.)."""
    result = AtomParseResult()
    raw_text = raw.get("raw_text") or ""
    vat_mapping = _parse_vat_analysis(raw_text)
    doc_language = raw.get("language")
    currency = raw.get("currency") or "EUR"

    # --- Vendor atom ---
    vendor_atom = _parse_vendor_atom(document_id, raw)
    if vendor_atom:
        result.atoms.append(vendor_atom)

    # --- Datetime atom ---
    dt_atom = _parse_datetime_atom(document_id, raw)
    if dt_atom:
        result.atoms.append(dt_atom)

    # --- Amount atoms (total, subtotal) ---
    amount_atoms = _parse_amount_atoms(document_id, raw, currency)
    result.atoms.extend(amount_atoms)

    # --- Tax atoms (document-level) ---
    tax_atom = _parse_document_tax_atom(document_id, raw, currency)
    if tax_atom:
        result.atoms.append(tax_atom)

    # --- Payment atom ---
    payment_atom = _parse_payment_atom(document_id, raw, currency)
    if payment_atom:
        result.atoms.append(payment_atom)

    # --- Item atoms (with full normalization) ---
    raw_items = raw.get("line_items") or raw.get("items") or []
    for raw_item in raw_items:
        item_atom = _parse_item_atom(
            document_id, raw_item, currency, doc_language, vat_mapping
        )
        if item_atom:
            result.atoms.append(item_atom)

    # --- Bundle formation ---
    bundle_type = _infer_bundle_type(raw)
    bundle = Bundle(
        id=str(uuid4()),
        document_id=document_id,
        bundle_type=bundle_type,
    )
    br = BundleResult(bundle=bundle)

    # --- Bundle-atom links ---
    for atom in result.atoms:
        role = _atom_role(atom.atom_type, bundle_type)
        br.bundle_atoms.append(
            BundleAtom(bundle_id=bundle.id, atom_id=atom.id, role=role)
        )

    result.bundles.append(br)
    return result


def _parse_statement(
    document_id: str,
    raw: dict[str, Any],
    transactions: list[dict[str, Any]],
) -> AtomParseResult:
    """Parse a bank statement into one STATEMENT_LINE bundle per transaction.

    Each transaction line becomes its own bundle with vendor, amount, and date
    atoms. These bundles then cluster into clouds via cloud formation, matching
    existing receipt/payment bundles naturally.
    """
    result = AtomParseResult()
    currency = raw.get("currency") or "EUR"

    for txn in transactions:
        bundle_atoms_list: list[BundleAtom] = []
        bundle = Bundle(
            id=str(uuid4()),
            document_id=document_id,
            bundle_type=BundleType.STATEMENT_LINE,
        )

        # Vendor from transaction description/vendor field
        vendor_name = txn.get("vendor") or txn.get("description")
        if vendor_name:
            vendor_atom = Atom(
                id=str(uuid4()),
                document_id=document_id,
                atom_type=AtomType.VENDOR,
                data={"name": str(vendor_name).strip()},
            )
            result.atoms.append(vendor_atom)
            bundle_atoms_list.append(
                BundleAtom(
                    bundle_id=bundle.id,
                    atom_id=vendor_atom.id,
                    role=BundleAtomRole.VENDOR_INFO,
                )
            )

        # Amount — try "amount" first, then debit/credit from text parser
        amount_val = txn.get("amount")
        if amount_val is None:
            debit = txn.get("debit")
            credit = txn.get("credit")
            if debit is not None:
                amount_val = abs(debit)  # Debits are negative, convert
            elif credit is not None:
                amount_val = credit
        if amount_val is not None:
            try:
                parsed_amount = Decimal(str(amount_val))
                # Statement amounts: debits are positive expenses
                if txn.get("type") == "debit" and parsed_amount < 0:
                    parsed_amount = abs(parsed_amount)
                amount_atom = Atom(
                    id=str(uuid4()),
                    document_id=document_id,
                    atom_type=AtomType.AMOUNT,
                    data={
                        "value": str(parsed_amount),
                        "currency": currency,
                        "semantic_type": "total",
                    },
                )
                result.atoms.append(amount_atom)
                bundle_atoms_list.append(
                    BundleAtom(
                        bundle_id=bundle.id,
                        atom_id=amount_atom.id,
                        role=BundleAtomRole.TOTAL,
                    )
                )
            except (InvalidOperation, ValueError):
                pass

        # Date
        date_str = txn.get("date") or txn.get("value_date")
        if date_str:
            dt_atom = Atom(
                id=str(uuid4()),
                document_id=document_id,
                atom_type=AtomType.DATETIME,
                data={
                    "value": str(date_str).strip()[:10],
                    "semantic_type": "statement_date",
                },
            )
            result.atoms.append(dt_atom)
            bundle_atoms_list.append(
                BundleAtom(
                    bundle_id=bundle.id,
                    atom_id=dt_atom.id,
                    role=BundleAtomRole.EVENT_TIME,
                )
            )

        br = BundleResult(bundle=bundle, bundle_atoms=bundle_atoms_list)
        result.bundles.append(br)

    return result


# ---------------------------------------------------------------------------
# Atom parsers
# ---------------------------------------------------------------------------


def _parse_vendor_atom(document_id: str, raw: dict[str, Any]) -> Atom | None:
    """Extract vendor observation from raw data.

    For invoices/credit notes, falls back to issuer_* fields when vendor_*
    fields are absent. The InvoiceRefiner normally handles this mapping,
    but this provides a safety net for direct YAML-to-atom paths (e.g.
    reingest without refiner).
    """
    doc_type = (raw.get("document_type") or "").lower()
    _is_invoice = doc_type in ("invoice", "credit_note")

    vendor = raw.get("vendor")
    if not vendor and _is_invoice:
        vendor = raw.get("issuer")
    if not vendor:
        return None

    vendor_name = str(vendor).strip()

    # Reject vendor names that look like barcodes (OCR misparse)
    _stripped = re.sub(r"[^0-9A-Za-z]", "", vendor_name)
    if re.match(r"^BARCODE[:\s]", vendor_name, re.IGNORECASE) or re.match(
        r"^\d{8,13}$", _stripped
    ):
        logger.warning(
            "Vendor name looks like a barcode, skipping: %r (doc=%s)",
            vendor_name,
            document_id[:8],
        )
        return None

    data: dict[str, Any] = {"name": vendor_name}

    # Vendor detail fields — fall back to issuer_* for invoices
    _vendor_fields = (
        "vendor_address",
        "vendor_phone",
        "vendor_website",
        "vendor_legal_name",
    )
    _issuer_fallback = {
        "vendor_address": "issuer_address",
        "vendor_phone": "issuer_phone",
        "vendor_website": "issuer_website",
        "vendor_legal_name": "issuer_legal_name",
    }
    for field_name in _vendor_fields:
        val = raw.get(field_name)
        if not val and _is_invoice:
            val = raw.get(_issuer_fallback[field_name])
        if val:
            key = field_name.replace("vendor_", "")
            data[key] = str(val).strip()

    vat = raw.get("vendor_vat")
    if not vat and _is_invoice:
        vat = raw.get("issuer_vat")
    if vat:
        data["vat_number"] = str(vat).strip().replace(" ", "")

    tax_id = raw.get("vendor_tax_id")
    if not tax_id and _is_invoice:
        tax_id = raw.get("issuer_tax_id")
    if tax_id:
        data["tax_id"] = str(tax_id).strip().replace(" ", "")

    return Atom(
        id=str(uuid4()),
        document_id=document_id,
        atom_type=AtomType.VENDOR,
        data=data,
    )


def _parse_datetime_atom(document_id: str, raw: dict[str, Any]) -> Atom | None:
    """Extract datetime observation from raw data."""
    date_val = raw.get("date") or raw.get("document_date")
    if not date_val:
        return None

    date_str = str(date_val)
    data: dict[str, Any] = {"value": date_str}

    # Warn on dates more than 1 year in the past (likely OCR misread)
    try:
        parsed = date.fromisoformat(date_str[:10])
        if parsed < date.today() - timedelta(days=365):
            logger.warning(
                "Extracted date %s is >1 year old — possible OCR error (doc=%s)",
                date_str,
                document_id[:8],
            )
            data["_suspicious_date"] = True
    except (ValueError, TypeError):
        pass  # Non-ISO date formats handled downstream

    time_val = raw.get("time")
    if time_val:
        data["value"] = f"{date_val} {time_val}"

    # Semantic type based on document type
    doc_type = (raw.get("document_type") or "").lower()
    if doc_type == "statement":
        data["semantic_type"] = "statement_date"
    elif doc_type in ("invoice", "contract"):
        data["semantic_type"] = "issue_date"
    else:
        data["semantic_type"] = "transaction_time"

    return Atom(
        id=str(uuid4()),
        document_id=document_id,
        atom_type=AtomType.DATETIME,
        data=data,
    )


def _parse_amount_atoms(
    document_id: str, raw: dict[str, Any], currency: str
) -> list[Atom]:
    """Extract amount observations (total, subtotal)."""
    atoms: list[Atom] = []

    total = _normalize_amount(raw.get("total") or raw.get("amount"))
    if total is not None:
        atoms.append(
            Atom(
                id=str(uuid4()),
                document_id=document_id,
                atom_type=AtomType.AMOUNT,
                data={
                    "value": str(total),
                    "currency": currency,
                    "semantic_type": "total",
                },
            )
        )

    subtotal = _normalize_amount(raw.get("subtotal"))
    if subtotal is not None:
        atoms.append(
            Atom(
                id=str(uuid4()),
                document_id=document_id,
                atom_type=AtomType.AMOUNT,
                data={
                    "value": str(subtotal),
                    "currency": currency,
                    "semantic_type": "subtotal",
                },
            )
        )

    return atoms


def _parse_document_tax_atom(
    document_id: str, raw: dict[str, Any], currency: str
) -> Atom | None:
    """Extract document-level tax observation."""
    tax_val = _normalize_amount(raw.get("tax") or raw.get("tax_amount"))
    if tax_val is None:
        return None

    return Atom(
        id=str(uuid4()),
        document_id=document_id,
        atom_type=AtomType.TAX,
        data={
            "amount": str(tax_val),
            "currency": currency,
            "type": "VAT",
        },
    )


def _parse_payment_atom(
    document_id: str, raw: dict[str, Any], currency: str
) -> Atom | None:
    """Extract payment observation from raw data."""
    method = raw.get("payment_method")
    card_last4 = raw.get("card_last4")
    auth_code = raw.get("authorization_code")
    total = _normalize_amount(raw.get("total") or raw.get("amount"))

    # Need at least one payment-specific field
    if not method and not card_last4 and not auth_code:
        return None

    data: dict[str, Any] = {}
    if method:
        data["method"] = str(method).strip().lower()
    if card_last4:
        data["card_last4"] = str(card_last4).strip()
    if auth_code:
        data["auth_code"] = str(auth_code).strip()
    if total is not None:
        data["amount"] = str(total)
        data["currency"] = currency

    for field_name in (
        "card_type",
        "terminal_id",
        "merchant_id",
        "reference_number",
        "iban",
        "bic",
    ):
        val = raw.get(field_name)
        if val:
            data[field_name] = str(val).strip()

    # Numeric payment fields: zero is a valid value (exact change)
    for field_name in ("amount_tendered", "change_due"):
        val = raw.get(field_name)
        if val is not None:
            data[field_name] = str(val).strip()

    return Atom(
        id=str(uuid4()),
        document_id=document_id,
        atom_type=AtomType.PAYMENT,
        data=data,
    )


def _parse_item_atom(
    document_id: str,
    raw_item: dict[str, Any],
    currency: str,
    doc_language: str | None,
    vat_mapping: dict[int, Decimal],
) -> Atom | None:
    """Parse a single line item into an item atom with full normalization.

    Absorbs normalization logic from PurchaseRefiner._parse_single_line_item().
    """
    name = raw_item.get("name") or raw_item.get("description") or raw_item.get("item")
    if not name:
        return None
    name = _clean_invoice_item_name(str(name).strip())
    if _is_non_item_name(name):
        return None

    data: dict[str, Any] = {"name": name, "currency": currency}

    # -- Quantity and unit (multi-source resolution) --
    unit_raw_from_llm = raw_item.get("unit_raw")
    quantity_raw = raw_item.get("quantity", "1")
    unit_from_qty = None

    if isinstance(quantity_raw, str):
        quantity, unit_from_qty = _parse_quantity_unit(quantity_raw)
    else:
        quantity = Decimal(str(quantity_raw)) if quantity_raw else Decimal("1")

    data["quantity"] = str(quantity)

    llm_unit_type = None
    if unit_raw_from_llm:
        llm_unit_type = normalize_unit(str(unit_raw_from_llm))

    llm_unit_quantity = raw_item.get("unit_quantity")

    if llm_unit_type and llm_unit_type not in (UnitType.PIECE, UnitType.OTHER):
        data["unit"] = llm_unit_type.value
        data["unit_raw"] = str(unit_raw_from_llm)
        extracted = _extract_unit_from_name(name)
        if extracted:
            data["name"] = extracted[2]
            if extracted[3] is not None:
                data["unit_quantity"] = str(extracted[3])
    elif unit_from_qty:
        data["unit_raw"] = unit_from_qty
        data["unit"] = normalize_unit(unit_from_qty).value
    else:
        extracted = _extract_unit_from_name(name)
        if extracted:
            data["unit_raw"] = extracted[0]
            data["unit"] = extracted[1].value
            data["name"] = extracted[2]
            if extracted[3] is not None:
                data["unit_quantity"] = str(extracted[3])
        elif llm_unit_type:
            data["unit_raw"] = str(unit_raw_from_llm)
            data["unit"] = llm_unit_type.value
        else:
            data["unit"] = UnitType.PIECE.value

    # LLM-provided unit_quantity fallback
    if "unit_quantity" not in data and llm_unit_quantity:
        try:
            data["unit_quantity"] = str(Decimal(str(llm_unit_quantity)))
        except (InvalidOperation, ValueError):
            pass

    # Extract unit_quantity from X<N> pattern (e.g., "X12" = pack of 12)
    if unit_raw_from_llm and "unit_quantity" not in data:
        x_match = re.match(r"^[xX](\d+)$", str(unit_raw_from_llm).strip())
        if x_match:
            data["unit_quantity"] = x_match.group(1)

    # -- Name normalization --
    name_en = raw_item.get("name_en")
    if name_en:
        data["name_normalized"] = str(name_en).strip()
        data["comparable_name"] = str(name_en).strip()

    language = raw_item.get("original_language") or raw_item.get("language")
    if not language and doc_language:
        language = doc_language
    if language:
        data["original_language"] = str(language).strip()

    # -- Prices --
    if "unit_price" in raw_item:
        up = _normalize_amount(raw_item.get("unit_price"))
        if up is not None:
            data["unit_price"] = str(up)

    total_raw = (
        raw_item.get("total_price") or raw_item.get("price") or raw_item.get("total")
    )
    if total_raw is not None:
        tp = _normalize_amount(total_raw)
        if tp is not None:
            data["total_price"] = str(tp)

    # Calculate missing price
    up_val = Decimal(data["unit_price"]) if "unit_price" in data else None
    tp_val = Decimal(data["total_price"]) if "total_price" in data else None
    qty_val = Decimal(data["quantity"])

    if up_val and not tp_val:
        data["total_price"] = str(up_val * qty_val)
    if tp_val and not up_val and qty_val and qty_val != 0:
        data["unit_price"] = str(tp_val / qty_val)

    # -- Tax --
    tax_info = _parse_item_tax(raw_item, vat_mapping)
    data.update(tax_info)

    # -- Discount --
    if "discount_amount" in raw_item:
        da = _normalize_amount(raw_item.get("discount_amount"))
        if da is not None:
            data["discount_amount"] = str(da)
    if "discount_percentage" in raw_item and raw_item["discount_percentage"]:
        try:
            data["discount_percentage"] = str(
                Decimal(str(raw_item["discount_percentage"]))
            )
        except (InvalidOperation, ValueError):
            pass

    # -- Fix weighed item errors --
    _fix_weighed_item_quantities(data)
    _infer_weighed_from_price_math(data)

    # -- Comparable unit price --
    _calculate_comparable_price(data)

    # -- Metadata --
    for key in ("category", "subcategory", "brand", "barcode"):
        if key in raw_item and raw_item[key]:
            val = raw_item[key]
            # Normalize category to title case for consistency
            if key in ("category", "subcategory") and isinstance(val, str):
                val = val.strip().title()
            data[key] = val

    return Atom(
        id=str(uuid4()),
        document_id=document_id,
        atom_type=AtomType.ITEM,
        data=data,
    )


# ---------------------------------------------------------------------------
# Bundle inference
# ---------------------------------------------------------------------------


def _infer_bundle_type(raw: dict[str, Any]) -> BundleType:
    """Infer bundle type from extraction content."""
    doc_type = (raw.get("document_type") or "").lower()

    if doc_type == "statement":
        return BundleType.STATEMENT_LINE
    if doc_type == "invoice":
        return BundleType.INVOICE
    if doc_type in ("payment_confirmation", "card_slip"):
        return BundleType.PAYMENT_RECORD

    # Default: basket (receipt / purchase)
    return BundleType.BASKET


def _atom_role(atom_type: AtomType, bundle_type: BundleType) -> BundleAtomRole:
    """Map atom type to its role within a bundle."""
    mapping: dict[AtomType, BundleAtomRole] = {
        AtomType.ITEM: BundleAtomRole.BASKET_ITEM,
        AtomType.VENDOR: BundleAtomRole.VENDOR_INFO,
        AtomType.PAYMENT: BundleAtomRole.PAYMENT_INFO,
        AtomType.DATETIME: BundleAtomRole.EVENT_TIME,
        AtomType.TAX: BundleAtomRole.TAX_DETAIL,
        AtomType.AMOUNT: BundleAtomRole.TOTAL,
    }
    return mapping.get(atom_type, BundleAtomRole.TOTAL)


# ---------------------------------------------------------------------------
# Item normalization helpers (absorbed from PurchaseRefiner)
# ---------------------------------------------------------------------------


def _extract_unit_from_name(
    name: str,
) -> tuple[str, UnitType, str, Decimal | None] | None:
    """Extract unit/volume from product name.

    Returns (unit_raw, UnitType, cleaned_name, unit_quantity) or None.
    """
    match = _NAME_UNIT_PATTERN.search(name)
    if match:
        qty_str = match.group(1).replace(",", ".")
        unit_str = match.group(2)
        unit_type = normalize_unit(unit_str)
        if unit_type != UnitType.OTHER:
            clean = name[: match.start()].strip().rstrip("-").strip()
            if not clean:
                clean = name[match.end() :].strip()
            return unit_str, unit_type, clean, Decimal(qty_str)

    match = _TRAILING_UNIT_PATTERN.search(name)
    if match:
        unit_str = match.group(1)
        unit_type = normalize_unit(unit_str)
        if unit_type != UnitType.OTHER:
            clean = name[: match.start()].strip()
            return unit_str, unit_type, clean, None

    match = _BARE_NUMBER_PATTERN.search(name)
    if match:
        qty = int(match.group(1))
        if qty in _COMMON_GRAM_VALUES:
            clean = name[: match.start()].strip()
            if clean:
                return "g", UnitType.GRAM, clean, Decimal(str(qty))

    # NOTE: bare 3-decimal numbers (e.g. "0.765") are ambiguous — could be
    # kg (produce) or L (beverages). Don't guess here; the item verifier
    # extracts the quantity and downstream LLM/context resolves the unit.

    return None


def _parse_item_tax(
    raw_item: dict[str, Any],
    vat_mapping: dict[int, Decimal],
) -> dict[str, str]:
    """Parse tax info from a raw line item. Returns string-valued dict for atom data."""
    result: dict[str, str] = {"tax_type": TaxType.NONE.value}

    try:
        tax_type_raw = raw_item.get("tax_type")
        if tax_type_raw:
            s = str(tax_type_raw).strip().lower()
            if "vat" in s:
                result["tax_type"] = TaxType.VAT.value
            elif "sales" in s:
                result["tax_type"] = TaxType.SALES_TAX.value
            elif "gst" in s:
                result["tax_type"] = TaxType.GST.value
            elif "exempt" in s:
                result["tax_type"] = TaxType.EXEMPT.value
            elif "included" in s:
                result["tax_type"] = TaxType.INCLUDED.value

        tax_rate_raw = raw_item.get("tax_rate")
        if tax_rate_raw is not None:
            rate_str = str(tax_rate_raw).strip().replace("%", "")
            if rate_str and rate_str[0].isdigit():
                rate = Decimal(rate_str)
                if rate > _MAX_REASONABLE_TAX_RATE:
                    code = int(rate)
                    if vat_mapping and code in vat_mapping:
                        result["tax_rate"] = str(vat_mapping[code])
                elif rate > 1:
                    result["tax_rate"] = str(rate)
                elif rate > 0:
                    result["tax_rate"] = str(rate * 100)
                else:
                    result["tax_rate"] = "0"

        if "tax_rate" in result and result["tax_type"] == TaxType.NONE.value:
            result["tax_type"] = TaxType.VAT.value

        tax_amount_raw = raw_item.get("tax_amount")
        if tax_amount_raw is not None:
            ta = _normalize_amount(tax_amount_raw)
            if ta is not None:
                result["tax_amount"] = str(ta)
    except Exception:
        logger.debug("Tax parsing failed for item", exc_info=True)

    return result


def _parse_vat_analysis(raw_text: str | None) -> dict[int, Decimal]:
    """Parse VAT analysis table from receipt text."""
    if not raw_text:
        return {}
    try:
        mapping: dict[int, Decimal] = {}
        for match in _VAT_ANALYSIS_PATTERN.finditer(raw_text):
            code = int(match.group(1))
            rate = Decimal(match.group(2))
            mapping[code] = rate
        return mapping
    except Exception:
        return {}


def _fix_weighed_item_quantities(data: dict[str, Any]) -> None:
    """Fix common LLM errors with weight/quantity fields.

    Operates on atom data dict with string values for Decimal fields.
    """
    unit_str = data.get("unit")
    if not unit_str:
        return

    try:
        unit = UnitType(unit_str)
    except ValueError:
        return

    if unit not in _WEIGHT_VOLUME_UNITS:
        return

    try:
        quantity = Decimal(data.get("quantity", "1"))
    except (InvalidOperation, ValueError):
        return

    uq_raw = data.get("unit_quantity")
    unit_quantity = None
    if uq_raw is not None:
        try:
            unit_quantity = Decimal(str(uq_raw))
        except (InvalidOperation, ValueError):
            pass

    # Case 1: conversion factor suspect
    suspects = _CONVERSION_FACTOR_SUSPECTS.get(unit)
    if suspects and unit_quantity is not None and unit_quantity in suspects:
        data["unit_quantity"] = str(quantity)
        data["quantity"] = "1"
        return

    # Case 2: fractional quantity with no unit_quantity for continuous units
    if (
        unit_quantity is None
        and unit in _CONTINUOUS_UNITS
        and quantity > 0
        and quantity < Decimal("100")
        and quantity % 1 != 0
    ):
        data["unit_quantity"] = str(quantity)
        data["quantity"] = "1"
        return

    # Case 3: weight duplicated in both fields
    if (
        unit_quantity is not None
        and unit in _CONTINUOUS_UNITS
        and quantity == unit_quantity
        and quantity > 0
        and quantity < Decimal("100")
        and quantity % 1 != 0
    ):
        data["quantity"] = "1"


def _infer_weighed_from_price_math(data: dict[str, Any]) -> None:
    """Infer unit=kg for items where price math proves they are weighed.

    Detects items stored as unit='pcs' (or 'other') but where the numbers
    show weight-based pricing: unit_quantity is fractional, in a plausible
    weight range, and unit_quantity * unit_price ≈ total_price.

    Example: receipt shows "1.94 Anna Chicken 4.95 9.60" — extracted as
    qty=1, unit=pcs, unit_quantity=1.94, unit_price=4.95, total_price=9.60.
    Since 1.94 * 4.95 = 9.60, this is clearly a weighed item priced per kg.
    """
    unit_str = data.get("unit", "")
    if unit_str not in ("pcs", "other"):
        return

    uq_raw = data.get("unit_quantity")
    if uq_raw is None:
        return

    try:
        uq = Decimal(str(uq_raw))
        up = Decimal(str(data.get("unit_price", "0")))
        tp = Decimal(str(data.get("total_price", "0")))
    except (InvalidOperation, ValueError):
        return

    if uq <= 0 or up <= 0 or tp <= 0:
        return

    # Must be fractional and in a plausible weight range (0, 10) kg
    if uq % 1 == 0 or uq >= 10:
        return

    # Price math check: unit_quantity * unit_price ≈ total_price
    computed = uq * up
    if tp == 0:
        return
    relative_error = abs(computed - tp) / tp
    if relative_error < Decimal("0.02"):
        data["unit"] = UnitType.KILOGRAM.value
        if not data.get("unit_raw"):
            data["unit_raw"] = "kg"


def _calculate_comparable_price(data: dict[str, Any]) -> None:
    """Calculate normalized price per standard unit. Mutates data dict."""
    try:
        total_price = Decimal(data["total_price"]) if "total_price" in data else None
        quantity = Decimal(data.get("quantity", "1"))
        unit_str = data.get("unit")
        uq_raw = data.get("unit_quantity")
    except (InvalidOperation, ValueError):
        return

    if not total_price or not quantity or quantity == 0 or not unit_str:
        return

    try:
        unit = UnitType(unit_str)
    except ValueError:
        return

    if uq_raw:
        try:
            unit_quantity = Decimal(str(uq_raw))
            total_content = quantity * unit_quantity if unit_quantity > 0 else quantity
        except (InvalidOperation, ValueError):
            total_content = quantity
    else:
        total_content = quantity

    if total_content == 0:
        return

    if unit in _WEIGHT_CONVERSIONS:
        factor, std_unit = _WEIGHT_CONVERSIONS[unit]
        raw_price = (total_price * factor) / total_content
        data["comparable_unit_price"] = str(raw_price.quantize(Decimal("0.01")))
        data["comparable_unit"] = std_unit
    elif unit in _VOLUME_CONVERSIONS:
        factor, std_unit = _VOLUME_CONVERSIONS[unit]
        raw_price = (total_price * factor) / total_content
        data["comparable_unit_price"] = str(raw_price.quantize(Decimal("0.01")))
        data["comparable_unit"] = std_unit
    else:
        raw_price = total_price / total_content
        data["comparable_unit_price"] = str(raw_price.quantize(Decimal("0.01")))
        data["comparable_unit"] = unit.value
