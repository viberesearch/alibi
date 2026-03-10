"""Heuristic text parser for OCR output (Stage 2a).

Converts raw OCR text into structured data deterministically, before
sending to the LLM for correction. This avoids the LLM having to
generate ~5800 tokens of structured JSON from scratch — instead it only
needs to correct OCR errors and fill gaps (translations, categories).

Supported document types:
- receipt / invoice: full parsing (header, dates, line items, totals, tax)
- payment_confirmation: simpler parsing (vendor, amount, card details)
- statement: dedicated parsing (institution, account, transactions)
- warranty / contract: minimal, needs_llm=True

Multilingual support: EN, DE, EL, RU markers for totals, tax, discounts.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from alibi.normalizers.dates import parse_date as _parse_date
from alibi.extraction.templates import ParserHints

logger = logging.getLogger(__name__)


@dataclass
class TextRegions:
    """Named regions of OCR text with line ranges.

    Materializes header/body/footer boundaries for targeted LLM repair.
    Handles mixed-content documents (e.g., payment header + receipt body).
    """

    header: str  # Vendor info, date, registration numbers
    body: str  # Line items / transaction details
    footer: str  # Totals, tax summary, payment info
    header_end: int  # Line index where header ends (exclusive)
    footer_start: int  # Line index where footer begins (inclusive)


@dataclass
class ParseResult:
    """Result of heuristic OCR text parsing."""

    data: dict[str, Any]
    confidence: float  # 0.0-1.0
    field_confidence: dict[str, float] = field(default_factory=dict)
    line_item_count: int = 0
    needs_llm: bool = True
    regions: TextRegions | None = None

    @property
    def gaps(self) -> list[str]:
        """Fields with zero confidence (backward compat)."""
        return [k for k, v in self.field_confidence.items() if v == 0.0]


# ---------------------------------------------------------------------------
# Header noise — lines to skip when looking for vendor name
# ---------------------------------------------------------------------------
_HEADER_NOISE = {
    # EN — payment terminal / POS noise
    "legal receipt",
    "jcc payment systems",
    "aid:",
    "versions:",
    "cardholder",
    "customer copy",
    "merchant copy",
    "sale",
    "purchase",
    "contactless",
    "chip and pin",
    "approved",
    "auth no",
    "payments",
    "worldline",
    "verifone",
    "ingenico",
    "member of",
    # EN — payment processors (should not be picked as vendor)
    "viva.com",
    "viva wallet",
    "sumup",
    "sum up",
    "square",
    "squareup",
    "stripe",
    "adyen",
    "fiserv",
    "toast",
    "clover",
    "zettle",
    "mypos",
    # EN — bank document titles
    "payment order",
    "credit transfer",
    "credit transfer order",
    "wire transfer",
    "wire transfer confirmation",
    "payment instruction",
    "funds transfer",
    # DE
    "kartenzahlung",
    "kundenbeleg",
    "händlerbeleg",
    "genehmigt",
    "zahlungsauftrag",
    "überweisung",
    # EL
    "αντιγραφο πελατη",
    "εγκεκριμενη",
    "αιτηση μεταφορας",
    "αιτηση μεταφορας κεφαλαιων",
    "εντολη πληρωμης",
    "εμβασμα",
    # RU
    "копия клиента",
    "одобрено",
    "платежная система",
    "платежное поручение",
    "перевод средств",
    # Barcode / metadata noise
    "barcode",
}

# Payment processor names — when detected as vendor, prefer vendor_legal_name
_PAYMENT_PROCESSORS = {
    "viva.com",
    "viva wallet",
    "viva",
    "sumup",
    "sum up",
    "square",
    "squareup",
    "stripe",
    "adyen",
    "fiserv",
    "toast",
    "clover",
    "zettle",
    "mypos",
    "worldline",
    "verifone",
    "ingenico",
    "jcc payment systems",
    "jcc",
}


def _swap_processor_vendor(data: dict[str, Any]) -> None:
    """Swap vendor with vendor_legal_name when vendor is a payment processor.

    Falls back to extracting the merchant from vendor_address when
    vendor_legal_name is empty (common on JCC/viva terminal slips where the
    LLM puts the actual merchant into the address field).
    """
    vendor = (data.get("vendor") or "").strip()
    if not vendor:
        return
    vendor_lower = vendor.lower()
    if not any(proc in vendor_lower for proc in _PAYMENT_PROCESSORS):
        return

    legal = (data.get("vendor_legal_name") or "").strip()
    if legal:
        data["vendor"] = legal
        data["vendor_legal_name"] = vendor
        return

    # Fallback: extract merchant from vendor_address.
    # JCC slips format: "24 HR KIOSK/CAVA AMPELAKION 1 LIMASSOL"
    # The merchant name is the first segment before a street-like token.
    address = (data.get("vendor_address") or "").strip()
    if not address:
        return

    # Split on commas first; if no commas, try to find where the street starts.
    parts = [p.strip() for p in address.split(",") if p.strip()]
    if len(parts) >= 2:
        merchant = parts[0]
    else:
        # Heuristic: street addresses typically have a number after the name.
        # Split at the first token that looks like "STREETNAME NUMBER".
        words = address.split()
        merchant_words = []
        for i, w in enumerate(words):
            # A bare number (street number) or a known city signals end of merchant name.
            if i > 0 and re.match(r"^\d+$", w):
                break
            merchant_words.append(w)
        merchant = " ".join(merchant_words) if merchant_words else ""

    if merchant and merchant.lower() not in _PAYMENT_PROCESSORS:
        data["vendor_legal_name"] = vendor
        data["vendor"] = merchant


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Month name abbreviations → month number
_MONTH_NAMES = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}

# Date patterns
_DATE_PATTERNS = [
    # DD/MM/YYYY HH:MM:SS or DD/MM/YYYY HH:MM (optional AM/PM)
    (
        r"(\d{1,2})[/.](\d{1,2})[/.](\d{4})\s+"
        r"(\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp][Mm])?)",
        "dmy_time",
    ),
    # DD/MM/YYYY or DD.MM.YYYY
    (r"(\d{1,2})[/.](\d{1,2})[/.](\d{4})", "dmy"),
    # YYYY-MM-DD
    (r"(\d{4})-(\d{2})-(\d{2})", "ymd"),
    # Mon DD, YYYY or Month DD, YYYY (e.g. "Feb 12, 2026")
    (
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+"
        r"(\d{1,2}),?\s+(\d{4})",
        "mdy_en",
    ),
    # DD Mon YYYY or DD Month YYYY (e.g. "12 Feb 2026")
    (
        r"(\d{1,2})\s+"
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{4})",
        "dmy_en",
    ),
]

# Time pattern (standalone, when not captured with date).
# Matches HH:MM or HH:MM:SS with optional AM/PM suffix.
_TIME_PATTERN = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?\s*(?:[AaPp][Mm])?)\b")

# Payment patterns
_CARD_TYPE_PATTERN = re.compile(
    r"\b(VISA|MASTERCARD|MASTER\s*CARD|AMEX|AMERICAN\s*EXPRESS|MAESTRO|DINERS)\b",
    re.IGNORECASE,
)
_CARD_LAST4_PATTERNS = [
    re.compile(r"\*{3,}\s*-?\s*(\d{4})"),  # ****-7201 or ***7201
    re.compile(r"VISA\s+(\d{4})\b"),  # VISA 7201
    re.compile(r"MASTERCARD\s+(\d{4})\b"),
    re.compile(r"<(\d{4})>"),  # <7201>
    re.compile(r"x{3,}(\d{4})", re.IGNORECASE),  # xxxx7201
]
_AUTH_CODE_PATTERN = re.compile(
    r"AUTH(?:ORISATION|ORIZATION)?\.?\s*(?:NO\.?|CODE)?:?\s*(\w+)", re.IGNORECASE
)
_TERMINAL_ID_PATTERN = re.compile(r"TERMINAL\s*(?:ID)?:?\s*([\w\-]+)", re.IGNORECASE)
_MERCHANT_ID_PATTERN = re.compile(
    r"(?:MERCHANT\s*(?:ID|NO\.?|NUMBER)?|MID)\s*:?\s*([\w\-]+)",
    re.IGNORECASE,
)

# Cash change tracking
_CHANGE_DUE_PATTERN = re.compile(
    r"(?:change\s*(?:due)?|wechselgeld|r[eé]sto)\s*[:.]?\s*(\d+[.,]\d{2})",
    re.IGNORECASE,
)
_CASH_TENDERED_PATTERN = re.compile(
    r"(?:cash|tendered|bar(?:zahlung)?)\s*[:.]?\s*(\d+[.,]\d{2})",
    re.IGNORECASE,
)

# Currency detection
_CURRENCY_PATTERNS = [
    re.compile(r"\bAMOUNT\s+(EUR|USD|GBP|CHF|CZK|PLN|RUB)\b", re.IGNORECASE),
    re.compile(r"\b(EUR|USD|GBP|CHF|CZK|PLN|RUB)\d", re.IGNORECASE),
    re.compile(r"\b(EUR|USD|GBP|CHF|CZK|PLN|RUB)\b"),
    # Russian: руб/RUB (Cyrillic)
    re.compile(r"\b(\d+[.,]\d{2})\s*(?:руб|рублей)\b"),
]

# Combined VAT + TIC on one line:
# - "VAT no. 10057000Y - TIC no. 12057000A" (dash-separated)
# - "VAT 10336127M TIC 123361270" (space-separated)
_VAT_TIC_COMBINED = re.compile(
    r"VAT\s*(?:NO\.?|REG\.?|NUMBER)?\s*\.?\s*([\w]+)"
    r"\s+[-–—]?\s*"
    r"TIC\s*(?:NO\.?|NUMBER)?\s*\.?\s*([\w]+)",
    re.IGNORECASE,
)

# VAT registration patterns (primary identifier on purchase documents)
_VAT_PATTERNS = [
    # "VAT NO:", "VAT REG:", "VAT NUMBER", "VAT N.:" (abbreviated)
    # NB: NUMBER must precede N\.? — otherwise N\.? matches just "n" from "number"
    # and the capture group grabs "umber" (e.g. "VAT number : CY10370773Q" → "umber")
    re.compile(r"VAT\s*(?:NO\.?|REG\.?|NUMBER|N\.?)\s*:?\s*([\w-]{5,})", re.IGNORECASE),
    # With country prefix: "VAT REG: CY - 99000110 H" → capture everything after label
    re.compile(
        r"VAT\s*(?:NO\.?|REG\.?|NUMBER|N\.?)\s*:?\s*"
        r"(?:[A-Z]{2}\s*[-–—]?\s*)?"  # optional country prefix (CY -)
        r"(\d[\d\s]{4,}\w)",  # digit-start, allow internal spaces, end with alnum
        re.IGNORECASE,
    ),
    # CY: "V.A.T. No.:" or "V. A. T. No. :" (spaced/dotted format)
    re.compile(r"V\.?\s*A\.?\s*T\.?\s*N[Oo]\.?\s*:?\s*([\w-]{5,})", re.IGNORECASE),
    re.compile(r"REG\.?\s*(?:NO\.?|NUMBER)\s*:?\s*([\w-]{5,})", re.IGNORECASE),
    # DE: USt-IdNr. (VAT ID)
    re.compile(r"USt[\-.]?(?:Id)?(?:Nr\.?)\s*:?\s*([\w/-]{5,})", re.IGNORECASE),
    # EL: ΑΦΜ (Greek VAT/TIN — serves as VAT on purchase documents)
    re.compile(r"ΑΦΜ\s*:?\s*([\w-]{5,})"),
    # CY/generic: "VAT 10336127M" (bare VAT + number, no qualifier)
    re.compile(r"^VAT\s+(\w{5,})", re.IGNORECASE),
    # CY bare: card terminal slips (JCC, viva.com) print VAT as a standalone line
    # with no label — exactly 8 digits followed by 1 uppercase letter.
    # e.g. "99000110H" or "10057000Y" or "10154482 N" (OCR space).
    # Anchored to full line to avoid false positives.
    re.compile(r"^(\d{8}\s*[A-Z])$"),
]

# Tax ID patterns (income tax / general tax identification)
_TAX_ID_PATTERNS = [
    # CY: TIC (Tax Identification Code)
    re.compile(r"TIC\s*(?:NO\.?|NUMBER)?\s*:?\s*([\w-]{5,})", re.IGNORECASE),
    re.compile(r"T\.?I\.?C\.?\s*:?\s*([\w-]{5,})", re.IGNORECASE),
    # CY: "T.T.C. No.:" variant (same as TIC)
    re.compile(r"T\.?\s*T\.?\s*C\.?\s*N[Oo]\.?\s*:?\s*([\w-]{5,})", re.IGNORECASE),
    # Generic TIN (Tax Identification Number)
    re.compile(r"T\.?I\.?N\.?\s*:?\s*([\w-]{5,})", re.IGNORECASE),
    # DE: Steuernummer (tax number, distinct from USt-IdNr VAT)
    re.compile(r"Steuernummer\s*:?\s*([\w/-]{5,})", re.IGNORECASE),
    # RU: ИНН (taxpayer identification number)
    re.compile(r"ИНН\s*:?\s*(\d{10,12})"),
]


def _clean_vat_value(raw: str) -> str:
    """Normalize extracted VAT value — collapse internal spaces.

    Preserves country prefixes that are part of the number (CY99000110H)
    but strips separated ones (from "CY - 99000110 H" → "99000110H").
    """
    v = raw.strip()
    # Strip separated country prefix: "CY - 99000110 H" or "CY 99000110H"
    # Only strip when there's a space/dash between country code and number
    v = re.sub(r"^[A-Z]{2}[\s-]+", "", v)
    # Collapse internal spaces: "99000110 H" → "99000110H"
    v = re.sub(r"\s+", "", v)
    return v


# Legal entity suffixes — for detecting legal names in header
_LEGAL_NAME_SUFFIXES = {
    "ltd",
    "limited",
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "gmbh",
    "ag",
    "sa",
    "s.a.",
    "sarl",
    "s.a.r.l",
    "srl",
    "s.r.l.",
    "bv",
    "b.v.",
    "nv",
    "n.v.",
    "plc",
    "oy",
    "oyj",
    "ab",
    "a/s",
    "as",
    "aps",
    "co",
    "company",
    # Greek
    "εε",
    "ε.ε.",
    "επε",
    "ε.π.ε.",
    "αε",
    "α.ε.",
    "ike",
    "ικε",
}
_LEGAL_NAME_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(s) for s in sorted(_LEGAL_NAME_SUFFIXES, key=len, reverse=True)
    )
    + r")\.?\s*$",
    re.IGNORECASE,
)

# Phone pattern (EN: Tel., DE: Tel., EL: ΤΗΛ, RU: Тел.)
_PHONE_PATTERN = re.compile(
    r"(?:TEL|ΤΗΛ|Тел)\.?\s*:?\s*([\d\s\-+()]{7,})", re.IGNORECASE
)

# Total-section markers (case-insensitive, multilingual)
_TOTAL_MARKERS = [
    # EN
    "total before disc",
    "subtotal",
    "sub total",
    "sub-total",
    "total",
    "amount due",
    "balance due",
    "grand total",
    "net total",
    "total payable",
    # DE
    "gesamt",
    "summe",
    "zwischensumme",
    "endbetrag",
    "zu zahlen",
    "gesamtbetrag",
    # EL
    "συνολο",  # SYNOLO
    "μερικο συνολο",  # MERIKO SYNOLO (subtotal)
    "γενικο συνολο",  # GENIKO SYNOLO
    # RU
    "итого",
    "всего",
    "к оплате",
    "всего к оплате",
    "промежуточный итог",
    # Section breaks that terminate item listing (EN + DE + EL + RU)
    "bank details",
    "payment details",
    "bankverbindung",
    "bankdaten",
    "τραπεζικα στοιχεια",
    "банковские реквизиты",
    "on behalf of",
    "im namen von",
    "εκ μερους",  # EK MEROUS
    "εκ μρους",  # OCR variant (μρους vs μέρους)
    "от имени",
]

# Discount markers
_DISCOUNT_MARKERS = [
    "discount",
    "member disc",
    "loyalty disc",
    "coupon",
    # DE
    "rabatt",
    "nachlass",
    # EL
    "εκπτωση",
    # RU
    "скидка",
]

# Item-level subtotal pattern: "Subtotal 3.00 eur" or "Subtotal 3.00"
# Distinct from section-level totals — used per item in some receipt formats.
_ITEM_SUBTOTAL_PATTERN = re.compile(
    r"^(?:subtotal|sub\s*total)\s+([\d,.]+)\s*(?:eur|usd|gbp|czk|rub|€|\$|£)?$",
    re.IGNORECASE,
)

# Tax markers
_TAX_MARKERS = [
    re.compile(
        r"(?:TOTAL\s+)?VAT(?:\s+AMOUNT)?(?:\s+\(?\d+(?:\.\d+)?%?\)?)?", re.IGNORECASE
    ),
    re.compile(r"TAX(?:\s+AMOUNT)?", re.IGNORECASE),
    # DE: MwSt (Mehrwertsteuer)
    re.compile(r"MwSt\.?(?:\s+\d+(?:[.,]\d+)?%?)?", re.IGNORECASE),
    re.compile(r"Mehrwertsteuer", re.IGNORECASE),
    # EL: ΦΠΑ
    re.compile(r"ΦΠΑ(?:\s+\d+(?:[.,]\d+)?%?)?"),
    # RU: НДС
    re.compile(r"НДС(?:\s+\d+(?:[.,]\d+)?%?)?"),
]

# Tax summary patterns: T1=19%, Taxable 1 VAT @ 19.00%
_TAX_SUMMARY_PATTERNS = [
    re.compile(
        r"(?:Taxable\s+)?(\d+)\s+(?:VAT\s+)?@\s*(\d+(?:\.\d+)?)\s*%", re.IGNORECASE
    ),
    re.compile(r"T(\d+)\s*=\s*(\d+(?:\.\d+)?)\s*%", re.IGNORECASE),
    re.compile(r"T(\d+)\s+.*?(\d+(?:\.\d+)?)\s*%", re.IGNORECASE),
    # DE: "A 19,0%" or "B  7,0%" — letter codes
    re.compile(r"^([A-E])\s+(\d+(?:[.,]\d+)?)\s*%", re.IGNORECASE | re.MULTILINE),
    # DE: "MwSt. A 19,0%"
    re.compile(r"MwSt\.?\s+([A-E])\s+(\d+(?:[.,]\d+)?)\s*%", re.IGNORECASE),
    # RU: "А 20,0%" or "Б 10,0%" — Cyrillic letter codes
    re.compile(r"^([АБВ])\s+(\d+(?:[.,]\d+)?)\s*%", re.MULTILINE),
]

# Tax code on line items (trailing T1, T2, TZ, A, B, etc., or raw 0/5/19)
_TAX_CODE_TRAILING = re.compile(r"\s+(T[0-9Z]|[A-EА-В]|\d{1,2})\s*$")

# Markdown table patterns (OCR sometimes renders tables as markdown)
_MD_TABLE_HEADER = re.compile(
    r"^\|?\s*(?:qty|quantity)\s*\|\s*(?:description|item|name)\s*\|"
    r"\s*(?:price|unit.?price)\s*\|.*\|\s*(?:total|amount)\s*\|?\s*$",
    re.IGNORECASE,
)
_MD_TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-+:?\s*\|")
_MD_TABLE_ROW = re.compile(r"^\|(.+)\|\s*$")

# Columnar table header (no pipe delimiters): "Qty Description Price Disc. Total"
_COLUMNAR_HEADER = re.compile(
    r"^\s*(?:qty|quantity)\s+(?:description|item|name)\s+(?:price|unit.?price)"
    r"\s+.*(?:total|amount)\s*$",
    re.IGNORECASE,
)
# Columnar row: leading qty + name + trailing price columns
# Matches: "1 Blue Green Wave 4.09 0.00 4.09" or "0.83 Lotoi 5.99 0.00 4.97"
_COLUMNAR_ROW = re.compile(
    r"^\s*(\d+(?:[.,]\d+)?)\s+(.+?)\s+(\d+[.,]\d{2})\s+(\d+[.,]\d{2})\s+(\d+[.,]\d{2})\s*$"
)
# 4-column variant (no discount column): "Qty Description Price Total"
# Matches: "1 Double Espresso 2.80 2.80" or "2 Latte 3.50 7.00"
_COLUMNAR_ROW_4 = re.compile(
    r"^\s*(\d+(?:[.,]\d+)?)\s+(.+?)\s+(\d+[.,]\d{2})\s+(\d+[.,]\d{2})\s*$"
)
# Guard: reject 4-col matches where "name" contains % (tax/VAT summary lines)
_TAX_LINE_GUARD = re.compile(r"\d+[.,]?\d*\s*%")


def _match_columnar(text: str) -> tuple[re.Match[str] | None, bool]:
    """Try 5-col columnar match first, then 4-col fallback.

    Returns (match, is_4col). 4-col rejects tax summary lines containing %.
    """
    m5 = _COLUMNAR_ROW.match(text)
    if m5:
        return m5, False
    m4 = _COLUMNAR_ROW_4.match(text)
    if m4 and not _TAX_LINE_GUARD.search(m4.group(2)):
        return m4, True
    return None, False


# Item-first columnar header: "Item Qty Price Disc Sub Total"
# Two-line format: name on line N, barcode+qty+prices on line N+1
_ITEM_FIRST_COLUMNAR_HEADER = re.compile(
    r"^\s*(?:item|description|name)\s+(?:qty|quantity)\s+(?:price|unit.?price)"
    r"\s+.*(?:total|amount|sub\s*total)\s*$",
    re.IGNORECASE,
)
# Two-line data row: barcode qty price disc subtotal (all numbers)
# Matches: "3886 1 9.45 0.00 9.45" or "90747 1 39.00 0.00 39.00"
_ITEM_FIRST_DATA_ROW = re.compile(
    r"^\s*(\d+)\s+(\d+(?:[.,]\d+)?)\s+(\d+[.,]\d{2})\s+(\d+[.,]\d{2})\s+(\d+[.,]\d{2})\s*$"
)

# Simple 3-column: "Name Qty Amount" (name-first, like MIRADAR UNION receipts)
_NAME_QTY_AMOUNT_HEADER = re.compile(
    r"^\s*(?:name|description|item)\s+(?:qty|quantity)\s+(?:amount|total|sum)\s*$",
    re.IGNORECASE,
)
# NQA row patterns: extract trailing price, optional qty in name part
# Matches any line ending with a price (dd.dd): "Salmon cold smoked 0.339 27.12"
_NQA_ROW = re.compile(r"^(.+?)\s+(\d+[.,]\d{2})\s*$")
# Trailing bare number before price in name part (weighed or integer qty):
# "Salmon cold smoked 0.339" → name="Salmon cold smoked", qty=0.339
_NQA_TRAILING_QTY = re.compile(r"^(.+?)\s+(\d+(?:[.,]\d+)?)\s*$")
# Unit+qty pattern: "Mustard 1pcs 2" → name="Mustard", qty=2
# Handles "Npcs N", "Npc N", "Nшт N" variants
_NQA_UNIT_QTY = re.compile(
    r"^(.+?)\s+\d+\s*(?:pcs|pc|шт)\s+(\d+(?:[.,]\d+)?)\s*$",
    re.IGNORECASE,
)

# Line item patterns
# Standard: NAME  PRICE [TAX_CODE]
# Tax codes: T1, T2, TZ, A-E, or raw numbers (0, 5, 19, 05)
_PRICE_PATTERN = re.compile(r"(\d+[.,]\d{2})\s*(T[0-9Z]|[A-EА-В]|\d{1,2})?\s*$")

# Weighed item: Qty 1.341 @ 2.99 each
_WEIGHED_PATTERN = re.compile(
    r"Qty\s+(\d+[.,]\d+)\s*[@#]\s*(\d+[.,]\d+)\s*each", re.IGNORECASE
)

# Broader weighed pattern: "1.535 kg @ 2.99" or "1.535kg @ 2.99" (no "Qty" prefix)
_WEIGHED_NO_QTY_PATTERN = re.compile(
    r"^(\d+[.,]\d+)\s*(?:kg|g)\s*[@#x×]\s*(\d+[.,]\d+)", re.IGNORECASE
)

# Multi-quantity: Qty 3.000 @ 1.99 each 5.97 T1
_MULTI_QTY_PATTERN = re.compile(
    r"Qty\s+(\d+[.,]\d+)\s*[@#]\s*(\d+[.,]\d+)\s*each\s+(\d+[.,]\d+)\s*(T[0-9Z]|[A-EА-В])?\s*$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Invoice-specific patterns
# ---------------------------------------------------------------------------

# Invoice number
_INVOICE_NUMBER_PATTERNS = [
    re.compile(r"INVOICE\s*(?:NO\.?|NUMBER|#)\s*:?\s*([\w/-]+)", re.IGNORECASE),
    re.compile(r"INV\.?\s*(?:NO\.?|#)\s*:?\s*([\w/-]+)", re.IGNORECASE),
    re.compile(r"INVOICE\s*:\s*([\w/-]{4,})", re.IGNORECASE),
]

# Due date
_DUE_DATE_PATTERNS = [
    re.compile(
        r"DUE\s*(?:DATE|BY|ON)?\s*:?\s*(\d{1,2}[/.]\d{1,2}[/.]\d{4})", re.IGNORECASE
    ),
    re.compile(r"PAYMENT\s*DUE\s*:?\s*(\d{1,2}[/.]\d{1,2}[/.]\d{4})", re.IGNORECASE),
    re.compile(r"PAY\s*BY\s*:?\s*(\d{1,2}[/.]\d{1,2}[/.]\d{4})", re.IGNORECASE),
    re.compile(r"DUE\s*(?:DATE)?\s*:?\s*(\d{4}-\d{2}-\d{2})", re.IGNORECASE),
]

# Payment terms
_PAYMENT_TERMS_PATTERNS = [
    re.compile(r"PAYMENT\s*TERMS?\s*:?\s*(.{4,40}?)(?:\n|$)", re.IGNORECASE),
    re.compile(
        r"TERMS?\s*:?\s*(NET\s*\d+|DUE\s*ON\s*RECEIPT|IMMEDIATE|COD)",
        re.IGNORECASE,
    ),
    re.compile(r"\b(NET\s*\d+|NET-\d+)\b", re.IGNORECASE),
    re.compile(
        r"\b(DUE\s*ON\s*RECEIPT|IMMEDIATE\s*PAYMENT|CASH\s*ON\s*DELIVERY)\b",
        re.IGNORECASE,
    ),
]

# PO / Purchase Order number
_PO_NUMBER_PATTERNS = [
    re.compile(r"P\.?O\.?\s*(?:NO\.?|NUMBER|#)\s*:?\s*([\w/-]+)", re.IGNORECASE),
    re.compile(r"PURCHASE\s*ORDER\s*(?:NO\.?|#)?\s*:?\s*([\w/-]+)", re.IGNORECASE),
]

# Customer / Bill-To block label
_CUSTOMER_LABEL_PATTERN = re.compile(
    r"^(BILL\s*TO|SHIP\s*TO|CLIENT|CUSTOMER|SOLD\s*TO)\s*:?", re.IGNORECASE
)

# Barcode-prefix line: "Barcode: 5290036000111" (no price, precedes product)
_BARCODE_PREFIX_PATTERN = re.compile(r"^Barcode\s*:\s*(\d{7,14})\s*$", re.IGNORECASE)

# Standalone EAN barcode: exactly 8 or 13 digits on their own line
_STANDALONE_BARCODE_PATTERN = re.compile(r"^\s*(\d{8}|\d{13})\s*$")

# Parenthetical product annotation at the end of an item name.
# Matches suffixes like "(takeaway)", "(250g, Ethiopia)", "(Cold pressed juice)".
# Minimum 2 chars inside parens; maximum 40 to avoid matching legal suffixes or
# long embedded clauses.
_PRODUCT_NOTE_PATTERN = re.compile(r"\s+(\([^)]{2,40}\))\s*$")


def _is_valid_ean(digits: str) -> bool:
    """Validate EAN-8 or EAN-13 check digit (GS1 mod-10 algorithm).

    Iterates from right to left: the check digit (rightmost) gets weight 1,
    then weights alternate 3, 1, 3, 1 ... going left.  The total must be
    divisible by 10.  This is the standard GS1 algorithm that works
    consistently for both EAN-8 and EAN-13.
    """
    if len(digits) not in (8, 13):
        return False
    if not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        weight = 1 if i % 2 == 0 else 3
        total += int(ch) * weight
    return total % 10 == 0


# ---------------------------------------------------------------------------
# Invoice item name cleaning patterns
# ---------------------------------------------------------------------------

# Leading line number + product code: "1 20026 " or "12 6048234 "
_INVOICE_ITEM_LEADING_CODE = re.compile(r"^\d{1,3}\s+\d{4,13}\s+")

# Trailing metadata block: qty + unit_price + remaining text (preservation
# method, vat rate, currency) -- the whole tail starting at qty decimal
_INVOICE_ITEM_TRAILING_META = re.compile(r"\s+\d+[.,]\d{2,3}\s+\d+[.,]\d{2}\s+.*$")

# Species/product code suffix: " - SAL", " - SOC", " - OCC", " - NIP"
_INVOICE_ITEM_SPECIES_SUFFIX = re.compile(r"\s*-\s*[A-Z]{2,4}\s*$")

# Trailing qty + EUR currency marker: " 5.00 EUR"
_INVOICE_ITEM_TRAILING_EUR = re.compile(r"\s+[\d.,]+\s+EUR\s*$", re.IGNORECASE)


def _clean_invoice_item_name(name: str) -> str:
    """Strip embedded invoice table metadata from item names.

    Handles OCR-wrapped invoice table rows where columns bleed into
    the item name field (line numbers, product codes, qty, unit price,
    currency, species codes).

    Example input:
        "1 20026 Salmon Steak (Salino Salar) - SAL 0.364 23.80 SomeText 5.00 EUR"
    Example output:
        "Salmon Steak (Salino Salar)"
    """
    if not name:
        return name

    # Strip leading line number + product code: "1 20026 "
    name = _INVOICE_ITEM_LEADING_CODE.sub("", name)

    # Strip trailing qty + unit_price + description + vat + EUR
    # Match: " 0.364 23.80 SomeText 5.00 EUR"
    name = _INVOICE_ITEM_TRAILING_EUR.sub("", name)

    # Strip trailing quantity + price pair: " 0.526 19.04 ..."
    # Only when followed by non-digit content (preservation method text)
    name = _INVOICE_ITEM_TRAILING_META.sub("", name)

    # Strip species code suffix: "- SAL", "- SOC", "- OCC", "- NIP"
    name = _INVOICE_ITEM_SPECIES_SUFFIX.sub("", name)

    return name.strip()


def _is_integer_qty(value: float) -> bool:
    """Check if a Qty value is effectively an integer (e.g. 3.000, 2.0).

    Used to distinguish multi-quantity (Qty 3.000 @ 1.99 each = buy 3)
    from weighed items (Qty 1.341 @ 2.99 each = 1.341 kg).
    """
    return abs(value - round(value)) < 0.01


def _parse_decimal(s: str) -> float | None:
    """Parse a decimal string, handling European comma format."""
    if not s:
        return None
    s = s.strip()
    # European format: comma as decimal separator (when no period present)
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _is_noise_line(line: str) -> bool:
    """Check if a line is header noise (not a vendor name candidate)."""
    low = line.strip().lower()
    if not low or len(low) < 3:
        return True
    for noise in _HEADER_NOISE:
        if noise in low:
            return True
    # Lines that are just numbers or codes
    if re.match(r"^[\d\s\-/.]+$", low):
        return True
    # Lines starting with ALL-CAPS keyword prefix (AID:, APP:, BARCODE:, TERMINAL:, etc.)
    if re.match(r"^[A-Z]{2,}:", line.strip()):
        return True
    # Page separators from multi-page OCR text
    if re.match(r"^---\s*page\s+\d+\s*---$", low):
        return True
    return False


# Patterns that should not be captured as vendor address
_ADDRESS_POLLUTION = {
    "customer receipt",
    "customer copy",
    "merchant copy",
    "dine in",
    "dine-in",
    "take away",
    "takeaway",
    "take-away",
    "eat in",
    "table no",
    "table:",
    "server:",
    "cashier:",
    "operator:",
    "pos:",
    "till:",
    "receipt",
    # DE
    "kundenbeleg",
    "händlerbeleg",
    "vor ort",
    "zum mitnehmen",
    "kassierer:",
    "tisch:",
    # EL
    "αντιγραφο πελατη",
    "ταμειο:",
    "τραπεζι:",
    # RU
    "копия клиента",
    "кассир:",
    "стол:",
}


def _is_address_pollution(line: str) -> bool:
    """Check if a line should NOT be captured as vendor address."""
    low = line.strip().lower()
    # Exact or substring matches for non-address content
    for term in _ADDRESS_POLLUTION:
        if term in low:
            return True
    # Registration lines (VAT, TIC) that weren't caught by regex
    if re.search(r"\bvat\b", low, re.IGNORECASE):
        return True
    if re.search(r"\btic\b", low, re.IGNORECASE):
        return True
    if re.search(r"\breg\b", low, re.IGNORECASE):
        return True
    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _preprocess_single_line(raw_text: str) -> str:
    """Re-insert line breaks if OCR text is a single long line.

    Some YAML caches or OCR backends produce a single continuous string.
    This attempts to re-insert newlines at item/section boundaries.
    """
    lines = raw_text.strip().split("\n")
    if len(lines) > 3 or len(raw_text) < 200:
        return raw_text  # Already multi-line or short

    text = raw_text

    # --- Header splits ---
    text = re.sub(r"\s+(TEL\.?\s*:?\s*\d)", r"\n\1", text)
    text = re.sub(r"\s+(T\.?I\.?N\.?\s)", r"\n\1", text)
    text = re.sub(r"\s+(VAT\s+NO)", r"\n\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(AID:\s)", r"\n\1", text)
    text = re.sub(r"\s+(AMOUNT\s)", r"\n\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(AUTH\s+NO)", r"\n\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(Versions:)", r"\n\1", text)
    text = re.sub(r"\s+(CARDHOLDER)", r"\n\1", text, flags=re.IGNORECASE)
    # Split "Item Price" header from first item
    text = re.sub(r"\s+(Item\s+Price)\s+", r"\n\1\n", text, flags=re.IGNORECASE)

    # --- Item splits ---
    # Split after price + tax code (T1/T2/TZ) before next item
    text = re.sub(r"(\d+[.,]\d{2}\s+T[0-9Z])\s+(?!Qty\b)", r"\1\n", text)
    # Split before Qty lines
    text = re.sub(r"\s+(Qty\s+)", r"\n\1", text)
    # Split after "Qty ... each [price tax]" when followed by non-digit (next item)
    text = re.sub(r"(each(?:\s+\d+[.,]\d{2}\s+T[0-9Z])?)\s+(?=[A-Z])", r"\1\n", text)

    # --- Footer splits ---
    text = re.sub(r"\s+(\(\s*\d+\s+Items?\))", r"\n\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(Total Before)", r"\n\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(LESS DISCOUNT)", r"\n\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(TOTAL\s)", r"\n\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(VISA\s+Card)", r"\n\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(Change\s)", r"\n\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(Taxable\s)", r"\n\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(Total VAT)", r"\n\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(Salesperson:)", r"\n\1", text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# Post-OCR text-based document classification
# ---------------------------------------------------------------------------

# Classification signal patterns
_INVOICE_SIGNALS = [
    re.compile(r"invoice\s*(?:no\.?|number|#)", re.IGNORECASE),
    re.compile(r"bill\s*to\b", re.IGNORECASE),
    re.compile(r"due\s*date\b", re.IGNORECASE),
    re.compile(r"payment\s*terms?\b", re.IGNORECASE),
    re.compile(r"purchase\s*order\b", re.IGNORECASE),
    re.compile(r"rechnung(?:snummer)?\b", re.IGNORECASE),  # DE
    re.compile(r"τιμολ[οό]γιο\b", re.IGNORECASE),  # EL
    re.compile(r"счет[- ]?фактура\b", re.IGNORECASE),  # RU
]

_STATEMENT_SIGNALS = [
    re.compile(r"account\s*(?:no\.?|number|activity|statement)", re.IGNORECASE),
    re.compile(r"\bIBAN\b"),
    re.compile(r"\bBIC\b"),
    re.compile(r"period\s*:?\s*\d", re.IGNORECASE),
    re.compile(r"\bdebit\b.*\bcredit\b", re.IGNORECASE),
    re.compile(r"kontoauszug\b", re.IGNORECASE),  # DE
    re.compile(r"выписка\b", re.IGNORECASE),  # RU
]

_PAYMENT_CONFIRMATION_SIGNALS = [
    re.compile(r"transaction\s*confirmation\b", re.IGNORECASE),
    re.compile(r"payment\s*confirmation\b", re.IGNORECASE),
    re.compile(r"confirmation\s*of\s*(?:transaction|payment)\b", re.IGNORECASE),
    re.compile(r"beneficiary\b", re.IGNORECASE),
    re.compile(r"debit\s*account\b", re.IGNORECASE),
    re.compile(r"transaction\s*reference\b", re.IGNORECASE),
    # Multilingual bank transfer document titles
    re.compile(r"αιτηση\s*μεταφορας\b", re.IGNORECASE),  # EL: request for transfer
    re.compile(r"εντολη\s*πληρωμης\b", re.IGNORECASE),  # EL: payment order
    re.compile(r"δικαιουχος\b", re.IGNORECASE),  # EL: beneficiary
    re.compile(r"платежное\s*поручение\b", re.IGNORECASE),  # RU: payment order
    re.compile(r"zahlungsauftrag\b", re.IGNORECASE),  # DE: payment order
    re.compile(r"payment\s*order\b", re.IGNORECASE),
    re.compile(r"wire\s*transfer\b", re.IGNORECASE),
]

# Bank/wire transfer document title prefixes — if _extract_header() picks one of
# these as the vendor, we discard it and look for the actual payee/beneficiary.
_BANK_DOC_TITLE_PREFIXES = {
    "transaction confirmation",
    "payment confirmation",
    "transfer confirmation",
    "payment order",
    "credit transfer",
    "wire transfer",
    "payment instruction",
    "funds transfer",
    "bank transfer",
    # EL
    "αιτηση μεταφορας",
    "εντολη πληρωμης",
    "εμβασμα",
    # RU
    "платежное поручение",
    "перевод средств",
    # DE
    "zahlungsauftrag",
    "überweisung",
    # Generic
    "account information",
    "transaction details",
    "customer receipt",
}

# Matches beneficiary / payee labels in bank confirmation documents.
_BENEFICIARY_PATTERN = re.compile(
    r"^(?:Beneficiary(?:\s+name)?|Payee|Recipient"
    r"|Δικαιούχος|Δικαιουχος"
    r"|Получатель"
    r"|Empfänger)\s*:\s*(.+)",
    re.IGNORECASE,
)

# Bank metadata line prefixes — lines starting with these are not vendor names.
_BANK_META_PREFIXES = (
    "iban",
    "account",
    "customer",
    "bank name",
    "bank:",
    "bic",
    "swift",
    "sort code",
    "payer",
    "sender",
    "ordering party",
    "debit account",
    "print date",
    "reference",
    "execution date",
    # DE
    "kontonummer",
    "auftraggeber",
    "absender",
    "konto",
    # EL
    "iban αποστολ",
    "λογαριασμος",
    "αποστολεας",
    # RU
    "плательщик",
    "отправитель",
    "счет плательщика",
    "номер счета",
    "дата",
)

_WARRANTY_SIGNALS = [
    re.compile(r"warranty\b", re.IGNORECASE),
    re.compile(r"guarantee\b", re.IGNORECASE),
    re.compile(r"serial\s*(?:no\.?|number)\b", re.IGNORECASE),
    re.compile(r"warranty\s*(?:end|expir|valid)", re.IGNORECASE),
    re.compile(r"Garantie\b", re.IGNORECASE),  # DE
    re.compile(r"гарантия\b", re.IGNORECASE),  # RU
]

_CONTRACT_SIGNALS = [
    re.compile(r"contract\s*(?:no\.?|number)\b", re.IGNORECASE),
    re.compile(r"(?:effective|commencement)\s*date\b", re.IGNORECASE),
    re.compile(r"parties?\s*(?:to|of)\b", re.IGNORECASE),
    re.compile(r"terms?\s*and\s*conditions?\b", re.IGNORECASE),
    re.compile(r"Vertrag\b", re.IGNORECASE),  # DE
    re.compile(r"договор\b", re.IGNORECASE),  # RU
]


def classify_ocr_text(raw_text: str) -> str:
    """Classify document type from OCR text using heuristic signals.

    Runs after Stage 1 OCR to determine document type without a separate
    vision model call. Uses pattern matching with weighted scoring.

    Returns:
        One of: receipt, invoice, payment_confirmation, statement,
        warranty, contract. Defaults to "receipt" when uncertain.
    """
    if not raw_text or not raw_text.strip():
        return "receipt"

    low = raw_text.lower()

    # Score each type by counting signal matches
    scores: dict[str, float] = {
        "invoice": 0.0,
        "statement": 0.0,
        "payment_confirmation": 0.0,
        "warranty": 0.0,
        "contract": 0.0,
        "receipt": 0.0,
    }

    for pat in _INVOICE_SIGNALS:
        if pat.search(raw_text):
            scores["invoice"] += 1.0

    for pat in _STATEMENT_SIGNALS:
        if pat.search(raw_text):
            scores["statement"] += 1.0

    for pat in _PAYMENT_CONFIRMATION_SIGNALS:
        if pat.search(raw_text):
            scores["payment_confirmation"] += 1.0

    for pat in _WARRANTY_SIGNALS:
        if pat.search(raw_text):
            scores["warranty"] += 1.0

    for pat in _CONTRACT_SIGNALS:
        if pat.search(raw_text):
            scores["contract"] += 1.0

    # Receipt signals: price patterns + total markers + no strong other signal
    lines = raw_text.split("\n")
    price_lines = sum(
        1
        for line in lines
        if _PRICE_PATTERN.search(line.strip()) and not _is_total_marker(line)
    )
    has_total = any(_is_total_marker(line) for line in lines)
    if price_lines >= 2:
        scores["receipt"] += 2.0
    if has_total:
        scores["receipt"] += 1.0

    # Card slip heuristic: card details but no real line items
    has_card = bool(
        _CARD_TYPE_PATTERN.search(raw_text)
        or any(p.search(raw_text) for p in _CARD_LAST4_PATTERNS)
    )
    has_auth = bool(_AUTH_CODE_PATTERN.search(raw_text))
    has_terminal = bool(_TERMINAL_ID_PATTERN.search(raw_text))
    card_signals = sum([has_card, has_auth, has_terminal])

    if card_signals >= 2 and price_lines <= 1:
        # Card slip with no real items → payment confirmation
        scores["payment_confirmation"] += 3.0
    elif card_signals >= 1 and price_lines >= 2:
        # Card details + line items → receipt (mixed content document)
        scores["receipt"] += 1.0

    # JCC / payment terminal noise boosts payment_confirmation
    jcc_noise = sum(
        1
        for kw in ["jcc payment", "cardholder", "customer copy", "merchant copy"]
        if kw in low
    )
    if jcc_noise >= 2:
        scores["payment_confirmation"] += 1.5

    # Find the highest-scoring type
    best_type = max(scores, key=lambda k: scores[k])
    best_score = scores[best_type]

    # Require minimum confidence to override receipt default
    if best_score < 1.5 and best_type != "receipt":
        return "receipt"

    return best_type


def parse_ocr_text(
    raw_text: str,
    doc_type: str = "receipt",
    hints: ParserHints | None = None,
) -> ParseResult:
    """Parse raw OCR text into structured data using heuristics.

    Args:
        raw_text: Raw OCR text from Stage 1.
        doc_type: Document type (receipt, invoice, payment_confirmation, etc.)
        hints: Optional parser hints from template learning system.

    Returns:
        ParseResult with structured data, confidence, and gap list.
    """
    if not raw_text or not raw_text.strip():
        return ParseResult(
            data={}, confidence=0.0, field_confidence={"empty_text": 0.0}
        )

    # Handle single-line OCR text (e.g. from YAML cache folding)
    raw_text = _preprocess_single_line(raw_text)

    try:
        if doc_type == "receipt":
            return _parse_receipt(raw_text, doc_type, hints=hints)
        elif doc_type == "invoice":
            return _parse_invoice(raw_text, hints=hints)
        elif doc_type == "payment_confirmation":
            return _parse_payment_confirmation(raw_text)
        elif doc_type == "statement":
            return _parse_statement(raw_text)
        else:
            # warranty, contract — minimal parsing
            return _parse_minimal(raw_text, doc_type)
    except Exception as e:
        logger.warning(f"Text parser failed for {doc_type}: {e}", exc_info=True)
        return ParseResult(
            data={},
            confidence=0.0,
            field_confidence={"parser_exception": 0.0},
            needs_llm=True,
        )


# ---------------------------------------------------------------------------
# Receipt / Invoice parser
# ---------------------------------------------------------------------------


def _try_layout(lines: list[str], layout_type: str) -> list[dict[str, Any]] | None:
    """Try a specific layout parser. Returns items or None if none found."""
    if layout_type == "markdown_table":
        return _extract_markdown_table_items(lines) or None
    elif layout_type in ("columnar", "columnar_4", "columnar_5"):
        return _extract_columnar_table_items(lines) or None
    elif layout_type == "item_first_columnar":
        return _extract_item_first_columnar_items(lines) or None
    elif layout_type == "nqa":
        return _extract_name_qty_amount_items(lines) or None
    return None


def _detect_barcode_position_from_lines(
    lines: list[str],
    items: list[dict[str, Any]],
) -> str | None:
    """Detect whether barcodes appear before or after items in OCR text.

    Scans for standalone barcode lines and checks whether the adjacent
    item name line is above (after_item) or below (before_item).
    Requires at least 2 consistent observations to report a pattern.
    """
    if not items:
        return None

    item_names = {str(it.get("name") or "").strip().lower() for it in items}
    if not item_names:
        return None

    before_count = 0
    after_count = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        bc_m = _STANDALONE_BARCODE_PATTERN.match(stripped)
        if not bc_m or not _is_valid_ean(bc_m.group(1)):
            continue

        # Check line AFTER barcode — if it contains an item name → before_item
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip().lower()
            for name in item_names:
                if name and len(name) > 3 and name in next_line:
                    before_count += 1
                    break

        # Check line BEFORE barcode — if it contains an item name → after_item
        if i > 0:
            prev_line = lines[i - 1].strip().lower()
            for name in item_names:
                if name and len(name) > 3 and name in prev_line:
                    after_count += 1
                    break

    if before_count >= 2 and before_count > after_count:
        return "before_item"
    if after_count >= 2 and after_count > before_count:
        return "after_item"
    return None


def _extract_line_items_with_hints(
    lines: list[str],
    tax_map: dict[str, float],
    hints: ParserHints | None,
    data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Extract line items, trying hinted layout first.

    Sets data["_layout_type"] to track which parser succeeded.
    """
    # Try hinted layout first
    if hints and hints.layout_type:
        hinted_items = _try_layout(lines, hints.layout_type)
        if hinted_items:
            if data is not None:
                data["_layout_type"] = hints.layout_type
            return hinted_items

    # Auto-detect: try each parser in order, track which succeeded
    md_items = _extract_markdown_table_items(lines)
    if md_items:
        if data is not None:
            data["_layout_type"] = "markdown_table"
        return md_items

    col_meta: dict[str, Any] = {}
    col_items = _extract_columnar_table_items(lines, meta=col_meta)
    if col_items:
        if data is not None:
            data["_layout_type"] = col_meta.get("columnar_variant", "columnar")
        return col_items

    ifc_items = _extract_item_first_columnar_items(lines)
    if ifc_items:
        if data is not None:
            data["_layout_type"] = "item_first_columnar"
        return ifc_items

    nqa_items = _extract_name_qty_amount_items(lines)
    if nqa_items:
        if data is not None:
            data["_layout_type"] = "nqa"
        return nqa_items

    # Standard parser (inline prices)
    if data is not None:
        data["_layout_type"] = "standard"
    bc_hint = hints.barcode_position if hints else None
    std_items = _extract_line_items(lines, tax_map, barcode_hint=bc_hint)

    # Detect barcode position pattern for template learning.
    # Scan OCR lines: when a standalone barcode line immediately precedes
    # a product line, that's "before_item"; when it follows, "after_item".
    if data is not None and not data.get("_barcode_position"):
        bc_pos = _detect_barcode_position_from_lines(lines, std_items)
        if bc_pos:
            data["_barcode_position"] = bc_pos

    return std_items


def _parse_receipt(
    raw_text: str, doc_type: str, hints: ParserHints | None = None
) -> ParseResult:
    """Full receipt/invoice parsing with line items."""
    lines = raw_text.split("\n")
    data: dict[str, Any] = {"document_type": doc_type}
    fc: dict[str, float] = {}  # per-field confidence
    old_gaps: list[str] = []  # used by _extract_header etc. (mutated)

    # Pass 1: Header — vendor, address, phone, registration
    _extract_header(lines, data, old_gaps, hints=hints)
    _swap_processor_vendor(data)
    # Apply hint vendor as fallback if header extraction missed it
    if not data.get("vendor") and hints and hints.vendor_name:
        data["vendor"] = hints.vendor_name
    fc["vendor"] = 1.0 if data.get("vendor") else 0.0

    # Pass 2: Date / Time
    _extract_date_time(raw_text, data, old_gaps, hints=hints)
    fc["date"] = 1.0 if data.get("date") else 0.0

    # Pass 3: Payment info
    _extract_payment(raw_text, data)
    fc["payment_method"] = 1.0 if data.get("payment_method") else 0.3

    # Pass 4: Currency
    _extract_currency(raw_text, data)
    if not data.get("currency") and hints and hints.currency:
        data["currency"] = hints.currency
    fc["currency"] = 0.8 if data.get("currency") else 0.0

    # Pass 5: Tax summary (before line items, so we can map codes)
    tax_map = _extract_tax_summary(raw_text)

    # Pass 6: Line items (prefer hinted layout, fallback to auto-detect)
    items = _extract_line_items_with_hints(lines, tax_map, hints, data)
    data["line_items"] = items
    item_count = len(items)
    fc["line_items"] = min(1.0, item_count / 3) if item_count > 0 else 0.0

    # Pass 7: Totals (total, subtotal, tax, discount)
    _extract_totals(lines, data, hints=hints)
    fc["total"] = 1.0 if data.get("total") else 0.2

    # Infer language from matched patterns
    if not data.get("_detected_language"):
        total_marker = (data.get("_total_marker") or "").lower()
        if any(m in total_marker for m in ("synolo", "σύνολο", "φπα")):
            data["_detected_language"] = "el"
        elif any(m in total_marker for m in ("gesamt", "summe", "mwst")):
            data["_detected_language"] = "de"
        elif any(m in total_marker for m in ("итого", "всего", "ндс")):
            data["_detected_language"] = "ru"
        else:
            data["_detected_language"] = "en"

    # Fields parser can't fill (enrichment only)
    fc["name_en"] = 0.0
    fc["category"] = 0.0
    fc["brand"] = 0.0
    fc["language"] = 0.0

    # Region splitting
    regions = _split_regions(lines)

    # Tag region boundary metadata for template learning
    if regions.header:
        data["_header_lines"] = len(regions.header.strip().splitlines())
    total_lines = len(lines)
    if regions.footer_start and total_lines > 0:
        data["_footer_ratio"] = round(regions.footer_start / total_lines, 2)

    # Compute confidence from core extraction fields (exclude semantic-only)
    core_fields = ("vendor", "date", "currency", "line_items", "total")
    core_signals = [fc[k] for k in core_fields if k in fc]
    confidence = sum(core_signals) / len(core_signals) if core_signals else 0.0
    # LLM needed only for enrichment when all core fields are present
    needs_llm = confidence < 0.9

    return ParseResult(
        data=data,
        confidence=round(confidence, 3),
        field_confidence=fc,
        line_item_count=item_count,
        needs_llm=needs_llm,
        regions=regions,
    )


def _has_legal_suffix(text: str) -> bool:
    """Check if text ends with a legal entity suffix (Ltd, GmbH, etc.)."""
    return bool(_LEGAL_NAME_PATTERN.search(text.strip()))


def _extract_header(
    lines: list[str],
    data: dict[str, Any],
    gaps: list[str],
    *,
    hints: ParserHints | None = None,
) -> None:
    """Extract vendor name, address, phone, VAT, tax ID from header lines.

    Detects both trade name (prominent header) and legal name (entity with
    legal suffix like Ltd, GmbH). Stores:
    - vendor: trade name (user-facing)
    - vendor_legal_name: legal entity name if different from trade name
    - vendor_address, vendor_phone: as before
    - vendor_vat: VAT registration number (primary ID for vendor matching)
    - vendor_tax_id: TIC/TIN tax identification (secondary)

    When ``hints.expected_header_lines`` is set, limits scanning to that
    many lines plus a small margin to prevent false vendor extraction
    from body text.
    """
    max_header_scan = len(lines)
    if hints and hints.expected_header_lines:
        max_header_scan = min(len(lines), hints.expected_header_lines + 3)

    vendor = None
    legal_name = None
    address_parts: list[str] = []

    for idx, line in enumerate(lines):
        if idx >= max_header_scan:
            break
        stripped = line.strip()
        if not stripped:
            continue

        # Check for phone
        phone_match = _PHONE_PATTERN.search(stripped)
        if phone_match:
            data["vendor_phone"] = phone_match.group(1).strip()
            continue

        # Check for registration numbers (VAT, TIC, combined)
        matched_reg = False

        # Try combined VAT + TIC first (e.g., "VAT no. X - TIC no. Y")
        combined = _VAT_TIC_COMBINED.search(stripped)
        if combined:
            data["vendor_vat"] = _clean_vat_value(combined.group(1))
            data["vendor_tax_id"] = combined.group(2).strip()
            matched_reg = True

        if not matched_reg:
            # Try VAT-specific patterns
            for pat in _VAT_PATTERNS:
                reg_match = pat.search(stripped)
                if reg_match:
                    data["vendor_vat"] = _clean_vat_value(reg_match.group(1))
                    matched_reg = True
                    break

        if not matched_reg:
            # Try Tax ID patterns (TIC, TIN, Steuernummer, ИНН)
            for pat in _TAX_ID_PATTERNS:
                tax_match = pat.search(stripped)
                if tax_match:
                    data["vendor_tax_id"] = tax_match.group(1).strip()
                    matched_reg = True
                    break

        if matched_reg:
            continue

        # Not a registration line — could be vendor or address
        if _is_noise_line(stripped):
            continue

        # Check if this looks like a price line (item section started)
        if _PRICE_PATTERN.search(stripped):
            break

        # Check if it looks like a date line
        if re.search(r"\d{1,2}[/.]\d{1,2}[/.]\d{4}", stripped):
            if vendor:
                break
            continue

        if vendor is None:
            vendor = stripped
            data["vendor"] = vendor
        elif legal_name is None and not address_parts and _has_legal_suffix(stripped):
            # Line immediately after vendor with legal suffix → legal name
            legal_name = stripped
            data["vendor_legal_name"] = legal_name
        elif len(address_parts) < 2 and not _is_address_pollution(stripped):
            address_parts.append(stripped)

    if address_parts:
        data["vendor_address"] = ", ".join(address_parts)

    if not vendor:
        gaps.append("vendor")


def _normalize_time(raw_time: str) -> str:
    """Normalize a time string to 24-hour format.

    Accepts "6:35 PM", "6:35PM", "14:30", "2:05:30 am", etc.
    Returns "HH:MM" or "HH:MM:SS" in 24-hour format.
    """
    s = raw_time.strip()
    # Detect AM/PM suffix
    ampm_match = re.match(r"(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([AaPp][Mm])", s)
    if ampm_match:
        hour = int(ampm_match.group(1))
        minute = ampm_match.group(2)
        second = ampm_match.group(3)
        meridiem = ampm_match.group(4).upper()
        if meridiem == "PM" and hour != 12:
            hour += 12
        elif meridiem == "AM" and hour == 12:
            hour = 0
        base = f"{hour:02d}:{minute}"
        return f"{base}:{second}" if second else base
    return s


def _extract_date_time(
    raw_text: str,
    data: dict[str, Any],
    gaps: list[str],
    *,
    hints: ParserHints | None = None,
) -> None:
    """Extract date and time from OCR text."""
    patterns = list(_DATE_PATTERNS)
    if hints and hints.date_format:
        hinted = [(p, f) for p, f in patterns if f == hints.date_format]
        rest = [(p, f) for p, f in patterns if f != hints.date_format]
        patterns = hinted + rest
    for pattern_str, fmt_type in patterns:
        match = re.search(pattern_str, raw_text, re.IGNORECASE)
        if match:
            if fmt_type == "dmy_time":
                raw = f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
                parsed = _parse_date(raw)
                if parsed:
                    data["date"] = parsed.isoformat()
                    data["time"] = _normalize_time(match.group(4))
            elif fmt_type == "dmy":
                raw = f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
                parsed = _parse_date(raw)
                if parsed:
                    data["date"] = parsed.isoformat()
            elif fmt_type == "ymd":
                raw = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                parsed = _parse_date(raw)
                if parsed:
                    data["date"] = parsed.isoformat()
            elif fmt_type == "mdy_en":
                month = _MONTH_NAMES.get(match.group(1).lower()[:3])
                d, y = match.group(2), match.group(3)
                if month:
                    data["date"] = f"{y}-{month}-{d.zfill(2)}"
            elif fmt_type == "dmy_en":
                d = match.group(1)
                month = _MONTH_NAMES.get(match.group(2).lower()[:3])
                y = match.group(3)
                if month:
                    data["date"] = f"{y}-{month}-{d.zfill(2)}"
            if "date" in data:
                data["_date_format"] = fmt_type
            break

    # If no time captured yet, look for standalone time
    if "time" not in data:
        time_match = _TIME_PATTERN.search(raw_text)
        if time_match:
            data["time"] = _normalize_time(time_match.group(1))

    if "date" not in data:
        gaps.append("date")


def _extract_payment(raw_text: str, data: dict[str, Any]) -> None:
    """Extract payment method, card type, last4, auth code."""
    # Card type
    card_match = _CARD_TYPE_PATTERN.search(raw_text)
    if card_match:
        card_type = card_match.group(1).lower()
        if "master" in card_type:
            card_type = "mastercard"
        elif "american" in card_type:
            card_type = "amex"
        data["card_type"] = card_type
        data["payment_method"] = "card"

    # Card last 4
    for pat in _CARD_LAST4_PATTERNS:
        m = pat.search(raw_text)
        if m:
            data["card_last4"] = m.group(1)
            if "payment_method" not in data:
                data["payment_method"] = "card"
            break

    # Auth code
    auth_match = _AUTH_CODE_PATTERN.search(raw_text)
    if auth_match:
        data["authorization_code"] = auth_match.group(1)

    # Terminal ID
    term_match = _TERMINAL_ID_PATTERN.search(raw_text)
    if term_match:
        data["terminal_id"] = term_match.group(1)

    # Merchant ID
    mid_match = _MERCHANT_ID_PATTERN.search(raw_text)
    if mid_match:
        data["merchant_id"] = mid_match.group(1)

    # Contactless / cash detection
    if "payment_method" not in data:
        low = raw_text.lower()
        if "contactless" in low:
            data["payment_method"] = "contactless"
        elif re.search(r"\bcash\b", low) and "cashier" not in low:
            data["payment_method"] = "cash"

    # Cash change tracking (only when payment is cash)
    if data.get("payment_method") == "cash":
        change_match = _CHANGE_DUE_PATTERN.search(raw_text)
        if change_match:
            data["change_due"] = _parse_decimal(change_match.group(1))
        tendered_match = _CASH_TENDERED_PATTERN.search(raw_text)
        if tendered_match:
            data["amount_tendered"] = _parse_decimal(tendered_match.group(1))


def _extract_currency(raw_text: str, data: dict[str, Any]) -> None:
    """Extract currency from OCR text."""
    # Check for Cyrillic currency markers first
    if re.search(r"руб(?:лей)?", raw_text, re.IGNORECASE):
        data["currency"] = "RUB"
        return

    # Check for currency symbols
    if "€" in raw_text:
        data["currency"] = "EUR"
        return
    if "£" in raw_text:
        data["currency"] = "GBP"
        return
    if "$" in raw_text and "EUR" not in raw_text:
        data["currency"] = "USD"
        return

    for pat in _CURRENCY_PATTERNS:
        m = pat.search(raw_text)
        if m:
            data["currency"] = m.group(1).upper()
            return


def _extract_tax_summary(raw_text: str) -> dict[str, float]:
    """Parse VAT summary table to build tax_code -> rate mapping.

    E.g. "Taxable 1 VAT @ 19.00%" -> {"T1": 19.0, "1": 19.0}

    When the same code appears multiple times (e.g. "Taxable 2 VAT @ 5.00%"
    and "Taxable 2 VAT @ 0.00%"), keep the highest rate (0% is usually
    exempt items within the same category).
    """
    tax_map: dict[str, float] = {}
    for pat in _TAX_SUMMARY_PATTERNS:
        for match in pat.finditer(raw_text):
            code = match.group(1)
            rate_val = _parse_decimal(match.group(2))
            if rate_val is None:
                continue
            key_t = f"T{code}"
            # Keep the highest rate for each code
            if rate_val > tax_map.get(key_t, -1):
                tax_map[key_t] = rate_val
            if rate_val > tax_map.get(code, -1):
                tax_map[code] = rate_val
    return tax_map


def _extract_markdown_table_items(
    lines: list[str],
) -> list[dict[str, Any]] | None:
    """Extract items from markdown table format (OCR renders some receipts this way).

    Detects tables like:
        | Qty | Description | Price | Disc. | Total |
        | :--- | :--- | :--- | :--- | :--- |
        | 1 | Blue Green Wave Frozen Blueberries 500 | 4.09 | 0.00 | 4.09 |

    Returns list of item dicts, or None if no markdown table found.
    """
    # Find table header
    header_idx = None
    for i, line in enumerate(lines):
        if _MD_TABLE_HEADER.match(line.strip()):
            header_idx = i
            break

    if header_idx is None:
        return None

    # Parse header to determine column mapping
    header_cells = [
        c.strip().lower() for c in lines[header_idx].strip().strip("|").split("|")
    ]

    col_map: dict[str, int] = {}
    for ci, cell in enumerate(header_cells):
        if cell in ("qty", "quantity"):
            col_map["qty"] = ci
        elif cell in ("description", "item", "name"):
            col_map["name"] = ci
        elif cell in ("price", "unit price", "unit_price"):
            col_map["unit_price"] = ci
        elif cell in ("total", "amount"):
            col_map["total"] = ci

    if "name" not in col_map or "total" not in col_map:
        return None

    items: list[dict[str, Any]] = []
    # Skip separator line(s) after header
    start = header_idx + 1
    while start < len(lines) and _MD_TABLE_SEPARATOR.match(lines[start].strip()):
        start += 1

    for i in range(start, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        # Stop at non-table lines (total section, etc.)
        m = _MD_TABLE_ROW.match(line)
        if not m:
            break

        cells = [c.strip() for c in m.group(1).split("|")]
        if len(cells) < len(header_cells):
            continue

        name = cells[col_map["name"]] if "name" in col_map else ""
        if not name:
            continue

        qty_str = cells[col_map["qty"]] if "qty" in col_map else "1"
        unit_price_str = (
            cells[col_map["unit_price"]] if "unit_price" in col_map else None
        )
        total_str = cells[col_map["total"]] if "total" in col_map else None

        qty = _parse_decimal(qty_str)
        unit_price = _parse_decimal(unit_price_str) if unit_price_str else None
        total_price = _parse_decimal(total_str) if total_str else None

        if total_price is None:
            continue

        item: dict[str, Any] = {
            "name": name,
            "quantity": qty if qty else 1,
            "unit_price": unit_price if unit_price else total_price,
            "total_price": total_price,
        }
        items.append(item)

    return items if items else None


def _extract_columnar_table_items(
    lines: list[str],
    meta: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    """Extract items from space-delimited columnar format.

    Detects tables like:
        Qty Description Price Disc. Total
        1 Blue Green Wave Frozen Blueberries 500 4.09 0.00 4.09
        5292006000152
        0.83 Lotoi 5.99 0.00 4.97

    Handles:
    - Standalone barcode lines (EAN-8/EAN-13) between items: attached to the
      preceding item, or carried forward to the next item if no item yet.
    - Multi-line item names: when a non-matching line appears that is not a
      barcode or total marker, it is treated as a continuation of the previous
      item's name (common in narrow receipt layouts).

    Returns list of item dicts, or None if no columnar table found.
    """
    header_idx = None
    for i, line in enumerate(lines):
        if _COLUMNAR_HEADER.match(line.strip()):
            header_idx = i
            break

    if header_idx is None:
        return None

    items: list[dict[str, Any]] = []
    pending_barcode: str | None = None
    col4_count = 0
    col5_count = 0
    # Partial row text for multi-line item names: "1 Blue Green Wave Frozen"
    # (has qty but no trailing price columns — continued on next line)
    pending_text: str | None = None
    for i in range(header_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        if _is_total_marker(line):
            break

        # Check for standalone barcode line (EAN-8 / EAN-13)
        bc_match = _STANDALONE_BARCODE_PATTERN.match(line)
        if bc_match and _is_valid_ean(bc_match.group(1)):
            bc_digits = bc_match.group(1)
            if items:
                # Attach to the most recently parsed item
                items[-1]["barcode"] = bc_digits
            else:
                # No item yet — carry forward
                pending_barcode = bc_digits
            continue

        # If we have a pending partial, try combining with current line
        if pending_text is not None:
            combined = pending_text + " " + line
            m, is_4col = _match_columnar(combined)
            if m:
                pending_text = None
                # Fall through to item extraction below
            else:
                # Combined didn't match either — discard pending, process
                # current line normally
                pending_text = None
                m, is_4col = _match_columnar(line)
        else:
            m, is_4col = _match_columnar(line)

        if not m:
            # Non-matching, non-barcode line: could be a partial row (has
            # leading qty but no trailing price columns) or a name
            # continuation for the previous item.
            if _is_non_item_line(line):
                continue
            # Check if it looks like a partial row: starts with a number
            # (potential qty) followed by text
            if re.match(r"^\s*\d+(?:[.,]\d+)?\s+\S", line):
                pending_text = line
            elif items:
                items[-1]["name"] = items[-1]["name"] + " " + line
            continue

        qty = _parse_decimal(m.group(1))
        name = m.group(2).strip()
        unit_price = _parse_decimal(m.group(3))
        if is_4col:
            total_price = _parse_decimal(m.group(4))
            col4_count += 1
        else:
            # group(4) is discount, skip
            total_price = _parse_decimal(m.group(5))
            col5_count += 1

        if not name or total_price is None:
            continue

        item: dict[str, Any] = {
            "name": name,
            "quantity": qty if qty else 1,
            "unit_price": unit_price if unit_price else total_price,
            "total_price": total_price,
        }
        if pending_barcode:
            item["barcode"] = pending_barcode
            pending_barcode = None
        items.append(item)

    if items and meta is not None:
        meta["columnar_variant"] = (
            "columnar_4" if col4_count >= col5_count else "columnar_5"
        )
    return items if items else None


def _extract_item_first_columnar_items(
    lines: list[str],
) -> list[dict[str, Any]] | None:
    """Extract items from item-first two-line columnar format.

    Handles receipts like (retailedge.io, pharmacy POS):
        Item Qty Price Disc Sub Total
        Tranexamic Acid 500Mg 20 Mull Tablets
        3886 1 9.45 0.00 9.45
        Fagron Derma Pack Trt 100Ml 100Ml Liquid
        90747 1 39.00 0.00 39.00

    Item name on line N, barcode+qty+price+disc+subtotal on line N+1.
    Returns list of item dicts, or None if no matching header found.
    """
    header_idx = None
    for i, line in enumerate(lines):
        if _ITEM_FIRST_COLUMNAR_HEADER.match(line.strip()):
            header_idx = i
            break

    if header_idx is None:
        return None

    items: list[dict[str, Any]] = []
    pending_name: str | None = None
    for i in range(header_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        if _is_total_marker(line):
            break

        # Try matching as a data row (barcode qty price disc subtotal)
        m = _ITEM_FIRST_DATA_ROW.match(line)
        if m:
            qty = _parse_decimal(m.group(2))
            unit_price = _parse_decimal(m.group(3))
            total_price = _parse_decimal(m.group(5))
            name = pending_name or f"Item {m.group(1)}"

            if total_price is not None:
                item: dict[str, Any] = {
                    "name": name,
                    "quantity": qty if qty else 1,
                    "unit_price": unit_price if unit_price else total_price,
                    "total_price": total_price,
                }
                items.append(item)
            pending_name = None
        else:
            # Text-only line = item name for the next data row
            pending_name = line

    return items if items else None


def _extract_name_qty_amount_items(
    lines: list[str],
) -> list[dict[str, Any]] | None:
    """Extract items from Name-Qty-Amount columnar format.

    Handles receipts like:
        Name Qty Amount
        Zakvaska Bread 4.00
        Salmon cold smoked 0.339 27.12
        Mustard 1pcs 2 3.00
        Sour cream 4.50

    Column order is Name (with embedded optional qty) then Amount.
    Qty may be absent (single item), a decimal weight (0.339 kg),
    an integer count (2), or prefixed with a unit spec like "1pcs".

    Returns list of item dicts, or None if no matching header found.
    """
    header_idx = None
    for i, line in enumerate(lines):
        if _NAME_QTY_AMOUNT_HEADER.match(line.strip()):
            header_idx = i
            break

    if header_idx is None:
        return None

    items: list[dict[str, Any]] = []
    for i in range(header_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        if _is_total_marker(line):
            break

        # Must end with a price (dd.dd)
        row_m = _NQA_ROW.match(line)
        if not row_m:
            continue

        name_part = row_m.group(1).strip()
        total_price = _parse_decimal(row_m.group(2))
        if total_price is None or not name_part:
            continue

        # Attempt to peel off a trailing qty from the name part.
        # Priority 1: unit+qty pattern — "Mustard 1pcs 2" → qty=2
        unit_qty_m = _NQA_UNIT_QTY.match(name_part)
        if unit_qty_m:
            name = unit_qty_m.group(1).strip()
            qty_val = _parse_decimal(unit_qty_m.group(2))
            if name and qty_val is not None and len(name) >= 2:
                item: dict[str, Any] = {
                    "name": name,
                    "name_en": None,
                    "quantity": round(qty_val) if _is_integer_qty(qty_val) else qty_val,
                    "unit_raw": None,
                    "unit_quantity": None,
                    "unit_price": total_price / qty_val if qty_val else total_price,
                    "total_price": total_price,
                    "tax_rate": None,
                    "tax_type": None,
                    "discount": None,
                    "brand": None,
                    "barcode": None,
                    "category": None,
                }
                items.append(item)
                continue

        # Priority 2: trailing bare number in name part — weighed or integer qty
        trailing_m = _NQA_TRAILING_QTY.match(name_part)
        if trailing_m:
            candidate_name = trailing_m.group(1).strip()
            qty_val = _parse_decimal(trailing_m.group(2))
            # Only treat as qty when the candidate name has >= 2 chars and
            # the trailing number is plausibly a quantity:
            #   - decimal fraction (weighed item, e.g. 0.339)
            #   - integer <= 99 (count)
            if (
                candidate_name
                and len(candidate_name) >= 2
                and qty_val is not None
                and (
                    not _is_integer_qty(qty_val)  # decimal weight
                    or (qty_val <= 99)  # integer count
                )
            ):
                is_weight = not _is_integer_qty(qty_val)
                item = {
                    "name": candidate_name,
                    "name_en": None,
                    # Weighed item: quantity=1, unit_quantity=weight
                    "quantity": 1 if is_weight else round(qty_val),
                    "unit_raw": "kg" if is_weight else None,
                    "unit_quantity": qty_val if is_weight else None,
                    "unit_price": (
                        total_price / qty_val
                        if (not is_weight and qty_val)
                        else total_price
                    ),
                    "total_price": total_price,
                    "tax_rate": None,
                    "tax_type": None,
                    "discount": None,
                    "brand": None,
                    "barcode": None,
                    "category": None,
                }
                items.append(item)
                continue

        # No qty found — plain name + amount
        item = {
            "name": name_part,
            "name_en": None,
            "quantity": 1,
            "unit_raw": None,
            "unit_quantity": None,
            "unit_price": total_price,
            "total_price": total_price,
            "tax_rate": None,
            "tax_type": None,
            "discount": None,
            "brand": None,
            "barcode": None,
            "category": None,
        }
        items.append(item)

    return items if items else None


def _extract_line_items(
    lines: list[str],
    tax_map: dict[str, float],
    barcode_hint: str | None = None,
) -> list[dict[str, Any]]:
    """Extract line items from receipt text.

    Handles:
    - Markdown tables: | Qty | Description | Price | Disc. | Total |
    - Columnar tables: Qty Description Price Disc. Total
    - Name-Qty-Amount tables: Name Qty Amount (MIRADAR UNION style)
    - Standard: NAME  12.99 T1
    - Weighed: NAME  PRICE T1 / Qty W @ UP each (next line)
    - Multi-qty: NAME / Qty N @ UP each TOTAL T1 (same or next line)

    Args:
        barcode_hint: "before_item" or "after_item" to control barcode
            attachment direction.  When "before_item", standalone barcodes
            are always held as pending for the next item.
    """
    # Try markdown table format first
    md_items = _extract_markdown_table_items(lines)
    if md_items:
        return md_items

    # Try columnar table format
    col_items = _extract_columnar_table_items(lines)
    if col_items:
        return col_items

    # Try Name-Qty-Amount table format
    nqa_items = _extract_name_qty_amount_items(lines)
    if nqa_items:
        return nqa_items

    items: list[dict[str, Any]] = []
    i = 0
    # Barcode seen before the first item; applied to the next item created.
    pending_barcode: str | None = None
    # Track barcode-to-item position for template learning
    _bc_before_count = 0  # barcodes consumed via pending (before_item)
    _bc_after_count = 0  # barcodes attached to previous item (after_item)

    # Find the item section: skip header lines until first price line
    # and stop at total markers
    item_start = _find_item_start(lines)

    while i < len(lines):
        if i < item_start:
            i += 1
            continue

        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Skip lines that are clearly not items (must come BEFORE total
        # marker check — column headers like "Name Quantity Price Subtotal"
        # contain the word "subtotal" and would falsely break the loop)
        if _is_non_item_line(line):
            i += 1
            continue

        # Stop at total markers (but not item-level subtotals like
        # "Subtotal 3.00 eur" which are per-item in some receipt formats)
        if _is_total_marker(line) and not _ITEM_SUBTOTAL_PATTERN.match(line):
            break

        # Standalone EAN-8 / EAN-13 barcode line (bare digit string, check-digit
        # validated).  Direction depends on barcode_hint:
        #   "before_item" — barcode precedes its item (Alphamega-style)
        #   "after_item" / default — attach to the most recently parsed item
        standalone_bc = _STANDALONE_BARCODE_PATTERN.match(line)
        if standalone_bc and _is_valid_ean(standalone_bc.group(1)):
            bc_digits = standalone_bc.group(1)
            if barcode_hint == "before_item":
                # Always hold for next item (barcode-precedes-item pattern)
                pending_barcode = bc_digits
                _bc_before_count += 1
            elif items:
                if not items[-1].get("barcode"):
                    items[-1]["barcode"] = bc_digits
                    _bc_after_count += 1
            else:
                pending_barcode = bc_digits
                _bc_before_count += 1
            i += 1
            continue

        # Try multi-qty pattern first (Qty N @ P each TOTAL [TX])
        multi_match = _MULTI_QTY_PATTERN.search(line)
        if multi_match:
            # This is a quantity detail line for the previous item
            qty = _parse_decimal(multi_match.group(1))
            unit_price = _parse_decimal(multi_match.group(2))
            total_price = _parse_decimal(multi_match.group(3))
            tax_code = multi_match.group(4)

            if items:
                last = items[-1]
                if qty is not None:
                    last["quantity"] = qty
                if unit_price is not None:
                    last["unit_price"] = unit_price
                if total_price is not None:
                    last["total_price"] = total_price
                if tax_code and tax_code in tax_map:
                    last["tax_rate"] = tax_map[tax_code]
            i += 1
            continue

        # Try weighed/multi-qty pattern (Qty W @ P each, no trailing total)
        weighed_match = _WEIGHED_PATTERN.search(line)
        if not weighed_match:
            weighed_match = _WEIGHED_NO_QTY_PATTERN.search(line)
        if weighed_match:
            qty_val = _parse_decimal(weighed_match.group(1))
            unit_price = _parse_decimal(weighed_match.group(2))

            if items and qty_val is not None:
                last = items[-1]
                if _is_integer_qty(qty_val):
                    # Multi-quantity (e.g. Qty 3.000 @ 1.99 each)
                    last["quantity"] = round(qty_val)
                    if unit_price is not None:
                        last["unit_price"] = unit_price
                else:
                    # Weighed item (e.g. Qty 1.341 @ 2.99 each)
                    last["quantity"] = 1
                    last["unit_quantity"] = qty_val
                    last["unit_raw"] = "kg"
                    if unit_price is not None:
                        last["unit_price"] = unit_price
            i += 1
            continue

        # Check if this is a name-only line followed by Qty + total or
        # a standalone price (narrow receipts wrap name/price across lines)
        price_match = _PRICE_PATTERN.search(line)

        if not price_match and i + 1 < len(lines) and not _is_non_item_line(line):
            # Find next non-empty line (skip blank lines between item name
            # and price/subtotal — common in spaced-out receipt formats)
            next_j = i + 1
            while next_j < len(lines) and not lines[next_j].strip():
                next_j += 1
            next_line = lines[next_j].strip() if next_j < len(lines) else ""
            multi_next = _MULTI_QTY_PATTERN.search(next_line)
            if multi_next and line:
                qty = _parse_decimal(multi_next.group(1))
                up = _parse_decimal(multi_next.group(2))
                tp = _parse_decimal(multi_next.group(3))
                tc = multi_next.group(4)
                # Check previous line for barcode prefix
                noi_barcode = None
                if i > 0:
                    prev_line = lines[i - 1].strip()
                    prev_bc = _BARCODE_PREFIX_PATTERN.match(prev_line)
                    if prev_bc:
                        noi_barcode = prev_bc.group(1)
                name_only_item: dict[str, Any] = {
                    "name": line,
                    "name_en": None,
                    "quantity": round(qty) if qty and _is_integer_qty(qty) else 1,
                    "unit_raw": None,
                    "unit_quantity": qty if qty and not _is_integer_qty(qty) else None,
                    "unit_price": up,
                    "total_price": tp,
                    "tax_rate": tax_map.get(tc) if tc else None,
                    "tax_type": None,
                    "discount": None,
                    "brand": None,
                    "barcode": noi_barcode or pending_barcode,
                    "category": None,
                }
                pending_barcode = None
                items.append(name_only_item)
                i = next_j + 1  # Skip name + blanks + price line
                continue

            # Item-level subtotal on next line: "Subtotal 3.00 eur"
            # (LITTLE SINS format — each item followed by its own subtotal)
            item_sub = _ITEM_SUBTOTAL_PATTERN.match(next_line)
            if item_sub and line:
                sub_total = _parse_decimal(item_sub.group(1))
                subtotal_item: dict[str, Any] = {
                    "name": line,
                    "name_en": None,
                    "quantity": 1,
                    "unit_raw": None,
                    "unit_quantity": None,
                    "unit_price": sub_total,
                    "total_price": sub_total,
                    "tax_rate": None,
                    "tax_type": None,
                    "discount": None,
                    "brand": None,
                    "barcode": pending_barcode,
                    "category": None,
                }
                pending_barcode = None
                items.append(subtotal_item)
                i = next_j + 1  # Skip name + blanks + subtotal line
                continue

            # Standalone price on next line (narrow receipt line-wrap)
            standalone_price = _PRICE_PATTERN.search(next_line)
            if (
                standalone_price
                and line
                and not _is_total_marker(next_line)
                and not _is_non_item_line(next_line)
                # Parenthetical-only lines are product annotations, not item
                # names — let them fall through to the continuation logic.
                and not (line.startswith("(") and line.endswith(")"))
            ):
                # Barcode-prefix: if current line is "Barcode: XXXX",
                # use next line's name (before price) instead.
                # Also handle 3-line pattern where barcode is on lines[i-1]:
                #   Barcode: XXXX
                #   PRODUCT NAME       <- current line (no price)
                #   qty price          <- next line (standalone price)
                bc_pfx = _BARCODE_PREFIX_PATTERN.match(line)
                if bc_pfx:
                    sp_name = next_line[: standalone_price.start()].strip()
                    sp_barcode = bc_pfx.group(1)
                else:
                    sp_name = line
                    sp_barcode = None
                    # Look one line back for barcode prefix
                    if i > 0:
                        prev_line = lines[i - 1].strip()
                        prev_bc = _BARCODE_PREFIX_PATTERN.match(prev_line)
                        if prev_bc:
                            sp_barcode = prev_bc.group(1)
                sp_str = standalone_price.group(1)
                sp_tax = standalone_price.group(2)
                sp_total = _parse_decimal(sp_str)
                standalone_item: dict[str, Any] = {
                    "name": sp_name or line,
                    "name_en": None,
                    "quantity": 1,
                    "unit_raw": None,
                    "unit_quantity": None,
                    "unit_price": sp_total,
                    "total_price": sp_total,
                    "tax_rate": tax_map.get(sp_tax) if sp_tax else None,
                    "tax_type": None,
                    "discount": None,
                    "brand": None,
                    "barcode": sp_barcode or pending_barcode,
                    "category": None,
                }
                pending_barcode = None
                items.append(standalone_item)
                i = next_j + 1  # Skip name + blanks + price line
                continue

            # 3-line name: name1 / name2 / price
            # When both current and next line have no price but the line
            # after next IS a price, treat name1 + name2 as the full name.
            # Skip when current line is a barcode prefix — that's handled
            # separately by the continuation/barcode-prefix logic.
            if (
                line
                and not _is_non_item_line(line)
                and not _BARCODE_PREFIX_PATTERN.match(line)
                and i + 2 < len(lines)
            ):
                part2 = lines[i + 1].strip()
                part3 = lines[i + 2].strip()
                part3_price = _PRICE_PATTERN.search(part3)
                if (
                    part2
                    and not _PRICE_PATTERN.search(part2)
                    and not _MULTI_QTY_PATTERN.search(part2)
                    and not _is_non_item_line(part2)
                    and not _is_total_marker(part2)
                    and part3_price
                    and not _is_total_marker(part3)
                    and not _is_non_item_line(part3)
                ):
                    tl_name = line + " " + part2
                    tl_str = part3_price.group(1)
                    tl_tax = part3_price.group(2)
                    tl_total = _parse_decimal(tl_str)
                    three_line_item: dict[str, Any] = {
                        "name": tl_name,
                        "name_en": None,
                        "quantity": 1,
                        "unit_raw": None,
                        "unit_quantity": None,
                        "unit_price": tl_total,
                        "total_price": tl_total,
                        "tax_rate": tax_map.get(tl_tax) if tl_tax else None,
                        "tax_type": None,
                        "discount": None,
                        "brand": None,
                        "barcode": pending_barcode,
                        "category": None,
                    }
                    pending_barcode = None
                    items.append(three_line_item)
                    i += 3  # Skip all three lines
                    continue

        # Try standard price line: NAME  PRICE [TX]
        if price_match:
            price_str = price_match.group(1)
            tax_code = price_match.group(2)
            name = line[: price_match.start()].strip()

            if not name:
                i += 1
                continue

            # Barcode-prefix: check if previous line was "Barcode: XXXX"
            # (Alphamega and similar POS formats print barcode on a
            # separate line before the product name + price).
            prev_barcode = None
            if i > 0:
                prev_line = lines[i - 1].strip()
                bc_match = _BARCODE_PREFIX_PATTERN.match(prev_line)
                if bc_match:
                    prev_barcode = bc_match.group(1)

            total_price = _parse_decimal(price_str)
            item: dict[str, Any] = {
                "name": name,
                "name_en": None,
                "quantity": 1,
                "unit_raw": None,
                "unit_quantity": None,
                "unit_price": total_price,
                "total_price": total_price,
                "tax_rate": None,
                "tax_type": None,
                "discount": None,
                "brand": None,
                "barcode": prev_barcode or pending_barcode,
                "category": None,
            }
            pending_barcode = None

            if tax_code and tax_code in tax_map:
                item["tax_rate"] = tax_map[tax_code]

            # Check if next line has Qty info
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                weighed_next = _WEIGHED_PATTERN.search(next_line) or (
                    _WEIGHED_NO_QTY_PATTERN.search(next_line)
                )
                multi_next = _MULTI_QTY_PATTERN.search(next_line)
                if multi_next:
                    qty = _parse_decimal(multi_next.group(1))
                    up = _parse_decimal(multi_next.group(2))
                    tp = _parse_decimal(multi_next.group(3))
                    tc = multi_next.group(4)
                    if qty is not None:
                        item["quantity"] = qty
                    if up is not None:
                        item["unit_price"] = up
                    if tp is not None:
                        item["total_price"] = tp
                    if tc and tc in tax_map:
                        item["tax_rate"] = tax_map[tc]
                    i += 1  # Skip the qty line
                elif weighed_next:
                    qty_val = _parse_decimal(weighed_next.group(1))
                    up = _parse_decimal(weighed_next.group(2))
                    if qty_val is not None:
                        if _is_integer_qty(qty_val):
                            # Multi-quantity
                            item["quantity"] = round(qty_val)
                        else:
                            # Weighed item
                            item["quantity"] = 1
                            item["unit_quantity"] = qty_val
                            item["unit_raw"] = "kg"
                    if up is not None:
                        item["unit_price"] = up
                    i += 1  # Skip the qty line

            items.append(item)
            i += 1
            continue

        # Continuation line: no price, no qty pattern, not a total marker.
        # Append to previous item's name (narrow receipts wrap descriptions).
        # Skip barcode-only lines — they prefix the next item, not continue
        # the previous one.
        if (
            items
            and line
            and not _is_total_marker(line)
            and not _BARCODE_PREFIX_PATTERN.match(line)
        ):
            items[-1]["name"] = items[-1]["name"] + " " + line

        i += 1

    _strip_leading_quantities(items)
    _extract_inline_weights(items)
    _strip_product_notes(items)
    return items


# Pattern: "1 Double Espresso" or "4 Red Bull" — digit(s) + space + uppercase letter
_LEADING_QTY_RE = re.compile(r"^(\d{1,2})\s+([A-ZΑ-ΩА-Я].*)$")


def _strip_leading_quantities(items: list[dict[str, Any]]) -> None:
    """Strip leading quantity prefix from item names like '1 Double Espresso'.

    Only activates when quantity == 1 (default) and name starts with
    a small integer followed by a space and an uppercase letter.
    Skips names that look like product codes ("1 kg", "100ml").
    """
    for item in items:
        name = str(item.get("name") or "")
        qty = item.get("quantity")
        if qty not in (1, "1", None):
            continue  # quantity already set from other source

        m = _LEADING_QTY_RE.match(name)
        if not m:
            continue

        candidate_qty = int(m.group(1))
        rest = m.group(2)

        # Skip if rest starts with a unit (e.g., "1 kg rice")
        if re.match(r"(?:kg|g|ml|l|pcs|pc|x)\b", rest, re.IGNORECASE):
            continue

        item["name"] = rest
        if candidate_qty > 1:
            item["quantity"] = candidate_qty
            # Adjust unit_price if total_price is set
            tp = item.get("total_price")
            if tp is not None:
                try:
                    item["unit_price"] = round(float(tp) / candidate_qty, 2)
                except (TypeError, ZeroDivisionError):
                    pass


def _strip_product_notes(items: list[dict[str, Any]]) -> None:
    """Extract parenthetical product annotations from the end of item names.

    Continuation lines like "(takeaway)" or "(250g, Ethiopia)" get fused into
    item names by the main loop.  This post-processing pass detects those
    trailing parentheticals and promotes them to a separate ``product_note``
    field, keeping item names clean for matching and display.

    Only end-of-string parentheticals are stripped.  An embedded parenthetical
    in the middle of the name — e.g. "Coffee (Fair Trade) Beans" — is left
    untouched because it is part of the product brand or variant designation.
    """
    for item in items:
        name = item.get("name", "")
        if not name:
            continue
        m = _PRODUCT_NOTE_PATTERN.search(name)
        if m:
            item["name"] = name[: m.start()].strip()
            item["product_note"] = m.group(1)


# Inline weight in item name: "PEPPERS 1.535 kg" or "Blueberries 500g" or "Salmon 200 g"
_INLINE_WEIGHT_RE = re.compile(
    r"^(.+?)\s+(\d+(?:[.,]\d+)?)\s*(kg|g|gr|ml|l|lt|cl)\b(.*)$",
    re.IGNORECASE,
)


def _extract_inline_weights(items: list[dict[str, Any]]) -> None:
    """Extract weight/package size embedded in item names.

    Handles:
    - "PEPPERS 1.535 kg" → name="PEPPERS", unit_quantity=1.535, unit_raw=kg
    - "Blueberries 500g" → name="Blueberries", unit_quantity=500, unit_raw=g
    - "Salmon 200 g" → name="Salmon", unit_quantity=200, unit_raw=g

    Only activates when unit_quantity/unit_raw are not set AND quantity is 1
    (default). Items with explicit qty from Qty lines are left to the atom
    parser's _extract_unit_from_name() which handles the full context.
    """
    for item in items:
        if item.get("unit_quantity") or item.get("unit_raw"):
            continue  # already has weight info
        # Only for default-quantity items; multi-qty items get downstream handling
        qty = item.get("quantity")
        if qty not in (1, "1", None):
            continue

        name = str(item.get("name") or "")
        m = _INLINE_WEIGHT_RE.match(name)
        if not m:
            continue

        clean_name = m.group(1).strip()
        qty_str = m.group(2).replace(",", ".")
        unit = m.group(3)
        trailing = m.group(4).strip()

        # If trailing text looks like a price or unit_price, skip
        if trailing and re.match(r"[\d.,@/]", trailing):
            continue

        # Ensure name isn't empty after stripping
        if not clean_name:
            continue

        try:
            weight_val = float(qty_str)
        except ValueError:
            continue

        item["name"] = clean_name
        item["unit_quantity"] = weight_val
        item["unit_raw"] = unit.lower()


def _find_item_start(lines: list[str]) -> int:
    """Find the index where line items begin."""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # First line with a price-like pattern
        if _PRICE_PATTERN.search(stripped) and not _is_total_marker(stripped):
            return i
        # Barcode prefix line also marks item section start
        if _BARCODE_PREFIX_PATTERN.match(stripped):
            return i
        # Standalone EAN also marks item section start
        sb = _STANDALONE_BARCODE_PATTERN.match(stripped)
        if sb and _is_valid_ean(sb.group(1)):
            return i
    return 0


def _find_footer_start(lines: list[str], body_start: int) -> int:
    """Find the index where the footer/totals section begins.

    Scans from body_start forward looking for the first total marker line.
    Returns len(lines) if no footer boundary found.
    """
    for i in range(body_start, len(lines)):
        if _is_total_marker(lines[i]):
            return i
    return len(lines)


def _split_regions(lines: list[str]) -> TextRegions:
    """Materialize header/body/footer as named text regions.

    Handles mixed-content documents where header and footer may belong
    to different document types (e.g., payment confirmation header with
    receipt line items in body).
    """
    body_start = _find_item_start(lines)
    footer_start = _find_footer_start(lines, body_start)

    # When _find_item_start returns 0, there's no clear body boundary.
    # Treat the entire text as header (no body/footer distinction).
    if body_start == 0 and not any(
        _PRICE_PATTERN.search(ln.strip())
        for ln in lines[:1]
        if ln.strip() and not _is_total_marker(ln.strip())
    ):
        return TextRegions(
            header="\n".join(lines),
            body="",
            footer="",
            header_end=len(lines),
            footer_start=len(lines),
        )

    header_text = "\n".join(lines[:body_start])
    body_text = "\n".join(lines[body_start:footer_start])
    footer_text = "\n".join(lines[footer_start:])

    return TextRegions(
        header=header_text,
        body=body_text,
        footer=footer_text,
        header_end=body_start,
        footer_start=footer_start,
    )


def _is_total_marker(line: str) -> bool:
    """Check if a line is a total/subtotal marker."""
    low = line.strip().lower()
    for marker in _TOTAL_MARKERS:
        if marker in low:
            return True
    return False


def _is_non_item_line(line: str) -> bool:
    """Check if a line is clearly not a line item."""
    low = line.strip().lower()

    # Payment / card info lines
    if any(
        kw in low
        for kw in [
            "visa",
            "mastercard",
            "auth no",
            "terminal",
            "cardholder",
            "approved",
            "contactless",
            "chip",
            "change due",
            "cash",
            "tendered",
            "amount eur",
            "amount usd",
            "amount gbp",
            "amount rub",
            "visa card",
            "change ",
            # DE
            "ec-karte",
            "kartennummer",
            "wechselgeld",
            "pfand",
            # RU
            "банковская карта",
            "карта:",
            "сдача",
        ]
    ):
        return True

    # Lines that are just dashes or equals
    if re.match(r"^[-=*_]{3,}$", line.strip()):
        return True

    # VAT summary lines (EN + DE + RU)
    if re.match(r"taxable\s+\d", low):
        return True
    if re.match(r"mwst\.?\s+[a-e]", low):
        return True
    if re.match(r"ндс\s+\d", low):
        return True
    # RU: "в т.ч. НДС" = "including VAT" — not a line item
    if "в т.ч." in low:
        return True
    # Garbled VAT summary headers from OCR (e.g., "VAT RatVAT Amount Amount")
    if re.match(r"vat\s*rat", low):
        return True
    # VAT summary table rows: "A 19% 12.34 77.34" or "VAT% Net VAT"
    if re.match(r"vat\s*%", low):
        return True
    # "VAT1 19.00% 0.02" or "VAT 5% 1.23" — numbered VAT breakdown lines
    if re.match(r"vat\d?\s+\d+[.,]?\d*\s*%", low):
        return True
    # Numeric VAT summary rows without "VAT" keyword:
    # "9.00% 0.66 8.00" or "19% 1.23" (rate% amount [total])
    if re.match(r"\d+(?:\.\d+)?\s*%\s+\d+[.,]\d{2}", low):
        return True
    # "Incl. VAT" / "Including VAT" / "Excl. VAT" summary lines
    if re.match(r"(?:incl|excl|including|excluding)\.?\s+vat", low):
        return True

    # Column headers (EN + DE + RU + EL)
    if low in (
        "item price",
        "item  price",
        "items price",
        "artikel preis",
        "наименование цена",
    ):
        return True

    # Page separators from multi-page OCR text
    if re.match(r"^---\s*page\s+\d+\s*---$", low):
        return True

    # Receipt ticket/transaction metadata
    if re.match(r"ticket\s*:", low):
        return True

    # Document metadata fields (invoice/receipt headers)
    if any(
        kw in low
        for kw in [
            "doc. date",
            "doc. number",
            "doc date",
            "doc number",
            "acc. no",
            "acc no",
            "store /",
            "driver/",
            "vehicle/",
            "route /",
        ]
    ):
        return True

    # Standalone column headers from garbled multi-page table layouts.
    # Only match short lines (< 40 chars) to avoid filtering real items.
    if len(low) < 40:
        _column_headers = (
            "description",
            "quantity",
            "amount",
            "price",
            "unit price",
            "item code",
            "vat code",
            "tax code",
            "comments",
            # DE
            "beschreibung",
            "menge",
            "betrag",
            "preis",
            "einzelpreis",
            # EL (common OCR variants)
            "περιγραφη",
            "ποσοτητα",
            "τιμη",
            "κωδικοσ",
            "σχολια",
        )
        if low in _column_headers:
            return True

    # Multi-word column header combos: "QTY DESCRIPTION PRICE AMOUNT VAT"
    _header_words = {
        "qty",
        "quantity",
        "description",
        "price",
        "amount",
        "unit price",
        "unit",
        "vat",
        "tax",
        "total",
        "subtotal",
        "item",
        "name",
        "code",
        "no",
        "rate",
        "disc.",
        "disc",
        "discount",
    }
    words = low.split()
    if len(words) >= 2 and all(w in _header_words for w in words):
        return True

    # Greek VAT: handle OCR mixing Greek Α (U+0391) / Latin A (U+0041) characters
    # Matches φπα, φπa, фπα etc. with either script variant for each character
    if re.search(r"[\u03c6f][\u03c0p][a\u03b1]", low):
        return True

    # Section headers that appear mid-document (not totals but not items)
    if any(
        kw in low
        for kw in [
            "vat analysis",
            "tax summary",
            "steuerübersicht",
            "ндс итого",
        ]
    ):
        return True

    return False


def _extract_totals(
    lines: list[str], data: dict[str, Any], *, hints: ParserHints | None = None
) -> None:
    """Extract total, subtotal, tax, discount amounts."""
    # Try hinted total marker first
    if hints and hints.total_marker and not data.get("total"):
        hint_low = hints.total_marker.lower()
        for line in lines:
            if hint_low in line.strip().lower():
                amount = _extract_amount_from_line(line.strip())
                if amount is not None:
                    data["total"] = amount
                    data["_total_marker"] = hints.total_marker
                    break

    for line in lines:
        stripped = line.strip()
        low = stripped.lower()

        # Look for "Total Before Disc" or "Subtotal" (multilingual)
        if any(
            kw in low
            for kw in [
                "total before disc",
                "subtotal",
                "sub total",
                "zwischensumme",
                "μερικο συνολο",
                "промежуточный итог",
            ]
        ):
            amount = _extract_amount_from_line(stripped)
            if amount is not None:
                data["subtotal"] = amount
            continue

        # Discount
        for marker in _DISCOUNT_MARKERS:
            if marker in low:
                amount = _extract_amount_from_line(stripped)
                if amount is not None:
                    data["discount"] = amount
                break

        # Tax / VAT amount (skip summary lines)
        if not re.match(r"taxable\s+\d", low) and not re.match(r"mwst\.?\s+[a-e]", low):
            for pat in _TAX_MARKERS:
                if pat.search(stripped):
                    amount = _extract_amount_from_line(stripped)
                    if amount is not None:
                        data["tax"] = amount
                    break

        # Grand total — EN + DE + EL + RU
        if (
            (re.match(r"^total\b", low) and "before" not in low and "vat" not in low)
            or re.match(r"^gesamt\b", low)
            or re.match(r"^gesamtbetrag\b", low)
        ):
            amount = _extract_amount_from_line(stripped)
            if amount is not None:
                data["total"] = amount
                data["_total_marker"] = stripped.split()[0]
        elif re.match(r"^(?:всего к оплате|всего|итого)\b", low):
            amount = _extract_amount_from_line(stripped)
            if amount is not None:
                data["total"] = amount
                data["_total_marker"] = stripped.split()[0]
        elif re.match(r"^(?:συνολο|συνοδο|γενικο συνολο)\b", low):
            amount = _extract_amount_from_line(stripped)
            if amount is not None:
                data["total"] = amount
                data["_total_marker"] = stripped.split()[0]

    # Fallback: JCC payment slip "AMOUNT EUR52,31" when no total found
    if not data.get("total"):
        for line in lines:
            m = re.match(r"^AMOUNT\s+EUR\s*(\d+[.,]\d{2})", line.strip(), re.IGNORECASE)
            if m:
                data["total"] = _parse_decimal(m.group(1))
                break

    # Extract declared item count: "Number of Items 25", "Number of Items: 13"
    # Also: "Total 2 48.45 0.00 48.45" (retailedge POS)
    for line in lines:
        stripped = line.strip()
        m = re.match(
            r"^(?:number\s+of\s+items|no\.?\s+of\s+items|"
            r"anzahl\s+artikel|αριθμ[οό]ς\s+ειδ[ωώ]ν|"
            r"количество\s+товаров)\s*:?\s*(\d+)\s*$",
            stripped,
            re.IGNORECASE,
        )
        if m:
            data["declared_item_count"] = int(m.group(1))
            break


def _extract_amount_from_line(line: str) -> float | None:
    """Extract the last decimal number from a line."""
    # Find all decimal numbers in the line
    amounts = re.findall(r"-?\d+[.,]\d{2}", line)
    if amounts:
        return _parse_decimal(amounts[-1])
    return None


# ---------------------------------------------------------------------------
# Invoice parser
# ---------------------------------------------------------------------------


def _parse_invoice(raw_text: str, hints: ParserHints | None = None) -> ParseResult:
    """Full invoice parsing with invoice-specific fields.

    Maps to INVOICE_PROMPT_V2 schema: issuer (not vendor), issue_date,
    due_date, invoice_number, customer, amount (not total), payment_terms.
    """
    lines = raw_text.split("\n")
    data: dict[str, Any] = {"document_type": "invoice"}
    fc: dict[str, float] = {}
    old_gaps: list[str] = []

    # Pass 1: Header — issuer + customer block
    _extract_invoice_header(lines, data, old_gaps, hints=hints)
    # Apply hint vendor as issuer fallback if header extraction missed it
    if not data.get("issuer") and hints and hints.vendor_name:
        data["issuer"] = hints.vendor_name
    fc["issuer"] = 1.0 if data.get("issuer") else 0.0

    # Pass 2: Date → issue_date
    _extract_date_time(raw_text, data, old_gaps, hints=hints)
    if "date" in data:
        data["issue_date"] = data.pop("date")
    data.pop("time", None)  # Not relevant for invoices
    fc["issue_date"] = 1.0 if data.get("issue_date") else 0.0

    # Pass 3: Invoice number
    _extract_invoice_number(raw_text, data, old_gaps)
    fc["invoice_number"] = 1.0 if data.get("invoice_number") else 0.0

    # Pass 4: Due date
    _extract_due_date(raw_text, data)
    fc["due_date"] = 0.8 if data.get("due_date") else 0.0

    # Pass 5: Payment terms
    _extract_payment_terms(raw_text, data)

    # Pass 6: PO number
    _extract_po_number(raw_text, data)

    # Pass 7: Currency
    _extract_currency(raw_text, data)
    if not data.get("currency") and hints and hints.currency:
        data["currency"] = hints.currency
    fc["currency"] = 0.8 if data.get("currency") else 0.0

    # Pass 8: Tax summary
    tax_map = _extract_tax_summary(raw_text)

    # Pass 9: Line items (reuse receipt extractor)
    items = _extract_line_items(lines, tax_map)
    data["line_items"] = items
    item_count = len(items)
    fc["line_items"] = min(1.0, item_count / 3) if item_count > 0 else 0.0

    # Pass 10: Totals → rename to invoice schema
    _extract_totals(lines, data, hints=hints)
    if "total" in data:
        data["amount"] = data.pop("total")
    fc["amount"] = 1.0 if data.get("amount") else 0.2

    # Fields parser can't fill
    fc["name_en"] = 0.0
    fc["category"] = 0.0
    fc["language"] = 0.0

    # Region splitting
    regions = _split_regions(lines)

    # Tag region boundary metadata for template learning
    if regions.header:
        data["_header_lines"] = len(regions.header.strip().splitlines())
    total_lines = len(lines)
    if regions.footer_start and total_lines > 0:
        data["_footer_ratio"] = round(regions.footer_start / total_lines, 2)

    # Compute confidence from core fields
    core_fields = (
        "issuer",
        "issue_date",
        "invoice_number",
        "currency",
        "line_items",
        "amount",
    )
    core_signals = [fc[k] for k in core_fields if k in fc]
    confidence = sum(core_signals) / len(core_signals) if core_signals else 0.0

    return ParseResult(
        data=data,
        confidence=round(confidence, 3),
        field_confidence=fc,
        line_item_count=item_count,
        needs_llm=True,
        regions=regions,
    )


def _extract_invoice_header(
    lines: list[str],
    data: dict[str, Any],
    gaps: list[str],
    *,
    hints: ParserHints | None = None,
) -> None:
    """Extract issuer and customer blocks from invoice header.

    Reuses _extract_header for the issuer block, then renames
    vendor→issuer keys to match INVOICE_PROMPT_V2 schema.
    Scans for BILL TO / CUSTOMER block separately.
    """
    _extract_header(lines, data, gaps, hints=hints)

    # Rename vendor → issuer keys
    if "vendor" in data:
        data["issuer"] = data.pop("vendor")
    if "vendor_address" in data:
        data["issuer_address"] = data.pop("vendor_address")
    if "vendor_phone" in data:
        data["issuer_phone"] = data.pop("vendor_phone")
    if "vendor_vat" in data:
        data["issuer_vat"] = data.pop("vendor_vat")
    if "vendor_tax_id" in data:
        data["issuer_tax_id"] = data.pop("vendor_tax_id")
    if "vendor" in gaps:
        gaps.remove("vendor")
        gaps.append("issuer")

    # Scan for BILL TO / CUSTOMER block
    in_customer_block = False
    customer_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if _CUSTOMER_LABEL_PATTERN.match(stripped):
            in_customer_block = True
            after_label = _CUSTOMER_LABEL_PATTERN.sub("", stripped).strip()
            if after_label:
                customer_lines.append(after_label)
            continue
        if in_customer_block:
            if not stripped or _is_noise_line(stripped):
                break
            if re.search(r"\d{1,2}[/.]\d{1,2}[/.]\d{4}", stripped):
                break
            customer_lines.append(stripped)
            if len(customer_lines) >= 3:
                break

    if customer_lines:
        data["customer"] = customer_lines[0]
        if len(customer_lines) > 1:
            data["billing_address"] = ", ".join(customer_lines[1:])


def _extract_invoice_number(
    raw_text: str, data: dict[str, Any], gaps: list[str]
) -> None:
    """Extract invoice number/ID from OCR text."""
    for pat in _INVOICE_NUMBER_PATTERNS:
        m = pat.search(raw_text)
        if m:
            data["invoice_number"] = m.group(1).strip()
            return
    gaps.append("invoice_number")


def _extract_due_date(raw_text: str, data: dict[str, Any]) -> None:
    """Extract due date from invoice text."""
    for pat in _DUE_DATE_PATTERNS:
        m = pat.search(raw_text)
        if m:
            date_str = m.group(1)
            parsed = _parse_date(date_str)
            if parsed:
                data["due_date"] = parsed.isoformat()
            return


def _extract_payment_terms(raw_text: str, data: dict[str, Any]) -> None:
    """Extract payment terms (Net 30, Due on receipt, etc.)."""
    for pat in _PAYMENT_TERMS_PATTERNS:
        m = pat.search(raw_text)
        if m:
            terms = m.group(1).strip().rstrip(".,")
            data["payment_terms"] = terms
            return


def _extract_po_number(raw_text: str, data: dict[str, Any]) -> None:
    """Extract PO/Purchase Order number."""
    for pat in _PO_NUMBER_PATTERNS:
        m = pat.search(raw_text)
        if m:
            data["po_number"] = m.group(1).strip()
            return


# ---------------------------------------------------------------------------
# Payment confirmation parser
# ---------------------------------------------------------------------------


def _parse_payment_confirmation(raw_text: str) -> ParseResult:
    """Parse payment confirmation / card slip."""
    lines = raw_text.split("\n")
    data: dict[str, Any] = {"document_type": "payment_confirmation", "line_items": []}
    fc: dict[str, float] = {}
    old_gaps: list[str] = []

    # Vendor — skip JCC header noise
    _extract_header(lines, data, old_gaps)

    # Bank transaction confirmations: vendor from payee/beneficiary field.
    # _extract_header() may pick a document title (e.g. "PAYMENT ORDER",
    # "ΑΙΤΗΣΗ ΜΕΤΑΦΟΡΑΣ ΚΕΦΑΛΑΙΩΝ") or a bank-metadata line as the vendor —
    # discard those and look for the actual payee.
    vendor = data.get("vendor", "")
    vendor_lower = vendor.lower().strip()
    _vendor_is_bank_metadata = (
        any(vendor_lower.startswith(p) for p in _BANK_META_PREFIXES)
        or bool(re.match(r"^[A-Z]{2}\d{2}", vendor))  # bare IBAN number
        or bool(re.match(r"^[A-Za-zА-Яа-яΑ-Ωα-ω][A-Za-zА-Яа-яΑ-Ωα-ω\s]+:\s+\S", vendor))
    )
    if (
        not vendor
        or _vendor_is_bank_metadata
        or any(vendor_lower.startswith(p) for p in _BANK_DOC_TITLE_PREFIXES)
    ):
        # 1. Try "Description:" label (legacy / EN banking)
        for line in lines:
            m = re.match(r"Description\s*:\s*(.+)", line.strip(), re.IGNORECASE)
            if m:
                data["vendor"] = m.group(1).strip()
                break
        else:
            # 2. Try Beneficiary / Payee / Δικαιούχος / Получатель / Empfänger
            for line in lines:
                m = _BENEFICIARY_PATTERN.match(line.strip())
                if m:
                    data["vendor"] = m.group(1).strip()
                    break

    # Payment processor swap: if vendor is a known processor but
    # vendor_legal_name contains the actual merchant, swap them.
    _swap_processor_vendor(data)

    fc["vendor"] = 1.0 if data.get("vendor") else 0.0

    # Date/Time
    _extract_date_time(raw_text, data, old_gaps)
    fc["date"] = 1.0 if data.get("date") else 0.0

    # Payment details
    _extract_payment(raw_text, data)
    fc["payment_method"] = (
        1.0 if (data.get("card_last4") or data.get("payment_method")) else 0.3
    )

    # Currency
    _extract_currency(raw_text, data)
    fc["currency"] = 0.8 if data.get("currency") else 0.0

    # Amount — prefer "Total Amount" over bare "Amount" (handles tips)
    # First pass: look for "total amount" or "total" lines
    for line in lines:
        stripped = line.strip()
        low = stripped.lower()
        if "total amount" in low or (low.startswith("total") and "tip" not in low):
            amount = _extract_amount_from_line(stripped)
            if amount is not None:
                data["total"] = amount
                break
    # Fallback: bare "amount" if no total found
    if not data.get("total"):
        for line in lines:
            stripped = line.strip()
            low = stripped.lower()
            if "amount" in low and "tip" not in low:
                amount = _extract_amount_from_line(stripped)
                if amount is not None:
                    data["total"] = amount
                    break

    fc["total"] = 1.0 if data.get("total") else 0.0
    fc["language"] = 0.0

    # Region splitting
    regions = _split_regions(lines)

    # Compute confidence from core fields
    core_fields = ("vendor", "date", "payment_method", "total")
    core_signals = [fc[k] for k in core_fields if k in fc]
    confidence = sum(core_signals) / len(core_signals) if core_signals else 0.0

    return ParseResult(
        data=data,
        confidence=round(confidence, 3),
        field_confidence=fc,
        line_item_count=0,
        needs_llm=True,
        regions=regions,
    )


# ---------------------------------------------------------------------------
# Statement parser
# ---------------------------------------------------------------------------

# Statement-specific patterns
_ACCOUNT_NO_PATTERNS = [
    re.compile(r"ACCOUNT\s*(?:NO\.?|NUMBER)\s*:?\s*([\d\-/*]+)", re.IGNORECASE),
    re.compile(r"(?:Р/?СЧЕТ|СЧЕТ)\s*(?:№|NO\.?)?\s*:?\s*([\d\-/*]+)", re.IGNORECASE),
    re.compile(r"KONTO(?:NR\.?)?\s*:?\s*([\d\-/*]+)", re.IGNORECASE),
]
_IBAN_PATTERN = re.compile(
    r"\b([A-Z]{2}\d{2}\s*(?:\d{4}\s*){4,8}\d{0,4})\b", re.IGNORECASE
)
_BIC_PATTERN = re.compile(r"\bBIC\s*:?\s*([A-Z]{6}[A-Z0-9]{2,5})\b", re.IGNORECASE)
_PERIOD_PATTERN = re.compile(
    r"(?:PERIOD|ΠΕΡΙΟΔΟΣ|ZEITRAUM|ПЕРИОД)\s*:?\s*"
    r"(\d{1,2}[/.]\d{1,2}[/.]\d{4})\s*[-–]\s*(\d{1,2}[/.]\d{1,2}[/.]\d{4})",
    re.IGNORECASE,
)
# Statement transaction line: DATE DESCRIPTION DEBIT CREDIT [VALUE_DATE]
_STMT_TXN_PATTERN = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{4})\s+"  # date
    r"(.+?)\s+"  # description
    r"(-?[\d.,]+)\s*$",  # amount (last number on line)
)


def _parse_date_dmy(date_str: str) -> str | None:
    """Parse DD/MM/YYYY or DD.MM.YYYY to YYYY-MM-DD."""
    m = re.match(r"(\d{1,2})[/.](\d{1,2})[/.](\d{4})", date_str)
    if m:
        return f"{m.group(3)}-{m.group(2).zfill(2)}-{m.group(1).zfill(2)}"
    return None


# Statement title lines to skip when looking for institution name
_STATEMENT_TITLE_NOISE = {
    "account activity",
    "account statement",
    "statement of account",
    "bank statement",
    "card statement",
    "credit card statement",
    "transaction history",
    "kontoauszug",
    "κίνηση λογαριασμού",
    "выписка по счету",
    "выписка",
}


def _parse_statement(raw_text: str) -> ParseResult:
    """Parse bank/card statement into structured data.

    Extracts: institution, account details, IBAN/BIC, period,
    individual transactions (date, description, debit, credit).
    """
    lines = raw_text.split("\n")
    data: dict[str, Any] = {"document_type": "statement"}
    fc: dict[str, float] = {}
    old_gaps: list[str] = []

    # Institution (first non-noise, non-title line)
    for line in lines:
        stripped = line.strip()
        if not stripped or _is_noise_line(stripped):
            continue
        low = stripped.lower()
        if low in _STATEMENT_TITLE_NOISE:
            continue
        # Skip lines that are just metadata (ACCOUNT NO, DATE, PAGE, etc.)
        if re.match(r"^(ACCOUNT|DATE|PAGE|CURRENCY|IBAN|BIC|PERIOD)\s", stripped):
            continue
        data["vendor"] = stripped
        break

    fc["institution"] = 1.0 if data.get("vendor") else 0.0

    # Account number
    for pat in _ACCOUNT_NO_PATTERNS:
        m = pat.search(raw_text)
        if m:
            data["account_number"] = m.group(1).strip()
            break

    # IBAN
    iban_match = _IBAN_PATTERN.search(raw_text)
    if iban_match:
        data["iban"] = iban_match.group(1).strip().replace(" ", "")

    # BIC
    bic_match = _BIC_PATTERN.search(raw_text)
    if bic_match:
        data["bic"] = bic_match.group(1)

    # Account type
    low_text = raw_text.lower()
    for acct_type in [
        "savings account",
        "current account",
        "checking account",
        "credit card",
        "sparkonto",
        "girokonto",
        "текущий счет",
        "сберегательный счет",
    ]:
        if acct_type in low_text:
            data["account_type"] = acct_type.title()
            break

    # Currency
    _extract_currency(raw_text, data)
    fc["currency"] = 0.8 if data.get("currency") else 0.0

    # Date (statement date)
    _extract_date_time(raw_text, data, old_gaps)
    if data.get("date"):
        data["document_date"] = data.pop("date")
    fc["document_date"] = 1.0 if data.get("document_date") else 0.0
    data.pop("time", None)

    # Period
    period_match = _PERIOD_PATTERN.search(raw_text)
    if period_match:
        start = _parse_date_dmy(period_match.group(1))
        end = _parse_date_dmy(period_match.group(2))
        if start and end:
            data["period_start"] = start
            data["period_end"] = end
    else:
        # Try to find period from date range pattern without label
        date_range = re.search(
            r"(\d{1,2}/\d{1,2}/\d{4})\s*[-–]\s*(\d{1,2}/\d{1,2}/\d{4})",
            raw_text,
        )
        if date_range:
            start = _parse_date_dmy(date_range.group(1))
            end = _parse_date_dmy(date_range.group(2))
            if start and end:
                data["period_start"] = start
                data["period_end"] = end

    # Transactions — parse table rows
    # Buffer for description prefix lines (text before the date+amount line)
    _AMOUNT_IN_LINE = re.compile(r"-?[\d.]+[,]\d{2}")
    _STMT_HEADER_LINE = re.compile(
        r"^(ACCOUNT|DATE|PAGE|CURRENCY|IBAN|BIC|PERIOD|BALANCE|TOTALS?)\s",
        re.IGNORECASE,
    )
    transactions: list[dict[str, Any]] = []
    desc_prefix_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Try to match a transaction line starting with a date
        date_match = re.match(r"^(\d{1,2}/\d{1,2}/\d{4})\s+(.+)", line)
        if date_match:
            txn_date_str = date_match.group(1)
            rest = date_match.group(2)
            txn_date = _parse_date_dmy(txn_date_str)

            # Prepend buffered description prefix
            if desc_prefix_lines:
                prefix = " ".join(desc_prefix_lines)
                rest = prefix + " " + rest
                desc_prefix_lines = []

            # Collect continuation lines (next lines without dates).
            # Collect continuation lines (amounts split from description).
            # Tolerate up to 1 blank line, but after a blank only continue
            # if the next line contains amounts (not description-only text
            # which is likely a prefix for the next transaction).
            full_rest = rest
            blank_count = 0
            while i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if not next_line:
                    blank_count += 1
                    if blank_count > 1:
                        break
                    i += 1
                    continue
                if re.match(r"^\d{1,2}/\d{1,2}/\d{4}", next_line):
                    break
                if _is_total_marker(next_line):
                    break
                # After a blank, only continue if line contains amounts
                if blank_count > 0 and not _AMOUNT_IN_LINE.search(next_line):
                    break
                blank_count = 0
                full_rest += " " + next_line
                i += 1

            # Extract amounts from the combined line
            # Look for debit/credit amounts (European format with comma)
            amounts = re.findall(r"-?[\d.]+[,]\d{2}", full_rest)
            if amounts:
                # Description is text before the first amount
                desc_end = full_rest.find(amounts[0])
                description = full_rest[:desc_end].strip()

                # Parse amounts
                debit = None
                credit = None
                value_date = None

                # Check for value date at end
                vd_match = re.search(r"(\d{1,2}/\d{1,2}/\d{4})\s*$", full_rest)
                if vd_match:
                    value_date = _parse_date_dmy(vd_match.group(1))

                for amt_str in amounts:
                    val = _parse_decimal(amt_str)
                    if val is not None:
                        if val < 0:
                            debit = val
                        elif val > 0:
                            credit = val
                        else:
                            # 0.00 — could be either
                            if credit is None:
                                credit = val

                txn: dict[str, Any] = {"date": txn_date, "description": description}
                if debit is not None:
                    txn["debit"] = debit
                if credit is not None:
                    txn["credit"] = credit
                if value_date:
                    txn["value_date"] = value_date

                transactions.append(txn)

        else:
            # Non-date line: buffer as potential description prefix
            if not line:
                desc_prefix_lines = []  # Reset on blank line
            elif (
                _is_total_marker(line)
                or _is_noise_line(line)
                or _STMT_HEADER_LINE.match(line)
                or _AMOUNT_IN_LINE.search(line)
                or len(line) > 80
            ):
                desc_prefix_lines = []
            else:
                desc_prefix_lines.append(line)

        i += 1

    if transactions:
        data["transactions"] = transactions
    fc["transactions"] = min(1.0, len(transactions) / 3) if transactions else 0.0

    # Totals line
    totals_match = re.search(
        r"TOTALS?\s*:?\s*(-?[\d.,]+)\s+(-?[\d.,]+)", raw_text, re.IGNORECASE
    )
    if totals_match:
        debit_total = _parse_decimal(totals_match.group(1))
        credit_total = _parse_decimal(totals_match.group(2))
        if debit_total is not None:
            data["total_debit"] = abs(debit_total)
        if credit_total is not None:
            data["total_credit"] = credit_total

    fc["language"] = 0.0

    # Compute confidence from core fields
    core_fields = ("institution", "currency", "document_date", "transactions")
    core_signals = [fc[k] for k in core_fields if k in fc]
    confidence = sum(core_signals) / len(core_signals) if core_signals else 0.0

    return ParseResult(
        data=data,
        confidence=round(confidence, 3),
        field_confidence=fc,
        line_item_count=len(transactions),
        needs_llm=True,
    )


# ---------------------------------------------------------------------------
# Minimal parser (warranty, contract)
# ---------------------------------------------------------------------------


def _parse_minimal(raw_text: str, doc_type: str) -> ParseResult:
    """Minimal parsing for complex document types."""
    data: dict[str, Any] = {"document_type": doc_type}
    fc: dict[str, float] = {"full_extraction": 0.0}
    old_gaps: list[str] = []

    # Try to get vendor and date at minimum
    lines = raw_text.split("\n")
    _extract_header(lines, data, old_gaps)
    _extract_date_time(raw_text, data, old_gaps)

    fc["vendor"] = 0.2 if data.get("vendor") else 0.0
    fc["date"] = 0.2 if data.get("date") else 0.0

    confidence = 0.15  # Low base — these types need full LLM
    if data.get("vendor"):
        confidence += 0.05
    if data.get("date"):
        confidence += 0.05

    return ParseResult(
        data=data,
        confidence=round(confidence, 3),
        field_confidence=fc,
        line_item_count=0,
        needs_llm=True,
    )
