"""Vendor name normalization — single source of truth.

Two output forms:
- normalize_vendor(name) -> title-case display name ("Acme Store")
- normalize_vendor_slug(name) -> lowercase slug for matching ("acmestore")

All other modules import from here. No duplicate suffix lists elsewhere.
"""

from __future__ import annotations

import re

# Unified legal suffixes — merged from all prior implementations.
# Sorted longest-first for correct regex stripping.
_LEGAL_SUFFIXES = [
    "incorporated",
    "corporation",
    "limited",
    "s.a.r.l",
    "s.r.l.",
    "gmbh",
    "corp",
    "sarl",
    "s.a.",
    "b.v.",
    "n.v.",
    "ltd",
    "llc",
    "inc",
    "srl",
    "a/s",
    "aps",
    "oyj",
    "ike",  # Greek: ΙΚΕ
    "επε",  # Greek: ΕΠΕ
    "epe",  # Greek ΕΠΕ, transliterated (slug path translits before stripping)
    "sa",
    "ag",
    "bv",
    "nv",
    "ab",
    "oy",
    "as",
    "co",
    "εε",  # Greek: Ε.Ε.
    "αε",  # Greek: Α.Ε.
    "ae",  # Greek ΑΕ, transliterated
    "ee",  # Greek ΕΕ, transliterated
]

_PREFIXES_TO_STRIP = [
    "the",
]

_LEGAL_PREFIXES = [
    "ооо",  # Russian: ООО
    "зао",  # Russian: ЗАО
    "оао",  # Russian: ОАО
]
# Transliterated forms for the slug path (which transliterates before stripping).
_LEGAL_PREFIXES_LATIN = ["ooo", "zao", "oao"]

_PUNCTUATION_TO_REMOVE = [".", ",", ":", ";", "'", '"', "!", "?"]

# Broad punctuation/whitespace pattern for slug form
_SLUG_STRIP_PATTERN = re.compile(r"[\s\-_.,;:!?'\"()&/]+")

# Greek -> Latin transliteration so a vendor OCR'd in Greek script slugs to the
# same string as its Latin spelling (e.g. "ΣΚΛΑΒΕΝΙΤΗΣ" -> "sklavenitis"). This
# lets a Greek-script receipt and its Latin-script card slip share a cloud key
# and collapse into one fact instead of two.
_GREEK_TO_LATIN = {
    "α": "a",
    "β": "v",
    "γ": "g",
    "δ": "d",
    "ε": "e",
    "ζ": "z",
    "η": "i",
    "θ": "th",
    "ι": "i",
    "κ": "k",
    "λ": "l",
    "μ": "m",
    "ν": "n",
    "ξ": "x",
    "ο": "o",
    "π": "p",
    "ρ": "r",
    "σ": "s",
    "ς": "s",
    "τ": "t",
    "υ": "y",
    "φ": "f",
    "χ": "ch",
    "ψ": "ps",
    "ω": "o",
    "ά": "a",
    "έ": "e",
    "ή": "i",
    "ί": "i",
    "ό": "o",
    "ύ": "y",
    "ώ": "o",
    "ϊ": "i",
    "ϋ": "y",
    "ΐ": "i",
    "ΰ": "y",
}


def _transliterate_greek(text: str) -> str:
    """Map lowercase Greek letters to Latin equivalents; leave others unchanged."""
    return "".join(_GREEK_TO_LATIN.get(ch, ch) for ch in text)


# Cyrillic -> Latin transliteration (Russian), same purpose as the Greek map:
# a vendor OCR'd in Cyrillic ("ПЯТЁРОЧКА" -> "pyatyorochka") slugs to the same
# string as its Latin spelling, so a Russian receipt and its card slip share a
# cloud key and collapse into one fact. Soft/hard signs fold to nothing.
_CYRILLIC_TO_LATIN = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "kh",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "shch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


def _transliterate_cyrillic(text: str) -> str:
    """Map lowercase Cyrillic letters to Latin equivalents; others unchanged."""
    return "".join(_CYRILLIC_TO_LATIN.get(ch, ch) for ch in text)


# Card acquirers / payment processors and ATM cash withdrawals are NOT real
# merchants -- on a lone card slip the printed "vendor" is the acquirer (JCC
# Payment Systems, P3T Payment Solutions) and an ATM line is a cash withdrawal,
# not a purchase. Recognizing them lets collapse prefer the real merchant when
# both are present, and lets spend/subscription analytics drop them.
_PAYMENT_INTERMEDIARY_RE = re.compile(
    r"\bPAYMENT\s+(?:SYSTEMS?|SOLUTIONS?|SERVICES?)\b"
    r"|\bACQUIR(?:ING|ER)\b"
    r"|\bATM\b"
    r"|\bJ[CG]C\s+PAYMENT\b"
    r"|\bP3T\b"
    # Russian acquirer banks whose name headlines the card slip while the
    # merchant sits on the line below — the slip must merge with the
    # merchant's receipt (same amount/date), not stand as its own vendor.
    r"|\bТ[-\s]?БАНК\b"
    r"|\bТИНЬКОФФ\b"
    r"|\bСБЕРБАНК\b"
    r"|\bАЛЬФА[-\s]?БАНК\b"
    r"|\bВТБ\b"
    r"|\bГАЗПРОМБАНК\b"
    r"|\bРАЙФФАЙЗЕН(?:БАНК)?\b"
    r"|\bРОСБАНК\b",
    re.IGNORECASE,
)


def is_payment_intermediary(name: str | None) -> bool:
    """Whether a vendor name is a card acquirer/processor or ATM, not a merchant."""
    if not name:
        return False
    return _PAYMENT_INTERMEDIARY_RE.search(name) is not None


# Sentinel strings that mean "no registration". An OCR/LLM structuring pass
# sometimes emits the literal "null"/"N/A"/"-" for a missing VAT; stored
# verbatim it becomes a vendor_key of "NULL", which then groups every keyless
# fact under one bogus vendor in analytics. Treat these as absent.
_REGISTRATION_SENTINELS = frozenset(
    {"", "null", "none", "nil", "n/a", "n.a.", "na", "nan", "-", "--", "unknown"}
)


def clean_registration(reg: str | None) -> str | None:
    """Normalize a raw VAT/tax-id string; return None for missing/sentinel values.

    Used everywhere a registration ID flows into a vendor_key so that literal
    "null"/"N/A"/"-" placeholders never become a real key.
    """
    if reg is None:
        return None
    cleaned = str(reg).strip()
    if cleaned.lower() in _REGISTRATION_SENTINELS:
        return None
    return cleaned


def normalize_vendor(name: str) -> str:
    """Clean and normalize vendor names.

    Operations:
        - Strip legal suffixes (Ltd, LLC, GmbH, etc.)
        - Strip common prefixes ("The ")
        - Normalize whitespace
        - Normalize case (title case)
        - Remove extra punctuation

    Examples:
        "ACME Corporation, Inc." -> "Acme"
        "The Best Shop Ltd" -> "Best Shop"
        "Μαγαζί Μου Ε.Ε." -> "Μαγαζί Μου"
        "ООО Компания" -> "Компания"

    Returns:
        Normalized vendor name.
    """
    if not name:
        return ""

    s = name.strip()

    # Normalize whitespace (multiple spaces to single)
    s = re.sub(r"\s+", " ", s)

    # Remove common punctuation
    for punct in _PUNCTUATION_TO_REMOVE:
        s = s.replace(punct, " ")

    # Split into tokens
    tokens = s.split()

    # Convert to lowercase for comparison
    lower_tokens = [t.lower() for t in tokens]

    # Strip prefixes (common English prefixes)
    while lower_tokens and lower_tokens[0] in _PREFIXES_TO_STRIP:
        lower_tokens.pop(0)
        tokens.pop(0)

    # Strip legal prefixes (e.g., Russian "ООО")
    while lower_tokens and lower_tokens[0] in _LEGAL_PREFIXES:
        lower_tokens.pop(0)
        tokens.pop(0)

    # Strip suffixes (including multi-token abbreviations like "Ε Ε")
    while lower_tokens:
        # Check single token match
        if lower_tokens[-1] in _LEGAL_SUFFIXES:
            lower_tokens.pop()
            tokens.pop()
            continue

        # Check if last 2 tokens form a legal suffix (e.g., "ε ε" -> "εε")
        if len(lower_tokens) >= 2:
            last_two = "".join(lower_tokens[-2:])
            if last_two in _LEGAL_SUFFIXES:
                lower_tokens.pop()
                lower_tokens.pop()
                tokens.pop()
                tokens.pop()
                continue

        # No match, stop stripping
        break

    # Rejoin and normalize case
    s = " ".join(tokens).strip()

    # Title case (capitalize each word)
    # Note: This may not work perfectly for all languages, but it's a reasonable default
    s = s.title()

    return s


def normalize_vendor_slug(name: str) -> str:
    """Normalize vendor name to a lowercase slug for matching.

    Strips legal suffixes, removes all punctuation/whitespace, lowercases.
    Used for vendor deduplication and cloud formation matching.

    Examples:
        "FreSko BUTANOLO LTD" -> "freskobutanolo"
        "THE NUT CRACKER HOUSE" -> "thenutcrackerhouse"
        "ACME Corp." -> "acme"
    """
    if not name:
        return ""

    result = _transliterate_cyrillic(_transliterate_greek(name.lower()))

    # Strip a leading legal-form prefix (Russian ООО/ЗАО/ОАО, now transliterated
    # to ooo/zao/oao) so "ООО Ромашка" and a slip's "Romashka" slug alike.
    for prefix in _LEGAL_PREFIXES_LATIN:
        result = re.sub(rf"^{re.escape(prefix)}\s+", "", result, count=1)

    # Strip legal suffixes (longest-first ordering prevents partial matches)
    for suffix in _LEGAL_SUFFIXES:
        pattern = rf"\s+{re.escape(suffix)}\.?\s*$"
        result = re.sub(pattern, "", result)

    # Remove all punctuation and whitespace
    result = _SLUG_STRIP_PATTERN.sub("", result)
    return result
