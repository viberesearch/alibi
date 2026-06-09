"""Jurisdiction inference: derive the transaction's country, canonical currency,
and tax regime from an extraction dict (address, language, currency cues, tax
terms, VAT-number format).

This is the single source of truth for country/currency disambiguation so the
logic is not scattered across the currency / tax / language normalizers. It is
applied once, post-extraction, before atoms are parsed (see
``ProcessingPipeline._fill_locale_gaps``).

Why this exists: the symbol-first currency heuristic maps a bare ``$`` to USD,
so a Canadian receipt (CAD) was recorded as USD; and Northern-Cyprus receipts
print the same total in Turkish Lira + EUR + USD, but the canonical amount is
always the Lira. Reading the jurisdiction first lets us resolve these correctly.

Jurisdiction codes are ISO 3166-1 alpha-2 where one exists, plus the sentinel
``CY-NORTH`` for the Turkish Republic of Northern Cyprus (TRNC), which has no
ISO code but a distinct currency (TRY) and tax regime (KDV) from the Republic
of Cyprus (CY / EUR / VAT-ΦΠΑ).
"""

from __future__ import annotations

import re
from typing import Any

# --- Jurisdiction -> canonical currency (ISO 4217) --------------------------
JURISDICTION_CURRENCY: dict[str, str] = {
    "CY": "EUR",  # Republic of Cyprus
    "CY-NORTH": "TRY",  # Northern Cyprus (TRNC) — Turkish Lira
    "AT": "EUR",  # Austria
    "DE": "EUR",
    "GR": "EUR",
    "FR": "EUR",
    "IT": "EUR",
    "ES": "EUR",
    "TR": "TRY",  # Turkey
    "CA": "CAD",  # Canada
    "US": "USD",
    "GB": "GBP",
    "RU": "RUB",
}

# --- Jurisdiction -> tax regime keyword (informational) ---------------------
JURISDICTION_TAX_REGIME: dict[str, str] = {
    "CY": "vat",
    "CY-NORTH": "vat",  # KDV is a VAT
    "AT": "vat",
    "DE": "vat",
    "GR": "vat",
    "TR": "vat",
    "CA": "sales_tax",  # GST/HST/PST
    "US": "sales_tax",
    "GB": "vat",
}

# Place / signal tokens (matched uppercase, accent-folded). Word-boundaried at
# use so short tokens like "TL" don't hit substrings.
_NORTH_CYPRUS_TOKENS = (
    "KKTC",
    "K.K.T.C",
    "KUZEY KIBRIS",
    "LEFKOSA",  # Lefkoşa (accent-folded)
    "GIRNE",
    "MAGUSA",
    "GAZIMAGUSA",
    "GUZELYURT",
    "ISKELE",
    "LEFKE",
    # Northern-Cyprus villages / districts (improve recall where no city shows)
    "KAPLICA",
    "KARPAZ",
    "DIPKARPAZ",
    "BAFRA",
    "ESENTEPE",
    "CATALKOY",
    "ALSANCAK",
    "LAPTA",
    "TATLISU",
    "BOGAZ",
    "YENIBOGAZICI",
)
# Strong Canada signals: provinces & cities (bare "CANADA" handled separately
# because drink brands like "Canada Dry" / "Canadian Club" trip it falsely).
_CANADA_PLACES = (
    "ONTARIO",
    "QUEBEC",
    "BRITISH COLUMBIA",
    "ALBERTA",
    "MANITOBA",
    "SASKATCHEWAN",
    "NOVA SCOTIA",
    "TORONTO",
    "MONTREAL",
    "VANCOUVER",
    "OTTAWA",
    "GUELPH",
    "CALGARY",
    "EDMONTON",
    "WINNIPEG",
    "HALIFAX",
    "WATERLOO",
    "KITCHENER",
)
# Brand/product phrases containing "Canada"/"Canadian" that are NOT a location.
_CANADA_BRAND_NOISE = (
    "CANADA DRY",
    "AIR CANADA",
    "CANADIAN CLUB",
    "CANADA GOOSE",
    "CANADA POST",
)
_AUSTRIA_TOKENS = (
    "AUSTRIA",
    "OSTERREICH",  # Österreich (accent-folded)
    "WIEN",
    "GRAZ",
    "LINZ",
    "SALZBURG",
    "INNSBRUCK",
    "KLAGENFURT",
    "VILLACH",
    "WELS",
)
# Mainland-Turkey tokens. "TURKEY" (the poultry product) and "ADANA" (a kebab
# style) are deliberately excluded — they appear as food items on non-Turkish
# receipts. "TÜRKİYE" (accent-folded TURKIYE) is the country, not the bird.
_TURKEY_TOKENS = (
    "TURKIYE",
    "ISTANBUL",
    "ANKARA",
    "IZMIR",
    "BURSA",
    "ANTALYA",
    "KONYA",
)
_CYPRUS_TOKENS = (
    "CYPRUS",
    "KYPROS",  # ΚΥΠΡΟΣ (accent-folded transliteration not used; Greek handled below)
    "LIMASSOL",
    "LEMESOS",
    "NICOSIA",
    "LEFKOSIA",
    "LARNACA",
    "LARNAKA",
    "PAPHOS",
    "PAFOS",
    "PARALIMNI",
    "AYIA NAPA",
    "STROVOLOS",
    "ENGOMI",
    "AGLANTZIA",
)
# Greek-script Cyprus place names (matched as-is, uppercased)
_CYPRUS_TOKENS_GREEK = (
    "ΚΥΠΡΟΣ",
    "ΛΕΜΕΣΟΣ",
    "ΛΕΥΚΩΣΙΑ",
    "ΛΑΡΝΑΚΑ",
    "ΠΑΦΟΣ",
)

# Canada-specific tax-term cues. HST/PST/QST and French-Canadian TPS/TVQ are
# unambiguous. Bare "GST" is excluded because on restaurant POS slips it
# abbreviates "GUESTS" ("GST 2"); a real GST *tax* line (followed by a percent
# or money amount) is matched separately by _CANADA_GST_RE below.
_CANADA_TAX = ("HST", "PST", "QST", "TPS", "TVQ")
_CANADA_GST_RE = re.compile(r"\bGST\b\s*[:#]?\s*(?:\d+[.,]\d|\$|\d+\s*%)")

# Strong Turkish-Lira currency cues (always count). Bare ISO "TRY" is excluded
# (homograph of English "try"); "TL" is handled separately because alone it is
# noisy -- it only counts when printed next to an amount ("250 TL" / "TL 250").
_TRY_CUES = ("₺", "KDV", "TURK LIRASI")
_TL_AMOUNT_RE = re.compile(
    r"(?<![A-Z0-9])(?:\d[\d.,]*\s*TL|TL\s*[:=]?\s*\d)(?![A-Z0-9])"
)

_ATU_VAT_RE = re.compile(r"\bATU\s?\d{8}\b")  # Austrian UID
# Greek VAT marker ΦΠΑ, tolerant of Latin/Greek OCR glyph mixing (Φ/F, Π/P,
# Α/A). Used in both Cyprus and Greece; in this Cyprus-based deployment (no
# Greek docs) it is a Cyprus fallback signal, and the currency is EUR either way.
_FPA_RE = re.compile(r"[ΦF][ΠP][AΑ]")
_GREEK_RE = re.compile(r"[Α-Ωα-ω]")  # Greek script presence
# Accent folding for uppercase Latin/Turkish/German diacritics (blob is
# uppercased before translation, so only uppercase forms are mapped).
_ACCENTS = str.maketrans(
    "ÄÖÜẞŞĞİÇÉÈÀÂÊÔÛŪ",
    "AOUSSGICEEAAEOUU",
)


def _haystack(extracted: dict[str, Any]) -> str:
    """Build an uppercase, accent-folded search blob from the extraction."""
    parts = [
        str(extracted.get("vendor") or ""),
        str(extracted.get("vendor_address") or ""),
        str(extracted.get("vendor_legal_name") or ""),
        str(extracted.get("vendor_vat") or ""),
        str(extracted.get("raw_text") or ""),
    ]
    blob = " ".join(parts).upper()
    return blob.translate(_ACCENTS)


def _contains_word(blob: str, token: str) -> bool:
    """Substring match with word boundaries (tokens may contain spaces/dots)."""
    return (
        re.search(r"(?<![A-Z0-9])" + re.escape(token) + r"(?![A-Z0-9])", blob)
        is not None
    )


def _any_token(blob: str, tokens: tuple[str, ...]) -> bool:
    return any(_contains_word(blob, t) for t in tokens)


def infer_jurisdiction(extracted: dict[str, Any]) -> str | None:
    """Infer the jurisdiction code from address / tax terms / currency cues.

    Returns a jurisdiction code (ISO alpha-2 or ``CY-NORTH``) or None when no
    signal is found. Order matters — more specific signals win first.
    """
    blob = _haystack(extracted)
    # Greek-script Cyprus tokens are checked against the un-folded raw text.
    raw_upper = (
        str(extracted.get("vendor_address") or "")
        + " "
        + str(extracted.get("raw_text") or "")
    ).upper()

    # 1. Northern Cyprus — most specific (shares Turkish Lira with Turkey).
    if _any_token(blob, _NORTH_CYPRUS_TOKENS):
        return "CY-NORTH"

    # 2. Canada — provinces/cities or HST-PST-QST(-TPS/TVQ) tax terms are strong.
    #    Bare "CANADA" only counts when it is not part of a drink/brand phrase.
    canada_strong = (
        _any_token(blob, _CANADA_PLACES)
        or _any_token(blob, _CANADA_TAX)
        or _CANADA_GST_RE.search(blob) is not None
    )
    canada_bare = _contains_word(blob, "CANADA") and not any(
        b in blob for b in _CANADA_BRAND_NOISE
    )
    if canada_strong or canada_bare:
        return "CA"

    # 3. Austria — places or the ATU VAT prefix.
    if _any_token(blob, _AUSTRIA_TOKENS) or _ATU_VAT_RE.search(blob):
        return "AT"

    # 4. Turkey mainland.
    if _any_token(blob, _TURKEY_TOKENS):
        return "TR"

    # 5. Republic of Cyprus: place names (Latin or Greek), or the Greek VAT
    #    marker ΦΠΑ that appears on essentially every Cyprus receipt.
    if (
        _any_token(blob, _CYPRUS_TOKENS)
        or any(t in raw_upper for t in _CYPRUS_TOKENS_GREEK)
        or _FPA_RE.search(blob)
        or _FPA_RE.search(raw_upper)
    ):
        return "CY"

    # 6. Currency-only fallback when no place is recognised.
    if _has_try_cue(blob):
        return "TR"
    if _contains_word(blob, "CAD") or _contains_word(blob, "C$"):
        return "CA"

    # 7. Greek-script fallback: in this Cyprus-based deployment (no Greek-
    #    jurisdiction documents) substantial Greek text with no other signal is
    #    a Republic-of-Cyprus receipt. N.Cyprus/Turkey/Canada are caught above.
    if len(_GREEK_RE.findall(blob)) >= 5:
        return "CY"

    return None


def _has_try_cue(blob: str) -> bool:
    return (
        "₺" in blob
        or _any_token(blob, _TRY_CUES)
        or _TL_AMOUNT_RE.search(blob) is not None
    )


def resolve_currency(extracted: dict[str, Any], jurisdiction: str | None) -> str | None:
    """Resolve the canonical currency given the inferred jurisdiction.

    Handles the two real failure modes:
      * ambiguous ``$`` -> USD when the jurisdiction is Canada (CAD),
      * multi-currency Northern-Cyprus receipts (TRY + EUR + USD) where the
        canonical/total currency is always the Turkish Lira.
    Returns an ISO 4217 code, or None to leave the existing value untouched.
    """
    # Lazy import to avoid a cycle (currency imports nothing heavy, but keep tidy).
    from alibi.normalizers.currency import normalize_currency

    blob = _haystack(extracted)
    current_raw = str(extracted.get("currency") or "").strip()
    current = normalize_currency(current_raw) if current_raw else ""

    # 1. A known jurisdiction's currency is canonical for a receipt issued
    #    there: it is the currency of the printed total. This overrides the
    #    extracted value -- fixing the symbol-derived USD->CAD bug and
    #    self-healing any earlier mis-tagging. Turkish jurisdictions -> TRY.
    if jurisdiction in ("TR", "CY-NORTH"):
        return "TRY"
    if jurisdiction and jurisdiction in JURISDICTION_CURRENCY:
        return JURISDICTION_CURRENCY[jurisdiction]

    # 2. Unknown jurisdiction: derive from unambiguous cues/symbols in the text
    #    rather than trusting a possibly-stale currency field.
    if _has_try_cue(blob):
        return "TRY"
    if "€" in blob:
        return "EUR"
    if "£" in blob:
        return "GBP"
    return current or None


def apply_jurisdiction(
    extracted: dict[str, Any], default_country: str | None = None
) -> None:
    """Mutate ``extracted`` in place: set ``country`` and canonical ``currency``.

    ``default_country`` (e.g. the inbox folder default) is used only as a weak
    fallback for the country field; it never overrides an extracted currency.
    """
    if not extracted:
        return

    jurisdiction = infer_jurisdiction(extracted)
    inferred = jurisdiction is not None
    if jurisdiction is None and default_country:
        jurisdiction = default_country.strip().upper() or None

    if jurisdiction:
        extracted["country"] = jurisdiction

    # Resolve currency authoritatively. Pass the jurisdiction only when it was
    # actually inferred -- a folder-default guess must not force its currency;
    # resolve_currency then falls back to text symbols/cues, leaving a genuinely
    # signal-less doc's currency untouched.
    resolved = resolve_currency(extracted, jurisdiction if inferred else None)
    if resolved:
        extracted["currency"] = resolved
