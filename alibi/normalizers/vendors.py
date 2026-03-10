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
]

_PREFIXES_TO_STRIP = [
    "the",
]

_LEGAL_PREFIXES = [
    "ооо",  # Russian: ООО
    "зао",  # Russian: ЗАО
    "оао",  # Russian: ОАО
]

_PUNCTUATION_TO_REMOVE = [".", ",", ":", ";", "'", '"', "!", "?"]

# Broad punctuation/whitespace pattern for slug form
_SLUG_STRIP_PATTERN = re.compile(r"[\s\-_.,;:!?'\"()&/]+")


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

    result = name.lower()

    # Strip legal suffixes (longest-first ordering prevents partial matches)
    for suffix in _LEGAL_SUFFIXES:
        pattern = rf"\s+{re.escape(suffix)}\.?\s*$"
        result = re.sub(pattern, "", result)

    # Remove all punctuation and whitespace
    result = _SLUG_STRIP_PATTERN.sub("", result)
    return result
