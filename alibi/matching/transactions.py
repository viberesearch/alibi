"""Vendor matching and pattern learning."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from alibi.normalizers.vendors import normalize_vendor_slug


def normalize_vendor(vendor: str) -> str:
    """Normalize vendor name for comparison.

    Delegates to the canonical slug normalizer.
    """
    return normalize_vendor_slug(vendor)


def fuzzy_match(s1: str, s2: str) -> float:
    """Calculate fuzzy match ratio between two strings.

    Returns a value between 0.0 and 1.0.
    """
    if not s1 or not s2:
        return 0.0

    # Use SequenceMatcher for fuzzy matching
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


@dataclass
class VendorPattern:
    """Pattern for auto-categorizing vendors."""

    pattern: str  # Regex or exact match
    vendor_name: str  # Normalized vendor name
    default_category: str | None = None
    default_tags: list[str] = field(default_factory=list)
    confidence: float = 0.8
    usage_count: int = 0


def learn_vendor_pattern(
    vendor_raw: str,
    vendor_normalized: str,
    category: str | None = None,
    tags: list[str] | None = None,
) -> VendorPattern:
    """Create a vendor pattern from a manual categorization.

    Args:
        vendor_raw: The raw vendor string from extraction
        vendor_normalized: The normalized vendor name
        category: Default category for this vendor
        tags: Default tags for this vendor

    Returns:
        VendorPattern that can be saved and reused
    """
    # Create pattern that matches similar strings
    normalized = normalize_vendor(vendor_raw)
    pattern = re.escape(normalized)

    return VendorPattern(
        pattern=pattern,
        vendor_name=vendor_normalized,
        default_category=category,
        default_tags=tags or [],
        confidence=0.9,  # High confidence since it's from user input
    )


def match_vendor_pattern(
    vendor: str,
    patterns: list[VendorPattern],
) -> VendorPattern | None:
    """Find a matching vendor pattern.

    Args:
        vendor: Vendor string to match
        patterns: List of known patterns

    Returns:
        Best matching pattern or None
    """
    normalized = normalize_vendor(vendor)

    best_match: VendorPattern | None = None
    best_score = 0.0

    for pattern in patterns:
        # Try regex match first
        if re.search(pattern.pattern, normalized, re.IGNORECASE):
            score = pattern.confidence
            if score > best_score:
                best_score = score
                best_match = pattern
        else:
            # Try fuzzy match
            ratio = fuzzy_match(normalized, normalize_vendor(pattern.vendor_name))
            if ratio > 0.8 and ratio * pattern.confidence > best_score:
                best_score = ratio * pattern.confidence
                best_match = pattern

    return best_match
