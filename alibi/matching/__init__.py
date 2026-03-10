"""Matching, deduplication, and vendor pattern learning."""

from alibi.matching.duplicates import (
    ComplementaryMatchResult,
    DuplicateCheckResult,
    check_content_duplicate,
    check_duplicate,
    compute_average_hash,
    compute_file_hash,
    compute_perceptual_hash,
    find_complementary_match,
    find_similar_images,
    get_file_fingerprint,
    hash_distance,
    is_image_file,
    normalize_vendor_name,
)
from alibi.matching.transactions import (
    VendorPattern,
    fuzzy_match,
    learn_vendor_pattern,
    match_vendor_pattern,
    normalize_vendor,
)

__all__ = [
    # Duplicates
    "ComplementaryMatchResult",
    "DuplicateCheckResult",
    "check_duplicate",
    "check_content_duplicate",
    "find_complementary_match",
    "compute_file_hash",
    "compute_perceptual_hash",
    "compute_average_hash",
    "hash_distance",
    "find_similar_images",
    "get_file_fingerprint",
    "is_image_file",
    "normalize_vendor_name",
    # Vendor patterns
    "VendorPattern",
    "normalize_vendor",
    "fuzzy_match",
    "learn_vendor_pattern",
    "match_vendor_pattern",
]
