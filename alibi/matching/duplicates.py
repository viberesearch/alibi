"""Duplicate detection and vendor canonicalization for artifacts."""

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

import imagehash
import yaml
from PIL import Image

from alibi.normalizers.vendors import normalize_vendor_slug

from alibi.db.models import Artifact

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vendor alias mappings (loaded from YAML at startup)
# ---------------------------------------------------------------------------

# Keys: normalized vendor name, Values: canonical display name
_vendor_mappings: dict[str, str] = {}


@dataclass
class DuplicateCheckResult:
    """Result of duplicate check."""

    is_duplicate: bool
    original_artifact: Optional[Artifact] = None
    match_type: Optional[str] = None  # 'exact_hash', 'perceptual', 'content_match'
    similarity_score: Optional[float] = None


@dataclass
class ComplementaryMatchResult:
    """Result of complementary proof matching.

    A complementary match means the same purchase (vendor+date+amount) has
    a document of a *different* type already stored, providing a different
    proof dimension for the same transaction.
    """

    is_match: bool
    original_artifact: Optional[Artifact] = None
    match_type: Optional[str] = None  # 'complementary'


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of file content.

    Args:
        file_path: Path to the file

    Returns:
        SHA-256 hash as hex string
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_perceptual_hash(image_path: Path) -> str:
    """Compute difference hash (dHash) for image similarity.

    Args:
        image_path: Path to the image file

    Returns:
        Perceptual hash as hex string
    """
    img = Image.open(image_path)
    return str(imagehash.dhash(img))


def compute_average_hash(image_path: Path) -> str:
    """Compute average hash (aHash) for image similarity.

    Args:
        image_path: Path to the image file

    Returns:
        Average hash as hex string
    """
    img = Image.open(image_path)
    return str(imagehash.average_hash(img))


def hash_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two perceptual hashes.

    Args:
        hash1: First hash string
        hash2: Second hash string

    Returns:
        Hamming distance (number of differing bits)
    """
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    return h1 - h2


def is_image_file(file_path: Path) -> bool:
    """Check if file is an image based on extension."""
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    return file_path.suffix.lower() in image_extensions


def check_duplicate(
    new_file_path: Path,
    existing_artifacts: list[Artifact],
    perceptual_threshold: int = 5,
) -> DuplicateCheckResult:
    """Check if a file is a duplicate of existing artifacts.

    Checks in order:
    1. Exact content hash match
    2. Perceptual hash similarity (for images)
    3. Content match (same vendor + date + amount + type)

    Args:
        new_file_path: Path to the new file
        existing_artifacts: List of existing artifacts to check against
        perceptual_threshold: Max hash distance for perceptual match (default 5)

    Returns:
        DuplicateCheckResult with match details
    """
    # Compute hash of new file
    new_hash = compute_file_hash(new_file_path)

    # 1. Check exact hash match
    for artifact in existing_artifacts:
        if artifact.file_hash == new_hash:
            return DuplicateCheckResult(
                is_duplicate=True,
                original_artifact=artifact,
                match_type="exact_hash",
                similarity_score=1.0,
            )

    # 2. Check perceptual similarity for images
    if is_image_file(new_file_path):
        try:
            new_perceptual = compute_perceptual_hash(new_file_path)

            for artifact in existing_artifacts:
                if artifact.perceptual_hash:
                    distance = hash_distance(new_perceptual, artifact.perceptual_hash)
                    if distance < perceptual_threshold:
                        similarity = 1.0 - (distance / 64.0)  # Normalize to 0-1
                        return DuplicateCheckResult(
                            is_duplicate=True,
                            original_artifact=artifact,
                            match_type="perceptual",
                            similarity_score=similarity,
                        )
        except Exception as e:
            logger.warning(f"Failed to compute perceptual hash: {e}")

    return DuplicateCheckResult(is_duplicate=False)


# Canonical slug normalizer imported from alibi.normalizers.vendors
normalize_vendor_name = normalize_vendor_slug


def vendors_match(name1: str, name2: str) -> bool:
    """Check if two normalized vendor names match.

    Uses exact equality first, then substring containment:
    if the shorter name is contained in the longer, it's likely
    the same vendor (e.g., "fresko" in "freskobutanolo").

    Requires the shorter name to be at least 4 chars to avoid
    false positives on very short strings.
    """
    if name1 == name2:
        return True
    if not name1 or not name2:
        return False
    shorter, longer = (name1, name2) if len(name1) <= len(name2) else (name2, name1)
    if len(shorter) >= 4 and shorter in longer:
        return True
    return False


def load_vendor_aliases(path: Path) -> dict[str, str]:
    """Read vendor alias YAML and return normalized-key → canonical-name mapping.

    YAML format is flat key:value where keys are any form of the vendor name
    (will be normalized for matching) and values are canonical display names.

    Args:
        path: Path to YAML file with vendor alias definitions.

    Returns:
        Dict mapping normalized vendor names to canonical display names.
        Returns empty dict on missing file, empty file, or parse error.
    """
    if not path.exists():
        return {}

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Cannot read vendor aliases from %s: %s", path, exc)
        return {}

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        logger.warning("Invalid YAML in %s: %s", path, exc)
        return {}

    if not isinstance(data, dict):
        return {}

    result: dict[str, str] = {}
    for key, value in data.items():
        normalized_key = normalize_vendor_name(str(key))
        canonical = str(value).strip()
        if normalized_key and canonical:
            result[normalized_key] = canonical
    return result


def init_vendor_mappings(path: Path | None = None) -> None:
    """Load vendor alias overrides into module cache.

    Call once at startup. If path is None, aliases are cleared.
    """
    global _vendor_mappings
    if path is None:
        _vendor_mappings = {}
        return
    _vendor_mappings = load_vendor_aliases(path)
    if _vendor_mappings:
        logger.info("Loaded %d vendor alias(es) from %s", len(_vendor_mappings), path)


def reset_vendor_mappings() -> None:
    """Clear vendor alias cache (for test isolation)."""
    global _vendor_mappings
    _vendor_mappings = {}


def canonicalize_vendor(raw_name: str | None) -> str | None:
    """Return canonical vendor name if known, else the raw name.

    Normalizes the input and checks against user-defined vendor
    aliases using fuzzy substring matching.

    Args:
        raw_name: Raw vendor name from extraction.

    Returns:
        Canonical display name if matched, else raw_name unchanged.
    """
    if not raw_name:
        return raw_name

    normalized = normalize_vendor_name(raw_name)
    if not normalized:
        return raw_name

    for alias_normalized, canonical in _vendor_mappings.items():
        if vendors_match(normalized, alias_normalized):
            return canonical

    return raw_name


def _time_diff_minutes(t1: str, t2: str) -> int:
    """Compute absolute difference in minutes between two HH:MM:SS times."""
    parts1 = t1.split(":")
    parts2 = t2.split(":")
    mins1 = int(parts1[0]) * 60 + int(parts1[1])
    mins2 = int(parts2[0]) * 60 + int(parts2[1])
    return abs(mins1 - mins2)


# Max time difference (minutes) for content duplicate matching.
# Card terminal slips and POS receipts from the same purchase
# typically differ by a few minutes.
CONTENT_DUPLICATE_TIME_TOLERANCE = 60


def check_content_duplicate(
    vendor: Optional[str],
    document_date: Optional[date],
    amount: Optional[Decimal],
    artifact_type: str,
    existing_artifacts: list[Artifact],
    transaction_time: Optional[str] = None,
) -> DuplicateCheckResult:
    """Check for duplicate based on content fields.

    Same vendor (fuzzy) + date + amount + type is likely a duplicate.
    When both documents have a time, uses a tolerance window instead
    of exact match (card slips and POS receipts differ by minutes).

    Args:
        vendor: Vendor name
        document_date: Document date
        amount: Document amount
        artifact_type: Type of artifact
        existing_artifacts: List of existing artifacts
        transaction_time: Transaction time (HH:MM:SS) for stricter matching

    Returns:
        DuplicateCheckResult with match details
    """
    if not all([vendor, document_date, amount]):
        return DuplicateCheckResult(is_duplicate=False)

    normalized_vendor = normalize_vendor_name(vendor)  # type: ignore[arg-type]  # guarded above

    for artifact in existing_artifacts:
        if not artifact.vendor:
            continue
        if (
            vendors_match(normalize_vendor_name(artifact.vendor), normalized_vendor)
            and artifact.document_date == document_date
            and artifact.amount == amount
            and artifact.type.value == artifact_type
        ):
            # If both have time, allow tolerance (card slip vs POS receipt)
            if transaction_time and artifact.transaction_time:
                diff = _time_diff_minutes(transaction_time, artifact.transaction_time)
                if diff > CONTENT_DUPLICATE_TIME_TOLERANCE:
                    continue
            return DuplicateCheckResult(
                is_duplicate=True,
                original_artifact=artifact,
                match_type="content_match",
                similarity_score=0.95,
            )

    return DuplicateCheckResult(is_duplicate=False)


# Max date difference (days) for complementary proof matching.
# Bank "value date" can differ from payment date by a few business days.
COMPLEMENTARY_DATE_TOLERANCE_DAYS = 3


def find_complementary_match(
    vendor: Optional[str],
    document_date: Optional[date],
    amount: Optional[Decimal],
    artifact_type: str,
    existing_artifacts: list[Artifact],
    transaction_time: Optional[str] = None,
) -> ComplementaryMatchResult:
    """Find a complementary proof for the same purchase.

    Matches artifacts with the same vendor+date+amount but a DIFFERENT type.
    This indicates the same purchase documented from a different proof
    dimension (e.g. receipt + payment confirmation for the same buy).

    Allows a date window of up to COMPLEMENTARY_DATE_TOLERANCE_DAYS to
    account for bank processing delays (value date vs payment date).

    Args:
        vendor: Vendor name
        document_date: Document date
        amount: Document amount
        artifact_type: Type of the NEW artifact being processed
        existing_artifacts: List of existing artifacts to check against
        transaction_time: Transaction time (HH:MM:SS) for stricter matching

    Returns:
        ComplementaryMatchResult with match details
    """
    if not all([vendor, document_date, amount]):
        return ComplementaryMatchResult(is_match=False)

    normalized_vendor = normalize_vendor_name(vendor)  # type: ignore[arg-type]

    for artifact in existing_artifacts:
        if not artifact.vendor:
            continue
        # Must be DIFFERENT type (same type = duplicate, not complementary)
        if artifact.type.value == artifact_type:
            continue
        if not vendors_match(normalize_vendor_name(artifact.vendor), normalized_vendor):
            continue
        if artifact.amount != amount:
            continue

        # Date matching with tolerance for bank processing delays
        if artifact.document_date and document_date:
            date_diff = abs((artifact.document_date - document_date).days)
            if date_diff > COMPLEMENTARY_DATE_TOLERANCE_DAYS:
                continue
        else:
            continue

        # If same day and both have time, apply time tolerance too
        if (
            artifact.document_date == document_date
            and transaction_time
            and artifact.transaction_time
        ):
            diff = _time_diff_minutes(transaction_time, artifact.transaction_time)
            if diff > CONTENT_DUPLICATE_TIME_TOLERANCE:
                continue

        return ComplementaryMatchResult(
            is_match=True,
            original_artifact=artifact,
            match_type="complementary",
        )

    return ComplementaryMatchResult(is_match=False)


def find_similar_images(
    image_path: Path,
    existing_artifacts: list[Artifact],
    max_distance: int = 10,
) -> list[tuple[Artifact, int]]:
    """Find images similar to the given image.

    Args:
        image_path: Path to the image
        existing_artifacts: List of existing artifacts
        max_distance: Maximum hash distance to consider similar

    Returns:
        List of (artifact, distance) tuples sorted by distance
    """
    if not is_image_file(image_path):
        return []

    try:
        new_hash = compute_perceptual_hash(image_path)
    except Exception as e:
        logger.warning(f"Failed to compute hash for {image_path}: {e}")
        return []

    similar = []
    for artifact in existing_artifacts:
        if artifact.perceptual_hash:
            try:
                distance = hash_distance(new_hash, artifact.perceptual_hash)
                if distance <= max_distance:
                    similar.append((artifact, distance))
            except Exception:
                continue

    return sorted(similar, key=lambda x: x[1])


def get_file_fingerprint(file_path: Path) -> dict[str, Any]:
    """Get a comprehensive fingerprint of a file for duplicate detection.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with file fingerprint data
    """
    fingerprint = {
        "file_hash": compute_file_hash(file_path),
        "file_size": file_path.stat().st_size,
        "extension": file_path.suffix.lower(),
    }

    if is_image_file(file_path):
        try:
            fingerprint["perceptual_hash"] = compute_perceptual_hash(file_path)
            fingerprint["average_hash"] = compute_average_hash(file_path)

            # Get image dimensions
            with Image.open(file_path) as img:
                fingerprint["width"] = img.width
                fingerprint["height"] = img.height
        except Exception as e:
            logger.warning(f"Failed to get image fingerprint: {e}")

    return fingerprint
