"""Cloud formation engine — unified cross-document clustering.

Replaces find_complementary_match() and _explode_statement_transactions()
with a single probabilistic clustering mechanism. Bundles from different
documents are matched into clouds based on vendor, amount, date, and item
overlap.

Temporal matching is asymmetric: receipt↔card_slip is tight (hours),
receipt↔bank_statement allows 1-3 business days, invoice↔payment allows
days to weeks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Any
from uuid import uuid4

from alibi.db.connection import DatabaseManager
from alibi.db.models import (
    AtomType,
    BundleType,
    Cloud,
    CloudBundle,
    CloudMatchType,
    CloudStatus,
)
from alibi.identities.matching import _strip_country_prefix
from alibi.normalizers.vendors import normalize_vendor_slug


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum substring length for fuzzy vendor matching
_MIN_VENDOR_SUBSTRING = 4

# Minimum SequenceMatcher ratio for fuzzy vendor_key match (OCR typo tolerance)
_VENDOR_KEY_FUZZY_THRESHOLD = 0.85

# Maximum absolute difference for near-match amounts (OCR rounding tolerance)
_AMOUNT_TOLERANCE = Decimal("0.02")

# normalize_vendor_slug imported from alibi.normalizers.vendors (canonical source)

# Date tolerances per bundle-type pair
_DATE_TOLERANCE: dict[tuple[BundleType, BundleType], int] = {
    # Same-type pairs: only cluster if same day (duplicate scans)
    (BundleType.BASKET, BundleType.BASKET): 0,
    (BundleType.INVOICE, BundleType.INVOICE): 0,
    (BundleType.PAYMENT_RECORD, BundleType.PAYMENT_RECORD): 0,
    (BundleType.STATEMENT_LINE, BundleType.STATEMENT_LINE): 0,
    # Receipt ↔ Payment record (card slip): tight, same day
    (BundleType.BASKET, BundleType.PAYMENT_RECORD): 1,
    (BundleType.PAYMENT_RECORD, BundleType.BASKET): 1,
    # Receipt ↔ Statement line: 1-3 business days for settlement
    (BundleType.BASKET, BundleType.STATEMENT_LINE): 5,
    (BundleType.STATEMENT_LINE, BundleType.BASKET): 5,
    # Invoice ↔ Payment: days to weeks
    (BundleType.INVOICE, BundleType.PAYMENT_RECORD): 60,
    (BundleType.PAYMENT_RECORD, BundleType.INVOICE): 60,
    # Invoice ↔ Statement: days to weeks
    (BundleType.INVOICE, BundleType.STATEMENT_LINE): 60,
    (BundleType.STATEMENT_LINE, BundleType.INVOICE): 60,
    # Payment ↔ Statement: 1-3 business days
    (BundleType.PAYMENT_RECORD, BundleType.STATEMENT_LINE): 5,
    (BundleType.STATEMENT_LINE, BundleType.PAYMENT_RECORD): 5,
}
_DEFAULT_DATE_TOLERANCE = 3  # days


# Confidence thresholds
_VENDOR_MATCH_CONFIDENCE = Decimal("0.3")
_AMOUNT_MATCH_CONFIDENCE = Decimal("0.4")
_DATE_MATCH_CONFIDENCE = Decimal("0.2")
_ITEM_OVERLAP_CONFIDENCE = Decimal("0.5")
_LOCATION_MATCH_CONFIDENCE = Decimal("0.15")


# ---------------------------------------------------------------------------
# Bundle summary — lightweight representation for matching
# ---------------------------------------------------------------------------


@dataclass
class BundleSummary:
    """Lightweight summary of a bundle for matching purposes."""

    bundle_id: str
    bundle_type: BundleType
    vendor: str | None = None
    vendor_normalized: str | None = None
    vendor_key: str | None = None
    vendor_legal_name: str | None = None
    vendor_legal_normalized: str | None = None
    amount: Decimal | None = None
    event_date: date | None = None
    currency: str = "EUR"
    item_names: list[str] = field(default_factory=list)
    cloud_id: str | None = None  # Already assigned cloud
    identity_id: str | None = None  # Vendor identity from registry
    lat: float | None = None  # Location latitude (from annotation)
    lng: float | None = None  # Location longitude (from annotation)


@dataclass
class MatchResult:
    """Result of matching a bundle against existing clouds."""

    cloud_id: str | None = None
    match_type: CloudMatchType | None = None
    confidence: Decimal = Decimal("0")
    is_new_cloud: bool = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_cloud_for_bundle(
    bundle: BundleSummary,
    existing_bundles: list[BundleSummary],
    db: DatabaseManager | None = None,
) -> MatchResult:
    """Find the best matching cloud for a bundle.

    Compares the new bundle against all existing bundles (which are
    already assigned to clouds) and returns the best match.

    Args:
        bundle: New bundle to place.
        existing_bundles: All bundles already assigned to clouds.
        db: Optional database manager for learned weight adjustments.

    Returns:
        MatchResult with cloud_id and confidence, or is_new_cloud=True.
    """
    best: MatchResult = MatchResult()

    for existing in existing_bundles:
        if existing.cloud_id is None:
            continue
        if existing.bundle_id == bundle.bundle_id:
            continue

        confidence, match_type = _score_match(bundle, existing, db=db)
        if confidence > best.confidence:
            best = MatchResult(
                cloud_id=existing.cloud_id,
                match_type=match_type,
                confidence=confidence,
                is_new_cloud=False,
            )

    # Threshold: vendor+amount+date needed for confident match
    if best.confidence <= Decimal("0.5"):
        return MatchResult()  # New cloud

    return best


def create_cloud_for_bundle(bundle_id: str) -> tuple[Cloud, CloudBundle]:
    """Create a new single-bundle cloud.

    Args:
        bundle_id: The bundle to start the cloud with.

    Returns:
        Tuple of (Cloud, CloudBundle) for the new cloud.
    """
    cloud = Cloud(id=str(uuid4()), status=CloudStatus.FORMING)
    link = CloudBundle(
        cloud_id=cloud.id,
        bundle_id=bundle_id,
        match_type=CloudMatchType.EXACT_AMOUNT,
        match_confidence=Decimal("1.0"),
    )
    return cloud, link


def add_bundle_to_cloud(
    cloud_id: str,
    bundle_id: str,
    match_type: CloudMatchType,
    confidence: Decimal,
) -> CloudBundle:
    """Add a bundle to an existing cloud.

    Args:
        cloud_id: Target cloud.
        bundle_id: Bundle to add.
        match_type: How the match was determined.
        confidence: Match confidence score.

    Returns:
        CloudBundle link record.
    """
    return CloudBundle(
        cloud_id=cloud_id,
        bundle_id=bundle_id,
        match_type=match_type,
        match_confidence=confidence,
    )


def extract_bundle_summary(
    bundle_id: str,
    bundle_type: BundleType,
    atoms: list[dict[str, Any]],
    cloud_id: str | None = None,
) -> BundleSummary:
    """Build a BundleSummary from atom data for matching.

    Args:
        bundle_id: Bundle ID.
        bundle_type: Type of bundle.
        atoms: List of atom data dicts (each with atom_type and data).
        cloud_id: Cloud ID if already assigned.

    Returns:
        BundleSummary ready for matching.
    """
    summary = BundleSummary(
        bundle_id=bundle_id,
        bundle_type=bundle_type,
        cloud_id=cloud_id,
    )

    for atom in atoms:
        atype = atom.get("atom_type", "")
        data = atom.get("data", {})

        if atype == AtomType.VENDOR.value or atype == AtomType.VENDOR:
            summary.vendor = data.get("name")
            if summary.vendor:
                summary.vendor_normalized = normalize_vendor_name(summary.vendor)
            legal = data.get("legal_name")
            if legal and str(legal).strip():
                summary.vendor_legal_name = str(legal).strip()
                summary.vendor_legal_normalized = normalize_vendor_name(
                    summary.vendor_legal_name
                )
            reg = data.get("vat_number") or data.get("tax_id")
            if reg and str(reg).strip():
                summary.vendor_key = str(reg).strip().upper().replace(" ", "")

        elif atype == AtomType.AMOUNT.value or atype == AtomType.AMOUNT:
            if data.get("semantic_type") == "total":
                try:
                    summary.amount = Decimal(str(data["value"]))
                except (InvalidOperation, ValueError, KeyError):
                    pass
                summary.currency = data.get("currency", "EUR")

        elif atype == AtomType.DATETIME.value or atype == AtomType.DATETIME:
            date_str = data.get("value", "")
            # Take just the date portion (YYYY-MM-DD)
            if date_str and len(date_str) >= 10:
                try:
                    summary.event_date = date.fromisoformat(date_str[:10])
                except ValueError:
                    pass

        elif atype == AtomType.ITEM.value or atype == AtomType.ITEM:
            name = data.get("name")
            if name:
                summary.item_names.append(str(name).strip().lower())

    return summary


# ---------------------------------------------------------------------------
# Vendor matching (replaces matching/duplicates.py logic)
# ---------------------------------------------------------------------------


# Alias for backward compatibility — all callers import this name
normalize_vendor_name = normalize_vendor_slug


def vendors_match(name1: str, name2: str) -> bool:
    """Check if two normalized vendor names match.

    Uses exact match first, then substring containment (min 4 chars).
    """
    if not name1 or not name2:
        return False

    n1 = normalize_vendor_name(name1)
    n2 = normalize_vendor_name(name2)

    if n1 == n2:
        return True

    shorter, longer = (n1, n2) if len(n1) <= len(n2) else (n2, n1)
    if len(shorter) >= _MIN_VENDOR_SUBSTRING and shorter in longer:
        return True

    return False


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------


def _score_match(
    new: BundleSummary,
    existing: BundleSummary,
    db: DatabaseManager | None = None,
) -> tuple[Decimal, CloudMatchType]:
    """Score match between two bundles.

    Returns (confidence, match_type). Confidence is 0-1.

    Args:
        new: New bundle to score.
        existing: Existing bundle already assigned to a cloud.
        db: Optional database manager for learned weight adjustments and
            false-positive pair lookups.
    """
    score = Decimal("0")
    match_type = CloudMatchType.VENDOR_DATE

    # Check for known false-positive pair (early exit)
    if db is not None and new.vendor_key and existing.vendor_key:
        try:
            from alibi.clouds.learning import is_known_false_positive_pair

            if is_known_false_positive_pair(db, new.vendor_key, existing.vendor_key):
                return Decimal("0"), CloudMatchType.VENDOR_DATE
        except Exception:
            pass  # Table may not exist yet

    # Load learned weight adjustments
    weight_adj: dict[str, float] = {}
    if db is not None:
        try:
            from alibi.clouds.learning import get_weight_adjustments

            weight_adj = get_weight_adjustments(db, vendor_key=new.vendor_key)
        except Exception:
            pass

    # Compute effective weights (learned multipliers applied on top of defaults)
    vendor_w = _VENDOR_MATCH_CONFIDENCE * Decimal(str(weight_adj.get("vendor", 1.0)))
    amount_w = _AMOUNT_MATCH_CONFIDENCE * Decimal(str(weight_adj.get("amount", 1.0)))
    date_w = _DATE_MATCH_CONFIDENCE * Decimal(str(weight_adj.get("date", 1.0)))
    item_w = _ITEM_OVERLAP_CONFIDENCE * Decimal(
        str(weight_adj.get("item_overlap", 1.0))
    )

    # Vendor matching — required for cloud formation
    vendor_score = _vendor_score(new, existing)
    if (
        vendor_score == Decimal("0")
        and new.vendor_normalized
        and existing.vendor_normalized
    ):
        return Decimal("0"), CloudMatchType.VENDOR_DATE
    score += vendor_score * vendor_w

    # Amount matching
    amount_score, amount_type = _amount_score(new, existing)
    score += amount_score * amount_w
    if amount_type:
        match_type = amount_type

    # Date matching (asymmetric per type pair)
    date_score = _date_score(new, existing)
    score += date_score * date_w

    # Item overlap bonus
    item_score = _item_overlap_score(new, existing)
    if item_score > 0:
        score += item_score * item_w
        match_type = CloudMatchType.ITEM_OVERLAP

    # Location proximity bonus (modest — not decisive alone)
    loc_score = _location_score(new.lat, new.lng, existing.lat, existing.lng)
    if loc_score > 0:
        score += loc_score * _LOCATION_MATCH_CONFIDENCE

    return min(score, Decimal("1.0")), match_type


def _vendor_score(a: BundleSummary, b: BundleSummary) -> Decimal:
    """Score vendor match (0, 0.8, or 1).

    Resolution order (strongest first):
    1. Identity registry: same identity_id = 1.0, different = 0.0
    2. vendor_key (registration ID): exact = 1.0, fuzzy = 0.9
    3. Normalized name comparison with substring containment
    """
    # Strongest: identity registry match
    if a.identity_id and b.identity_id:
        if a.identity_id == b.identity_id:
            return Decimal("1")
        # Explicitly different identities
        return Decimal("0")

    # Strong signal: matching vendor_key (registration ID)
    if a.vendor_key and b.vendor_key:
        if a.vendor_key == b.vendor_key:
            return Decimal("1")
        # Country-prefix-stripped comparison (e.g., "CY10370773Q" == "10370773Q")
        if _strip_country_prefix(a.vendor_key) == _strip_country_prefix(b.vendor_key):
            return Decimal("1")
        # Allow for OCR typos in registration IDs
        similarity = SequenceMatcher(None, a.vendor_key, b.vendor_key).ratio()
        if similarity >= _VENDOR_KEY_FUZZY_THRESHOLD:
            return Decimal("0.9")
        # Clearly different registration IDs = different vendors
        return Decimal("0")

    # Collect all normalized names for each side (trade name + legal name)
    a_names = {n for n in (a.vendor_normalized, a.vendor_legal_normalized) if n}
    b_names = {n for n in (b.vendor_normalized, b.vendor_legal_normalized) if n}

    if not a_names or not b_names:
        return Decimal("0")

    # Exact match on any combination of trade/legal names
    if a_names & b_names:
        return Decimal("1")

    # Substring containment across all name combinations
    for na in a_names:
        for nb in b_names:
            shorter = min(na, nb, key=len)
            longer = max(na, nb, key=len)
            if len(shorter) >= _MIN_VENDOR_SUBSTRING and shorter in longer:
                return Decimal("0.8")

    return Decimal("0")


def _amount_score(
    a: BundleSummary, b: BundleSummary
) -> tuple[Decimal, CloudMatchType | None]:
    """Score amount match with small tolerance for OCR rounding."""
    if a.amount is None or b.amount is None:
        return Decimal("0"), None

    if a.currency != b.currency:
        return Decimal("0"), None

    diff = abs(a.amount - b.amount)

    if diff == Decimal("0"):
        return Decimal("1"), CloudMatchType.EXACT_AMOUNT

    if diff <= _AMOUNT_TOLERANCE:
        return Decimal("0.8"), CloudMatchType.NEAR_AMOUNT

    return Decimal("0"), None


def _date_score(a: BundleSummary, b: BundleSummary) -> Decimal:
    """Score date proximity with asymmetric tolerance."""
    if a.event_date is None or b.event_date is None:
        return Decimal("0.5")  # Unknown date — neutral

    pair = (a.bundle_type, b.bundle_type)
    tolerance = _DATE_TOLERANCE.get(pair, _DEFAULT_DATE_TOLERANCE)
    diff = abs((a.event_date - b.event_date).days)

    if diff == 0:
        return Decimal("1")
    if diff <= tolerance:
        # Linear decay
        return Decimal(str(1 - diff / (tolerance + 1))).quantize(Decimal("0.01"))

    return Decimal("0")


def _location_score(
    lat1: float | None,
    lng1: float | None,
    lat2: float | None,
    lng2: float | None,
) -> Decimal:
    """Score location proximity between two points.

    Returns 1.0 for <100m, 0.8 for <500m, 0.5 for <2km, 0.2 for <5km,
    0 otherwise. Returns 0 if either point has missing coordinates
    (no penalty — just no bonus).
    """
    if lat1 is None or lng1 is None or lat2 is None or lng2 is None:
        return Decimal("0")

    from alibi.utils.map_url import haversine_distance

    dist = haversine_distance(lat1, lng1, lat2, lng2)

    if dist < 100:
        return Decimal("1")
    if dist < 500:
        return Decimal("0.8")
    if dist < 2000:
        return Decimal("0.5")
    if dist < 5000:
        return Decimal("0.2")

    return Decimal("0")


def _item_overlap_score(a: BundleSummary, b: BundleSummary) -> Decimal:
    """Score item name overlap between two bundles.

    Most useful for invoice↔receipt matching where items should overlap.
    """
    if not a.item_names or not b.item_names:
        return Decimal("0")

    set_a = set(a.item_names)
    set_b = set(b.item_names)
    overlap = len(set_a & set_b)

    if overlap == 0:
        return Decimal("0")

    # Jaccard similarity
    union = len(set_a | set_b)
    if union == 0:
        return Decimal("0")

    return Decimal(str(overlap / union)).quantize(Decimal("0.01"))
