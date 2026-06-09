"""Duplicate-fact detection and safe merging.

A single transaction can be ingested more than once (an archive copy plus a
later reprocess, or two photos of the same receipt). Cloud formation does not
always merge them — most often because OCR variance in the vendor registration
ID (``300108234`` vs ``30010823A``) keeps the twin out of the candidate set —
leaving two facts for one transaction.

This module finds those duplicate groups and resolves them *safely*. The hard
lesson behind the safety gate: grouping on extracted ``(vendor, date, total)``
alone false-merges when an extraction is wrong. A mis-read receipt once landed
on the same ``2025-12-29 / 85.10`` signature as a genuinely different receipt
and the two were merged, deleting the real one.

So a candidate pair is auto-merged **only** when an independent signal
corroborates that they are the same physical transaction:

* one twin has zero line items (a poor/empty extraction — safe to drop), OR
* their document perceptual hashes near-match (the same image re-ingested), OR
* their item *price multisets* substantially overlap.

Price overlap is the workhorse: item *names* are garbled differently by OCR on
each scan (``bazania`` vs ``bazanta``), but the printed prices are stable. On
the confirmed regression pairs the price overlap is 0.71 and 1.00; on the known
false pair it is 0.08 — so a 0.5 threshold separates them cleanly. Anything that
does not clear the gate is flagged ``REVIEW`` rather than merged.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Iterable

from alibi.db.connection import DatabaseManager
from alibi.identities.matching import _strip_country_prefix
from alibi.normalizers.vendors import normalize_vendor_slug

# ---------------------------------------------------------------------------
# Tunables (calibrated on the corpus regression pairs)
# ---------------------------------------------------------------------------

# Min Jaccard overlap of the two item price multisets to call a pair duplicates.
PRICE_OVERLAP_THRESHOLD = 0.5

# Max Hamming distance between two 64-bit dHash strings to call them the same
# image. dHash of distinct photos of the same receipt differs widely, so this
# only fires on a literal re-ingest of the same file.
PHASH_MAX_DISTANCE = 6

# Min SequenceMatcher ratio for two vendor registration IDs to be "compatible"
# (tolerates a one-character OCR slip). Mirrors formation's fuzzy key threshold.
_VENDOR_KEY_FUZZY_THRESHOLD = 0.85

# Min length for a normalized vendor-name substring containment match.
_MIN_VENDOR_SUBSTRING = 4


# ---------------------------------------------------------------------------
# Pure helpers — price multiset overlap and perceptual-hash distance
# ---------------------------------------------------------------------------


def _round_price(value: Any) -> Decimal | None:
    """Coerce a price (Decimal/float/str) to 2dp Decimal, or None."""
    if value is None:
        return None
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError):
        return None


def price_multiset(prices: Iterable[Any]) -> "collections.Counter[Decimal]":
    """Build a Counter of 2dp prices, dropping None/unparseable values."""
    counter: collections.Counter[Decimal] = collections.Counter()
    for p in prices:
        rp = _round_price(p)
        if rp is not None:
            counter[rp] += 1
    return counter


def price_multiset_overlap(prices_a: Iterable[Any], prices_b: Iterable[Any]) -> float:
    """Jaccard overlap of two price multisets (intersection / union by count).

    Robust to OCR name garbling because prices survive OCR far better than
    non-Latin item names. Returns 0.0 when either side has no priced items.
    """
    ca = price_multiset(prices_a)
    cb = price_multiset(prices_b)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / union if union else 0.0


def hamming_distance(h1: str | None, h2: str | None) -> int | None:
    """Bit-level Hamming distance between two equal-length hex hash strings.

    Returns None when either hash is missing or they are not comparable
    (different length / not valid hex).
    """
    if not h1 or not h2 or len(h1) != len(h2):
        return None
    try:
        return bin(int(h1, 16) ^ int(h2, 16)).count("1")
    except ValueError:
        return None


def phash_near_match(
    hashes_a: Iterable[str | None],
    hashes_b: Iterable[str | None],
    max_distance: int = PHASH_MAX_DISTANCE,
) -> bool:
    """True if any cross pair of perceptual hashes is within ``max_distance``."""
    la = [h for h in hashes_a if h]
    lb = [h for h in hashes_b if h]
    for ha in la:
        for hb in lb:
            d = hamming_distance(ha, hb)
            if d is not None and d <= max_distance:
                return True
    return False


# ---------------------------------------------------------------------------
# Vendor compatibility (candidate gating — broad on purpose)
# ---------------------------------------------------------------------------


def vendors_compatible(
    vendor_a: str | None,
    key_a: str | None,
    vendor_b: str | None,
    key_b: str | None,
) -> bool:
    """Whether two facts could be the same vendor despite OCR variance.

    Deliberately permissive: registration-ID fuzzy match (one-char OCR slip),
    country-prefix-insensitive key equality, or normalized-name equality /
    substring containment. The duplicate *decision* gate does the real safety
    work; this only decides whether a pair is worth gating at all.
    """
    ka = (key_a or "").upper().replace(" ", "")
    kb = (key_b or "").upper().replace(" ", "")
    if ka and kb:
        if ka == kb:
            return True
        if _strip_country_prefix(ka) == _strip_country_prefix(kb):
            return True
        if SequenceMatcher(None, ka, kb).ratio() >= _VENDOR_KEY_FUZZY_THRESHOLD:
            return True
        # Two clearly different registration IDs => different vendors.
        return False

    na = normalize_vendor_slug(vendor_a or "")
    nb = normalize_vendor_slug(vendor_b or "")
    if not na or not nb:
        return False
    if na == nb:
        return True
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    return len(shorter) >= _MIN_VENDOR_SUBSTRING and shorter in longer


# ---------------------------------------------------------------------------
# Duplicate decision gate
# ---------------------------------------------------------------------------


class DuplicateVerdict(str, Enum):
    """Outcome of comparing two candidate-duplicate facts."""

    MERGE = "merge"  # Safe to auto-merge (corroborated)
    REVIEW = "review"  # Looks like a dup but needs a human glance


@dataclass
class FactDupInfo:
    """Minimal fact view needed to decide and act on a duplicate."""

    fact_id: str
    cloud_id: str
    vendor: str | None
    vendor_key: str | None
    event_date: str | None
    event_time: str | None
    total_amount: Decimal | None
    currency: str | None
    n_items: int
    item_prices: list[Any] = field(default_factory=list)
    perceptual_hashes: list[str | None] = field(default_factory=list)


def decide_duplicate(
    a: FactDupInfo,
    b: FactDupInfo,
    *,
    price_threshold: float = PRICE_OVERLAP_THRESHOLD,
    phash_max_distance: int = PHASH_MAX_DISTANCE,
) -> tuple[DuplicateVerdict, str]:
    """Decide whether two candidate-duplicate facts may be auto-merged.

    Returns (verdict, reason). MERGE only when an independent signal
    corroborates the duplicate; otherwise REVIEW.
    """
    if a.n_items == 0 or b.n_items == 0:
        return DuplicateVerdict.MERGE, "zero-item twin"

    if phash_near_match(a.perceptual_hashes, b.perceptual_hashes, phash_max_distance):
        return DuplicateVerdict.MERGE, "perceptual-hash match"

    overlap = price_multiset_overlap(a.item_prices, b.item_prices)
    if overlap >= price_threshold:
        return DuplicateVerdict.MERGE, f"price overlap {overlap:.2f}"

    return DuplicateVerdict.REVIEW, f"price overlap {overlap:.2f} < {price_threshold}"


# ---------------------------------------------------------------------------
# Collapse helper — pick item atoms from non-duplicate bundles
# ---------------------------------------------------------------------------


def select_item_atoms(
    bundles: list[dict[str, Any]],
    *,
    price_threshold: float = PRICE_OVERLAP_THRESHOLD,
) -> list[dict[str, Any]]:
    """Choose which bundles' item atoms to use when collapsing a cloud.

    Complementary bundles (a basket plus its card slip) contribute distinct
    information and are all kept. But two *duplicate* baskets — the same receipt
    scanned twice, which a broadened formation candidate set now merges into one
    cloud — would otherwise double the line items. So among item-bearing
    bundles, keep the richest and drop any later one whose item price multiset
    overlaps an already-kept bundle (>= ``price_threshold``).

    Returns the flat list of item atoms to feed to item extraction.
    """
    # Bundles that carry at least one item atom, richest first.
    item_bundles: list[tuple[list[dict[str, Any]], collections.Counter[Decimal]]] = []
    for b in bundles:
        atoms = [a for a in b.get("atoms", []) if _is_item_atom(a)]
        if atoms:
            prices = price_multiset(a.get("data", {}).get("total_price") for a in atoms)
            item_bundles.append((atoms, prices))
    item_bundles.sort(key=lambda ba: len(ba[0]), reverse=True)

    if len(item_bundles) <= 1:
        # Fast path: nothing to dedup — keep every item atom.
        return [a for atoms, _ in item_bundles for a in atoms]

    kept: list[tuple[list[dict[str, Any]], collections.Counter[Decimal]]] = []
    for atoms, prices in item_bundles:
        duplicates_kept = False
        for _, kept_prices in kept:
            union = sum((prices | kept_prices).values())
            inter = sum((prices & kept_prices).values())
            if union and inter / union >= price_threshold:
                duplicates_kept = True
                break
        if not duplicates_kept:
            kept.append((atoms, prices))

    return [a for atoms, _ in kept for a in atoms]


# AtomType.ITEM may surface as the enum or its "item" value depending on the
# call site; accept either when scanning atoms.
try:  # pragma: no cover - trivial import guard
    from alibi.db.models import AtomType as _AtomType

    _ITEM_ATOM_TYPES: tuple[Any, ...] = (_AtomType.ITEM, _AtomType.ITEM.value)
except Exception:  # pragma: no cover
    _ITEM_ATOM_TYPES = ("item",)


def _is_item_atom(atom: dict[str, Any]) -> bool:
    return atom.get("atom_type") in _ITEM_ATOM_TYPES


# ---------------------------------------------------------------------------
# Group discovery over the live fact table
# ---------------------------------------------------------------------------


def _load_fact_info(db: DatabaseManager, fact_row: dict[str, Any]) -> FactDupInfo:
    """Hydrate a FactDupInfo with item prices and document perceptual hashes."""
    fid = fact_row["id"]
    cloud_id = fact_row["cloud_id"]
    item_rows = db.fetchall(
        "SELECT total_price FROM fact_items WHERE fact_id = ?", (fid,)
    )
    prices = [r["total_price"] for r in item_rows]
    phash_rows = db.fetchall(
        "SELECT DISTINCT d.perceptual_hash FROM documents d "
        "JOIN bundles b ON b.document_id = d.id "
        "WHERE b.cloud_id = ? AND d.perceptual_hash IS NOT NULL",
        (cloud_id,),
    )
    total = _round_price(fact_row["total_amount"])
    return FactDupInfo(
        fact_id=fid,
        cloud_id=cloud_id,
        vendor=fact_row["vendor"],
        vendor_key=fact_row["vendor_key"],
        event_date=fact_row["event_date"],
        event_time=fact_row["event_time"],
        total_amount=total,
        currency=fact_row["currency"],
        n_items=len(prices),
        item_prices=prices,
        perceptual_hashes=[r["perceptual_hash"] for r in phash_rows],
    )


def _components(facts: list[FactDupInfo]) -> list[list[FactDupInfo]]:
    """Group facts in one (date, amount, currency) bucket by vendor compatibility.

    Builds connected components over the "could be the same vendor" relation so a
    chain of OCR-variant keys (A~B, B~C) stays in one candidate group.
    """
    n = len(facts)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if vendors_compatible(
                facts[i].vendor,
                facts[i].vendor_key,
                facts[j].vendor,
                facts[j].vendor_key,
            ):
                parent[find(i)] = find(j)

    groups: dict[int, list[FactDupInfo]] = {}
    for i, f in enumerate(facts):
        groups.setdefault(find(i), []).append(f)
    return [g for g in groups.values() if len(g) > 1]


def find_duplicate_groups(db: DatabaseManager) -> list[list[FactDupInfo]]:
    """Find candidate-duplicate fact groups.

    A candidate group shares an exact ``(event_date, total_amount, currency)``
    signature and a compatible vendor. Exact amount/date is just the *join* key;
    a wrong extraction can share it by coincidence, so the safety call is left to
    :func:`decide_duplicate`, not to the grouping.
    """
    buckets = db.fetchall(
        "SELECT event_date, total_amount, currency FROM facts "
        "WHERE total_amount IS NOT NULL "
        "GROUP BY event_date, total_amount, currency HAVING COUNT(*) > 1",
        (),
    )

    groups: list[list[FactDupInfo]] = []
    for bucket in buckets:
        rows = db.fetchall(
            "SELECT id, cloud_id, vendor, vendor_key, event_date, event_time, "
            "total_amount, currency FROM facts "
            "WHERE total_amount = ? AND currency = ? "
            "AND (event_date = ? OR (event_date IS NULL AND ? IS NULL))",
            (
                bucket["total_amount"],
                bucket["currency"],
                bucket["event_date"],
                bucket["event_date"],
            ),
        )
        facts = [_load_fact_info(db, dict(r)) for r in rows]
        groups.extend(_components(facts))
    return groups


# ---------------------------------------------------------------------------
# Dedup pass — decide and (optionally) merge
# ---------------------------------------------------------------------------


def _keeper_sort_key(f: FactDupInfo) -> tuple[int, int, str]:
    """Richest first: most items, then has-event_time, then stable id."""
    return (-f.n_items, 0 if f.event_time else 1, f.fact_id)


@dataclass
class DedupAction:
    """A single keeper/redundant resolution within a duplicate group."""

    verdict: DuplicateVerdict
    reason: str
    keeper: FactDupInfo
    redundant: FactDupInfo
    applied: bool = False


@dataclass
class DedupReport:
    """Outcome of a dedup pass."""

    resolved: list[DedupAction] = field(default_factory=list)
    review: list[DedupAction] = field(default_factory=list)

    @property
    def resolved_count(self) -> int:
        return len(self.resolved)

    @property
    def review_count(self) -> int:
        return len(self.review)


def dedup_pass(db: DatabaseManager, *, apply: bool = False) -> DedupReport:
    """Find duplicate facts and (optionally) resolve the safe ones.

    Each non-keeper fact in a group is compared against the group's richest
    (keeper) fact. A MERGE verdict resolves the duplicate by deleting the
    *redundant* twin (its fact and source-document chain) while leaving the
    keeper fact untouched; REVIEW verdicts are reported only.

    Why delete rather than re-collapse the two bundles into one cloud: surgical
    fact edits decouple ``fact_items`` from their bundle's item atoms (a fixed
    fact can have many items over a bundle with one stale atom). Re-collapsing
    such a keeper would regenerate items from the stale atoms and silently lose
    the correction. Deleting the redundant twin preserves the keeper exactly.

    Idempotent: a clean corpus yields an empty report. ``apply=False`` mutates
    nothing.
    """
    report = DedupReport()
    for group in find_duplicate_groups(db):
        ordered = sorted(group, key=_keeper_sort_key)
        keeper = ordered[0]
        for redundant in ordered[1:]:
            verdict, reason = decide_duplicate(keeper, redundant)
            action = DedupAction(
                verdict=verdict, reason=reason, keeper=keeper, redundant=redundant
            )
            if verdict is DuplicateVerdict.MERGE:
                if apply:
                    _delete_redundant(db, redundant)
                    action.applied = True
                report.resolved.append(action)
            else:
                report.review.append(action)
    return report


def _documents_for_cloud(db: DatabaseManager, cloud_id: str) -> list[str]:
    rows = db.fetchall(
        "SELECT DISTINCT d.id FROM documents d "
        "JOIN bundles b ON b.document_id = d.id "
        "WHERE b.cloud_id = ?",
        (cloud_id,),
    )
    return [r["id"] for r in rows]


def _delete_redundant(db: DatabaseManager, redundant: FactDupInfo) -> None:
    """Remove the redundant twin: its source document chain, then its fact/cloud.

    The keeper fact is never touched (no re-collapse), so its items — including
    any surgical corrections — survive intact.
    """
    from alibi.services.query import delete_document, delete_fact

    for doc_id in _documents_for_cloud(db, redundant.cloud_id):
        delete_document(db, doc_id)
    delete_fact(db, redundant.fact_id)
