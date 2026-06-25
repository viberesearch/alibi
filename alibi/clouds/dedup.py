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


@dataclass
class _ItemBundle:
    """An item-bearing bundle viewed for duplicate-photo de-duplication."""

    atoms: list[dict[str, Any]]
    prices: "collections.Counter[Decimal]"
    signature: "_BundleSignature | None"
    # Distance of this bundle's priced item sum from its own stated total.
    # The faithful transcription of a duplicate photo is the one whose items
    # sum nearest the printed total; ``None`` when the total is unknown.
    coverage_distance: Decimal | None

    @property
    def n_items(self) -> int:
        return len(self.atoms)


@dataclass(frozen=True)
class _BundleSignature:
    """A bundle's receipt identity — vendor slug + total, with an optional date.

    The date is corroborating only: a duplicate photo sometimes loses its date
    to OCR, so it may be ``None``. Two signatures with *conflicting* dates are
    never the same receipt; missing dates simply do not contradict.
    """

    vendor_slug: str
    total: Decimal
    event_date: str | None


def _atom_type(atom: dict[str, Any]) -> str:
    """Normalize an atom's type to its string value (enum or str accepted)."""
    t = atom.get("atom_type", "")
    return str(getattr(t, "value", t))


def _bundle_total(bundle: dict[str, Any]) -> Decimal | None:
    """The bundle's printed total (the ``total`` semantic amount atom)."""
    for atom in bundle.get("atoms", []):
        if _atom_type(atom) == "amount":
            data = atom.get("data", {})
            if data.get("semantic_type") == "total":
                return _round_price(data.get("value"))
    return None


def _bundle_date(bundle: dict[str, Any]) -> str | None:
    """The bundle's event date (YYYY-MM-DD) from its datetime atom."""
    for atom in bundle.get("atoms", []):
        if _atom_type(atom) == "datetime":
            value = str(atom.get("data", {}).get("value", "")).strip()
            if len(value) >= 10:
                return value[:10]
    return None


def _bundle_vendor_slug(bundle: dict[str, Any]) -> str | None:
    """The bundle's normalized vendor slug, from its vendor atom."""
    for atom in bundle.get("atoms", []):
        if _atom_type(atom) == "vendor":
            slug = normalize_vendor_slug(atom.get("data", {}).get("name") or "")
            return slug or None
    return None


def _bundle_signature(bundle: dict[str, Any]) -> _BundleSignature | None:
    """A bundle's identity (vendor + total, optional date), or None.

    Requires a vendor and a printed total; the date is optional corroboration.
    """
    slug = _bundle_vendor_slug(bundle)
    total = _bundle_total(bundle)
    if slug and total is not None:
        return _BundleSignature(
            vendor_slug=slug, total=total, event_date=_bundle_date(bundle)
        )
    return None


def _same_receipt(a: _BundleSignature, b: _BundleSignature) -> bool:
    """Whether two bundle signatures denote the same physical receipt.

    Same printed total, a compatible vendor (equal slug, or one a >= 4-char
    substring of the other for OCR variants like "LIDL" / "LIDL Cyprus"), and
    dates that do not conflict (equal, or at least one missing). This catches
    duplicate photos whose OCR prices diverged too far for price overlap and
    whose images differ too much for a perceptual-hash match.

    Same-vendor/same-total baskets from *different* days are not merged here —
    but they no longer share a cloud either: formation vetoes same-type merges
    across a date gap, so a surviving multi-basket cloud is same-date (or
    date-incomplete) by construction.
    """
    if a.total != b.total:
        return False
    if a.event_date and b.event_date and a.event_date != b.event_date:
        return False
    sa, sb = a.vendor_slug, b.vendor_slug
    if sa == sb:
        return True
    shorter, longer = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    return len(shorter) >= 4 and shorter in longer


def select_item_atoms(
    bundles: list[dict[str, Any]],
    *,
    price_threshold: float = PRICE_OVERLAP_THRESHOLD,
) -> list[dict[str, Any]]:
    """Choose which bundles' item atoms to use when collapsing a cloud.

    Complementary bundles (a basket plus its card slip) contribute distinct
    information and are all kept. But two *duplicate* baskets — the same receipt
    scanned twice, which a broadened formation candidate set now merges into one
    cloud — would otherwise double the line items. Among item-bearing bundles,
    keep one per physical receipt and drop the duplicates.

    A later bundle is a duplicate of an already-kept one when **either** their
    item price multisets overlap (>= ``price_threshold``) **or** they share a
    ``(vendor, date, total)`` signature. The signature catches duplicate photos
    whose OCR prices diverged so far that price overlap alone fails (e.g. one
    photo read 56.22 as a single line, the other as seventeen) and whose images
    differ too much in resolution for a perceptual-hash match.

    Keeper choice favours the faithful transcription: among duplicates, the
    bundle whose priced items sum nearest its printed total is kept (most items
    breaks ties, or stands in when the total is unknown).

    Returns the flat list of item atoms to feed to item extraction.
    """
    # Bundles that carry at least one item atom. Each atom is tagged with its
    # source bundle so downstream de-duplication can keep within-bundle repeats
    # while collapsing cross-bundle duplicates.
    item_bundles: list[_ItemBundle] = []
    for b in bundles:
        bundle_id = b.get("bundle_id")
        atoms = [
            {**a, "_bundle_id": bundle_id}
            for a in b.get("atoms", [])
            if _is_item_atom(a)
        ]
        if not atoms:
            continue
        prices = price_multiset(a.get("data", {}).get("total_price") for a in atoms)
        total = _bundle_total(b)
        coverage_distance = (
            abs(sum(prices.elements(), Decimal("0")) - total)
            if total is not None
            else None
        )
        item_bundles.append(
            _ItemBundle(
                atoms=atoms,
                prices=prices,
                signature=_bundle_signature(b),
                coverage_distance=coverage_distance,
            )
        )

    # Keeper-best first: smallest coverage distance, then most items. Bundles
    # with an unknown total (distance None) sort after, ranked by item count —
    # which preserves the legacy "keep the richest" behaviour.
    _far = Decimal("Infinity")
    item_bundles.sort(
        key=lambda ib: (
            ib.coverage_distance if ib.coverage_distance is not None else _far,
            -ib.n_items,
        )
    )

    if len(item_bundles) <= 1:
        # Fast path: nothing to dedup — keep every item atom.
        return [a for ib in item_bundles for a in ib.atoms]

    kept: list[_ItemBundle] = []
    for ib in item_bundles:
        is_duplicate = False
        for keeper in kept:
            union = sum((ib.prices | keeper.prices).values())
            inter = sum((ib.prices & keeper.prices).values())
            price_dup = bool(union) and inter / union >= price_threshold
            signature_dup = (
                ib.signature is not None
                and keeper.signature is not None
                and _same_receipt(ib.signature, keeper.signature)
            )
            if price_dup or signature_dup:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(ib)

    return [a for ib in kept for a in ib.atoms]


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
# Document-level duplicate-PHOTO detection (proactive, read-only)
# ---------------------------------------------------------------------------


@dataclass
class DuplicatePhoto:
    """A document that is part of a near-duplicate-photo group."""

    document_id: str
    file_path: str | None
    perceptual_hash: str | None
    cloud_id: str | None  # the doc's cloud (via its bundle), if assigned
    fact_id: str | None  # the collapsed fact for that cloud, if any


@dataclass
class DuplicatePhotoGroup:
    """A set of documents whose photos perceptually near-match."""

    documents: list[DuplicatePhoto] = field(default_factory=list)

    @property
    def distinct_clouds(self) -> set[str]:
        return {d.cloud_id for d in self.documents if d.cloud_id}

    @property
    def collapsed_together(self) -> bool:
        """True when every document already resolves to one shared cloud.

        Such a group is benign — formation/dedup already merged the duplicate.
        A group spanning several clouds (or with unassigned documents) is the
        actionable case: the same photo produced separate, un-merged facts.
        """
        clouds = {d.cloud_id for d in self.documents}
        return len(clouds) == 1 and None not in clouds


def find_duplicate_photos(
    db: DatabaseManager,
    max_distance: int = PHASH_MAX_DISTANCE,
) -> list[DuplicatePhotoGroup]:
    """Group documents whose stored perceptual hashes near-match.

    Proactively surfaces duplicate photos straight from
    ``documents.perceptual_hash`` instead of relying on the amount+date
    coincidence that fact dedup needs — so a re-ingested or alternate-resolution
    copy of a receipt is flagged even when the two scans' extractions diverged
    enough that their facts never shared a ``(date, total)`` bucket.

    Read-only. Returns groups of size >= 2 (most actionable first: groups that
    span multiple clouds, then larger groups). NOTE: dHash near-match catches
    near-identical images (re-ingests, alternate-resolution copies) but NOT a
    fresh re-shoot of the same receipt from a different angle — those differ
    widely in dHash and remain an amount+date / item-price concern for fact
    dedup.
    """
    rows = db.fetchall(
        "SELECT d.id AS id, d.file_path AS file_path, "
        "       d.perceptual_hash AS perceptual_hash, "
        "       (SELECT b.cloud_id FROM bundles b "
        "        WHERE b.document_id = d.id AND b.cloud_id IS NOT NULL "
        "        LIMIT 1) AS cloud_id "
        "FROM documents d "
        "WHERE d.perceptual_hash IS NOT NULL AND d.perceptual_hash != ''",
        (),
    )
    docs = [dict(r) for r in rows]
    n = len(docs)
    if n < 2:
        return []

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_distance(docs[i]["perceptual_hash"], docs[j]["perceptual_hash"])
            if d is not None and d <= max_distance:
                parent[find(i)] = find(j)

    # Resolve each cloud's collapsed fact once.
    fact_by_cloud: dict[str, str] = {}
    for cloud_id in {d["cloud_id"] for d in docs if d["cloud_id"]}:
        fact = db.fetchone(
            "SELECT id FROM facts WHERE cloud_id = ? LIMIT 1", (cloud_id,)
        )
        if fact:
            fact_by_cloud[cloud_id] = fact["id"]

    buckets: dict[int, list[DuplicatePhoto]] = {}
    for i, doc in enumerate(docs):
        buckets.setdefault(find(i), []).append(
            DuplicatePhoto(
                document_id=doc["id"],
                file_path=doc["file_path"],
                perceptual_hash=doc["perceptual_hash"],
                cloud_id=doc["cloud_id"],
                fact_id=fact_by_cloud.get(doc["cloud_id"]) if doc["cloud_id"] else None,
            )
        )

    groups = [DuplicatePhotoGroup(documents=g) for g in buckets.values() if len(g) > 1]
    # Most actionable first: multi-cloud groups, then larger groups.
    groups.sort(key=lambda g: (g.collapsed_together, -len(g.documents)))
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
