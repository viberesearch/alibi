"""Merge-preserving re-extraction for under-extracted facts.

Re-runs Stage-3 LLM structuring on a document's CACHED OCR text
(``documents.raw_extraction['raw_text']`` -- NEVER re-OCR) to recover line
items the original local pass missed, then splices the richer items into the
EXISTING fact via :func:`alibi.clouds.correction.recollapse_cloud`.

Why this is safe (merge-preserving):

* ``recollapse_cloud`` regenerates the fact from the cloud's *existing* bundles
  -- it does NOT re-run cloud formation. So a reconciled ``vendor_key`` and any
  multi-document membership (a receipt collapsed with its payment slip) survive;
  the fact is never re-split into a new cloud.
* Only ITEM atoms are swapped. Vendor / amount / date atoms are left untouched,
  so the ``vendor_key`` derivation in collapse is identical by construction.

The plain ``lt reingest`` path cannot be used here: it calls
``cleanup_document`` + full cloud re-formation, which can scatter the bundle
into a new cloud and discard the manual reconciliation.

Dry-run (the default) still calls the structurer so the projected item delta is
real, but performs no DB or YAML mutation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from alibi.atoms.parser import _parse_item_atom, _parse_vat_analysis
from alibi.clouds.correction import recollapse_cloud
from alibi.db import v2_store
from alibi.db.connection import DatabaseManager
from alibi.db.models import Atom, BundleAtomRole
from alibi.enrichment.coverage import item_coverage_report
from alibi.normalizers.currency import normalize_currency

logger = logging.getLogger(__name__)

# Stage-3 structurer signature: (raw_text, doc_type) -> extraction dict.
# Injectable so tests can supply a deterministic stub instead of calling Gemini.
Structurer = Callable[[str, str], dict[str, Any]]

# Bundle types that carry line items (a payment slip never should).
_ITEM_BUNDLE_TYPES = ("basket", "invoice")


@dataclass
class DocReextract:
    """Per-document outcome inside a single fact re-extraction."""

    document_id: str
    bundle_id: str
    items_before: int = 0
    items_after: int = 0
    skipped: str | None = None


@dataclass
class ReextractResult:
    """Outcome of re-extracting one fact."""

    fact_id: str
    cloud_id: str | None = None
    vendor: str | None = None
    vendor_key_before: str | None = None
    vendor_key_after: str | None = None
    total: float = 0.0
    item_sum_before: float = 0.0
    item_sum_after: float = 0.0
    items_before: int = 0
    items_after: int = 0
    coverage_before: float = 0.0
    coverage_after: float = 0.0
    new_fact_id: str | None = None
    documents: list[DocReextract] = field(default_factory=list)
    would_change: bool = False
    applied: bool = False
    error: str | None = None

    @property
    def vendor_key_preserved(self) -> bool:
        """True when the reconciled vendor_key is unchanged after re-collapse."""
        return self.vendor_key_before == self.vendor_key_after


def _default_structurer(raw_text: str, doc_type: str) -> dict[str, Any]:
    """Production structurer: Gemini Stage-3 over cached OCR text."""
    from alibi.extraction.gemini_structurer import structure_ocr_text_gemini

    return structure_ocr_text_gemini(raw_text, doc_type=doc_type)


def select_queue(
    db: DatabaseManager,
    queue: str = "partial",
    limit: int = 5,
    threshold_pct: float = 92.0,
) -> list[Any]:
    """Pick the worst under-extracted facts as a re-extraction work-list.

    ``queue``:
        * ``partial``  -- facts WITH some items but below threshold (items short).
        * ``item-less`` -- facts with a total but zero items.
        * ``all``      -- everything below threshold, worst first.

    Returns ItemCoverageRow objects, lowest coverage first.
    """
    report = item_coverage_report(db, threshold_pct=threshold_pct, worst_limit=10_000)
    rows = list(report.worst)
    q = queue.replace("_", "-").lower()
    if q == "partial":
        rows = [r for r in rows if r.n_items > 0]
    elif q in ("item-less", "itemless", "no-items"):
        rows = [r for r in rows if r.n_items == 0]
    # "all" -> keep every below-threshold row
    rows.sort(key=lambda r: r.coverage_pct)
    return rows[:limit]


def _fact_coverage(
    db: DatabaseManager, fact_id: str
) -> tuple[int, float, float, float]:
    """Return (n_items, item_sum, total, coverage_pct) for one fact."""
    row = db.fetchone(
        "SELECT f.total_amount AS total, COUNT(fi.id) AS n_items, "
        "COALESCE(SUM(fi.total_price), 0) AS item_sum "
        "FROM facts f LEFT JOIN fact_items fi ON fi.fact_id = f.id "
        "WHERE f.id = ? GROUP BY f.id",
        (fact_id,),
    )
    if not row:
        return 0, 0.0, 0.0, 0.0
    total = float(row["total"] or 0)
    item_sum = float(row["item_sum"] or 0)
    n_items = int(row["n_items"] or 0)
    cov = (item_sum / total * 100.0) if total else 0.0
    return n_items, round(item_sum, 2), round(total, 2), round(cov, 1)


def _item_bundles(db: DatabaseManager, cloud_id: str) -> list[Any]:
    """Item-bearing bundles (basket/invoice) in a cloud, with document ids."""
    placeholders = ",".join("?" for _ in _ITEM_BUNDLE_TYPES)
    return db.fetchall(
        f"SELECT b.id AS bundle_id, b.document_id, b.bundle_type "
        f"FROM bundles b WHERE b.cloud_id = ? "
        f"AND b.bundle_type IN ({placeholders})",
        (cloud_id, *_ITEM_BUNDLE_TYPES),
    )


def _doc_extraction(
    db: DatabaseManager, document_id: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Load a document's cached extraction dict and its yaml_path."""
    row = db.fetchone(
        "SELECT raw_extraction, yaml_path FROM documents WHERE id = ?",
        (document_id,),
    )
    if not row or not row["raw_extraction"]:
        return None, None
    data = row["raw_extraction"]
    if isinstance(data, str):
        data = json.loads(data)
    return data, row["yaml_path"]


def _count_item_atoms(db: DatabaseManager, bundle_id: str) -> int:
    """Count BASKET_ITEM atoms currently linked to a bundle."""
    row = db.fetchone(
        "SELECT COUNT(*) AS n FROM bundle_atoms WHERE bundle_id = ? AND role = ?",
        (bundle_id, BundleAtomRole.BASKET_ITEM.value),
    )
    return int(row["n"]) if row else 0


def _build_item_atoms(
    document_id: str,
    line_items: list[dict[str, Any]],
    currency: str,
    language: str | None,
    raw_text: str,
) -> list[Atom]:
    """Parse re-extracted line items into normalized ITEM atoms."""
    vat_mapping = _parse_vat_analysis(raw_text or "")
    atoms: list[Atom] = []
    for raw_item in line_items:
        if not isinstance(raw_item, dict):
            continue
        atom = _parse_item_atom(document_id, raw_item, currency, language, vat_mapping)
        if atom is not None:
            atoms.append(atom)
    return atoms


def _swap_item_atoms(
    db: DatabaseManager, bundle_id: str, new_atoms: list[Atom]
) -> None:
    """Replace a bundle's ITEM atoms with ``new_atoms`` (single transaction)."""
    old = db.fetchall(
        "SELECT a.id FROM atoms a JOIN bundle_atoms ba ON a.id = ba.atom_id "
        "WHERE ba.bundle_id = ? AND ba.role = ?",
        (bundle_id, BundleAtomRole.BASKET_ITEM.value),
    )
    old_ids = [r["id"] for r in old]
    with db.transaction() as cur:
        for aid in old_ids:
            cur.execute(
                "DELETE FROM bundle_atoms WHERE bundle_id = ? AND atom_id = ?",
                (bundle_id, aid),
            )
            cur.execute("DELETE FROM atoms WHERE id = ?", (aid,))
        for atom in new_atoms:
            cur.execute(
                "INSERT INTO atoms (id, document_id, atom_type, data, confidence) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    atom.id,
                    atom.document_id,
                    atom.atom_type.value,
                    json.dumps(atom.data),
                    float(atom.confidence),
                ),
            )
            cur.execute(
                "INSERT OR IGNORE INTO bundle_atoms (bundle_id, atom_id, role) "
                "VALUES (?, ?, ?)",
                (bundle_id, atom.id, BundleAtomRole.BASKET_ITEM.value),
            )


def _sync_yaml(
    db: DatabaseManager,
    document_id: str,
    yaml_path: str | None,
    extraction: dict[str, Any],
    new_line_items: list[dict[str, Any]],
) -> None:
    """Write richer line_items back to the DB extraction JSON and YAML SSOT.

    The YAML store stays the human-editable record of *what was extracted*; the
    DB stays the record of *how it collapsed*. Re-collapse (not reingest) is the
    application path, so this only keeps the SSOT honest -- it does not change
    how the fact collapsed.
    """
    import yaml as _yaml

    # 1. DB mirror (documents.raw_extraction)
    updated = dict(extraction)
    updated["line_items"] = new_line_items
    updated["_pipeline"] = "reextract_merge_preserving"
    with db.transaction() as cur:
        cur.execute(
            "UPDATE documents SET raw_extraction = ? WHERE id = ?",
            (json.dumps(updated), document_id),
        )

    # 2. YAML store file (best-effort: keep all keys, replace line_items)
    if not yaml_path:
        return
    path = Path(yaml_path)
    if not path.exists():
        logger.warning("reextract: yaml_path missing on disk, skipped: %s", yaml_path)
        return
    try:
        with open(path) as f:
            doc = _yaml.safe_load(f) or {}
        if not isinstance(doc, dict):
            return
        doc["line_items"] = new_line_items
        doc["_pipeline"] = "reextract_merge_preserving"
        with open(path, "w") as f:
            _yaml.dump(doc, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:  # never let SSOT sync abort the re-extraction
        logger.warning("reextract: YAML sync failed for %s: %s", yaml_path, e)


def reextract_fact(
    db: DatabaseManager,
    fact_id: str,
    apply: bool = False,
    structurer: Structurer | None = None,
    sync_yaml: bool = True,
) -> ReextractResult:
    """Re-extract one fact's items from cached OCR, preserving its identity.

    Args:
        db: Database manager.
        fact_id: Target purchase fact.
        apply: When False (default) compute the projected delta without mutating.
        structurer: Stage-3 structurer (defaults to Gemini). Injectable for tests.
        sync_yaml: When applying, also write richer items back to the YAML SSOT.

    Returns:
        ReextractResult. On apply, ``new_fact_id`` holds the re-collapsed fact id
        and ``vendor_key_preserved`` must be True for a clean merge.
    """
    structure = structurer or _default_structurer

    fact = v2_store.get_fact_by_id(db, fact_id)
    if not fact:
        return ReextractResult(fact_id=fact_id, error="fact not found")
    if fact.get("fact_type") != "purchase":
        return ReextractResult(
            fact_id=fact_id, error="not a purchase fact (re-extract is items-only)"
        )

    cloud_id = fact.get("cloud_id")
    fact_currency = normalize_currency(fact.get("currency") or "EUR")
    n0, sum0, total, cov0 = _fact_coverage(db, fact_id)
    result = ReextractResult(
        fact_id=fact_id,
        cloud_id=cloud_id,
        vendor=fact.get("vendor"),
        vendor_key_before=fact.get("vendor_key"),
        total=total,
        item_sum_before=sum0,
        items_before=n0,
        coverage_before=cov0,
    )

    bundles = _item_bundles(db, cloud_id) if cloud_id else []
    if not bundles:
        result.error = "no item-bearing (basket/invoice) bundle in cloud"
        return result

    # Phase 1 -- gather per-document plans (calls the structurer, NO mutation).
    # A "plan" is an improving document whose re-extraction recovered strictly
    # more items than it currently has.
    plans: list[
        tuple[str, str, list[Atom], list[dict[str, Any]], dict[str, Any], str | None]
    ] = []
    for b in bundles:
        bundle_id = b["bundle_id"]
        document_id = b["document_id"]
        before = _count_item_atoms(db, bundle_id)
        doc_rec = DocReextract(
            document_id=document_id, bundle_id=bundle_id, items_before=before
        )
        result.documents.append(doc_rec)

        extraction, yaml_path = _doc_extraction(db, document_id)
        raw_text = (extraction or {}).get("raw_text") or ""
        if not raw_text:
            doc_rec.skipped = "no cached raw_text"
            continue

        doc_type = (extraction or {}).get("document_type") or "receipt"
        language = (extraction or {}).get("language")
        currency = normalize_currency(
            (extraction or {}).get("currency") or fact_currency
        )

        try:
            structured = structure(raw_text, doc_type)
        except Exception as e:  # one bad doc must not abort the whole fact
            doc_rec.skipped = f"structurer error: {e}"
            continue

        new_items = structured.get("line_items") or []
        new_atoms = _build_item_atoms(
            document_id, new_items, currency, language, raw_text
        )
        doc_rec.items_after = len(new_atoms)

        # Only act when re-extraction recovered MORE items than we already have
        # (never regress a fact by overwriting good items with a worse pass).
        if len(new_atoms) <= before:
            doc_rec.skipped = (
                f"no improvement ({len(new_atoms)} <= {before} existing items)"
            )
            continue

        plans.append(
            (bundle_id, document_id, new_atoms, new_items, extraction or {}, yaml_path)
        )

    result.would_change = bool(plans)

    if not plans or not apply:
        # No mutation. Project the fact-level item count honestly: each improving
        # document contributes its EXTRA atoms (new minus current bundle atoms);
        # a fact whose bundle already holds the items (collapse-limited, not
        # extraction-limited) contributes 0 and is reported as no change.
        delta = sum(
            d.items_after - d.items_before
            for d in result.documents
            if d.skipped is None and d.items_after > d.items_before
        )
        result.items_after = result.items_before + delta
        result.vendor_key_after = result.vendor_key_before
        return result

    # Phase 2 -- apply. Delete the existing fact FIRST so its fact_items stop
    # referencing the old item atoms (FK), then swap atoms and re-collapse the
    # SAME cloud (preserves vendor_key + multi-doc membership, no re-formation).
    assert cloud_id is not None  # guaranteed: plans require item-bearing bundles
    v2_store.delete_fact(db, fact_id)
    for bundle_id, document_id, new_atoms, new_items, extraction, yaml_path in plans:
        _swap_item_atoms(db, bundle_id, new_atoms)
        if sync_yaml:
            _sync_yaml(db, document_id, yaml_path, extraction, new_items)

    new_fact_id = recollapse_cloud(db, cloud_id)
    result.new_fact_id = new_fact_id
    result.applied = True
    if new_fact_id:
        # recollapse_cloud re-derives vendor_key from the vendor atom via
        # make_vendor_key -- it does NOT apply the canonical-identity override
        # the ingestion pipeline uses (fact.vendor_key = get_canonical_vendor_key).
        # We left the vendor atoms untouched, so the fact's identity is unchanged;
        # restore the ORIGINAL key so the reconciliation from fuzzy vendor-identity
        # clustering (PR #87) is preserved rather than reverted to the raw key.
        with db.transaction() as cur:
            cur.execute(
                "UPDATE facts SET vendor_key = ? WHERE id = ?",
                (result.vendor_key_before, new_fact_id),
            )
        n1, sum1, _t, cov1 = _fact_coverage(db, new_fact_id)
        new_fact = v2_store.get_fact_by_id(db, new_fact_id)
        result.vendor_key_after = (new_fact or {}).get("vendor_key")
        result.items_after = n1
        result.item_sum_after = sum1
        result.coverage_after = cov1
    return result


# ---------------------------------------------------------------------------
# Recollapse-only remediation (no Gemini) -- re-apply collapse logic to
# already-collapsed facts so a fixed collapse rule (e.g. bundle-aware item
# de-duplication) is reflected in existing data, preserving the vendor_key.
# ---------------------------------------------------------------------------


@dataclass
class RecollapseResult:
    """Outcome of recollapsing one fact (no re-extraction)."""

    fact_id: str
    cloud_id: str | None = None
    vendor: str | None = None
    vendor_key_before: str | None = None
    vendor_key_after: str | None = None
    items_before: int = 0
    items_after: int = 0
    coverage_before: float = 0.0
    coverage_after: float = 0.0
    new_fact_id: str | None = None
    applied: bool = False
    error: str | None = None

    @property
    def vendor_key_preserved(self) -> bool:
        return self.vendor_key_before == self.vendor_key_after


def select_recollapse_candidates(db: DatabaseManager, limit: int = 50) -> list[str]:
    """Fact ids whose bundles contain within-bundle duplicate item lines.

    These are exactly the facts the bundle-aware de-duplication fix changes:
    a single bundle holding several identical (name, total_price, quantity)
    item atoms used to collapse to one fact_item. Read-only.
    """
    rows = db.fetchall(
        "SELECT f.id AS fact_id, "
        "       json_extract(a.data,'$.name') AS name, "
        "       json_extract(a.data,'$.total_price') AS price, "
        "       json_extract(a.data,'$.quantity') AS qty, "
        "       ba.bundle_id AS bundle_id, COUNT(*) AS n "
        "FROM facts f "
        "JOIN bundles b ON b.cloud_id = f.cloud_id "
        "JOIN bundle_atoms ba ON ba.bundle_id = b.id AND ba.role = 'basket_item' "
        "JOIN atoms a ON a.id = ba.atom_id "
        "WHERE f.fact_type = 'purchase' "
        "GROUP BY f.id, ba.bundle_id, name, price, qty "
        "HAVING n > 1 AND name IS NOT NULL AND price IS NOT NULL"
    )
    seen: list[str] = []
    for r in rows:
        fid = r["fact_id"]
        if fid not in seen:
            seen.append(fid)
        if len(seen) >= limit:
            break
    return seen


def select_overcount_candidates(
    db: DatabaseManager, ratio: float = 1.15, limit: int = 50
) -> list[str]:
    """Fact ids whose items over-sum the total in a multi-basket cloud (Type A).

    Two photos of one receipt left two BASKET bundles whose line items were
    summed, so ``item_sum`` exceeds the printed total. The collapse fix drops
    the duplicate basket by a (vendor, total) signature; re-collapsing these
    facts applies it. Restricted to clouds with >= 2 item-bearing bundles --
    a single-bundle over-count is a phantom OCR price (a separate data clean
    that re-collapse cannot fix). Worst ratio first. Read-only.
    """
    rows = db.fetchall(
        "SELECT f.id AS fact_id "
        "FROM facts f JOIN fact_items fi ON fi.fact_id = f.id "
        "WHERE f.fact_type = 'purchase' AND f.total_amount > 0 "
        "AND (SELECT COUNT(*) FROM bundles b "
        "     WHERE b.cloud_id = f.cloud_id "
        "     AND b.bundle_type IN ('basket', 'invoice')) >= 2 "
        "GROUP BY f.id "
        "HAVING SUM(fi.total_price) > f.total_amount * ? "
        "ORDER BY SUM(fi.total_price) / f.total_amount DESC",
        (ratio,),
    )
    return [r["fact_id"] for r in rows[:limit]]


def _simulate_collapse_item_count(db: DatabaseManager, cloud_id: str) -> int | None:
    """Run try_collapse in-memory (no DB write) and return its item count."""
    from alibi.clouds.collapse import try_collapse
    from alibi.db.models import Cloud, CloudStatus

    bundle_data = v2_store.get_cloud_bundle_data(db, cloud_id)
    if not bundle_data:
        return None
    result = try_collapse(Cloud(id=cloud_id, status=CloudStatus.FORMING), bundle_data)
    if not result.collapsed:
        return None
    return len(result.items)


def recollapse_fact(
    db: DatabaseManager,
    fact_id: str,
    apply: bool = False,
    allow_reduce: bool = False,
) -> RecollapseResult:
    """Re-collapse a fact's cloud to pick up collapse-rule fixes, keep its key.

    Dry-run (default) simulates the collapse in-memory and reports the projected
    item count without mutating. On ``apply`` the fact is deleted and the cloud
    re-collapsed (recollapse_cloud), then the original vendor_key is restored --
    only the line items change, the vendor identity does not.

    ``allow_reduce`` controls the safety gate. By default a re-collapse that
    would yield FEWER items than the fact has is skipped (the additive
    remediation for within-bundle dedup, where losing items would be a
    regression). For the duplicate-photo over-count remediation a reduction is
    exactly the fix -- pass ``allow_reduce=True`` to apply it.
    """
    fact = v2_store.get_fact_by_id(db, fact_id)
    if not fact:
        return RecollapseResult(fact_id=fact_id, error="fact not found")
    if fact.get("fact_type") != "purchase":
        return RecollapseResult(fact_id=fact_id, error="not a purchase fact")

    cloud_id = fact.get("cloud_id")
    n0, _s0, _t0, cov0 = _fact_coverage(db, fact_id)
    result = RecollapseResult(
        fact_id=fact_id,
        cloud_id=cloud_id,
        vendor=fact.get("vendor"),
        vendor_key_before=fact.get("vendor_key"),
        items_before=n0,
        coverage_before=cov0,
    )
    if not cloud_id:
        result.error = "fact has no cloud"
        return result

    projected = _simulate_collapse_item_count(db, cloud_id)
    projected = projected if projected is not None else n0

    if not apply:
        result.items_after = projected
        result.vendor_key_after = result.vendor_key_before
        return result

    # Safety: the additive remediation skips reductions (losing items in a
    # recovery pass is not worth the risk). The duplicate-photo over-count
    # remediation opts into reductions, where dropping the doubled basket's
    # items IS the fix.
    if projected < n0 and not allow_reduce:
        result.items_after = n0
        result.vendor_key_after = result.vendor_key_before
        result.error = f"would reduce {n0}->{projected}, skipped"
        return result

    v2_store.delete_fact(db, fact_id)
    new_fact_id = recollapse_cloud(db, cloud_id)
    result.new_fact_id = new_fact_id
    result.applied = True
    if new_fact_id:
        with db.transaction() as cur:
            cur.execute(
                "UPDATE facts SET vendor_key = ? WHERE id = ?",
                (result.vendor_key_before, new_fact_id),
            )
        n1, _s1, _t1, cov1 = _fact_coverage(db, new_fact_id)
        new_fact = v2_store.get_fact_by_id(db, new_fact_id)
        result.vendor_key_after = (new_fact or {}).get("vendor_key")
        result.items_after = n1
        result.coverage_after = cov1
    return result


# ---------------------------------------------------------------------------
# Date-split remediation -- undo Type-B mis-merges where formation clustered
# same-vendor/same-amount bundles from DIFFERENT days into one cloud. Splits
# each distinct-date basket group into its own cloud, routing same-date payment
# slips along, preserving the canonical vendor_key on same-vendor groups.
# ---------------------------------------------------------------------------

from datetime import date as _date  # noqa: E402
from alibi.normalizers.vendors import normalize_vendor_slug as _slug  # noqa: E402

_SPLIT_BUNDLE_TYPES = ("basket", "invoice", "payment_record")


@dataclass
class SplitResult:
    cloud_id: str
    vendor: str | None = None
    vendor_key_before: str | None = None
    dates: list[str] = field(default_factory=list)
    new_clouds: int = 0
    new_fact_ids: list[str] = field(default_factory=list)
    applied: bool = False
    skipped: str | None = None
    error: str | None = None


def _cloud_bundle_dates(db: DatabaseManager, cloud_id: str) -> list[Any]:
    """All bundles in a cloud with their document date (YYYY-MM-DD)."""
    return db.fetchall(
        "SELECT b.id AS bundle_id, b.bundle_type, "
        "substr(COALESCE(json_extract(d.raw_extraction,'$.date'), "
        "json_extract(d.raw_extraction,'$.document_date')),1,10) AS dt "
        "FROM bundles b JOIN documents d ON d.id = b.document_id "
        "WHERE b.cloud_id = ?",
        (cloud_id,),
    )


def _safe_date_groups(
    basket_dates: list[str | None], grace_days: int
) -> tuple[list[str] | None, str | None]:
    """Distinct sorted dates if it is SAFE to split, else (None, reason).

    Conservative: refuses to split when two basket dates are within
    ``grace_days`` (consecutive-day ambiguity) or share a month-day across
    different years (a likely OCR year misread of one receipt).
    """
    parsed: list[_date] = []
    for ds in basket_dates:
        if not ds:
            return None, "a basket has no date"
        try:
            parsed.append(_date.fromisoformat(ds))
        except (ValueError, TypeError):
            return None, f"unparseable date {ds!r}"
    distinct = sorted(set(parsed))
    if len(distinct) < 2:
        return None, "single date group"
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            d1, d2 = distinct[i], distinct[j]
            if abs((d2 - d1).days) <= grace_days:
                return None, f"dates {d1}/{d2} within {grace_days}d grace"
            if (d1.month, d1.day) == (d2.month, d2.day):
                return None, f"same month-day {d1}/{d2} (likely OCR year)"
    return [d.isoformat() for d in distinct], None


def select_date_split_candidates(
    db: DatabaseManager, grace_days: int = 3, limit: int = 50
) -> list[str]:
    """Cloud ids with >=2 baskets spanning safely-distinct dates (Type-B)."""
    rows = db.fetchall(
        "SELECT b.cloud_id AS cloud_id, "
        "substr(COALESCE(json_extract(d.raw_extraction,'$.date'), "
        "json_extract(d.raw_extraction,'$.document_date')),1,10) AS dt "
        "FROM bundles b JOIN documents d ON d.id = b.document_id "
        "WHERE b.bundle_type = 'basket'"
    )
    by_cloud: dict[str, list[str | None]] = {}
    for r in rows:
        by_cloud.setdefault(r["cloud_id"], []).append(r["dt"])
    out: list[str] = []
    for cid, dates in by_cloud.items():
        if len(dates) < 2:
            continue
        groups, _reason = _safe_date_groups(dates, grace_days)
        if groups is not None:
            out.append(cid)
        if len(out) >= limit:
            break
    return out


def split_cloud_by_date(
    db: DatabaseManager, cloud_id: str, apply: bool = False, grace_days: int = 3
) -> SplitResult:
    """Split a date-mis-merged cloud into one cloud per distinct basket date.

    The earliest date group stays in the original cloud; each later group's
    bundles (baskets + same-date slips) move to a new cloud (move_bundle
    re-collapses both). The canonical vendor_key is restored on each resulting
    fact whose vendor matches the original (a cross-vendor mis-merge lets the
    moved fact keep its own derived key).
    """
    from alibi.services.correction import move_bundle

    orig = v2_store.get_fact_for_cloud(db, cloud_id)
    result = SplitResult(
        cloud_id=cloud_id,
        vendor=(orig or {}).get("vendor"),
        vendor_key_before=(orig or {}).get("vendor_key"),
    )
    bundles = _cloud_bundle_dates(db, cloud_id)
    baskets = [b for b in bundles if b["bundle_type"] in ("basket", "invoice")]
    groups, reason = _safe_date_groups([b["dt"] for b in baskets], grace_days)
    if groups is None:
        result.skipped = reason
        return result
    result.dates = groups

    if not apply:
        return result

    orig_slug = _slug(result.vendor or "")
    orig_key = result.vendor_key_before

    def _restore_key_if_same_vendor(fid: str | None) -> None:
        if not fid or orig_key is None:
            return
        f = v2_store.get_fact_by_id(db, fid)
        if f and _slug(f.get("vendor") or "") == orig_slug:
            with db.transaction() as cur:
                cur.execute(
                    "UPDATE facts SET vendor_key = ? WHERE id = ?", (orig_key, fid)
                )

    # Earliest group anchors the original cloud; move the rest out.
    for grp_date in groups[1:]:
        movers = [
            b
            for b in bundles
            if b["dt"] == grp_date and b["bundle_type"] in _SPLIT_BUNDLE_TYPES
        ]
        target: str | None = None
        for mb in movers:
            res = move_bundle(db, mb["bundle_id"], target)
            if not res.success:
                result.error = f"move failed for {mb['bundle_id'][:8]}: {res.error}"
                return result
            target = res.target_cloud_id
        if target:
            result.new_clouds += 1
            nf = v2_store.get_fact_for_cloud(db, target)
            if nf:
                _restore_key_if_same_vendor(nf["id"])
                result.new_fact_ids.append(nf["id"])

    # The anchor cloud re-collapsed as bundles left; restore its canonical key.
    af = v2_store.get_fact_for_cloud(db, cloud_id)
    if af:
        _restore_key_if_same_vendor(af["id"])

    result.applied = True
    return result
