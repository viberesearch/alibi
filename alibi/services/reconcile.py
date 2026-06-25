"""Price reconciliation -- repair item totals that don't sum to the receipt.

A recurring OCR/structuring defect on weighed grocery lines: the printed second
number on a line is the LINE TOTAL, but the structurer reads it as a unit price
and stores ``total_price = quantity x unit_price`` -- inflating the line (e.g.
"MELI 4.00 5.10" becomes 4 x 5.10 = 20.40 instead of 5.10). Both the local model
and the cloud fallback make this mistake, because the OCR text itself is
column-ambiguous, so it cannot be fixed reliably at parse time.

It CAN be repaired afterwards, because the receipt carries an authoritative
checksum the line items must sum to: the printed total. For each item we know
two or three candidate line totals -- the stored ``total_price``, the
``unit_price`` (when the printed number was actually the total), and
``total_price / quantity`` (undoing the multiply). We search for the assignment
whose sum reconciles to the printed total.

The gate is deliberately strict. A line total is GROSS (VAT included), as is the
receipt total, so a correct assignment reconciles to within a cent or two. We
auto-apply ONLY when a single assignment lands inside a tight tolerance; anything
looser -- a residual gap (a missing or spurious line), or two different
assignments that both fit -- is reported as REVIEW rather than guessed. The grand
total alone has just enough VAT-rounding slack that a loose match could pick wrong
line values that coincidentally sum closer, so we refuse to apply those.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from itertools import product
from typing import Any

from alibi.clouds.correction import recollapse_cloud
from alibi.db import v2_store
from alibi.db.connection import DatabaseManager

_CENT = Decimal("0.01")

# Acceptance tolerance: a correct assignment of gross line totals reconciles to
# the gross receipt total within rounding. Looser gaps are left for review.
_ABS_TOL = Decimal("0.05")
_PCT_TOL = Decimal("0.006")  # 0.6% of the receipt total

# Search guards -- a fact with too many ambiguous lines is reported, not solved.
_MAX_FREE_LINES = 16
_MAX_COMBINATIONS = 100_000


def _dec(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value)).quantize(_CENT)
    except (InvalidOperation, ValueError):
        return None


@dataclass
class ReconcileLine:
    """One item line, as the reconciler sees it."""

    atom_id: str
    name: str
    quantity: Decimal
    unit_price: Decimal | None
    total_price: Decimal | None

    def candidates(self) -> list[Decimal]:
        """Distinct candidate line totals, current value first.

        The current ``total_price`` plus the two ways it is commonly wrong: the
        ``unit_price`` (the printed number was the line total, not the unit) and
        ``total_price / quantity`` (the structurer multiplied a total by qty).
        """
        base = self.total_price if self.total_price is not None else Decimal("0")
        cands = [base]
        if self.unit_price is not None and self.unit_price != base:
            cands.append(self.unit_price)
        if self.total_price is not None and self.quantity not in (
            Decimal("0"),
            Decimal("1"),
        ):
            undo = (self.total_price / self.quantity).quantize(_CENT)
            if undo not in cands:
                cands.append(undo)
        return cands


@dataclass
class LineFix:
    """A proposed change to one line's total (and derived unit price)."""

    atom_id: str
    name: str
    old_total: Decimal | None
    new_total: Decimal
    new_unit: Decimal | None


@dataclass
class ReconcilePlan:
    """The reconciler's verdict for one fact."""

    reconciles: bool
    reason: str
    printed_total: Decimal
    old_sum: Decimal
    new_sum: Decimal
    fixes: list[LineFix] = field(default_factory=list)


def plan_reconcile(
    lines: list[ReconcileLine],
    printed_total: Decimal,
    *,
    abs_tol: Decimal = _ABS_TOL,
    pct_tol: Decimal = _PCT_TOL,
    max_free: int = _MAX_FREE_LINES,
    max_combos: int = _MAX_COMBINATIONS,
) -> ReconcilePlan:
    """Find the line-total assignment that reconciles to ``printed_total``.

    Returns ``reconciles=True`` only when exactly one assignment lands within
    tolerance (a unique, confident repair). Multiple fitting assignments, none
    fitting, or too many ambiguous lines all yield ``reconciles=False`` with a
    reason -- those are for a human, not an automatic guess.
    """
    old_sum = sum(
        (ln.total_price or Decimal("0") for ln in lines), Decimal("0")
    ).quantize(_CENT)
    plan = ReconcilePlan(
        reconciles=False,
        reason="",
        printed_total=printed_total,
        old_sum=old_sum,
        new_sum=old_sum,
    )

    per_line = [(ln, ln.candidates()) for ln in lines]
    free = [(ln, cs) for ln, cs in per_line if len(cs) > 1]
    fixed_sum = sum((cs[0] for ln, cs in per_line if len(cs) == 1), Decimal("0"))

    if not free:
        plan.reason = "no adjustable lines"
        return plan
    if len(free) > max_free:
        plan.reason = f"too ambiguous ({len(free)} free lines)"
        return plan

    n_combos = 1
    for _ln, cs in free:
        n_combos *= len(cs)
    if n_combos > max_combos:
        plan.reason = f"too many combinations ({n_combos})"
        return plan

    tol = max(abs_tol, (pct_tol * printed_total).quantize(_CENT))

    best_dist: Decimal | None = None
    passing: list[tuple[Decimal, ...]] = []
    for combo in product(*(cs for _ln, cs in free)):
        total = fixed_sum + sum(combo, Decimal("0"))
        dist = abs(total - printed_total)
        if best_dist is None or dist < best_dist:
            best_dist = dist
        if dist <= tol:
            passing.append(combo)

    if not passing:
        plan.reason = f"best off by {best_dist} (> {tol})"
        return plan
    if len(passing) > 1:
        plan.reason = f"ambiguous: {len(passing)} assignments within {tol}"
        return plan

    combo = passing[0]
    fixes: list[LineFix] = []
    new_sum = fixed_sum
    for (ln, _cs), chosen in zip(free, combo):
        new_sum += chosen
        if chosen != (ln.total_price or Decimal("0")):
            new_unit = (
                (chosen / ln.quantity).quantize(_CENT)
                if ln.quantity not in (Decimal("0"),)
                else chosen
            )
            fixes.append(
                LineFix(
                    atom_id=ln.atom_id,
                    name=ln.name,
                    old_total=ln.total_price,
                    new_total=chosen,
                    new_unit=new_unit,
                )
            )
    if not fixes:
        plan.reason = "already reconciled"
        return plan

    plan.reconciles = True
    plan.reason = f"reconciled to within {best_dist}"
    plan.new_sum = new_sum.quantize(_CENT)
    plan.fixes = fixes
    return plan


# ---------------------------------------------------------------------------
# Fact-level application
# ---------------------------------------------------------------------------


@dataclass
class ReconcileResult:
    """Outcome of reconciling one fact."""

    fact_id: str
    vendor: str | None = None
    vendor_key_before: str | None = None
    vendor_key_after: str | None = None
    total: Decimal = Decimal("0")
    item_sum_before: Decimal = Decimal("0")
    item_sum_after: Decimal = Decimal("0")
    n_fixes: int = 0
    reconciles: bool = False
    reason: str = ""
    new_fact_id: str | None = None
    applied: bool = False
    error: str | None = None

    @property
    def vendor_key_preserved(self) -> bool:
        return self.vendor_key_before == self.vendor_key_after


_ITEM_BUNDLE_TYPES = ("basket", "invoice")


def _fact_item_lines(db: DatabaseManager, cloud_id: str) -> list[ReconcileLine]:
    """The item atoms of a cloud's item-bearing bundles, as reconcile lines."""
    placeholders = ",".join("?" for _ in _ITEM_BUNDLE_TYPES)
    rows = db.fetchall(
        f"SELECT a.id AS atom_id, "
        f"  json_extract(a.data,'$.name') AS name, "
        f"  json_extract(a.data,'$.quantity') AS quantity, "
        f"  json_extract(a.data,'$.unit_price') AS unit_price, "
        f"  json_extract(a.data,'$.total_price') AS total_price "
        f"FROM bundles b "
        f"JOIN bundle_atoms ba ON ba.bundle_id = b.id AND ba.role = 'basket_item' "
        f"JOIN atoms a ON a.id = ba.atom_id "
        f"WHERE b.cloud_id = ? AND b.bundle_type IN ({placeholders})",
        (cloud_id, *_ITEM_BUNDLE_TYPES),
    )
    lines: list[ReconcileLine] = []
    for r in rows:
        if r["total_price"] is None:
            continue  # nothing to reconcile on a price-less line
        qty = _dec(r["quantity"]) or Decimal("1")
        lines.append(
            ReconcileLine(
                atom_id=r["atom_id"],
                name=r["name"] or "",
                quantity=qty if qty > 0 else Decimal("1"),
                unit_price=_dec(r["unit_price"]),
                total_price=_dec(r["total_price"]),
            )
        )
    return lines


def reconcile_fact(
    db: DatabaseManager, fact_id: str, apply: bool = False
) -> ReconcileResult:
    """Reconcile one fact's item totals to its printed total.

    Dry-run (default) computes the plan without mutating. On ``apply`` -- only
    when the plan reconciles -- the flagged atoms' ``total_price``/``unit_price``
    are corrected, the cloud re-collapsed, and the original vendor_key restored.
    """
    fact = v2_store.get_fact_by_id(db, fact_id)
    if not fact:
        return ReconcileResult(fact_id=fact_id, error="fact not found")
    if fact.get("fact_type") != "purchase":
        return ReconcileResult(fact_id=fact_id, error="not a purchase fact")

    total = _dec(fact.get("total_amount"))
    cloud_id = fact.get("cloud_id")
    result = ReconcileResult(
        fact_id=fact_id,
        vendor=fact.get("vendor"),
        vendor_key_before=fact.get("vendor_key"),
        total=total or Decimal("0"),
    )
    if total is None or total <= 0:
        result.error = "fact has no usable total"
        return result
    if not cloud_id:
        result.error = "fact has no cloud"
        return result

    lines = _fact_item_lines(db, cloud_id)
    if not lines:
        result.error = "no item-bearing bundle"
        return result

    plan = plan_reconcile(lines, total)
    result.item_sum_before = plan.old_sum
    result.item_sum_after = plan.new_sum
    result.reconciles = plan.reconciles
    result.reason = plan.reason
    result.n_fixes = len(plan.fixes)

    if not plan.reconciles or not apply:
        result.vendor_key_after = result.vendor_key_before
        return result

    # Apply: delete the fact first (so fact_items stop referencing the atoms),
    # correct each flagged atom, then re-collapse and restore the vendor_key.
    v2_store.delete_fact(db, fact_id)
    with db.transaction() as cur:
        for fix in plan.fixes:
            cur.execute(
                "UPDATE atoms SET data = json_set("
                "  json_set(data, '$.total_price', ?), '$.unit_price', ?) "
                "WHERE id = ?",
                (
                    float(fix.new_total),
                    float(fix.new_unit or fix.new_total),
                    fix.atom_id,
                ),
            )
    new_fact_id = recollapse_cloud(db, cloud_id)
    result.new_fact_id = new_fact_id
    result.applied = True
    if new_fact_id:
        with db.transaction() as cur:
            cur.execute(
                "UPDATE facts SET vendor_key = ? WHERE id = ?",
                (result.vendor_key_before, new_fact_id),
            )
        nf = v2_store.get_fact_by_id(db, new_fact_id)
        result.vendor_key_after = (nf or {}).get("vendor_key")
        row = db.fetchone(
            "SELECT COALESCE(SUM(total_price), 0) AS s FROM fact_items WHERE fact_id = ?",
            (new_fact_id,),
        )
        result.item_sum_after = (_dec(row["s"]) if row else None) or Decimal("0")
    return result


def select_overcount_facts(
    db: DatabaseManager, ratio: float = 1.15, limit: int = 100
) -> list[str]:
    """Fact ids whose item totals over-sum the receipt total (any cloud shape).

    Unlike the duplicate-photo selector this is not restricted to multi-basket
    clouds -- a single-bundle receipt whose weighed lines were qty x unit
    inflated is exactly what reconciliation targets. Worst ratio first.
    """
    rows = db.fetchall(
        "SELECT f.id AS fact_id "
        "FROM facts f JOIN fact_items fi ON fi.fact_id = f.id "
        "WHERE f.fact_type = 'purchase' AND f.total_amount > 0 "
        "GROUP BY f.id "
        "HAVING SUM(fi.total_price) > f.total_amount * ? "
        "ORDER BY SUM(fi.total_price) / f.total_amount DESC",
        (ratio,),
    )
    return [r["fact_id"] for r in rows[:limit]]
