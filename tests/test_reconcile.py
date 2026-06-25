"""Tests for price reconciliation (alibi/services/reconcile.py).

The solver repairs item totals that over-sum the receipt total -- the weighed-
line qty x unit_price double-multiply -- but only when a single assignment fits
tightly. Ambiguous or loose fits are left for review, never guessed.
"""

from __future__ import annotations

from decimal import Decimal

from alibi.clouds.correction import recollapse_cloud
from alibi.db import v2_store
from alibi.db.connection import DatabaseManager
from alibi.db.models import BundleType
from alibi.services import reconcile as rc

# Reuse the synthetic seeding helpers from the reextract suite.
from tests.test_reextract import _make_atoms, _make_bundle, _make_cloud, _make_document


def _line(name, qty, unit, total):
    return rc.ReconcileLine(
        atom_id=name,
        name=name,
        quantity=Decimal(str(qty)),
        unit_price=Decimal(str(unit)) if unit is not None else None,
        total_price=Decimal(str(total)) if total is not None else None,
    )


# ---------------------------------------------------------------------------
# Pure solver
# ---------------------------------------------------------------------------


class TestPlanReconcile:
    def test_clean_double_multiply_reconciles(self):
        # B was qty 2 x unit 4.00 = 8.00; the real line total is 4.00.
        lines = [
            _line("A", 1, 3.00, 3.00),  # correct
            _line("B", 2, 4.00, 8.00),  # double-multiplied
        ]
        plan = rc.plan_reconcile(lines, Decimal("7.00"))
        assert plan.reconciles
        assert len(plan.fixes) == 1
        fix = plan.fixes[0]
        assert fix.atom_id == "B"
        assert fix.new_total == Decimal("4.00")
        assert fix.new_unit == Decimal("2.00")  # 4.00 / qty 2
        assert plan.new_sum == Decimal("7.00")

    def test_undo_multiply_candidate_when_unit_absent(self):
        # No unit_price, but total / quantity recovers the real line total.
        lines = [
            _line("A", 1, None, 5.00),
            _line("B", 4, None, 20.00),  # real total 5.00 = 20.00 / 4
        ]
        plan = rc.plan_reconcile(lines, Decimal("10.00"))
        assert plan.reconciles
        assert plan.fixes[0].new_total == Decimal("5.00")

    def test_ambiguous_two_assignments_reviewed(self):
        # 8+4 and 4+8 both total 12.00 with different per-line picks.
        lines = [
            _line("A", 2, 4.00, 8.00),
            _line("B", 2, 4.00, 8.00),
        ]
        plan = rc.plan_reconcile(lines, Decimal("12.00"))
        assert not plan.reconciles
        assert "ambiguous" in plan.reason

    def test_no_adjustable_lines_reviewed(self):
        # Every line has a single candidate; nothing to try.
        lines = [_line("A", 1, 3.00, 3.00), _line("B", 1, 2.00, 2.00)]
        plan = rc.plan_reconcile(lines, Decimal("9.00"))
        assert not plan.reconciles
        assert plan.reason == "no adjustable lines"

    def test_no_fit_within_tolerance_reviewed(self):
        lines = [_line("A", 2, 4.00, 8.00), _line("B", 1, 1.00, 1.00)]
        plan = rc.plan_reconcile(lines, Decimal("50.00"))
        assert not plan.reconciles
        assert "best off by" in plan.reason

    def test_already_correct_is_not_a_fix(self):
        # Sum already matches: the only assignment that fits is the current one.
        lines = [_line("A", 1, 3.00, 3.00), _line("B", 2, 2.00, 4.00)]
        plan = rc.plan_reconcile(lines, Decimal("7.00"))
        # B's candidates are {4.00, 2.00}; 3+4=7 (current) fits, 3+2=5 does not.
        assert not plan.reconciles  # no change needed -> not an applied fix
        assert plan.reason in ("already reconciled", "no adjustable lines")

    def test_tolerance_allows_small_vat_rounding(self):
        # Real total 6.99 vs printed 7.00 -> within 0.6% tolerance.
        lines = [_line("A", 1, 2.99, 2.99), _line("B", 2, 2.00, 8.00)]
        plan = rc.plan_reconcile(lines, Decimal("7.00"))
        assert plan.reconciles
        assert plan.fixes[0].new_total == Decimal("4.00")  # 2.99 + 4.00 = 6.99

    def test_too_many_free_lines_reviewed(self):
        lines = [_line(f"L{i}", 2, 1.00, 2.00) for i in range(20)]
        plan = rc.plan_reconcile(lines, Decimal("20.00"))
        assert not plan.reconciles
        assert "too ambiguous" in plan.reason


# ---------------------------------------------------------------------------
# DB-backed application
# ---------------------------------------------------------------------------


def _seed_overcount_weighed_fact(db: DatabaseManager) -> tuple[str, str]:
    """A single-basket fact with one qty x unit double-multiplied line.

    Items: Milk 3.00 (correct) + Apples qty 4 stored as 4 x 5.10 = 20.40 whose
    real line total is 5.10. Receipt total 8.10. Returns (fact_id, cloud_id).
    """
    doc_id = _make_document(db)
    atoms = _make_atoms(
        db, doc_id, amount=8.10, items=[("Milk", 3.00), ("Apples", 20.40)]
    )
    # Stamp Apples with qty 4 and unit_price 5.10 (the misread line total).
    import json as _j

    for a in atoms:
        if a.atom_type.value == "item" and a.data.get("name") == "Apples":
            a.data["quantity"] = "4"
            a.data["unit_price"] = "5.10"
            with db.transaction() as cur:
                cur.execute(
                    "UPDATE atoms SET data = ? WHERE id = ?",
                    (_j.dumps(a.data), a.id),
                )
    bundle_id = _make_bundle(db, doc_id, atoms, BundleType.BASKET)
    cloud_id = _make_cloud(db, bundle_id)
    fact_id = recollapse_cloud(db, cloud_id)
    assert fact_id
    return fact_id, cloud_id


def test_select_overcount_facts_finds_single_bundle(db: DatabaseManager) -> None:
    fact_id, _ = _seed_overcount_weighed_fact(db)
    assert fact_id in rc.select_overcount_facts(db)


def test_reconcile_fact_repairs_and_preserves_key(db: DatabaseManager) -> None:
    fact_id, _ = _seed_overcount_weighed_fact(db)
    before = v2_store.get_fact_by_id(db, fact_id)
    key_before = before["vendor_key"]
    assert key_before

    # Dry run mutates nothing.
    dry = rc.reconcile_fact(db, fact_id, apply=False)
    assert dry.reconciles
    assert dry.item_sum_before == Decimal("23.40")  # 3.00 + 20.40
    assert v2_store.get_fact_by_id(db, fact_id) is not None

    res = rc.reconcile_fact(db, fact_id, apply=True)
    assert res.applied and res.reconciles
    assert res.vendor_key_preserved
    assert res.item_sum_after == Decimal("8.10")  # 3.00 + 5.10
    new_fact = v2_store.get_fact_by_id(db, res.new_fact_id)
    assert new_fact["vendor_key"] == key_before


def test_reconcile_fact_balanced_is_noop(db: DatabaseManager) -> None:
    doc_id = _make_document(db)
    atoms = _make_atoms(db, doc_id, amount=5.00, items=[("Bread", 5.00)])
    bundle_id = _make_bundle(db, doc_id, atoms, BundleType.BASKET)
    cloud_id = _make_cloud(db, bundle_id)
    fact_id = recollapse_cloud(db, cloud_id)
    res = rc.reconcile_fact(db, fact_id, apply=True)
    assert not res.reconciles
    assert not res.applied
