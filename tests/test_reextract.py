"""Tests for merge-preserving re-extraction (alibi/services/reextract.py).

Seeds synthetic documents/atoms/bundles/clouds programmatically (no fixtures,
no live Gemini), creates the initial fact via ``recollapse_cloud`` so the
vendor_key is collapse-derived, then re-extracts with a stub structurer and
asserts:

* richer items land in the SAME fact lineage (cloud unchanged),
* the reconciled vendor_key is preserved (never re-split),
* multi-document collapse (receipt + payment slip) survives,
* dry-run mutates nothing and a no-improvement pass is skipped.
"""

from decimal import Decimal
from uuid import uuid4

from alibi.clouds.correction import recollapse_cloud
from alibi.db import v2_store
from alibi.db.connection import DatabaseManager
from alibi.db.models import (
    Atom,
    AtomType,
    Bundle,
    BundleAtom,
    BundleAtomRole,
    BundleType,
    Cloud,
    CloudBundle,
    CloudMatchType,
    CloudStatus,
    Document,
)
from alibi.services import reextract as rx

# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------

_RAW_TEXT = (
    "BIG STORE LTD\nVAT 10033253D\nMilk 3.00\nBread 4.00\nEggs 3.00\nTOTAL 10.00"
)


def _make_document(
    db: DatabaseManager,
    *,
    raw_text: str = _RAW_TEXT,
    line_items: list[dict] | None = None,
    doc_type: str = "receipt",
) -> str:
    doc_id = str(uuid4())
    doc = Document(
        id=doc_id,
        file_path=f"{doc_id}.jpg",
        file_hash=str(uuid4())[:16],
        raw_extraction={
            "vendor": "Big Store Ltd",
            "currency": "EUR",
            "language": "en",
            "document_type": doc_type,
            "raw_text": raw_text,
            "line_items": line_items or [],
        },
    )
    v2_store.store_document(db, doc)
    return doc_id


def _make_atoms(
    db: DatabaseManager,
    document_id: str,
    *,
    vendor: str = "Big Store Ltd",
    tax_id: str = "10033253D",
    amount: float = 10.0,
    items: list[tuple[str, float]] | None = None,
) -> list[Atom]:
    atoms = [
        Atom(
            id=str(uuid4()),
            document_id=document_id,
            atom_type=AtomType.VENDOR,
            data={"name": vendor, "tax_id": tax_id},
        ),
        Atom(
            id=str(uuid4()),
            document_id=document_id,
            atom_type=AtomType.AMOUNT,
            data={"value": amount, "currency": "EUR", "semantic_type": "total"},
        ),
        Atom(
            id=str(uuid4()),
            document_id=document_id,
            atom_type=AtomType.DATETIME,
            data={"value": "2025-03-15T10:30:00"},
        ),
    ]
    for name, price in items or []:
        atoms.append(
            Atom(
                id=str(uuid4()),
                document_id=document_id,
                atom_type=AtomType.ITEM,
                data={
                    "name": name,
                    "quantity": "1",
                    "unit": "pcs",
                    "total_price": str(price),
                    "currency": "EUR",
                },
            )
        )
    v2_store.store_atoms(db, atoms)
    return atoms


def _make_bundle(
    db: DatabaseManager,
    document_id: str,
    atoms: list[Atom],
    bundle_type: BundleType = BundleType.BASKET,
) -> str:
    bundle_id = str(uuid4())
    bundle = Bundle(id=bundle_id, document_id=document_id, bundle_type=bundle_type)
    role_map = {
        AtomType.VENDOR: BundleAtomRole.VENDOR_INFO,
        AtomType.AMOUNT: BundleAtomRole.TOTAL,
        AtomType.DATETIME: BundleAtomRole.EVENT_TIME,
        AtomType.ITEM: BundleAtomRole.BASKET_ITEM,
        AtomType.PAYMENT: BundleAtomRole.PAYMENT_INFO,
    }
    bundle_atoms = [
        BundleAtom(
            bundle_id=bundle_id,
            atom_id=a.id,
            role=role_map.get(a.atom_type, BundleAtomRole.BASKET_ITEM),
        )
        for a in atoms
    ]
    v2_store.store_bundle(db, bundle, bundle_atoms)
    return bundle_id


def _make_cloud(db: DatabaseManager, bundle_id: str) -> str:
    cloud_id = str(uuid4())
    cloud = Cloud(id=cloud_id, status=CloudStatus.FORMING)
    link = CloudBundle(
        cloud_id=cloud_id,
        bundle_id=bundle_id,
        match_type=CloudMatchType.EXACT_AMOUNT,
        match_confidence=Decimal("1.0"),
    )
    v2_store.store_cloud(db, cloud, link)
    return cloud_id


def _add_bundle_to_cloud(db: DatabaseManager, cloud_id: str, bundle_id: str) -> None:
    v2_store.add_cloud_bundle(
        db,
        CloudBundle(
            cloud_id=cloud_id,
            bundle_id=bundle_id,
            match_type=CloudMatchType.EXACT_AMOUNT,
            match_confidence=Decimal("1.0"),
        ),
    )
    with db.transaction() as cur:
        cur.execute(
            "UPDATE bundles SET cloud_id = ? WHERE id = ?", (cloud_id, bundle_id)
        )


def _seed_under_extracted_fact(
    db: DatabaseManager, *, n_seed_items: int = 1, amount: float = 10.0
) -> tuple[str, str, str]:
    """Build a 1-item under-extracted purchase fact. Returns (fact_id, cloud_id, doc_id)."""
    doc_id = _make_document(db)
    seed_items = [("Milk", 3.0)][:n_seed_items]
    atoms = _make_atoms(db, doc_id, amount=amount, items=seed_items)
    bundle_id = _make_bundle(db, doc_id, atoms, BundleType.BASKET)
    cloud_id = _make_cloud(db, bundle_id)
    fact_id = recollapse_cloud(db, cloud_id)
    assert fact_id, "collapse should produce a fact"
    return fact_id, cloud_id, doc_id


def _richer_structurer(raw_text: str, doc_type: str) -> dict:
    """Stub Stage-3: recovers all three line items from the cached OCR."""
    return {
        "line_items": [
            {"name": "Milk", "quantity": "1", "total_price": "3.00"},
            {"name": "Bread", "quantity": "1", "total_price": "4.00"},
            {"name": "Eggs", "quantity": "1", "total_price": "3.00"},
        ]
    }


def _poorer_structurer(raw_text: str, doc_type: str) -> dict:
    """Stub Stage-3 that recovers no more than already present."""
    return {"line_items": [{"name": "Milk", "quantity": "1", "total_price": "3.00"}]}


# ---------------------------------------------------------------------------
# Apply: recovers items, preserves vendor_key + cloud
# ---------------------------------------------------------------------------


def test_reextract_recovers_items_preserving_key(db: DatabaseManager) -> None:
    fact_id, cloud_id, _doc = _seed_under_extracted_fact(db)

    before = v2_store.get_fact_by_id(db, fact_id)
    key_before = before["vendor_key"]
    assert key_before, "seeded fact should have a real (tax-id-derived) vendor_key"

    res = rx.reextract_fact(
        db, fact_id, apply=True, structurer=_richer_structurer, sync_yaml=False
    )

    assert res.applied
    assert res.error is None
    assert res.items_before == 1
    assert res.items_after == 3
    # Same cloud (no re-split / re-formation)
    assert res.cloud_id == cloud_id
    new_fact = v2_store.get_fact_by_id(db, res.new_fact_id)
    assert new_fact is not None
    assert new_fact["cloud_id"] == cloud_id
    # vendor_key preserved by construction (vendor atom untouched)
    assert res.vendor_key_preserved
    assert new_fact["vendor_key"] == key_before
    # coverage materially improved
    assert res.coverage_after > res.coverage_before


def test_reextract_restores_canonical_vendor_key(db: DatabaseManager) -> None:
    """A canonical key from identity clustering must survive re-collapse.

    recollapse_cloud re-derives the key from the vendor atom (raw key); the
    ingestion pipeline instead overrides it with the canonical identity key.
    Simulate that override, then assert re-extraction restores the canonical
    key rather than reverting to the collapse-derived one.
    """
    fact_id, cloud_id, _doc = _seed_under_extracted_fact(db)

    # Simulate the pipeline's canonical-identity override (PR #87): a key that
    # is NOT what make_vendor_key would derive from the vendor atom's tax_id.
    canonical = "CANON-CLUSTERED-KEY"
    with db.transaction() as cur:
        cur.execute(
            "UPDATE facts SET vendor_key = ? WHERE id = ?", (canonical, fact_id)
        )

    res = rx.reextract_fact(
        db, fact_id, apply=True, structurer=_richer_structurer, sync_yaml=False
    )

    assert res.applied
    assert res.vendor_key_before == canonical
    assert res.vendor_key_preserved
    new_fact = v2_store.get_fact_by_id(db, res.new_fact_id)
    assert new_fact["vendor_key"] == canonical  # restored, not the raw tax-id key
    assert new_fact["vendor_key"] != "10033253D"


def test_reextract_new_fact_has_three_items(db: DatabaseManager) -> None:
    fact_id, cloud_id, _doc = _seed_under_extracted_fact(db)
    res = rx.reextract_fact(
        db, fact_id, apply=True, structurer=_richer_structurer, sync_yaml=False
    )
    rows = db.fetchall(
        "SELECT name FROM fact_items WHERE fact_id = ? ORDER BY name",
        (res.new_fact_id,),
    )
    names = {r["name"] for r in rows}
    assert {"Bread", "Eggs", "Milk"} <= names


# ---------------------------------------------------------------------------
# Multi-document collapse (receipt + payment slip) survives
# ---------------------------------------------------------------------------


def test_reextract_preserves_multi_doc_collapse(db: DatabaseManager) -> None:
    # Receipt basket (under-extracted) + payment slip in the SAME cloud.
    receipt_doc = _make_document(db)
    r_atoms = _make_atoms(db, receipt_doc, amount=10.0, items=[("Milk", 3.0)])
    receipt_bundle = _make_bundle(db, receipt_doc, r_atoms, BundleType.BASKET)
    cloud_id = _make_cloud(db, receipt_bundle)

    slip_doc = _make_document(db, doc_type="payment_confirmation")
    s_atoms = _make_atoms(db, slip_doc, amount=10.0, items=None)
    slip_bundle = _make_bundle(db, slip_doc, s_atoms, BundleType.PAYMENT_RECORD)
    _add_bundle_to_cloud(db, cloud_id, slip_bundle)

    fact_id = recollapse_cloud(db, cloud_id)
    assert fact_id

    res = rx.reextract_fact(
        db, fact_id, apply=True, structurer=_richer_structurer, sync_yaml=False
    )

    assert res.applied
    assert res.items_after == 3
    assert res.cloud_id == cloud_id
    # The payment-slip bundle is STILL in the cloud (membership preserved).
    bundle_ids = set(v2_store.get_bundles_in_cloud(db, cloud_id))
    assert slip_bundle in bundle_ids
    assert receipt_bundle in bundle_ids
    assert res.vendor_key_preserved


# ---------------------------------------------------------------------------
# Dry-run: no mutation
# ---------------------------------------------------------------------------


def test_reextract_dry_run_no_mutation(db: DatabaseManager) -> None:
    fact_id, cloud_id, _doc = _seed_under_extracted_fact(db)
    items_before = db.fetchone(
        "SELECT COUNT(*) AS n FROM fact_items WHERE fact_id = ?", (fact_id,)
    )["n"]

    res = rx.reextract_fact(
        db, fact_id, apply=False, structurer=_richer_structurer, sync_yaml=False
    )

    assert res.applied is False
    assert res.would_change is True
    # Projection reflects the richer extraction...
    assert res.items_after == 3
    # ...but nothing in the DB changed.
    items_after = db.fetchone(
        "SELECT COUNT(*) AS n FROM fact_items WHERE fact_id = ?", (fact_id,)
    )["n"]
    assert items_after == items_before == 1
    assert v2_store.get_fact_by_id(db, fact_id) is not None  # fact untouched


# ---------------------------------------------------------------------------
# No-improvement pass is skipped (never regress)
# ---------------------------------------------------------------------------


def test_reextract_skips_no_improvement(db: DatabaseManager) -> None:
    fact_id, cloud_id, _doc = _seed_under_extracted_fact(db)
    res = rx.reextract_fact(
        db, fact_id, apply=True, structurer=_poorer_structurer, sync_yaml=False
    )
    # Nothing improved → no new fact, original kept intact.
    assert res.new_fact_id is None
    assert res.would_change is False
    assert res.items_after == res.items_before == 1
    assert any("no improvement" in (d.skipped or "") for d in res.documents)
    items_after = db.fetchone(
        "SELECT COUNT(*) AS n FROM fact_items WHERE fact_id = ?", (fact_id,)
    )["n"]
    assert items_after == 1


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_reextract_unknown_fact(db: DatabaseManager) -> None:
    res = rx.reextract_fact(db, "does-not-exist", structurer=_richer_structurer)
    assert res.error == "fact not found"
    assert res.applied is False


# ---------------------------------------------------------------------------
# Queue selection
# ---------------------------------------------------------------------------


def _seed_fact_with_repeated_atoms(db: DatabaseManager) -> tuple[str, str]:
    """A fact whose bundle holds 3 identical 'Activia 1.80' item atoms."""
    doc_id = _make_document(db)
    atoms = _make_atoms(
        db,
        doc_id,
        amount=10.0,
        items=[("Activia", 1.8), ("Activia", 1.8), ("Activia", 1.8)],
    )
    bundle_id = _make_bundle(db, doc_id, atoms, BundleType.BASKET)
    cloud_id = _make_cloud(db, bundle_id)
    fact_id = recollapse_cloud(db, cloud_id)
    assert fact_id
    return fact_id, cloud_id


def test_recollapse_recovers_within_bundle_repeats(db: DatabaseManager) -> None:
    fact_id, cloud_id = _seed_fact_with_repeated_atoms(db)
    # With the bundle-aware dedup, all 3 repeats are already kept on first
    # collapse; recollapse is a no-op that preserves them and the vendor_key.
    before = v2_store.get_fact_by_id(db, fact_id)
    key_before = before["vendor_key"]
    assert key_before

    res = rx.recollapse_fact(db, fact_id, apply=True)
    assert res.applied
    assert res.vendor_key_preserved
    new_fact = v2_store.get_fact_by_id(db, res.new_fact_id)
    assert new_fact["vendor_key"] == key_before
    n = db.fetchone(
        "SELECT COUNT(*) AS n FROM fact_items WHERE fact_id = ?", (res.new_fact_id,)
    )["n"]
    assert n == 3  # all three Activia kept


def test_recollapse_dry_run_no_mutation(db: DatabaseManager) -> None:
    fact_id, cloud_id = _seed_fact_with_repeated_atoms(db)
    res = rx.recollapse_fact(db, fact_id, apply=False)
    assert res.applied is False
    assert res.items_after == 3  # simulated count
    # Fact untouched
    assert v2_store.get_fact_by_id(db, fact_id) is not None


def test_select_recollapse_candidates_finds_repeats(db: DatabaseManager) -> None:
    fact_id, _ = _seed_fact_with_repeated_atoms(db)
    # A clean fact with no repeats should NOT be selected.
    clean_fact, _c, _d = _seed_under_extracted_fact(db)
    cands = rx.select_recollapse_candidates(db, limit=50)
    assert fact_id in cands
    assert clean_fact not in cands


# ---------------------------------------------------------------------------
# Duplicate-photo over-count remediation (allow_reduce)
# ---------------------------------------------------------------------------


def _seed_overcount_dup_photo_fact(db: DatabaseManager) -> tuple[str, str]:
    """A multi-basket fact whose items over-sum the total (Type A over-count).

    Two BASKET bundles (two photos of one receipt, same vendor+date+total) that
    the fixed collapse de-duplicates to a single basket. The fact is then forced
    into the pre-fix doubled state by copying its fact_items, so its item_sum
    exceeds the total. Returns (fact_id, cloud_id).
    """
    items = [("Milk", 3.0), ("Bread", 4.0), ("Eggs", 3.0)]  # sum 10 == total
    d1 = _make_document(db)
    a1 = _make_atoms(db, d1, amount=10.0, items=items)
    b1 = _make_bundle(db, d1, a1, BundleType.BASKET)
    cloud_id = _make_cloud(db, b1)
    d2 = _make_document(db)
    a2 = _make_atoms(db, d2, amount=10.0, items=items)
    b2 = _make_bundle(db, d2, a2, BundleType.BASKET)
    _add_bundle_to_cloud(db, cloud_id, b2)
    fact_id = recollapse_cloud(db, cloud_id)
    assert fact_id
    # Force the doubled (over-count) state: copy each fact_item once.
    rows = db.fetchall("SELECT * FROM fact_items WHERE fact_id = ?", (fact_id,))
    with db.transaction() as cur:
        for r in rows:
            cur.execute(
                "INSERT INTO fact_items "
                "(id, fact_id, atom_id, name, total_price, quantity, unit) "
                "VALUES (?, ?, ?, ?, ?, ?, 'pcs')",
                (
                    str(uuid4()),
                    fact_id,
                    r["atom_id"],
                    r["name"],
                    r["total_price"],
                    r["quantity"],
                ),
            )
    return fact_id, cloud_id


def test_select_overcount_candidates_finds_multibasket_overcount(
    db: DatabaseManager,
) -> None:
    fact_id, _ = _seed_overcount_dup_photo_fact(db)
    # A clean, balanced fact must NOT be selected.
    clean_fact, _c, _d = _seed_under_extracted_fact(db)
    cands = rx.select_overcount_candidates(db, limit=50)
    assert fact_id in cands
    assert clean_fact not in cands


def test_recollapse_without_allow_reduce_skips_reduction(
    db: DatabaseManager,
) -> None:
    fact_id, _ = _seed_overcount_dup_photo_fact(db)
    res = rx.recollapse_fact(db, fact_id, apply=True, allow_reduce=False)
    assert res.error and "would reduce" in res.error
    # Untouched: still doubled (6 items).
    n = db.fetchone(
        "SELECT COUNT(*) AS n FROM fact_items WHERE fact_id = ?", (fact_id,)
    )["n"]
    assert n == 6


def test_recollapse_allow_reduce_drops_duplicate_basket(
    db: DatabaseManager,
) -> None:
    fact_id, _ = _seed_overcount_dup_photo_fact(db)
    before = v2_store.get_fact_by_id(db, fact_id)
    key_before = before["vendor_key"]
    res = rx.recollapse_fact(db, fact_id, apply=True, allow_reduce=True)
    assert res.applied
    assert res.items_before == 6
    assert res.items_after == 3  # one basket's worth, de-duplicated
    assert res.vendor_key_preserved
    new_fact = v2_store.get_fact_by_id(db, res.new_fact_id)
    assert new_fact["vendor_key"] == key_before


def _seed_two_date_basket_cloud(
    db: DatabaseManager, vendor: str = "Big Store Ltd", tax_id: str = "10033253D"
) -> tuple[str, str, str]:
    """One cloud with two baskets on DIFFERENT dates (a Type-B mis-merge).

    Returns (cloud_id, bundle1_id, bundle2_id). Both baskets same vendor+amount.
    """
    from datetime import date as _d
    from alibi.db.models import Atom, AtomType

    def _basket(doc_date: str) -> tuple[str, list]:
        doc_id = _make_document(db)
        atoms = _make_atoms(
            db,
            doc_id,
            vendor=vendor,
            tax_id=tax_id,
            amount=3.0,
            items=[("Coffee", 3.0)],
        )
        # override the datetime atom's date
        for a in atoms:
            if a.atom_type == AtomType.DATETIME:
                a.data["value"] = f"{doc_date}T10:00:00"
        # re-store atoms with the new date
        with db.transaction() as cur:
            for a in atoms:
                if a.atom_type == AtomType.DATETIME:
                    import json as _j

                    cur.execute(
                        "UPDATE atoms SET data=? WHERE id=?", (_j.dumps(a.data), a.id)
                    )
        bundle_id = _make_bundle(db, doc_id, atoms, BundleType.BASKET)
        # stamp the document date used by the split selector
        import json as _j

        with db.transaction() as cur:
            cur.execute(
                "UPDATE documents SET raw_extraction=json_set(raw_extraction,'$.date',?)"
                " WHERE id=?",
                (doc_date, doc_id),
            )
        return bundle_id, atoms

    b1, _a1 = _basket("2026-02-22")
    cloud_id = _make_cloud(db, b1)
    b2, _a2 = _basket("2026-03-03")
    _add_bundle_to_cloud(db, cloud_id, b2)
    recollapse_cloud(db, cloud_id)
    return cloud_id, b1, b2


def test_split_dates_separates_distinct_day_baskets(db: DatabaseManager) -> None:
    cloud_id, b1, b2 = _seed_two_date_basket_cloud(db)
    res = rx.split_cloud_by_date(db, cloud_id, apply=True, grace_days=3)
    assert res.applied
    assert res.new_clouds == 1
    # The two baskets now live in different clouds.
    c1 = v2_store.get_cloud_for_bundle(db, b1)
    c2 = v2_store.get_cloud_for_bundle(db, b2)
    assert c1 != c2


def test_split_dates_dry_run_no_mutation(db: DatabaseManager) -> None:
    cloud_id, b1, b2 = _seed_two_date_basket_cloud(db)
    res = rx.split_cloud_by_date(db, cloud_id, apply=False, grace_days=3)
    assert res.applied is False
    assert len(res.dates) == 2
    # Both baskets still in the same cloud.
    assert v2_store.get_cloud_for_bundle(db, b1) == v2_store.get_cloud_for_bundle(
        db, b2
    )


def test_split_dates_skips_within_grace(db: DatabaseManager) -> None:
    # Two baskets one day apart -> within the default 3-day grace -> skip.
    cloud_id, b1, b2 = _seed_two_date_basket_cloud(db)
    # Re-point b2's date to 1 day after b1.
    with db.transaction() as cur:
        cur.execute(
            "UPDATE documents SET raw_extraction=json_set(raw_extraction,'$.date','2026-02-23') "
            "WHERE id=(SELECT document_id FROM bundles WHERE id=?)",
            (b2,),
        )
    res = rx.split_cloud_by_date(db, cloud_id, apply=True, grace_days=3)
    assert res.applied is False
    assert res.skipped is not None and "grace" in res.skipped


def test_select_queue_partitions_partial_vs_itemless(db: DatabaseManager) -> None:
    # Partial fact: total 10, one 3.0 item (30% coverage).
    partial_fact, _c1, _d1 = _seed_under_extracted_fact(db)

    # Item-less fact: total 20, zero items.
    doc2 = _make_document(db)
    atoms2 = _make_atoms(db, doc2, amount=20.0, items=None)
    bundle2 = _make_bundle(db, doc2, atoms2, BundleType.BASKET)
    cloud2 = _make_cloud(db, bundle2)
    itemless_fact = recollapse_cloud(db, cloud2)
    assert itemless_fact

    partial_ids = {r.fact_id for r in rx.select_queue(db, queue="partial", limit=50)}
    itemless_ids = {r.fact_id for r in rx.select_queue(db, queue="item-less", limit=50)}
    all_ids = {r.fact_id for r in rx.select_queue(db, queue="all", limit=50)}

    assert partial_fact in partial_ids
    assert partial_fact not in itemless_ids
    assert itemless_fact in itemless_ids
    assert itemless_fact not in partial_ids
    assert {partial_fact, itemless_fact} <= all_ids
