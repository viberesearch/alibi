"""Tests for duplicate-fact detection and safe resolution (alibi.clouds.dedup).

Covers the safety gate that stops a wrong extraction from false-merging two real
receipts, the price-multiset overlap signal that survives OCR name garbling, and
the delete-redundant resolution that preserves the keeper (including surgical
edits) intact. The two confirmed corpus regression pairs are encoded as tests.
"""

from __future__ import annotations

import json
from decimal import Decimal
from uuid import uuid4

from alibi.clouds.collapse import _one_vendor
from alibi.clouds.dedup import (
    DuplicateVerdict,
    FactDupInfo,
    decide_duplicate,
    dedup_pass,
    find_duplicate_groups,
    hamming_distance,
    phash_near_match,
    price_multiset_overlap,
    select_item_atoms,
    vendors_compatible,
)
from alibi.db.connection import DatabaseManager

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestPriceMultisetOverlap:
    def test_identical_multisets(self):
        assert price_multiset_overlap([1.0, 2.0, 2.0], [2.0, 1.0, 2.0]) == 1.0

    def test_disjoint(self):
        assert price_multiset_overlap([1.0, 2.0], [3.0, 4.0]) == 0.0

    def test_partial(self):
        # multiset inter={1,2}=2, union={1,2,3}=3 -> 2/3
        assert price_multiset_overlap([1.0, 2.0, 3.0], [1.0, 2.0]) == 2 / 3

    def test_empty_side_is_zero(self):
        assert price_multiset_overlap([], [1.0]) == 0.0

    def test_none_prices_dropped(self):
        assert price_multiset_overlap([1.0, None], [1.0]) == 1.0

    def test_rounds_to_two_dp(self):
        assert price_multiset_overlap([1.001], [1.004]) == 1.0


class TestHamming:
    def test_identical(self):
        assert hamming_distance("ff00", "ff00") == 0

    def test_one_bit(self):
        assert hamming_distance("0", "1") == 1

    def test_length_mismatch_returns_none(self):
        assert hamming_distance("ff", "f") is None

    def test_missing_returns_none(self):
        assert hamming_distance(None, "ff") is None

    def test_non_hex_returns_none(self):
        assert hamming_distance("zz", "ff") is None


class TestPhashNearMatch:
    def test_near(self):
        assert phash_near_match(["fcfef19090f8f8f8"], ["fcfef19090f8f8f8"]) is True

    def test_far(self):
        # The two confirmed dup photos have very different dHashes -> not a match.
        assert phash_near_match(["fcfef19090f8f8f8"], ["c8c0901000481800"]) is False

    def test_any_cross_pair(self):
        assert phash_near_match([None, "ff00"], ["ff01"], max_distance=2) is True


class TestVendorsCompatible:
    def test_ocr_variant_key(self):
        # One-character OCR slip in the registration ID.
        assert vendors_compatible("LIDL", "30010823A", "LIDL Cyprus", "300108234")

    def test_country_prefix_key(self):
        assert vendors_compatible("X", "CY10370773Q", "Y", "10370773Q")

    def test_different_keys(self):
        assert not vendors_compatible("A", "11111111", "B", "99999999")

    def test_name_substring_when_no_keys(self):
        assert vendors_compatible("LIDL", None, "LIDL Cyprus", None)

    def test_different_names_no_keys(self):
        assert not vendors_compatible("LIDL", None, "SPAR", None)


def _info(fid, *, n_items=1, prices=None, phashes=None, time="10:00:00"):
    prices = prices if prices is not None else [1.0] * n_items
    return FactDupInfo(
        fact_id=fid,
        cloud_id="c-" + fid,
        vendor="LIDL",
        vendor_key="300108234",
        event_date="2025-12-29",
        event_time=time,
        total_amount=Decimal("56.22"),
        currency="EUR",
        n_items=len(prices),
        item_prices=prices,
        perceptual_hashes=phashes or [],
    )


class TestDecideDuplicate:
    def test_zero_item_twin_merges(self):
        a = _info("a", prices=[1.0, 2.0])
        b = _info("b", prices=[])
        verdict, reason = decide_duplicate(a, b)
        assert verdict is DuplicateVerdict.MERGE
        assert "zero-item" in reason

    def test_phash_match_merges(self):
        a = _info("a", prices=[1.0], phashes=["ff00ff00ff00ff00"])
        b = _info("b", prices=[9.0], phashes=["ff00ff00ff00ff00"])
        verdict, _ = decide_duplicate(a, b)
        assert verdict is DuplicateVerdict.MERGE

    def test_price_overlap_merges(self):
        a = _info("a", prices=[1.0, 2.0, 3.0])
        b = _info("b", prices=[1.0, 2.0, 3.0])
        verdict, reason = decide_duplicate(a, b)
        assert verdict is DuplicateVerdict.MERGE
        assert "price overlap 1.00" in reason

    def test_low_overlap_reviews(self):
        a = _info("a", prices=[1.0, 2.0, 3.0])
        b = _info("b", prices=[8.0, 9.0])
        verdict, _ = decide_duplicate(a, b)
        assert verdict is DuplicateVerdict.REVIEW


class TestSelectItemAtoms:
    @staticmethod
    def _bundle(prices):
        return {
            "atoms": [
                {"atom_type": "item", "data": {"name": f"i{i}", "total_price": p}}
                for i, p in enumerate(prices)
            ]
        }

    def test_single_bundle_keeps_all(self):
        atoms = select_item_atoms([self._bundle([1.0, 2.0, 3.0])])
        assert len(atoms) == 3

    def test_duplicate_baskets_keep_richest(self):
        # Two overlapping baskets (same receipt scanned twice) -> not doubled.
        atoms = select_item_atoms(
            [self._bundle([1.0, 2.0, 3.0]), self._bundle([1.0, 2.0, 3.0, 4.0])]
        )
        assert len(atoms) == 4  # richest only

    def test_complementary_baskets_kept(self):
        # Disjoint item sets -> genuinely different baskets, keep both.
        atoms = select_item_atoms(
            [self._bundle([1.0, 2.0]), self._bundle([100.0, 200.0])]
        )
        assert len(atoms) == 4

    def test_non_item_bundles_ignored(self):
        payment = {"atoms": [{"atom_type": "amount", "data": {"value": 5}}]}
        atoms = select_item_atoms([self._bundle([1.0, 2.0]), payment])
        assert len(atoms) == 2


class TestSelectItemAtomsDuplicatePhoto:
    """Signature-based duplicate-photo de-duplication (Type A over-count).

    Two photos of one receipt make two BASKET bundles with the same
    (vendor, date, total) but OCR prices that diverged too far for price
    overlap to catch, so the items were summed (e.g. LIDL 56.22 -> 224.88).
    """

    @staticmethod
    def _basket(prices, *, vendor="LIDL", date="2025-12-29", total=56.22):
        """A basket bundle with item, vendor, datetime and total atoms."""
        atoms = [
            {
                "atom_type": "item",
                "data": {"name": f"i{i}", "total_price": p},
            }
            for i, p in enumerate(prices)
        ]
        atoms.append({"atom_type": "vendor", "data": {"name": vendor}})
        atoms.append({"atom_type": "datetime", "data": {"value": f"{date} 12:00:00"}})
        atoms.append(
            {"atom_type": "amount", "data": {"semantic_type": "total", "value": total}}
        )
        return {"bundle_id": str(uuid4()), "atoms": atoms}

    def test_diverged_prices_same_signature_deduped(self):
        # One photo read the receipt as one line, the other as several; prices
        # don't overlap but vendor+date+total match -> keep one basket only.
        a = self._basket([56.22], total=56.22)
        b = self._basket([10.0, 20.0, 26.22], total=56.22)
        atoms = select_item_atoms([a, b])
        # Kept basket is the one whose items sum nearest the total (56.22):
        # b sums to 56.22 exactly; a is also 56.22 -> a (fewer items) loses tie.
        assert len(atoms) == 3

    def test_keeper_is_closest_to_total_not_most_items(self):
        # Faithful 17-line basket (sums to the total) must win over a noisier
        # 23-line basket that only summed to half (the 6217e6af LIDL case).
        faithful = self._basket([5.0] * 17, total=85.0)  # sums to 85.0 == total
        noisy = self._basket([2.0] * 21 + [0.1] * 2, total=85.0)  # sums to 42.2
        atoms = select_item_atoms([noisy, faithful])
        assert len(atoms) == 17

    def test_different_total_kept(self):
        # Genuinely different receipts (different totals) are not merged.
        a = self._basket([1.0, 2.0], total=3.0)
        b = self._basket([10.0, 20.0], total=30.0)
        atoms = select_item_atoms([a, b])
        assert len(atoms) == 4

    def test_ocr_variant_vendor_name_still_deduped(self):
        a = self._basket([56.22], vendor="LIDL", total=56.22)
        b = self._basket([10.0, 46.22], vendor="LIDL Cyprus", total=56.22)
        atoms = select_item_atoms([a, b])
        assert len(atoms) == 2  # one basket kept despite the name variant

    def test_missing_date_on_one_photo_still_deduped(self):
        # A duplicate photo lost its date to OCR; vendor+total still match and
        # the dates do not conflict -> dedup (the Blue Island case).
        a = self._basket([1.0, 47.14], total=45.62)
        b = self._basket([10.0, 20.0, 32.04], total=45.62)
        # Drop b's datetime atom (OCR missed it).
        b["atoms"] = [at for at in b["atoms"] if at["atom_type"] != "datetime"]
        atoms = select_item_atoms([a, b])
        assert len(atoms) == 2  # one basket kept

    def test_conflicting_dates_not_deduped(self):
        # Same vendor+total but genuinely different days -> keep both.
        a = self._basket([1.0, 2.0], date="2026-01-12", total=3.0)
        b = self._basket([1.5, 1.5], date="2026-02-20", total=3.0)
        atoms = select_item_atoms([a, b])
        assert len(atoms) == 4

    def test_missing_total_falls_back_to_price_overlap(self):
        # No total atom -> no signature; disjoint prices -> both kept.
        a = self._basket([1.0, 2.0], total=None)
        b = self._basket([100.0, 200.0], total=None)
        # Strip the total atoms the helper added (total=None still appends one).
        for bundle in (a, b):
            bundle["atoms"] = [
                at
                for at in bundle["atoms"]
                if not (
                    at["atom_type"] == "amount"
                    and at["data"].get("semantic_type") == "total"
                )
            ]
        atoms = select_item_atoms([a, b])
        assert len(atoms) == 4


class TestOneVendor:
    def test_ocr_variant_names(self):
        assert _one_vendor(["LIDL", "LIDL Cyprus"]) is True

    def test_same_name(self):
        assert _one_vendor(["LIDL", "lidl"]) is True

    def test_different_vendors(self):
        assert _one_vendor(["LIDL", "SPAR"]) is False

    def test_empty(self):
        assert _one_vendor([]) is False

    def test_single(self):
        assert _one_vendor(["LIDL Cyprus"]) is True


# ---------------------------------------------------------------------------
# DB-backed: find_duplicate_groups + dedup_pass
# ---------------------------------------------------------------------------


def _insert_fact(
    db: DatabaseManager,
    *,
    vendor: str,
    vendor_key: str | None,
    prices: list[float],
    phash: str | None = None,
    event_date: str = "2025-12-29",
    event_time: str | None = "12:00:00",
    total: float = 56.22,
    currency: str = "EUR",
    fact_id: str | None = None,
) -> tuple[str, str]:
    """Insert a complete fact chain (cloud/document/bundle/atoms/fact/items)."""
    fid = fact_id or str(uuid4())
    cid = str(uuid4())
    doc_id = str(uuid4())
    bun_id = str(uuid4())
    with db.transaction() as cur:
        cur.execute("INSERT INTO clouds (id, status) VALUES (?, ?)", (cid, "collapsed"))
        cur.execute(
            "INSERT INTO documents (id, file_path, file_hash, perceptual_hash) "
            "VALUES (?, ?, ?, ?)",
            (doc_id, f"/tmp/{doc_id}.jpg", doc_id, phash),
        )
        cur.execute(
            "INSERT INTO bundles (id, document_id, bundle_type, cloud_id) "
            "VALUES (?, ?, ?, ?)",
            (bun_id, doc_id, "basket", cid),
        )
        cur.execute(
            "INSERT INTO cloud_bundles (cloud_id, bundle_id, match_type, "
            "match_confidence) VALUES (?, ?, ?, ?)",
            (cid, bun_id, "exact_amount", 1.0),
        )
        cur.execute(
            "INSERT INTO facts (id, cloud_id, fact_type, vendor, vendor_key, "
            "total_amount, currency, event_date, event_time, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fid,
                cid,
                "purchase",
                vendor,
                vendor_key,
                total,
                currency,
                event_date,
                event_time,
                "confirmed",
            ),
        )
        for i, p in enumerate(prices):
            aid = str(uuid4())
            cur.execute(
                "INSERT INTO atoms (id, document_id, atom_type, data) "
                "VALUES (?, ?, ?, ?)",
                (aid, doc_id, "item", json.dumps({"name": f"i{i}", "total_price": p})),
            )
            cur.execute(
                "INSERT INTO bundle_atoms (bundle_id, atom_id, role) VALUES (?, ?, ?)",
                (bun_id, aid, "basket_item"),
            )
            cur.execute(
                "INSERT INTO fact_items (id, fact_id, atom_id, name, total_price) "
                "VALUES (?, ?, ?, ?, ?)",
                (str(uuid4()), fid, aid, f"i{i}", p),
            )
    return fid, cid


def _fact_count(db: DatabaseManager) -> int:
    return db.fetchone("SELECT COUNT(*) c FROM facts")["c"]


def _items_for(db: DatabaseManager, fid: str) -> int:
    return db.fetchone("SELECT COUNT(*) c FROM fact_items WHERE fact_id = ?", (fid,))[
        "c"
    ]


class TestDedupPass:
    def test_true_pair_merges_keeping_richer(self, db: DatabaseManager):
        # Regression: IMG_0994 ~ IMG_0512 — OCR-variant key, overlapping prices.
        shared = [float(i) for i in range(1, 13)]
        _insert_fact(
            db,
            vendor="LIDL Cyprus",
            vendor_key="300108234",
            prices=shared + [13.0, 14.0],  # 14 items
        )
        keep, _ = _insert_fact(
            db,
            vendor="LIDL",
            vendor_key="30010823A",
            prices=shared + [13.5, 14.5, 15.5],  # 15 items (richer)
        )

        report = dedup_pass(db, apply=False)
        assert report.resolved_count == 1
        assert report.review_count == 0
        action = report.resolved[0]
        assert action.keeper.fact_id == keep  # richer twin kept
        assert action.verdict is DuplicateVerdict.MERGE

        dedup_pass(db, apply=True)
        assert _fact_count(db) == 1
        assert _items_for(db, keep) == 15  # keeper preserved exactly

    def test_false_pair_reviewed_not_merged(self, db: DatabaseManager):
        # Same vendor/date/total but disjoint items (a wrong extraction colliding
        # on the header) -> must NOT auto-merge.
        a, _ = _insert_fact(
            db,
            vendor="LIDL",
            vendor_key="300108234",
            prices=[100.0, 101.0, 102.0],
        )
        b, _ = _insert_fact(
            db,
            vendor="LIDL Cyprus",
            vendor_key="300108234",
            prices=[200.0, 201.0, 202.0],
        )
        report = dedup_pass(db, apply=False)
        assert report.resolved_count == 0
        assert report.review_count == 1
        dedup_pass(db, apply=True)
        assert _fact_count(db) == 2  # nothing deleted

    def test_zero_item_twin_merges(self, db: DatabaseManager):
        keep, _ = _insert_fact(
            db, vendor="SPAR", vendor_key="11111111Z", prices=[1.0, 2.0, 3.0]
        )
        _insert_fact(db, vendor="SPAR", vendor_key="11111111Z", prices=[])
        report = dedup_pass(db, apply=True)
        assert report.resolved_count == 1
        assert _fact_count(db) == 1
        assert _items_for(db, keep) == 3

    def test_distinct_transactions_untouched(self, db: DatabaseManager):
        # Different totals -> not even candidate-grouped.
        _insert_fact(
            db, vendor="LIDL", vendor_key="300108234", prices=[1.0], total=10.0
        )
        _insert_fact(
            db, vendor="LIDL", vendor_key="300108234", prices=[1.0], total=20.0
        )
        report = dedup_pass(db, apply=False)
        assert report.resolved_count == 0
        assert report.review_count == 0

    def test_idempotent(self, db: DatabaseManager):
        shared = [float(i) for i in range(1, 13)]
        _insert_fact(db, vendor="LIDL", vendor_key="300108234", prices=shared)
        _insert_fact(db, vendor="LIDL Cyprus", vendor_key="30010823A", prices=shared)
        dedup_pass(db, apply=True)
        again = dedup_pass(db, apply=True)
        assert again.resolved_count == 0

    def test_find_groups_requires_compatible_vendor(self, db: DatabaseManager):
        # Same date/total/currency but unrelated vendors -> not grouped.
        _insert_fact(db, vendor="LIDL", vendor_key="11111111", prices=[1.0])
        _insert_fact(db, vendor="SPAR", vendor_key="99999999", prices=[1.0])
        assert find_duplicate_groups(db) == []


# ---------------------------------------------------------------------------
# Formation candidate broadening (Mode A prevention)
# ---------------------------------------------------------------------------


class TestCandidateBroadening:
    """get_bundle_summaries_for_vendor must surface an OCR-variant-key twin.

    The root cause of duplicate facts: an exact vendor_key pre-filter excludes a
    twin whose key was OCR'd differently. Widening by normalized vendor name
    brings it back into the candidate set so cloud formation can compare them.
    """

    @staticmethod
    def _insert_vendor_bundle(db, vendor_name, vat):
        doc_id = str(uuid4())
        bun_id = str(uuid4())
        atom_id = str(uuid4())
        with db.transaction() as cur:
            cur.execute(
                "INSERT INTO documents (id, file_path, file_hash) VALUES (?, ?, ?)",
                (doc_id, f"/tmp/{doc_id}.jpg", doc_id),
            )
            cur.execute(
                "INSERT INTO bundles (id, document_id, bundle_type) VALUES (?, ?, ?)",
                (bun_id, doc_id, "basket"),
            )
            cur.execute(
                "INSERT INTO atoms (id, document_id, atom_type, data) "
                "VALUES (?, ?, ?, ?)",
                (
                    atom_id,
                    doc_id,
                    "vendor",
                    json.dumps({"name": vendor_name, "vat_number": vat}),
                ),
            )
            cur.execute(
                "INSERT INTO bundle_atoms (bundle_id, atom_id, role) VALUES (?, ?, ?)",
                (bun_id, atom_id, "vendor_info"),
            )
        return bun_id

    def test_variant_key_twin_in_candidate_set(self, db: DatabaseManager):
        from alibi.db.v2_store import get_bundle_summaries_for_vendor

        twin = self._insert_vendor_bundle(db, "LIDL", "30010823A")
        self._insert_vendor_bundle(db, "SPAR", "55555555")  # noise

        # New bundle's key was OCR'd as 300108234 (one char off) — the exact-key
        # filter alone would miss the twin, but the name widening includes it.
        summaries = get_bundle_summaries_for_vendor(
            db, vendor_key="300108234", vendor_name="lidl"
        )
        ids = {s["bundle_id"] for s in summaries}
        assert twin in ids

    def test_name_widening_excludes_unrelated_vendor(self, db: DatabaseManager):
        from alibi.db.v2_store import get_bundle_summaries_for_vendor

        self._insert_vendor_bundle(db, "LIDL", "30010823A")
        spar = self._insert_vendor_bundle(db, "SPAR", "55555555")

        summaries = get_bundle_summaries_for_vendor(
            db, vendor_key="300108234", vendor_name="lidl"
        )
        ids = {s["bundle_id"] for s in summaries}
        assert spar not in ids
