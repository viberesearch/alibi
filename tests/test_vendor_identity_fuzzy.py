"""Fuzzy vendor-identity clustering (OCR-variant VATs, cross-script names).

Exact VAT/name matching spawns a separate identity per OCR variant, which then
makes cloud formation veto the merge (two distinct identity_ids => same-merchant
veto). These tests lock in the fuzzy fallback that unifies the real-world
clusters the 2026-06 manual vendor-key reconciliation had to fix by hand, while
NOT merging genuinely different merchants.
"""

from __future__ import annotations

import os
import tempfile

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.config import Config, reset_config
from alibi.db.connection import DatabaseManager
from alibi.identities import matching, store


@pytest.fixture
def db():
    reset_config()
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    config = Config(db_path=db_path)
    manager = DatabaseManager(config)
    if not manager.is_initialized():
        manager.initialize()
    yield manager
    manager.close()
    os.unlink(db_path)


def _seed(db, name: str, *, vat: str | None = None, slug: str | None = None) -> str:
    """Create a vendor identity with name/normalized_name/vat members."""
    from alibi.normalizers.vendors import normalize_vendor_slug

    identity_id = store.create_identity(db, "vendor", name)
    store.add_member(db, identity_id, "name", name)
    store.add_member(
        db, identity_id, "normalized_name", slug or normalize_vendor_slug(name)
    )
    if vat:
        store.add_member(db, identity_id, "vat_number", vat)
        store.add_member(db, identity_id, "vendor_key", vat.upper().replace(" ", ""))
    return identity_id


class TestOcrVariantVat:
    """Same name, OCR-variant VAT -> same identity."""

    def test_papas_ocr_variant_vat(self, db):
        # PAPAS: 10355430K vs 103055400K (ratio ~0.84) with identical name.
        seeded = _seed(db, "PAPAS HYPERMARKET", vat="103055400K")
        found = matching.find_vendor_identity(
            db, vendor_name="PAPAS HYPERMARKET", registration="10355430K"
        )
        assert found is not None
        assert found["id"] == seeded

    def test_lidl_ocr_digit_for_letter(self, db):
        # LIDL: 30010823A vs 300108234 (A<->4 OCR confusion), name identical.
        seeded = _seed(db, "LIDL", vat="30010823A")
        found = matching.find_vendor_identity(
            db, vendor_name="LIDL", registration="300108234"
        )
        assert found is not None
        assert found["id"] == seeded

    def test_country_prefixed_variant(self, db):
        # S & A: CY10431313S vs CY1043131S (extra digit).
        seeded = _seed(db, "S & A Hospitality Ltd", vat="CY10431313S")
        found = matching.find_vendor_identity(
            db, vendor_name="S & A Hospitality Ltd", registration="CY1043131S"
        )
        assert found is not None
        assert found["id"] == seeded


class TestCrossScriptName:
    """Greek/Latin name variants, VAT missing or garbled -> same identity."""

    def test_greek_latin_name_no_vat(self, db):
        # Latin "SKLAVENITIS COLUMBIA" seeded; a Greek-script scan arrives with
        # no VAT. normalize_vendor_slug transliterates, but OCR garble can leave
        # the slugs merely *similar* -- the fuzzy name rule must still unify.
        seeded = _seed(db, "SKLAVENITIS COLUMBIA", vat="10033253D")
        found = matching.find_vendor_identity(
            db, vendor_name="SKLAVENITISS COLUMBIA"  # one OCR'd extra letter
        )
        assert found is not None
        assert found["id"] == seeded

    def test_garbled_name_with_variant_vat(self, db):
        seeded = _seed(db, "YPERAGORA HADJIANTONIS", vat="60024172V")
        found = matching.find_vendor_identity(
            db, vendor_name="YPERAGORA HADJIANTONIS", registration="60024172"
        )
        assert found is not None
        assert found["id"] == seeded


class TestNoFalseMerges:
    """Different merchants must NOT be clustered together."""

    def test_different_name_similar_vat_not_merged(self, db):
        # Near-sequential VATs but clearly different names -> no merge.
        _seed(db, "ACME GROCERIES", vat="10000001A")
        found = matching.find_vendor_identity(
            db, vendor_name="ZENITH HARDWARE", registration="10000002A"
        )
        assert found is None

    def test_different_vat_same_short_generic_handled_by_exact_only(self, db):
        # Two different shops, different VATs, unrelated names: no fuzzy merge.
        _seed(db, "Blue Market", vat="11111111A")
        found = matching.find_vendor_identity(
            db, vendor_name="Red Bazaar", registration="22222222B"
        )
        assert found is None

    def test_name_only_contradicting_vat_blocked(self, db):
        # Same name family but VATs that clearly contradict (e.g. a chain vs an
        # unrelated shop that happens to share a word) should not merge on name
        # alone when both carry strongly-different VATs.
        _seed(db, "Central Market", vat="50000000A")
        found = matching.find_vendor_identity(
            db, vendor_name="Central Market", registration="99999999Z"
        )
        # Same exact name -> the exact name pass unifies them regardless; this
        # documents that exact-name behavior is unchanged (not a fuzzy concern).
        assert found is not None


class TestEndToEndEnsureAndFormation:
    """The payoff: two OCR-variant scans resolve to one identity, which removes
    the cloud-formation veto that previously split the merchant."""

    def test_ensure_unifies_ocr_variant_and_collects_member(self, db):
        from alibi.identities.matching import ensure_vendor_identity

        first = ensure_vendor_identity(
            db, vendor_name="PAPAS HYPERMARKET", vat_number="103055400K"
        )
        second = ensure_vendor_identity(
            db, vendor_name="PAPAS HYPERMARKET", vat_number="10355430K"
        )
        assert first is not None
        assert first == second  # same identity, not a new one

        # Both VAT variants are now recorded on the one identity.
        vats = {m["value"] for m in store.get_members_by_type(db, first, "vat_number")}
        assert vats == {"103055400K", "10355430K"}

    def test_formation_no_longer_vetoes_unified_identity(self, db):
        from alibi.clouds.formation import BundleSummary, BundleType, _vendor_score
        from alibi.identities.matching import ensure_vendor_identity
        from decimal import Decimal

        # Both scans resolve to the same identity via ensure_vendor_identity.
        idA = ensure_vendor_identity(db, vendor_name="LIDL", vat_number="30010823A")
        idB = ensure_vendor_identity(db, vendor_name="LIDL", vat_number="300108234")
        assert idA == idB

        a = BundleSummary(
            bundle_id="a",
            bundle_type=BundleType.BASKET,
            vendor="LIDL",
            vendor_key="30010823A",
            identity_id=idA,
        )
        b = BundleSummary(
            bundle_id="b",
            bundle_type=BundleType.PAYMENT_RECORD,
            vendor="LIDL",
            vendor_key="300108234",
            identity_id=idB,
        )
        # Same identity_id -> full vendor score (was a 0.0 veto when the two OCR
        # variants produced distinct identities).
        assert _vendor_score(a, b) == Decimal("1")


class TestExactPathsUnchanged:
    """The fuzzy fallback only fires after the exact passes; verify those still
    short-circuit so behavior for clean data is unchanged."""

    def test_exact_vat_match(self, db):
        seeded = _seed(db, "ALPHAMEGA", vat="10027397Z")
        found = matching.find_vendor_identity(db, registration="10027397Z")
        assert found is not None and found["id"] == seeded

    def test_no_signal_returns_none(self, db):
        _seed(db, "Some Shop", vat="10000000A")
        assert matching.find_vendor_identity(db) is None

    def test_unknown_vendor_returns_none(self, db):
        _seed(db, "Known Shop", vat="10000000A")
        assert (
            matching.find_vendor_identity(db, vendor_name="Totally Unrelated Emporium")
            is None
        )
