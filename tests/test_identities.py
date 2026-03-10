"""Tests for the identity system (manual entity grouping)."""

import json
import os
import tempfile

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.config import Config, reset_config
from alibi.db.connection import DatabaseManager
from alibi.identities import store, matching


@pytest.fixture
def db():
    """Create a fresh temp database with identity tables."""
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


class TestIdentityStore:
    """Tests for identity CRUD operations."""

    def test_create_vendor_identity(self, db):
        identity_id = store.create_identity(
            db, "vendor", "FreSko", metadata={"legal_name": "FRESKO BUTANOLO LTD"}
        )
        assert identity_id

        identity = store.get_identity(db, identity_id)
        assert identity is not None
        assert identity["entity_type"] == "vendor"
        assert identity["canonical_name"] == "FreSko"
        assert identity["metadata"]["legal_name"] == "FRESKO BUTANOLO LTD"
        assert identity["active"]
        assert identity["members"] == []

    def test_create_item_identity(self, db):
        identity_id = store.create_identity(
            db,
            "item",
            "Happy Cow Milk",
            metadata={"barcode": "5290004000123", "brand": "Happy Cow"},
        )
        identity = store.get_identity(db, identity_id)
        assert identity["entity_type"] == "item"
        assert identity["canonical_name"] == "Happy Cow Milk"
        assert identity["metadata"]["barcode"] == "5290004000123"

    def test_add_member(self, db):
        identity_id = store.create_identity(db, "vendor", "FreSko")
        store.add_member(db, identity_id, "name", "FreSko")
        store.add_member(db, identity_id, "name", "FRESKO")
        store.add_member(db, identity_id, "name", "FRESKO BUTANOLO LTD")
        store.add_member(db, identity_id, "vat_number", "10336127M")

        identity = store.get_identity(db, identity_id)
        assert len(identity["members"]) == 4

        names = [m["value"] for m in identity["members"] if m["member_type"] == "name"]
        assert set(names) == {"FreSko", "FRESKO", "FRESKO BUTANOLO LTD"}

    def test_add_member_duplicate_ignored(self, db):
        identity_id = store.create_identity(db, "vendor", "FreSko")
        store.add_member(db, identity_id, "name", "FreSko")
        store.add_member(db, identity_id, "name", "FreSko")  # Duplicate

        identity = store.get_identity(db, identity_id)
        # Should only have 1 member (UNIQUE constraint with INSERT OR IGNORE)
        name_members = [m for m in identity["members"] if m["member_type"] == "name"]
        assert len(name_members) == 1

    def test_remove_member(self, db):
        identity_id = store.create_identity(db, "vendor", "FreSko")
        member_id = store.add_member(db, identity_id, "name", "FreSko")

        assert store.remove_member(db, member_id)

        identity = store.get_identity(db, identity_id)
        assert len(identity["members"]) == 0

    def test_delete_identity_cascades(self, db):
        identity_id = store.create_identity(db, "vendor", "FreSko")
        store.add_member(db, identity_id, "name", "FreSko")
        store.add_member(db, identity_id, "vat_number", "10336127M")

        assert store.delete_identity(db, identity_id)
        assert store.get_identity(db, identity_id) is None

        # Members should be gone too (CASCADE)
        conn = db.get_connection()
        rows = conn.execute(
            "SELECT COUNT(*) FROM identity_members WHERE identity_id = ?",
            (identity_id,),
        ).fetchone()
        assert rows[0] == 0

    def test_update_identity_name(self, db):
        identity_id = store.create_identity(db, "vendor", "FreSko")
        store.update_identity(db, identity_id, canonical_name="FreSko Market")

        identity = store.get_identity(db, identity_id)
        assert identity["canonical_name"] == "FreSko Market"

    def test_update_identity_active(self, db):
        identity_id = store.create_identity(db, "vendor", "FreSko")
        store.update_identity(db, identity_id, active=False)

        identity = store.get_identity(db, identity_id)
        assert not identity["active"]

    def test_list_identities_all(self, db):
        store.create_identity(db, "vendor", "FreSko")
        store.create_identity(db, "vendor", "Arab Butchery")
        store.create_identity(db, "item", "Happy Cow Milk")

        all_ids = store.list_identities(db)
        assert len(all_ids) == 3

    def test_list_identities_by_type(self, db):
        store.create_identity(db, "vendor", "FreSko")
        store.create_identity(db, "vendor", "Arab Butchery")
        store.create_identity(db, "item", "Happy Cow Milk")

        vendors = store.list_identities(db, entity_type="vendor")
        assert len(vendors) == 2

        items = store.list_identities(db, entity_type="item")
        assert len(items) == 1

    def test_list_identities_active_only(self, db):
        id1 = store.create_identity(db, "vendor", "FreSko")
        store.create_identity(db, "vendor", "Arab Butchery")
        store.update_identity(db, id1, active=False)

        active = store.list_identities(db, active_only=True)
        assert len(active) == 1
        assert active[0]["canonical_name"] == "Arab Butchery"

    def test_get_members_by_type(self, db):
        identity_id = store.create_identity(db, "vendor", "FreSko")
        store.add_member(db, identity_id, "name", "FreSko")
        store.add_member(db, identity_id, "name", "FRESKO")
        store.add_member(db, identity_id, "vat_number", "10336127M")

        names = store.get_members_by_type(db, identity_id, "name")
        assert len(names) == 2

        regs = store.get_members_by_type(db, identity_id, "vat_number")
        assert len(regs) == 1
        assert regs[0]["value"] == "10336127M"

    def test_add_barcode_member(self, db):
        identity_id = store.create_identity(db, "item", "Happy Cow Milk")
        store.add_member(db, identity_id, "barcode", "5290004000123")

        identity = store.get_identity(db, identity_id)
        barcodes = [m for m in identity["members"] if m["member_type"] == "barcode"]
        assert len(barcodes) == 1
        assert barcodes[0]["value"] == "5290004000123"

    def test_add_member_with_source(self, db):
        identity_id = store.create_identity(db, "vendor", "FreSko")
        store.add_member(db, identity_id, "name", "FreSko", source="user")
        store.add_member(db, identity_id, "name", "FRESKO MARKET", source="auto")

        identity = store.get_identity(db, identity_id)
        sources = {m["value"]: m["source"] for m in identity["members"]}
        assert sources["FreSko"] == "user"
        assert sources["FRESKO MARKET"] == "auto"


class TestIdentityMatching:
    """Tests for identity resolution/matching."""

    def test_find_vendor_by_name(self, db):
        identity_id = store.create_identity(db, "vendor", "FreSko")
        store.add_member(db, identity_id, "name", "FreSko")
        store.add_member(db, identity_id, "name", "FRESKO")

        result = matching.find_vendor_identity(db, vendor_name="FRESKO")
        assert result is not None
        assert result["canonical_name"] == "FreSko"

    def test_find_vendor_by_registration(self, db):
        identity_id = store.create_identity(db, "vendor", "FreSko")
        store.add_member(db, identity_id, "vat_number", "10336127M")

        result = matching.find_vendor_identity(db, registration="10336127M")
        assert result is not None
        assert result["canonical_name"] == "FreSko"

    def test_find_vendor_by_vendor_key(self, db):
        identity_id = store.create_identity(db, "vendor", "FreSko")
        store.add_member(db, identity_id, "vendor_key", "10336127M")

        result = matching.find_vendor_identity(db, vendor_key="10336127M")
        assert result is not None
        assert result["canonical_name"] == "FreSko"

    def test_find_vendor_by_normalized_name(self, db):
        identity_id = store.create_identity(db, "vendor", "FreSko")
        store.add_member(db, identity_id, "normalized_name", "fresko")

        result = matching.find_vendor_identity(db, vendor_name="FRESKO")
        # Should find via normalized_name after exact name fails
        assert result is not None
        assert result["canonical_name"] == "FreSko"

    def test_find_vendor_not_found(self, db):
        store.create_identity(db, "vendor", "FreSko")
        store.add_member(db, store.list_identities(db)[0]["id"], "name", "FreSko")

        result = matching.find_vendor_identity(db, vendor_name="Unknown Store")
        assert result is None

    def test_find_vendor_inactive_not_returned(self, db):
        identity_id = store.create_identity(db, "vendor", "FreSko")
        store.add_member(db, identity_id, "name", "FreSko")
        store.update_identity(db, identity_id, active=False)

        result = matching.find_vendor_identity(db, vendor_name="FreSko")
        assert result is None

    def test_find_item_by_barcode(self, db):
        identity_id = store.create_identity(
            db,
            "item",
            "Happy Cow Milk",
            metadata={"barcode": "5290004000123"},
        )
        store.add_member(db, identity_id, "barcode", "5290004000123")

        result = matching.find_item_identity(db, barcode="5290004000123")
        assert result is not None
        assert result["canonical_name"] == "Happy Cow Milk"

    def test_find_item_by_name(self, db):
        identity_id = store.create_identity(db, "item", "Happy Cow Milk")
        store.add_member(db, identity_id, "name", "Happy Cow Milk")
        store.add_member(db, identity_id, "name", "Happy Milk")

        result = matching.find_item_identity(db, item_name="Happy Milk")
        assert result is not None
        assert result["canonical_name"] == "Happy Cow Milk"

    def test_find_item_by_normalized_name(self, db):
        identity_id = store.create_identity(db, "item", "Happy Cow Milk")
        store.add_member(db, identity_id, "normalized_name", "happy cow milk")

        result = matching.find_item_identity(db, item_name="Happy Cow Milk 1L")
        # normalized "happy cow milk 1l" → strips trailing unit → "happy cow milk"
        assert result is not None
        assert result["canonical_name"] == "Happy Cow Milk"

    def test_resolve_vendor(self, db):
        identity_id = store.create_identity(db, "vendor", "FreSko")
        store.add_member(db, identity_id, "name", "FRESKO BUTANOLO LTD")

        canonical = matching.resolve_vendor(db, vendor_name="FRESKO BUTANOLO LTD")
        assert canonical == "FreSko"

    def test_resolve_vendor_not_found(self, db):
        canonical = matching.resolve_vendor(db, vendor_name="Unknown")
        assert canonical is None

    def test_resolve_item(self, db):
        identity_id = store.create_identity(db, "item", "Happy Cow Milk")
        store.add_member(db, identity_id, "name", "Happy Milk")

        canonical = matching.resolve_item(db, item_name="Happy Milk")
        assert canonical == "Happy Cow Milk"

    def test_find_identities_for_fact(self, db):
        vid = store.create_identity(db, "vendor", "FreSko")
        store.add_member(db, vid, "name", "FreSko")

        iid = store.create_identity(db, "item", "Milk")
        store.add_member(db, iid, "name", "Milk")

        result = matching.find_identities_for_fact(
            db, vendor="FreSko", vendor_key=None, item_names=["Milk", "Bread"]
        )
        assert result["vendor_identity"] is not None
        assert result["vendor_identity"]["canonical_name"] == "FreSko"
        assert "Milk" in result["item_identities"]
        assert "Bread" not in result["item_identities"]

    def test_vat_number_priority_over_name(self, db):
        """VAT number match should win over name match."""
        id1 = store.create_identity(db, "vendor", "Store A")
        store.add_member(db, id1, "name", "STORE")
        store.add_member(db, id1, "vat_number", "REG_A")

        id2 = store.create_identity(db, "vendor", "Store B")
        store.add_member(db, id2, "name", "STORE")
        store.add_member(db, id2, "vat_number", "REG_B")

        # Should find by vat_number, not by ambiguous name
        result = matching.find_vendor_identity(
            db, vendor_name="STORE", registration="REG_B"
        )
        assert result["canonical_name"] == "Store B"


class TestIdentityNormalization:
    """Tests for item name normalization in matching."""

    def test_normalize_strips_trailing_unit(self):
        assert matching._normalize_item_name("Milk 1L") == "milk"
        assert matching._normalize_item_name("Butter 500g") == "butter"
        assert matching._normalize_item_name("Juice 250ml") == "juice"

    def test_normalize_preserves_name_without_unit(self):
        assert matching._normalize_item_name("Bread") == "bread"
        assert matching._normalize_item_name("Happy Cow Milk") == "happy cow milk"

    def test_normalize_collapses_whitespace(self):
        assert matching._normalize_item_name("Happy  Cow   Milk") == "happy cow milk"


class TestVatCountryPrefixMatching:
    """Tests for prefix-stripped VAT/vendor_key lookup in _find_by_member."""

    def test_find_by_prefixed_vat_stored_bare(self, db):
        """Lookup with 'CY10370773Q' finds identity stored as '10370773Q'."""
        identity_id = store.create_identity(db, "vendor", "Acme CY")
        store.add_member(db, identity_id, "vat_number", "10370773Q")

        result = matching.find_vendor_identity(db, registration="CY10370773Q")
        assert result is not None
        assert result["canonical_name"] == "Acme CY"

    def test_find_by_bare_vat_stored_prefixed(self, db):
        """Lookup with '10370773Q' finds identity stored as 'CY10370773Q'."""
        identity_id = store.create_identity(db, "vendor", "Acme CY")
        store.add_member(db, identity_id, "vat_number", "CY10370773Q")

        result = matching.find_vendor_identity(db, registration="10370773Q")
        assert result is not None
        assert result["canonical_name"] == "Acme CY"

    def test_find_by_prefixed_vendor_key_stored_bare(self, db):
        """Lookup with 'CY10370773Q' vendor_key finds identity stored as '10370773Q'."""
        identity_id = store.create_identity(db, "vendor", "Acme CY")
        store.add_member(db, identity_id, "vendor_key", "10370773Q")

        result = matching.find_vendor_identity(db, vendor_key="CY10370773Q")
        assert result is not None
        assert result["canonical_name"] == "Acme CY"

    def test_find_by_bare_vendor_key_stored_prefixed(self, db):
        """Lookup with bare key finds identity stored with country prefix."""
        identity_id = store.create_identity(db, "vendor", "Acme CY")
        store.add_member(db, identity_id, "vendor_key", "CY10370773Q")

        result = matching.find_vendor_identity(db, vendor_key="10370773Q")
        assert result is not None
        assert result["canonical_name"] == "Acme CY"

    def test_non_eu_prefix_not_stripped(self, db):
        """A non-EU two-letter prefix is NOT stripped; lookup fails correctly."""
        identity_id = store.create_identity(db, "vendor", "US Corp")
        store.add_member(db, identity_id, "vat_number", "US12345678")

        # Exact match succeeds
        result = matching.find_vendor_identity(db, registration="US12345678")
        assert result is not None

        # "US" is not an EU code; "12345678" should NOT match "US12345678"
        result_bare = matching.find_vendor_identity(db, registration="12345678")
        assert result_bare is None

    def test_different_bare_numbers_no_match(self, db):
        """Two VATs with different bare numbers do not collide."""
        identity_id = store.create_identity(db, "vendor", "Acme CY")
        store.add_member(db, identity_id, "vat_number", "CY10370773Q")

        result = matching.find_vendor_identity(db, registration="CY99999999X")
        assert result is None

    def test_inactive_identity_not_returned_via_prefix_match(self, db):
        """Inactive identity is not returned even if bare VAT matches."""
        identity_id = store.create_identity(db, "vendor", "Acme CY")
        store.add_member(db, identity_id, "vat_number", "10370773Q")
        store.update_identity(db, identity_id, active=False)

        result = matching.find_vendor_identity(db, registration="CY10370773Q")
        assert result is None


class TestEnsureVendorIdentityCanonicalName:
    """Tests for canonical_name propagation in ensure_vendor_identity."""

    def test_correction_updates_canonical_name_via_vendor_key(self, db):
        """ensure_vendor_identity with source=correction updates canonical_name (vendor_key match)."""
        from alibi.normalizers.vendors import normalize_vendor

        # Create an identity with a stale canonical name, linked via vendor_key
        identity_id = store.create_identity(db, "vendor", "Mediterranean Hospital C")
        store.add_member(db, identity_id, "vendor_key", "CY10180201N")

        new_name = "Mediterranean Hospital of Cyprus"

        # Correction call: new name + existing vendor_key
        result_id = matching.ensure_vendor_identity(
            db,
            vendor_name=new_name,
            vendor_key="CY10180201N",
            source="correction",
        )

        assert result_id == identity_id
        identity = store.get_identity(db, identity_id)
        assert identity is not None
        # canonical_name must be updated to the normalized form of the corrected name
        assert identity["canonical_name"] == normalize_vendor(new_name)
        # canonical_name must NOT be the stale value
        assert identity["canonical_name"] != "Mediterranean Hospital C"

    def test_correction_updates_canonical_name_via_exact_name_match(self, db):
        """ensure_vendor_identity with source=correction and name match updates canonical_name."""
        from alibi.normalizers.vendors import normalize_vendor

        # Old identity has a different canonical_name from the raw member value
        identity_id = store.create_identity(db, "vendor", "FreSko")
        store.add_member(db, identity_id, "name", "FRESKO MARKET")

        result_id = matching.ensure_vendor_identity(
            db,
            vendor_name="FRESKO MARKET",
            source="correction",
        )

        assert result_id == identity_id
        identity = store.get_identity(db, identity_id)
        assert identity is not None
        # canonical_name updated to the normalized form of "FRESKO MARKET"
        assert identity["canonical_name"] == normalize_vendor("FRESKO MARKET")

    def test_extraction_does_not_update_canonical_name(self, db):
        """ensure_vendor_identity with source=extraction leaves canonical_name unchanged."""
        identity_id = store.create_identity(db, "vendor", "FreSko")
        store.add_member(db, identity_id, "vendor_key", "CY10180201N")
        store.add_member(db, identity_id, "name", "FRESKO MARKET")

        result_id = matching.ensure_vendor_identity(
            db,
            vendor_name="Fresko Hypermarket",
            vendor_key="CY10180201N",
            source="extraction",
        )

        assert result_id == identity_id
        identity = store.get_identity(db, identity_id)
        assert identity is not None
        # canonical_name must remain unchanged for extraction source
        assert identity["canonical_name"] == "FreSko"

    def test_correction_same_name_no_update(self, db):
        """ensure_vendor_identity with source=correction and same canonical_name skips update."""
        from alibi.normalizers.vendors import normalize_vendor

        raw_name = "FreSko Market"
        normalized = normalize_vendor(raw_name)
        identity_id = store.create_identity(db, "vendor", normalized)
        store.add_member(db, identity_id, "name", raw_name)

        # Correction with name that normalizes to the same canonical_name already stored
        result_id = matching.ensure_vendor_identity(
            db,
            vendor_name=raw_name,
            source="correction",
        )

        assert result_id == identity_id
        identity = store.get_identity(db, identity_id)
        assert identity is not None
        # canonical_name unchanged (same display value, no DB write needed)
        assert identity["canonical_name"] == normalized

    def test_correction_creates_new_identity_with_correct_name(self, db):
        """When no existing identity, source=correction creates one with normalized name."""
        from alibi.normalizers.vendors import normalize_vendor

        raw_name = "Mediterranean Hospital of Cyprus"
        result_id = matching.ensure_vendor_identity(
            db,
            vendor_name=raw_name,
            source="correction",
        )

        assert result_id is not None
        identity = store.get_identity(db, result_id)
        assert identity is not None
        assert identity["canonical_name"] == normalize_vendor(raw_name)

    def test_correction_uses_vendor_key_to_find_and_update(self, db):
        """Correction matches by vendor_key and updates stale canonical_name."""
        from alibi.normalizers.vendors import normalize_vendor

        new_name = "Mediterranean Hospital of Cyprus"
        identity_id = store.create_identity(db, "vendor", "Med Hospital")
        store.add_member(db, identity_id, "vendor_key", "CY10180201N")

        result_id = matching.ensure_vendor_identity(
            db,
            vendor_name=new_name,
            vendor_key="CY10180201N",
            source="correction",
        )

        assert result_id == identity_id
        identity = store.get_identity(db, identity_id)
        assert identity is not None
        assert identity["canonical_name"] == normalize_vendor(new_name)
        assert identity["canonical_name"] != "Med Hospital"
