"""Tests for ensure_item_identity — find-or-create item identities."""

import os
import tempfile

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.config import Config, reset_config
from alibi.db.connection import DatabaseManager
from alibi.identities import matching, store


@pytest.fixture
def db():
    """Create a fresh temp database."""
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


class TestEnsureItemIdentity:
    def test_creates_new_identity_from_name(self, db):
        identity_id = matching.ensure_item_identity(db, item_name="Fresh Milk 1L")
        assert identity_id is not None

        identity = store.get_identity(db, identity_id)
        assert identity["entity_type"] == "item"
        assert identity["canonical_name"] == "Fresh Milk 1L"
        # Should have name + normalized_name members
        member_types = {m["member_type"] for m in identity["members"]}
        assert "name" in member_types
        assert "normalized_name" in member_types

    def test_creates_new_identity_from_barcode(self, db):
        identity_id = matching.ensure_item_identity(db, barcode="5290004000123")
        assert identity_id is not None

        identity = store.get_identity(db, identity_id)
        assert identity["canonical_name"] == "5290004000123"
        member_types = {m["member_type"] for m in identity["members"]}
        assert "barcode" in member_types

    def test_creates_identity_with_name_and_barcode(self, db):
        identity_id = matching.ensure_item_identity(
            db, item_name="Happy Cow Milk", barcode="5290004000123"
        )
        assert identity_id is not None

        identity = store.get_identity(db, identity_id)
        assert identity["canonical_name"] == "Happy Cow Milk"
        member_types = {m["member_type"] for m in identity["members"]}
        assert "name" in member_types
        assert "normalized_name" in member_types
        assert "barcode" in member_types

    def test_returns_none_without_signal(self, db):
        assert matching.ensure_item_identity(db) is None

    def test_finds_existing_by_barcode(self, db):
        id1 = matching.ensure_item_identity(
            db, item_name="Milk A", barcode="1234567890123"
        )
        id2 = matching.ensure_item_identity(
            db, item_name="Milk B", barcode="1234567890123"
        )
        assert id1 == id2

    def test_adds_barcode_to_existing_name_identity(self, db):
        # First call: name only
        id1 = matching.ensure_item_identity(db, item_name="Fresh Milk 1L")
        # Second call: same name + barcode
        id2 = matching.ensure_item_identity(
            db, item_name="Fresh Milk 1L", barcode="5290004000123"
        )
        assert id1 == id2

        identity = store.get_identity(db, id2)
        member_types = {m["member_type"] for m in identity["members"]}
        assert "barcode" in member_types

    def test_adds_new_name_to_existing_barcode_identity(self, db):
        id1 = matching.ensure_item_identity(db, barcode="5290004000123")
        id2 = matching.ensure_item_identity(
            db, item_name="Happy Cow Milk", barcode="5290004000123"
        )
        assert id1 == id2

        identity = store.get_identity(db, id2)
        member_types = {m["member_type"] for m in identity["members"]}
        assert "name" in member_types
        assert "barcode" in member_types

    def test_does_not_duplicate_members(self, db):
        matching.ensure_item_identity(db, item_name="Bread", barcode="1234567890123")
        matching.ensure_item_identity(db, item_name="Bread", barcode="1234567890123")

        # Should have exactly 3 members: name, normalized_name, barcode
        identities = store.list_identities(db, entity_type="item")
        assert len(identities) == 1
        assert len(identities[0]["members"]) == 3

    def test_source_parameter(self, db):
        identity_id = matching.ensure_item_identity(
            db, item_name="Test Item", source="correction"
        )
        identity = store.get_identity(db, identity_id)
        sources = {m["source"] for m in identity["members"]}
        assert "correction" in sources
