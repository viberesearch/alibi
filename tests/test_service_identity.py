"""Tests for alibi.services.identity."""

from unittest.mock import MagicMock, call, patch

import pytest

from alibi.services import identity as svc

MODULE = "alibi.services.identity"


@pytest.fixture
def db() -> MagicMock:
    return MagicMock()


def _make_identity(identity_id: str, name: str = "Acme") -> dict:
    return {
        "id": identity_id,
        "entity_type": "vendor",
        "canonical_name": name,
        "active": True,
        "members": [],
    }


# ---------------------------------------------------------------------------
# resolve_vendor
# ---------------------------------------------------------------------------


def test_resolve_vendor_delegates_to_find_vendor_identity(db: MagicMock) -> None:
    identity = _make_identity("id-001")
    with patch(f"{MODULE}.find_vendor_identity", return_value=identity) as mock:
        result = svc.resolve_vendor(db, vendor_name="Acme", vendor_key="CY10057000Y")

    assert result == identity
    mock.assert_called_once_with(
        db,
        vendor_name="Acme",
        vendor_key="CY10057000Y",
        registration=None,
    )


def test_resolve_vendor_returns_none_when_not_found(db: MagicMock) -> None:
    with patch(f"{MODULE}.find_vendor_identity", return_value=None):
        result = svc.resolve_vendor(db, vendor_name="Unknown")

    assert result is None


def test_resolve_vendor_passes_registration(db: MagicMock) -> None:
    with patch(f"{MODULE}.find_vendor_identity", return_value=None) as mock:
        svc.resolve_vendor(db, registration="CY10057000Y")

    mock.assert_called_once_with(
        db,
        vendor_name=None,
        vendor_key=None,
        registration="CY10057000Y",
    )


# ---------------------------------------------------------------------------
# list_identities
# ---------------------------------------------------------------------------


def test_list_identities_delegates_to_store(db: MagicMock) -> None:
    rows = [_make_identity("id-001"), _make_identity("id-002", "Beta Corp")]
    with patch("alibi.identities.store.list_identities", return_value=rows) as mock:
        result = svc.list_identities(db, entity_type="vendor")

    assert result == rows
    mock.assert_called_once_with(db, entity_type="vendor")


def test_list_identities_no_filter(db: MagicMock) -> None:
    with patch("alibi.identities.store.list_identities", return_value=[]) as mock:
        svc.list_identities(db)

    mock.assert_called_once_with(db, entity_type=None)


# ---------------------------------------------------------------------------
# get_identity
# ---------------------------------------------------------------------------


def test_get_identity_returns_dict(db: MagicMock) -> None:
    identity = _make_identity("id-001")
    with patch("alibi.identities.store.get_identity", return_value=identity) as mock:
        result = svc.get_identity(db, "id-001")

    assert result == identity
    mock.assert_called_once_with(db, "id-001")


def test_get_identity_returns_none_when_missing(db: MagicMock) -> None:
    with patch("alibi.identities.store.get_identity", return_value=None):
        result = svc.get_identity(db, "nonexistent")

    assert result is None


# ---------------------------------------------------------------------------
# merge_vendors
# ---------------------------------------------------------------------------


def test_merge_vendors_success(db: MagicMock) -> None:
    id_a = "aaaa-0001"
    id_b = "bbbb-0002"
    identity_a = _make_identity(id_a, "Acme")
    identity_b = _make_identity(id_b, "ACME LTD")

    # get_identity is called twice (once for each ID)
    with patch(
        "alibi.identities.store.get_identity", side_effect=[identity_a, identity_b]
    ):
        ctx = MagicMock()
        db.transaction.return_value.__enter__ = MagicMock(return_value=ctx)
        db.transaction.return_value.__exit__ = MagicMock(return_value=False)

        result = svc.merge_vendors(db, id_a, id_b)

    assert result is True
    # Check member re-assignment and deletion were issued
    assert ctx.execute.call_count == 2
    member_update = ctx.execute.call_args_list[0]
    assert "UPDATE identity_members" in member_update[0][0]
    assert member_update[0][1] == (id_a, id_b)

    deletion = ctx.execute.call_args_list[1]
    assert "DELETE FROM identities" in deletion[0][0]
    assert deletion[0][1] == (id_b,)


def test_merge_vendors_returns_false_if_a_missing(db: MagicMock) -> None:
    id_b = _make_identity("bbbb-0002")
    with patch("alibi.identities.store.get_identity", side_effect=[None, id_b]):
        result = svc.merge_vendors(db, "missing-a", "bbbb-0002")

    assert result is False
    db.transaction.assert_not_called()


def test_merge_vendors_returns_false_if_b_missing(db: MagicMock) -> None:
    id_a = _make_identity("aaaa-0001")
    with patch("alibi.identities.store.get_identity", side_effect=[id_a, None]):
        result = svc.merge_vendors(db, "aaaa-0001", "missing-b")

    assert result is False
    db.transaction.assert_not_called()


def test_merge_vendors_returns_false_if_both_missing(db: MagicMock) -> None:
    with patch("alibi.identities.store.get_identity", side_effect=[None, None]):
        result = svc.merge_vendors(db, "missing-a", "missing-b")

    assert result is False
    db.transaction.assert_not_called()
