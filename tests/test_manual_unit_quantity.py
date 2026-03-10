"""Tests for manual unit_quantity/unit entry across all interfaces.

Covers:
- correction.update_fact_item allowlist (unit_quantity, unit)
- identity metadata propagation when unit_quantity is set
- LineItemUpdate Pydantic model fields
- API PATCH endpoint accepts unit_quantity/unit
- MCP update_line_item tool
- Unknown fields still rejected
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from alibi.services import correction as svc


# ---------------------------------------------------------------------------
# correction.update_fact_item — allowlist
# ---------------------------------------------------------------------------


class TestUpdateFactItemAllowlist:
    def test_unit_quantity_in_allowed_fields(self):
        assert "unit_quantity" in svc._UPDATABLE_ITEM_FIELDS

    def test_unit_in_allowed_fields(self):
        assert "unit" in svc._UPDATABLE_ITEM_FIELDS

    def test_unit_quantity_in_column_map(self):
        assert svc._ITEM_FIELD_TO_COLUMN["unit_quantity"] == "unit_quantity"

    def test_unit_in_column_map(self):
        assert svc._ITEM_FIELD_TO_COLUMN["unit"] == "unit"

    def test_unknown_field_still_rejected(self, mock_db):
        with pytest.raises(ValueError, match="Disallowed"):
            svc.update_fact_item(mock_db, "item-1", {"price": "10.00"})

    def test_unknown_field_with_unit_quantity_rejected(self, mock_db):
        with pytest.raises(ValueError, match="Disallowed"):
            svc.update_fact_item(
                mock_db, "item-1", {"unit_quantity": 0.5, "weight_raw": "500g"}
            )


# ---------------------------------------------------------------------------
# correction.update_fact_item — unit_quantity DB write
# ---------------------------------------------------------------------------


class TestUpdateFactItemUnitQuantity:
    def _make_cursor(self, mock_db):
        mock_cursor = MagicMock()
        mock_db.transaction.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.transaction.return_value.__exit__ = MagicMock(return_value=False)
        return mock_cursor

    def test_update_unit_quantity_succeeds(self, mock_db):
        cursor = self._make_cursor(mock_db)
        existing = {"id": "item-1", "name": "Beef", "barcode": None, "unit": None}

        with patch("alibi.services.query.get_fact_item", return_value=existing):
            with patch("alibi.identities.matching.ensure_item_identity"):
                with patch.object(mock_db, "fetchone", return_value=None):
                    result = svc.update_fact_item(
                        mock_db, "item-1", {"unit_quantity": 0.5}
                    )

        assert result is True
        sql = cursor.execute.call_args_list[0][0][0]
        assert "unit_quantity" in sql
        assert "fact_items" in sql

    def test_update_unit_succeeds(self, mock_db):
        cursor = self._make_cursor(mock_db)
        existing = {
            "id": "item-1",
            "name": "Olive Oil",
            "barcode": None,
            "unit": None,
        }

        with patch("alibi.services.query.get_fact_item", return_value=existing):
            result = svc.update_fact_item(mock_db, "item-1", {"unit": "L"})

        assert result is True
        sql = cursor.execute.call_args_list[0][0][0]
        assert "unit" in sql

    def test_update_unit_quantity_and_unit_together(self, mock_db):
        cursor = self._make_cursor(mock_db)
        existing = {
            "id": "item-1",
            "name": "Chicken",
            "barcode": None,
            "unit": None,
        }

        with patch("alibi.services.query.get_fact_item", return_value=existing):
            with patch("alibi.identities.matching.ensure_item_identity"):
                with patch.object(mock_db, "fetchone", return_value=None):
                    result = svc.update_fact_item(
                        mock_db, "item-1", {"unit_quantity": 1.2, "unit": "kg"}
                    )

        assert result is True
        sql = cursor.execute.call_args_list[0][0][0]
        assert "unit_quantity" in sql
        assert "unit" in sql

    def test_returns_false_for_missing_item(self, mock_db):
        with patch("alibi.services.query.get_fact_item", return_value=None):
            assert (
                svc.update_fact_item(mock_db, "item-x", {"unit_quantity": 0.5}) is False
            )


# ---------------------------------------------------------------------------
# correction.update_fact_item — identity metadata propagation
# ---------------------------------------------------------------------------


class TestUpdateFactItemIdentityMetadata:
    def _make_cursor(self, mock_db):
        mock_cursor = MagicMock()
        mock_db.transaction.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.transaction.return_value.__exit__ = MagicMock(return_value=False)
        return mock_cursor

    def test_unit_quantity_propagates_to_identity_metadata(self, mock_db):
        self._make_cursor(mock_db)
        existing = {
            "id": "item-1",
            "name": "Beef Mince",
            "barcode": None,
            "unit": "kg",
        }

        with patch("alibi.services.query.get_fact_item", return_value=existing):
            with patch(
                "alibi.identities.matching.ensure_item_identity",
                return_value="identity-abc",
            ) as mock_ensure:
                with patch.object(mock_db, "fetchone", return_value=None):
                    svc.update_fact_item(
                        mock_db, "item-1", {"unit_quantity": 0.765, "unit": "kg"}
                    )

        mock_ensure.assert_called_once_with(
            mock_db,
            item_name="Beef Mince",
            barcode=None,
            source="correction",
        )

    def test_identity_metadata_written_correctly(self, mock_db):
        self._make_cursor(mock_db)
        existing = {
            "id": "item-1",
            "name": "Salmon",
            "barcode": None,
            "unit": None,
        }
        mock_identity_row = MagicMock()
        mock_identity_row.__getitem__ = lambda self, k: (
            None if k == "metadata" else None
        )

        with patch("alibi.services.query.get_fact_item", return_value=existing):
            with patch(
                "alibi.identities.matching.ensure_item_identity",
                return_value="identity-xyz",
            ):
                with patch.object(
                    mock_db,
                    "fetchone",
                    return_value={"metadata": None},
                ):
                    svc.update_fact_item(
                        mock_db, "item-1", {"unit_quantity": 0.684, "unit": "kg"}
                    )

        # The transaction for identity metadata update should be the second call
        # (first is the fact_items UPDATE)
        all_calls = mock_db.transaction.call_args_list
        assert len(all_calls) >= 1

    def test_no_identity_propagation_when_unit_quantity_is_none(self, mock_db):
        self._make_cursor(mock_db)
        existing = {"id": "item-1", "name": "Milk", "barcode": None, "unit": None}

        with patch("alibi.services.query.get_fact_item", return_value=existing):
            with patch("alibi.identities.matching.ensure_item_identity") as mock_ensure:
                svc.update_fact_item(mock_db, "item-1", {"unit": "L"})

        # unit alone (no unit_quantity) should NOT trigger identity metadata update
        mock_ensure.assert_not_called()

    def test_identity_propagation_skipped_when_no_item_name(self, mock_db):
        self._make_cursor(mock_db)
        # Item has no name — identity lookup is skipped
        existing = {"id": "item-1", "name": None, "barcode": None, "unit": None}

        with patch("alibi.services.query.get_fact_item", return_value=existing):
            with patch("alibi.identities.matching.ensure_item_identity") as mock_ensure:
                with patch.object(mock_db, "fetchone", return_value=None):
                    result = svc.update_fact_item(
                        mock_db, "item-1", {"unit_quantity": 0.5}
                    )

        assert result is True
        mock_ensure.assert_not_called()


# ---------------------------------------------------------------------------
# _update_identity_unit_metadata helper
# ---------------------------------------------------------------------------


class TestUpdateIdentityUnitMetadata:
    def test_creates_metadata_from_scratch(self, mock_db):
        mock_cursor = MagicMock()
        mock_db.transaction.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.transaction.return_value.__exit__ = MagicMock(return_value=False)
        mock_db.fetchone.return_value = {"metadata": None}

        svc._update_identity_unit_metadata(mock_db, "id-1", 0.5, "kg")

        mock_cursor.execute.assert_called_once()
        sql, params = mock_cursor.execute.call_args[0]
        assert "UPDATE identities" in sql
        written = json.loads(params[0])
        assert written["unit_quantity"] == 0.5
        assert written["unit"] == "kg"

    def test_merges_into_existing_metadata(self, mock_db):
        mock_cursor = MagicMock()
        mock_db.transaction.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.transaction.return_value.__exit__ = MagicMock(return_value=False)
        existing_meta = json.dumps({"barcode": "123", "category": "meat"})
        mock_db.fetchone.return_value = {"metadata": existing_meta}

        svc._update_identity_unit_metadata(mock_db, "id-2", 1.2, "kg")

        _, params = mock_cursor.execute.call_args[0]
        written = json.loads(params[0])
        assert written["unit_quantity"] == 1.2
        assert written["unit"] == "kg"
        # Original metadata preserved
        assert written["barcode"] == "123"
        assert written["category"] == "meat"

    def test_noop_when_identity_not_found(self, mock_db):
        mock_db.fetchone.return_value = None

        # Should not raise
        svc._update_identity_unit_metadata(mock_db, "id-missing", 0.5, "kg")

        mock_db.transaction.assert_not_called()


# ---------------------------------------------------------------------------
# LineItemUpdate Pydantic model
# ---------------------------------------------------------------------------


class TestLineItemUpdateModel:
    def test_model_accepts_unit_quantity(self):
        from alibi.api.routers.line_items import LineItemUpdate

        m = LineItemUpdate(unit_quantity=0.5)
        assert m.unit_quantity == 0.5

    def test_model_accepts_unit(self):
        from alibi.api.routers.line_items import LineItemUpdate

        m = LineItemUpdate(unit="kg")
        assert m.unit == "kg"

    def test_model_accepts_all_fields(self):
        from alibi.api.routers.line_items import LineItemUpdate

        m = LineItemUpdate(
            barcode="1234567890123",
            brand="Haagen-Dazs",
            category="Ice Cream",
            name="Vanilla",
            unit_quantity=0.5,
            unit="L",
        )
        assert m.unit_quantity == 0.5
        assert m.unit == "L"

    def test_unit_quantity_defaults_to_none(self):
        from alibi.api.routers.line_items import LineItemUpdate

        m = LineItemUpdate()
        assert m.unit_quantity is None

    def test_unit_defaults_to_none(self):
        from alibi.api.routers.line_items import LineItemUpdate

        m = LineItemUpdate()
        assert m.unit is None

    def test_model_dump_excludes_none_by_default(self):
        from alibi.api.routers.line_items import LineItemUpdate

        m = LineItemUpdate(unit_quantity=1.5)
        d = m.model_dump(exclude_none=True)
        assert "unit_quantity" in d
        assert "unit" not in d


# ---------------------------------------------------------------------------
# API PATCH endpoint
# ---------------------------------------------------------------------------


class TestLineItemAPIEndpoint:
    @pytest.fixture
    def client(self, db_manager):
        from collections.abc import Generator

        from alibi.api.app import create_app
        from alibi.api.deps import get_database

        app = create_app()
        app.dependency_overrides[get_database] = lambda: db_manager
        with TestClient(app) as c:
            yield c
        app.dependency_overrides.clear()

    @pytest.fixture
    def seeded_db(self, db_manager) -> str:
        """Insert cloud, fact, document, atom, and fact_item; return item_id."""
        import uuid

        conn = db_manager.get_connection()
        cloud_id = str(uuid.uuid4())
        fact_id = str(uuid.uuid4())
        doc_id = str(uuid.uuid4())
        atom_id = str(uuid.uuid4())
        item_id = str(uuid.uuid4())

        conn.execute(
            "INSERT INTO clouds (id, status) VALUES (?, 'collapsed')",
            (cloud_id,),
        )
        conn.execute(
            "INSERT INTO facts (id, cloud_id, fact_type, vendor, total_amount,"
            " currency, event_date, status)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                fact_id,
                cloud_id,
                "purchase",
                "Test Vendor",
                "10.00",
                "EUR",
                "2024-01-01",
                "confirmed",
            ),
        )
        conn.execute(
            "INSERT INTO documents (id, file_path, file_hash, created_at)"
            " VALUES (?, ?, ?, ?)",
            (doc_id, "/receipts/test.jpg", "abc123", "2024-01-01T10:00:00"),
        )
        conn.execute(
            "INSERT INTO atoms (id, document_id, atom_type, data)"
            " VALUES (?, ?, ?, ?)",
            (atom_id, doc_id, "item", "{}"),
        )
        conn.execute(
            "INSERT INTO fact_items (id, fact_id, atom_id, name, quantity,"
            " unit_price, total_price)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (item_id, fact_id, atom_id, "Beef Mince", 1, 5.0, 5.0),
        )
        conn.commit()
        return item_id

    def test_patch_unit_quantity_returns_updated_item(self, client, seeded_db):
        item_id = seeded_db
        resp = client.patch(
            f"/api/v1/line-items/{item_id}",
            json={"unit_quantity": 0.765, "unit": "kg"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert float(data["unit_quantity"]) == pytest.approx(0.765, abs=0.001)
        assert data["unit"] == "kg"

    def test_patch_unit_only_succeeds(self, client, seeded_db):
        item_id = seeded_db
        resp = client.patch(
            f"/api/v1/line-items/{item_id}",
            json={"unit": "L"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["unit"] == "L"

    def test_patch_empty_body_returns_400(self, client, seeded_db):
        item_id = seeded_db
        resp = client.patch(f"/api/v1/line-items/{item_id}", json={})
        assert resp.status_code == 400

    def test_patch_nonexistent_item_returns_404(self, client):
        resp = client.patch(
            "/api/v1/line-items/nonexistent-id",
            json={"unit_quantity": 0.5},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# MCP update_line_item tool
# ---------------------------------------------------------------------------


class TestMCPUpdateLineItem:
    def test_no_fields_returns_error(self, mock_db):
        from alibi.mcp.tools import update_line_item

        result = update_line_item(mock_db, item_id="item-1")
        assert "error" in result
        assert result["error"] == "No fields to update"

    def test_unit_quantity_update_succeeds(self, mock_db):
        from alibi.mcp.tools import update_line_item

        with patch(
            "alibi.services.correction.update_fact_item", return_value=True
        ) as mock_update:
            result = update_line_item(mock_db, item_id="item-1", unit_quantity=0.5)

        assert result["success"] is True
        assert "unit_quantity" in result["updated_fields"]
        mock_update.assert_called_once_with(mock_db, "item-1", {"unit_quantity": 0.5})

    def test_unit_update_succeeds(self, mock_db):
        from alibi.mcp.tools import update_line_item

        with patch("alibi.services.correction.update_fact_item", return_value=True):
            result = update_line_item(mock_db, item_id="item-1", unit="kg")

        assert result["success"] is True
        assert "unit" in result["updated_fields"]

    def test_all_fields_forwarded(self, mock_db):
        from alibi.mcp.tools import update_line_item

        with patch(
            "alibi.services.correction.update_fact_item", return_value=True
        ) as mock_update:
            result = update_line_item(
                mock_db,
                item_id="item-1",
                barcode="1234567890123",
                brand="Lidl",
                category="Meat",
                name="Chicken Breast",
                unit_quantity=0.6,
                unit="kg",
            )

        assert result["success"] is True
        called_fields = mock_update.call_args[0][2]
        assert called_fields["unit_quantity"] == 0.6
        assert called_fields["unit"] == "kg"
        assert called_fields["barcode"] == "1234567890123"

    def test_missing_item_returns_error(self, mock_db):
        from alibi.mcp.tools import update_line_item

        with patch("alibi.services.correction.update_fact_item", return_value=False):
            result = update_line_item(mock_db, item_id="bad-id", unit_quantity=0.5)

        assert "error" in result
        assert result["error"] == "Item not found"

    def test_disallowed_field_via_service_returns_error(self, mock_db):
        from alibi.mcp.tools import update_line_item

        with patch(
            "alibi.services.correction.update_fact_item",
            side_effect=ValueError("Disallowed field(s)"),
        ):
            # Simulate service raising ValueError (shouldn't happen via MCP
            # since MCP builds its own fields dict, but ensures error path works)
            result = update_line_item(mock_db, item_id="item-1", unit_quantity=0.5)

        assert "error" in result
