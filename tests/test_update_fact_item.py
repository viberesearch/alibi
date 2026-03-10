"""Tests for update_fact_item service function."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from alibi.services import correction as svc


class TestUpdateFactItem:
    def test_rejects_empty_fields(self, mock_db):
        with pytest.raises(ValueError, match="must not be empty"):
            svc.update_fact_item(mock_db, "item-1", {})

    def test_rejects_unknown_fields(self, mock_db):
        with pytest.raises(ValueError, match="Disallowed"):
            svc.update_fact_item(mock_db, "item-1", {"price": "10.00"})

    def test_returns_false_for_missing_item(self, mock_db):
        with patch("alibi.services.query.get_fact_item", return_value=None):
            assert svc.update_fact_item(mock_db, "item-1", {"barcode": "123"}) is False

    def test_updates_barcode(self, mock_db):
        mock_cursor = MagicMock()
        mock_db.transaction.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.transaction.return_value.__exit__ = MagicMock(return_value=False)

        existing = {"id": "item-1", "name": "Milk", "barcode": None}

        with patch("alibi.services.query.get_fact_item", return_value=existing):
            with patch("alibi.identities.matching.ensure_item_identity"):
                result = svc.update_fact_item(
                    mock_db, "item-1", {"barcode": "5290004000123"}
                )

        assert result is True
        # First call is the UPDATE, subsequent calls may be correction event logging
        assert mock_cursor.execute.call_count >= 1
        sql = mock_cursor.execute.call_args_list[0][0][0]
        assert "barcode" in sql
        assert "fact_items" in sql

    def test_updates_multiple_fields(self, mock_db):
        mock_cursor = MagicMock()
        mock_db.transaction.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.transaction.return_value.__exit__ = MagicMock(return_value=False)

        existing = {"id": "item-1", "name": "Milk", "barcode": None}

        with patch("alibi.services.query.get_fact_item", return_value=existing):
            result = svc.update_fact_item(
                mock_db, "item-1", {"brand": "Happy Cow", "category": "Dairy"}
            )

        assert result is True
        # First execute call is the UPDATE; subsequent calls are correction event logging
        sql = mock_cursor.execute.call_args_list[0][0][0]
        assert "brand" in sql
        assert "category" in sql

    def test_barcode_triggers_identity_update(self, mock_db):
        mock_cursor = MagicMock()
        mock_db.transaction.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.transaction.return_value.__exit__ = MagicMock(return_value=False)

        existing = {"id": "item-1", "name": "Milk", "barcode": None}

        with patch("alibi.services.query.get_fact_item", return_value=existing):
            with patch("alibi.identities.matching.ensure_item_identity") as mock_ensure:
                svc.update_fact_item(mock_db, "item-1", {"barcode": "5290004000123"})

        mock_ensure.assert_called_once_with(
            mock_db,
            item_name="Milk",
            barcode="5290004000123",
            source="correction",
        )

    def test_updates_comparable_name(self, mock_db):
        mock_cursor = MagicMock()
        mock_db.transaction.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.transaction.return_value.__exit__ = MagicMock(return_value=False)

        existing = {"id": "item-1", "name": "Milk", "barcode": None}

        with patch("alibi.services.query.get_fact_item", return_value=existing):
            result = svc.update_fact_item(
                mock_db, "item-1", {"comparable_name": "Full Fat Milk"}
            )

        assert result is True
        sql = mock_cursor.execute.call_args_list[0][0][0]
        assert "comparable_name" in sql

    def test_brand_does_not_trigger_identity_update(self, mock_db):
        mock_cursor = MagicMock()
        mock_db.transaction.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_db.transaction.return_value.__exit__ = MagicMock(return_value=False)

        existing = {"id": "item-1", "name": "Milk", "barcode": None}

        with patch("alibi.services.query.get_fact_item", return_value=existing):
            with patch("alibi.identities.matching.ensure_item_identity") as mock_ensure:
                svc.update_fact_item(mock_db, "item-1", {"brand": "Happy Cow"})

        mock_ensure.assert_not_called()
