"""Tests for alibi.telegram.handlers.barcode_scan."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram.types import InlineKeyboardMarkup, Message

from alibi.extraction.barcode_detector import BarcodeResult
from alibi.telegram.handlers.barcode_scan import (
    ScanAction,
    _lookup_product,
    scan_handler,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_message():
    """A mock Telegram Message with no photo attached."""
    msg = AsyncMock(spec=Message)
    msg.answer = AsyncMock(return_value=AsyncMock(spec=Message))
    msg.chat = AsyncMock()
    msg.chat.do = AsyncMock()
    msg.photo = None
    msg.reply_to_message = None
    msg.bot = AsyncMock()
    return msg


@pytest.fixture
def mock_status_msg():
    """A mock status message returned by message.answer('Scanning...')."""
    status = AsyncMock(spec=Message)
    status.edit_text = AsyncMock()
    status.delete = AsyncMock()
    return status


# ---------------------------------------------------------------------------
# ScanAction CallbackData
# ---------------------------------------------------------------------------


class TestScanCallbackData:
    def test_pack_unpack_enrich(self):
        action = ScanAction(action="enrich", barcode="5901234123457")
        packed = action.pack()
        unpacked = ScanAction.unpack(packed)
        assert unpacked.action == "enrich"
        assert unpacked.barcode == "5901234123457"

    def test_pack_unpack_lookup(self):
        action = ScanAction(action="lookup", barcode="96385074")
        packed = action.pack()
        unpacked = ScanAction.unpack(packed)
        assert unpacked.action == "lookup"
        assert unpacked.barcode == "96385074"

    def test_enrich_within_64_bytes(self):
        # Telegram callback_data limit is 64 bytes
        action = ScanAction(action="enrich", barcode="5901234123457")
        assert len(action.pack().encode("utf-8")) <= 64

    def test_lookup_within_64_bytes(self):
        action = ScanAction(action="lookup", barcode="5901234123457")
        assert len(action.pack().encode("utf-8")) <= 64

    def test_ean8_within_64_bytes(self):
        action = ScanAction(action="enrich", barcode="96385074")
        assert len(action.pack().encode("utf-8")) <= 64


# ---------------------------------------------------------------------------
# scan_handler — no photo
# ---------------------------------------------------------------------------


class TestScanHandler:
    @pytest.mark.asyncio
    async def test_no_photo_sends_usage_instructions(self, mock_message):
        """When there is no photo and no reply photo, sends usage instructions."""
        with (
            patch(
                "alibi.telegram.handlers.barcode_scan.has_barcode_support",
                return_value=True,
            ),
        ):
            await scan_handler(mock_message)

        mock_message.answer.assert_called_once()
        text = mock_message.answer.call_args[0][0]
        assert "/scan" in text

    @pytest.mark.asyncio
    async def test_pyzbar_unavailable_sends_not_available_message(self, mock_message):
        """When pyzbar is not installed, handler notifies the user."""
        with patch(
            "alibi.telegram.handlers.barcode_scan.has_barcode_support",
            return_value=False,
        ):
            await scan_handler(mock_message)

        mock_message.answer.assert_called_once()
        text = mock_message.answer.call_args[0][0]
        assert "not available" in text.lower() or "pyzbar" in text.lower()

    @pytest.mark.asyncio
    async def test_photo_no_barcodes_sends_no_barcodes_message(
        self, mock_message, mock_status_msg
    ):
        """Photo present but detect_barcodes returns [] → 'No barcodes detected'."""
        # Set up a photo on the message
        photo_size = MagicMock()
        photo_size.file_id = "file123"
        mock_message.photo = [photo_size]

        mock_message.answer.return_value = mock_status_msg

        with (
            patch(
                "alibi.telegram.handlers.barcode_scan.has_barcode_support",
                return_value=True,
            ),
            patch(
                "alibi.telegram.handlers.barcode_scan._download_photo",
                new_callable=AsyncMock,
                return_value=b"\xff\xd8\xff",  # fake JPEG bytes
            ),
            patch(
                "alibi.telegram.handlers.barcode_scan.detect_barcodes",
                return_value=[],
            ),
        ):
            await scan_handler(mock_message)

        mock_status_msg.edit_text.assert_called_once()
        text = mock_status_msg.edit_text.call_args[0][0]
        assert "no barcodes" in text.lower()

    @pytest.mark.asyncio
    async def test_photo_with_barcode_sends_result_with_keyboard(
        self, mock_message, mock_status_msg
    ):
        """Photo with a valid EAN-13 barcode → sends result message with inline keyboard."""
        photo_size = MagicMock()
        photo_size.file_id = "file456"
        mock_message.photo = [photo_size]

        mock_message.answer.side_effect = [
            mock_status_msg,  # first call: "Scanning..."
            AsyncMock(spec=Message),  # second call: result message
        ]

        barcode = BarcodeResult(data="5901234123457", type="EAN13", valid_ean=True)

        with (
            patch(
                "alibi.telegram.handlers.barcode_scan.has_barcode_support",
                return_value=True,
            ),
            patch(
                "alibi.telegram.handlers.barcode_scan._download_photo",
                new_callable=AsyncMock,
                return_value=b"\xff\xd8\xff",
            ),
            patch(
                "alibi.telegram.handlers.barcode_scan.detect_barcodes",
                return_value=[barcode],
            ),
            patch(
                "alibi.telegram.handlers.barcode_scan.get_db",
                return_value=MagicMock(),
            ),
            patch(
                "alibi.telegram.handlers.barcode_scan._lookup_product",
                return_value=None,
            ),
        ):
            await scan_handler(mock_message)

        # The result answer call (after status deleted) should include a keyboard
        result_call = mock_message.answer.call_args
        keyboard = result_call[1].get("reply_markup")
        assert isinstance(keyboard, InlineKeyboardMarkup)

    @pytest.mark.asyncio
    async def test_photo_with_barcode_result_text_contains_barcode_value(
        self, mock_message, mock_status_msg
    ):
        """Result message text contains the detected barcode value."""
        photo_size = MagicMock()
        photo_size.file_id = "file789"
        mock_message.photo = [photo_size]

        mock_message.answer.side_effect = [
            mock_status_msg,
            AsyncMock(spec=Message),
        ]

        barcode = BarcodeResult(data="5901234123457", type="EAN13", valid_ean=True)

        with (
            patch(
                "alibi.telegram.handlers.barcode_scan.has_barcode_support",
                return_value=True,
            ),
            patch(
                "alibi.telegram.handlers.barcode_scan._download_photo",
                new_callable=AsyncMock,
                return_value=b"\xff\xd8\xff",
            ),
            patch(
                "alibi.telegram.handlers.barcode_scan.detect_barcodes",
                return_value=[barcode],
            ),
            patch(
                "alibi.telegram.handlers.barcode_scan.get_db",
                return_value=MagicMock(),
            ),
            patch(
                "alibi.telegram.handlers.barcode_scan._lookup_product",
                return_value=None,
            ),
        ):
            await scan_handler(mock_message)

        result_call = mock_message.answer.call_args
        text = result_call[0][0]
        assert "5901234123457" in text

    @pytest.mark.asyncio
    async def test_reply_to_photo_triggers_detection(
        self, mock_message, mock_status_msg
    ):
        """Reply to a photo message with /scan → downloads the replied-to photo."""
        # This message has no photo itself
        mock_message.photo = None

        # But it replies to a message that has a photo
        replied = AsyncMock(spec=Message)
        photo_size = MagicMock()
        photo_size.file_id = "reply_file_001"
        replied.photo = [photo_size]
        mock_message.reply_to_message = replied

        mock_message.answer.side_effect = [
            mock_status_msg,
            AsyncMock(spec=Message),
        ]

        barcode = BarcodeResult(data="96385074", type="EAN8", valid_ean=True)

        download_mock = AsyncMock(return_value=b"\xff\xd8\xff")

        with (
            patch(
                "alibi.telegram.handlers.barcode_scan.has_barcode_support",
                return_value=True,
            ),
            patch(
                "alibi.telegram.handlers.barcode_scan._download_photo",
                download_mock,
            ),
            patch(
                "alibi.telegram.handlers.barcode_scan.detect_barcodes",
                return_value=[barcode],
            ),
            patch(
                "alibi.telegram.handlers.barcode_scan.get_db",
                return_value=MagicMock(),
            ),
            patch(
                "alibi.telegram.handlers.barcode_scan._lookup_product",
                return_value=None,
            ),
        ):
            await scan_handler(mock_message)

        # _download_photo must have been called with the replied-to message
        download_mock.assert_called_once_with(replied)

    @pytest.mark.asyncio
    async def test_inline_keyboard_has_two_buttons(self, mock_message, mock_status_msg):
        """Inline keyboard for a scan result has exactly two buttons."""
        photo_size = MagicMock()
        photo_size.file_id = "file_btn"
        mock_message.photo = [photo_size]

        mock_message.answer.side_effect = [
            mock_status_msg,
            AsyncMock(spec=Message),
        ]

        barcode = BarcodeResult(data="5901234123457", type="EAN13", valid_ean=True)

        with (
            patch(
                "alibi.telegram.handlers.barcode_scan.has_barcode_support",
                return_value=True,
            ),
            patch(
                "alibi.telegram.handlers.barcode_scan._download_photo",
                new_callable=AsyncMock,
                return_value=b"\xff\xd8\xff",
            ),
            patch(
                "alibi.telegram.handlers.barcode_scan.detect_barcodes",
                return_value=[barcode],
            ),
            patch(
                "alibi.telegram.handlers.barcode_scan.get_db",
                return_value=MagicMock(),
            ),
            patch(
                "alibi.telegram.handlers.barcode_scan._lookup_product",
                return_value=None,
            ),
        ):
            await scan_handler(mock_message)

        result_call = mock_message.answer.call_args
        keyboard = result_call[1].get("reply_markup")
        assert len(keyboard.inline_keyboard) == 1
        assert len(keyboard.inline_keyboard[0]) == 2


# ---------------------------------------------------------------------------
# _lookup_product
# ---------------------------------------------------------------------------


class TestLookupProduct:
    def test_product_found_in_cache(self, db_manager):
        """Product in product_cache returns dict with product_name, brand, category."""
        db_manager.execute(
            "INSERT INTO product_cache (barcode, data) VALUES (?, ?)",
            (
                "5901234123457",
                json.dumps(
                    {
                        "product_name": "Test Cookies",
                        "brands": "TestBrand",
                        "categories_tags": ["en:snacks", "en:biscuits"],
                    }
                ),
            ),
        )

        result = _lookup_product(db_manager, "5901234123457")

        assert result is not None
        assert result["product_name"] == "Test Cookies"
        assert result["brand"] == "TestBrand"
        assert "en:snacks" in result["category"]

    def test_product_not_in_cache_returns_none(self, db_manager):
        """Barcode absent from cache returns None."""
        result = _lookup_product(db_manager, "9999999999994")
        assert result is None

    def test_not_found_sentinel_returns_none(self, db_manager):
        """Cache entry with _not_found=True treated as absent."""
        db_manager.execute(
            "INSERT INTO product_cache (barcode, data) VALUES (?, ?)",
            ("4006381333931", json.dumps({"_not_found": True})),
        )
        result = _lookup_product(db_manager, "4006381333931")
        assert result is None

    def test_category_built_from_tags(self, db_manager):
        """categories_tags list is joined into a category string (max 3 tags)."""
        db_manager.execute(
            "INSERT INTO product_cache (barcode, data) VALUES (?, ?)",
            (
                "40170725",
                json.dumps(
                    {
                        "product_name": "Snack",
                        "brands": "Acme",
                        "categories_tags": [
                            "en:food",
                            "en:snacks",
                            "en:chips",
                            "en:extras",
                        ],
                    }
                ),
            ),
        )
        result = _lookup_product(db_manager, "40170725")
        assert result is not None
        # Only first 3 tags included
        parts = result["category"].split(", ")
        assert len(parts) == 3

    def test_missing_categories_tags_returns_empty_category(self, db_manager):
        """Product without categories_tags returns empty string for category."""
        db_manager.execute(
            "INSERT INTO product_cache (barcode, data) VALUES (?, ?)",
            (
                "96385074",
                json.dumps({"product_name": "Basic Item", "brands": "Basic Co"}),
            ),
        )
        result = _lookup_product(db_manager, "96385074")
        assert result is not None
        assert result["category"] == ""

    def test_corrupted_json_returns_none(self, db_manager):
        """Corrupted JSON data in cache returns None without raising."""
        db_manager.execute(
            "INSERT INTO product_cache (barcode, data) VALUES (?, ?)",
            ("5901234123457", "not-valid-json{{{"),
        )
        result = _lookup_product(db_manager, "5901234123457")
        assert result is None
