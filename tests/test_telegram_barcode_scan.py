"""Tests for the thin Telegram barcode scan handler.

Barcode detection and product lookup now live on the host: ``scan_handler``
and the enrich/lookup callbacks are thin HTTP clients of the host barcode
endpoints. These tests mock the shared :class:`AlibiAPIClient` (patched as
``alibi.telegram.handlers.barcode_scan.client``) and the module's
``_download_photo`` helper, then assert the right calls are made and replies
are formatted from the API response.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram.types import InlineKeyboardMarkup, Message as TgMessage

from alibi.telegram.api_client import AlibiAPIError
from alibi.telegram.handlers.barcode_scan import (
    ScanAction,
    _esc,
    handle_enrich,
    handle_lookup,
    scan_handler,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_status_msg():
    """A mock status message returned by message.answer('Scanning...')."""
    status = MagicMock(spec=TgMessage)
    status.edit_text = AsyncMock()
    status.delete = AsyncMock()
    return status


@pytest.fixture
def mock_message(mock_status_msg):
    """A mock Telegram Message with a photo attached.

    ``message.answer`` returns the status message so the handler can
    edit/delete it; later result-card answers reuse the same return value.
    """
    msg = MagicMock()
    msg.answer = AsyncMock(return_value=mock_status_msg)
    msg.chat = MagicMock()
    msg.chat.do = AsyncMock()
    msg.photo = [MagicMock(file_id="f")]
    msg.reply_to_message = None
    msg.bot = MagicMock()
    return msg


def _make_client(scan=None, enrich=None, lookup=None) -> MagicMock:
    """Build a MagicMock AlibiAPIClient with async barcode methods stubbed."""
    client = MagicMock()
    client.scan_barcode = AsyncMock(
        return_value=scan if scan is not None else {"count": 0, "barcodes": []}
    )
    client.enrich_by_barcode = AsyncMock(
        return_value=enrich if enrich is not None else {"matched": 0, "enriched": 0}
    )
    client.barcode_lookup = AsyncMock(return_value=lookup)
    return client


def _barcode(
    data: str = "4006381333931",
    btype: str = "EAN13",
    valid_ean: bool = True,
    product: dict | None = None,
) -> dict:
    return {
        "data": data,
        "type": btype,
        "valid_ean": valid_ean,
        "product": product,
    }


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
# _esc helper
# ---------------------------------------------------------------------------


class TestEsc:
    def test_esc_empty(self):
        assert _esc("") == ""

    def test_esc_plain(self):
        assert _esc("Plain Text") == "Plain Text"

    def test_esc_specials(self):
        assert _esc("a_b*c`d[e") == "a\\_b\\*c\\`d\\[e"


# ---------------------------------------------------------------------------
# scan_handler
# ---------------------------------------------------------------------------


class TestScanHandler:
    @pytest.mark.asyncio
    async def test_no_photo_sends_usage_instructions(self, mock_message):
        """No photo and no reply photo → sends usage instructions."""
        mock_message.photo = None
        mock_message.reply_to_message = None
        client = _make_client()

        with (
            patch("alibi.telegram.handlers.barcode_scan.client", new=client),
            patch(
                "alibi.telegram.handlers.barcode_scan._download_photo",
                AsyncMock(return_value=None),
            ),
        ):
            await scan_handler(mock_message)

        mock_message.answer.assert_called_once()
        text = mock_message.answer.call_args[0][0]
        assert "/scan" in text
        client.scan_barcode.assert_not_called()

    @pytest.mark.asyncio
    async def test_photo_no_barcodes_sends_no_barcodes_message(
        self, mock_message, mock_status_msg
    ):
        """Photo present but API returns no barcodes → 'No barcodes detected'."""
        client = _make_client(scan={"count": 0, "barcodes": []})

        with (
            patch("alibi.telegram.handlers.barcode_scan.client", new=client),
            patch(
                "alibi.telegram.handlers.barcode_scan._download_photo",
                AsyncMock(return_value=b"img"),
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
        """Valid EAN-13 barcode → sends result message with inline keyboard."""
        client = _make_client(
            scan={"count": 1, "barcodes": [_barcode("5901234123457")]}
        )

        with (
            patch("alibi.telegram.handlers.barcode_scan.client", new=client),
            patch(
                "alibi.telegram.handlers.barcode_scan._download_photo",
                AsyncMock(return_value=b"img"),
            ),
        ):
            await scan_handler(mock_message)

        # The result answer call (after status deleted) should include a keyboard
        result_call = mock_message.answer.call_args
        keyboard = result_call[1].get("reply_markup")
        assert isinstance(keyboard, InlineKeyboardMarkup)
        mock_status_msg.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_result_text_contains_barcode_value(
        self, mock_message, mock_status_msg
    ):
        """Result message text contains the detected barcode value."""
        client = _make_client(
            scan={"count": 1, "barcodes": [_barcode("5901234123457")]}
        )

        with (
            patch("alibi.telegram.handlers.barcode_scan.client", new=client),
            patch(
                "alibi.telegram.handlers.barcode_scan._download_photo",
                AsyncMock(return_value=b"img"),
            ),
        ):
            await scan_handler(mock_message)

        result_call = mock_message.answer.call_args
        text = result_call[0][0]
        assert "5901234123457" in text

    @pytest.mark.asyncio
    async def test_result_text_shows_product_found(self, mock_message, mock_status_msg):
        """When the API returns a product, the card says 'Product found'."""
        product = {
            "product_name": "Test Cookies",
            "brand": "TestBrand",
            "category": "snacks",
        }
        client = _make_client(
            scan={
                "count": 1,
                "barcodes": [_barcode("5901234123457", product=product)],
            }
        )

        with (
            patch("alibi.telegram.handlers.barcode_scan.client", new=client),
            patch(
                "alibi.telegram.handlers.barcode_scan._download_photo",
                AsyncMock(return_value=b"img"),
            ),
        ):
            await scan_handler(mock_message)

        result_call = mock_message.answer.call_args
        text = result_call[0][0]
        assert "Product found" in text
        assert "Test Cookies" in text

    @pytest.mark.asyncio
    async def test_inline_keyboard_has_two_buttons(self, mock_message, mock_status_msg):
        """Inline keyboard for a scan result has exactly two buttons."""
        client = _make_client(
            scan={"count": 1, "barcodes": [_barcode("5901234123457")]}
        )

        with (
            patch("alibi.telegram.handlers.barcode_scan.client", new=client),
            patch(
                "alibi.telegram.handlers.barcode_scan._download_photo",
                AsyncMock(return_value=b"img"),
            ),
        ):
            await scan_handler(mock_message)

        result_call = mock_message.answer.call_args
        keyboard = result_call[1].get("reply_markup")
        assert len(keyboard.inline_keyboard) == 1
        assert len(keyboard.inline_keyboard[0]) == 2

    @pytest.mark.asyncio
    async def test_scan_503_reports_unavailable(self, mock_message, mock_status_msg):
        """A 503 from the host edits the status to 'not available'."""
        client = _make_client()
        client.scan_barcode = AsyncMock(
            side_effect=AlibiAPIError("503: pyzbar missing")
        )

        with (
            patch("alibi.telegram.handlers.barcode_scan.client", new=client),
            patch(
                "alibi.telegram.handlers.barcode_scan._download_photo",
                AsyncMock(return_value=b"img"),
            ),
        ):
            await scan_handler(mock_message)

        mock_status_msg.edit_text.assert_called_once()
        text = mock_status_msg.edit_text.call_args[0][0]
        assert "not available" in text.lower()


# ---------------------------------------------------------------------------
# handle_enrich callback
# ---------------------------------------------------------------------------


class TestHandleEnrich:
    @pytest.fixture
    def mock_callback(self):
        cb = AsyncMock()
        cb.message = MagicMock(spec=TgMessage)
        cb.message.text = "Barcode card"
        cb.message.edit_text = AsyncMock()
        cb.answer = AsyncMock()
        return cb

    @pytest.mark.asyncio
    async def test_no_match_shows_alert(self, mock_callback):
        """matched==0 → alert, no message edit."""
        callback_data = ScanAction(action="enrich", barcode="4006381333931")
        client = _make_client(enrich={"matched": 0, "enriched": 0})

        with patch("alibi.telegram.handlers.barcode_scan.client", new=client):
            await handle_enrich(mock_callback, callback_data)

        mock_callback.message.edit_text.assert_not_called()
        mock_callback.answer.assert_any_call(
            "No unenriched items found with barcode 4006381333931", show_alert=True
        )

    @pytest.mark.asyncio
    async def test_match_edits_message_with_counts(self, mock_callback):
        """matched>0 → message edited with 'Enriched M/N items'."""
        callback_data = ScanAction(action="enrich", barcode="4006381333931")
        client = _make_client(enrich={"matched": 4, "enriched": 3})

        with patch("alibi.telegram.handlers.barcode_scan.client", new=client):
            await handle_enrich(mock_callback, callback_data)

        mock_callback.message.edit_text.assert_called_once()
        text = mock_callback.message.edit_text.call_args[0][0]
        assert "Enriched 3/4 items" in text
        assert mock_callback.answer.called


# ---------------------------------------------------------------------------
# handle_lookup callback
# ---------------------------------------------------------------------------


class TestHandleLookup:
    @pytest.fixture
    def mock_callback(self):
        cb = AsyncMock()
        cb.message = MagicMock(spec=TgMessage)
        cb.message.text = "Barcode card"
        cb.message.edit_text = AsyncMock()
        cb.answer = AsyncMock()
        return cb

    @pytest.mark.asyncio
    async def test_product_found_edits_message(self, mock_callback):
        """A found OFF product appends product info to the message."""
        callback_data = ScanAction(action="lookup", barcode="5901234123457")
        product = {
            "product_name": "Test Cookies",
            "brands": "TestBrand",
            "nutriscore_grade": "b",
            "categories_tags": ["en:snacks", "en:biscuits"],
        }
        client = _make_client(lookup=product)

        with patch("alibi.telegram.handlers.barcode_scan.client", new=client):
            await handle_lookup(mock_callback, callback_data)

        # First answer dismisses the spinner with "Looking up product..."
        mock_callback.answer.assert_any_call("Looking up product...")
        mock_callback.message.edit_text.assert_called_once()
        text = mock_callback.message.edit_text.call_args[0][0]
        assert "Open Food Facts" in text
        assert "Test Cookies" in text

    @pytest.mark.asyncio
    async def test_product_not_found_edits_message(self, mock_callback):
        """None from the API → 'Product not found' appended."""
        callback_data = ScanAction(action="lookup", barcode="9999999999994")
        client = _make_client(lookup=None)

        with patch("alibi.telegram.handlers.barcode_scan.client", new=client):
            await handle_lookup(mock_callback, callback_data)

        mock_callback.message.edit_text.assert_called_once()
        text = mock_callback.message.edit_text.call_args[0][0]
        assert "not found" in text.lower()
