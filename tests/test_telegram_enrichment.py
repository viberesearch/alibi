"""Tests for Telegram enrichment review handler with inline keyboards."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alibi.telegram.handlers.enrichment import (
    EnrichAction,
    _esc,
    _fmt_conf,
    enrich_handler,
    handle_confirm,
    handle_reject,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_message():
    """Create a mock Telegram Message object."""
    msg = AsyncMock()
    msg.answer = AsyncMock()
    return msg


@pytest.fixture
def mock_db():
    """Create a mock DatabaseManager."""
    db = MagicMock()
    db.is_initialized.return_value = True
    db.fetchall.return_value = []
    return db


@pytest.fixture
def mock_callback():
    """Create a mock CallbackQuery object."""
    from aiogram.types import Message as TgMessage

    cb = AsyncMock()
    cb.message = AsyncMock(spec=TgMessage)
    cb.message.text = "Some item text"
    cb.message.edit_text = AsyncMock()
    cb.answer = AsyncMock()
    cb.data = "enrich:confirm:some-uuid"
    return cb


def _make_queue_item(
    item_id: str = "aaaa-bbbb-cccc-dddd-eeee00000001",
    name: str = "Olive Oil",
    brand: str | None = "Minerva",
    category: str | None = "groceries",
    source: str = "openfoodfacts",
    confidence: float = 0.6,
    vendor: str | None = "Fresko",
) -> dict:
    return {
        "id": item_id,
        "name": name,
        "brand": brand,
        "category": category,
        "enrichment_source": source,
        "enrichment_confidence": confidence,
        "fact_id": "fact-001",
        "vendor": vendor,
    }


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_esc_empty_string(self):
        assert _esc("") == ""

    def test_esc_plain_text(self):
        assert _esc("Hello World") == "Hello World"

    def test_esc_underscores(self):
        assert _esc("some_item_name") == "some\\_item\\_name"

    def test_esc_asterisks(self):
        assert _esc("bold*text") == "bold\\*text"

    def test_esc_backticks(self):
        assert _esc("code`block") == "code\\`block"

    def test_esc_brackets(self):
        assert _esc("[link]") == "\\[link]"

    def test_fmt_conf_none(self):
        assert _fmt_conf(None) == "—"

    def test_fmt_conf_zero(self):
        assert _fmt_conf(0.0) == "0%"

    def test_fmt_conf_half(self):
        assert _fmt_conf(0.5) == "50%"

    def test_fmt_conf_full(self):
        assert _fmt_conf(1.0) == "100%"

    def test_fmt_conf_rounds(self):
        assert _fmt_conf(0.666) == "67%"


# ---------------------------------------------------------------------------
# EnrichAction CallbackData tests
# ---------------------------------------------------------------------------


class TestEnrichAction:
    def test_pack_unpack_confirm(self):
        action = EnrichAction(action="confirm", item_id="aaaa-bbbb-cccc-dddd-1234")
        packed = action.pack()
        unpacked = EnrichAction.unpack(packed)
        assert unpacked.action == "confirm"
        assert unpacked.item_id == "aaaa-bbbb-cccc-dddd-1234"

    def test_pack_unpack_reject(self):
        action = EnrichAction(action="reject", item_id="ffff-eeee-dddd-cccc-5678")
        packed = action.pack()
        unpacked = EnrichAction.unpack(packed)
        assert unpacked.action == "reject"
        assert unpacked.item_id == "ffff-eeee-dddd-cccc-5678"

    def test_packed_string_within_64_bytes(self):
        # Telegram inline callback_data limit is 64 bytes
        action = EnrichAction(
            action="confirm", item_id="aaaa-bbbb-cccc-dddd-eeee00000001"
        )
        assert len(action.pack().encode("utf-8")) <= 64

    def test_packed_string_reject_within_64_bytes(self):
        action = EnrichAction(
            action="reject", item_id="aaaa-bbbb-cccc-dddd-eeee00000001"
        )
        assert len(action.pack().encode("utf-8")) <= 64


# ---------------------------------------------------------------------------
# /enrich command handler tests
# ---------------------------------------------------------------------------


class TestEnrichHandler:
    @pytest.mark.asyncio
    async def test_empty_queue_shows_no_items_message(self, mock_message, mock_db):
        """Empty queue returns a helpful 'no pending items' message."""
        with (
            patch("alibi.telegram.handlers.enrichment.get_db", return_value=mock_db),
            patch("alibi.telegram.handlers.enrichment.enrichment_review") as mock_svc,
        ):
            mock_svc.get_review_queue.return_value = []

            await enrich_handler(mock_message)

        mock_message.answer.assert_called_once()
        text = mock_message.answer.call_args[0][0]
        assert "No items pending" in text
        mock_svc.get_review_stats.assert_not_called()

    @pytest.mark.asyncio
    async def test_queue_with_items_sends_header_and_item_messages(
        self, mock_message, mock_db
    ):
        """One item in queue: sends header message + one item message."""
        item = _make_queue_item()
        stats = {"pending_review": 1}

        with (
            patch("alibi.telegram.handlers.enrichment.get_db", return_value=mock_db),
            patch("alibi.telegram.handlers.enrichment.enrichment_review") as mock_svc,
        ):
            mock_svc.get_review_queue.return_value = [item]
            mock_svc.get_review_stats.return_value = stats

            await enrich_handler(mock_message)

        # One header + one item card = 2 calls
        assert mock_message.answer.call_count == 2

    @pytest.mark.asyncio
    async def test_item_message_contains_name_brand_source(self, mock_message, mock_db):
        """Item card message contains the item name, brand, and source."""
        item = _make_queue_item(
            name="Olive Oil", brand="Minerva", source="openfoodfacts"
        )
        stats = {"pending_review": 1}

        with (
            patch("alibi.telegram.handlers.enrichment.get_db", return_value=mock_db),
            patch("alibi.telegram.handlers.enrichment.enrichment_review") as mock_svc,
        ):
            mock_svc.get_review_queue.return_value = [item]
            mock_svc.get_review_stats.return_value = stats

            await enrich_handler(mock_message)

        # Second call is the item card (index 1)
        item_call = mock_message.answer.call_args_list[1]
        text = item_call[0][0]
        assert "Olive Oil" in text
        assert "Minerva" in text
        assert "openfoodfacts" in text

    @pytest.mark.asyncio
    async def test_item_message_has_inline_keyboard(self, mock_message, mock_db):
        """Item card message includes an InlineKeyboardMarkup."""
        from aiogram.types import InlineKeyboardMarkup

        item = _make_queue_item()
        stats = {"pending_review": 1}

        with (
            patch("alibi.telegram.handlers.enrichment.get_db", return_value=mock_db),
            patch("alibi.telegram.handlers.enrichment.enrichment_review") as mock_svc,
        ):
            mock_svc.get_review_queue.return_value = [item]
            mock_svc.get_review_stats.return_value = stats

            await enrich_handler(mock_message)

        item_call = mock_message.answer.call_args_list[1]
        keyboard = item_call[1].get("reply_markup")
        assert isinstance(keyboard, InlineKeyboardMarkup)
        # Should have one row with two buttons
        assert len(keyboard.inline_keyboard) == 1
        assert len(keyboard.inline_keyboard[0]) == 2

    @pytest.mark.asyncio
    async def test_inline_keyboard_button_labels(self, mock_message, mock_db):
        """Confirm and Reject buttons have correct text labels."""
        item = _make_queue_item()
        stats = {"pending_review": 1}

        with (
            patch("alibi.telegram.handlers.enrichment.get_db", return_value=mock_db),
            patch("alibi.telegram.handlers.enrichment.enrichment_review") as mock_svc,
        ):
            mock_svc.get_review_queue.return_value = [item]
            mock_svc.get_review_stats.return_value = stats

            await enrich_handler(mock_message)

        item_call = mock_message.answer.call_args_list[1]
        keyboard = item_call[1].get("reply_markup")
        buttons = keyboard.inline_keyboard[0]
        button_texts = [btn.text for btn in buttons]
        assert any("Confirm" in t for t in button_texts)
        assert any("Reject" in t for t in button_texts)

    @pytest.mark.asyncio
    async def test_multiple_items_in_queue(self, mock_message, mock_db):
        """Three items in queue produces header + 3 item messages."""
        items = [
            _make_queue_item(item_id=f"id-{i}", name=f"Item {i}") for i in range(3)
        ]
        stats = {"pending_review": 3}

        with (
            patch("alibi.telegram.handlers.enrichment.get_db", return_value=mock_db),
            patch("alibi.telegram.handlers.enrichment.enrichment_review") as mock_svc,
        ):
            mock_svc.get_review_queue.return_value = items
            mock_svc.get_review_stats.return_value = stats

            await enrich_handler(mock_message)

        # 1 header + 3 items
        assert mock_message.answer.call_count == 4

    @pytest.mark.asyncio
    async def test_null_brand_renders_dash(self, mock_message, mock_db):
        """Item with no brand renders em-dash placeholder."""
        item = _make_queue_item(brand=None, category=None)
        stats = {"pending_review": 1}

        with (
            patch("alibi.telegram.handlers.enrichment.get_db", return_value=mock_db),
            patch("alibi.telegram.handlers.enrichment.enrichment_review") as mock_svc,
        ):
            mock_svc.get_review_queue.return_value = [item]
            mock_svc.get_review_stats.return_value = stats

            await enrich_handler(mock_message)

        item_call = mock_message.answer.call_args_list[1]
        text = item_call[0][0]
        assert "—" in text

    @pytest.mark.asyncio
    async def test_get_review_queue_called_with_limit_5(self, mock_message, mock_db):
        """enrich_handler calls get_review_queue with limit=5."""
        with (
            patch("alibi.telegram.handlers.enrichment.get_db", return_value=mock_db),
            patch("alibi.telegram.handlers.enrichment.enrichment_review") as mock_svc,
        ):
            mock_svc.get_review_queue.return_value = []

            await enrich_handler(mock_message)

        mock_svc.get_review_queue.assert_called_once_with(mock_db, limit=5)


# ---------------------------------------------------------------------------
# Callback query handler tests
# ---------------------------------------------------------------------------


class TestHandleConfirm:
    @pytest.mark.asyncio
    async def test_confirm_success_edits_message(self, mock_callback, mock_db):
        """Successful confirm edits message text and removes keyboard."""
        callback_data = EnrichAction(action="confirm", item_id="item-uuid-001")

        with (
            patch("alibi.telegram.handlers.enrichment.get_db", return_value=mock_db),
            patch("alibi.telegram.handlers.enrichment.enrichment_review") as mock_svc,
        ):
            mock_svc.confirm_enrichment.return_value = True

            await handle_confirm(mock_callback, callback_data)

        mock_svc.confirm_enrichment.assert_called_once_with(mock_db, "item-uuid-001")
        mock_callback.message.edit_text.assert_called_once()
        edit_call = mock_callback.message.edit_text.call_args
        new_text = edit_call[0][0]
        assert "Confirmed" in new_text
        # reply_markup should be None (keyboard removed)
        assert edit_call[1].get("reply_markup") is None

    @pytest.mark.asyncio
    async def test_confirm_success_appends_to_original_text(
        self, mock_callback, mock_db
    ):
        """Confirmed status is appended to the existing message text."""
        mock_callback.message.text = "Olive Oil\nBrand: Minerva"
        callback_data = EnrichAction(action="confirm", item_id="item-uuid-001")

        with (
            patch("alibi.telegram.handlers.enrichment.get_db", return_value=mock_db),
            patch("alibi.telegram.handlers.enrichment.enrichment_review") as mock_svc,
        ):
            mock_svc.confirm_enrichment.return_value = True

            await handle_confirm(mock_callback, callback_data)

        edit_call = mock_callback.message.edit_text.call_args
        new_text = edit_call[0][0]
        assert new_text.startswith("Olive Oil\nBrand: Minerva")

    @pytest.mark.asyncio
    async def test_confirm_not_found_shows_alert(self, mock_callback, mock_db):
        """When item not found, shows alert and skips edit."""
        callback_data = EnrichAction(action="confirm", item_id="missing-uuid")

        with (
            patch("alibi.telegram.handlers.enrichment.get_db", return_value=mock_db),
            patch("alibi.telegram.handlers.enrichment.enrichment_review") as mock_svc,
        ):
            mock_svc.confirm_enrichment.return_value = False

            await handle_confirm(mock_callback, callback_data)

        mock_callback.message.edit_text.assert_not_called()
        mock_callback.answer.assert_any_call("Item not found", show_alert=True)

    @pytest.mark.asyncio
    async def test_confirm_always_calls_answer(self, mock_callback, mock_db):
        """callback.answer() is always called to dismiss loading spinner."""
        callback_data = EnrichAction(action="confirm", item_id="item-uuid-001")

        with (
            patch("alibi.telegram.handlers.enrichment.get_db", return_value=mock_db),
            patch("alibi.telegram.handlers.enrichment.enrichment_review") as mock_svc,
        ):
            mock_svc.confirm_enrichment.return_value = True

            await handle_confirm(mock_callback, callback_data)

        # answer() must be called (to dismiss the spinner)
        assert mock_callback.answer.called


class TestHandleReject:
    @pytest.mark.asyncio
    async def test_reject_success_edits_message(self, mock_callback, mock_db):
        """Successful reject edits message text and removes keyboard."""
        callback_data = EnrichAction(action="reject", item_id="item-uuid-002")

        with (
            patch("alibi.telegram.handlers.enrichment.get_db", return_value=mock_db),
            patch("alibi.telegram.handlers.enrichment.enrichment_review") as mock_svc,
        ):
            mock_svc.reject_enrichment.return_value = True

            await handle_reject(mock_callback, callback_data)

        mock_svc.reject_enrichment.assert_called_once_with(mock_db, "item-uuid-002")
        mock_callback.message.edit_text.assert_called_once()
        edit_call = mock_callback.message.edit_text.call_args
        new_text = edit_call[0][0]
        assert "Rejected" in new_text
        assert edit_call[1].get("reply_markup") is None

    @pytest.mark.asyncio
    async def test_reject_success_appends_to_original_text(
        self, mock_callback, mock_db
    ):
        """Rejected status is appended to the existing message text."""
        mock_callback.message.text = "Olive Oil\nBrand: Minerva"
        callback_data = EnrichAction(action="reject", item_id="item-uuid-002")

        with (
            patch("alibi.telegram.handlers.enrichment.get_db", return_value=mock_db),
            patch("alibi.telegram.handlers.enrichment.enrichment_review") as mock_svc,
        ):
            mock_svc.reject_enrichment.return_value = True

            await handle_reject(mock_callback, callback_data)

        edit_call = mock_callback.message.edit_text.call_args
        new_text = edit_call[0][0]
        assert new_text.startswith("Olive Oil\nBrand: Minerva")

    @pytest.mark.asyncio
    async def test_reject_not_found_shows_alert(self, mock_callback, mock_db):
        """When item not found, shows alert and skips edit."""
        callback_data = EnrichAction(action="reject", item_id="missing-uuid")

        with (
            patch("alibi.telegram.handlers.enrichment.get_db", return_value=mock_db),
            patch("alibi.telegram.handlers.enrichment.enrichment_review") as mock_svc,
        ):
            mock_svc.reject_enrichment.return_value = False

            await handle_reject(mock_callback, callback_data)

        mock_callback.message.edit_text.assert_not_called()
        mock_callback.answer.assert_any_call("Item not found", show_alert=True)

    @pytest.mark.asyncio
    async def test_reject_always_calls_answer(self, mock_callback, mock_db):
        """callback.answer() is always called to dismiss loading spinner."""
        callback_data = EnrichAction(action="reject", item_id="item-uuid-002")

        with (
            patch("alibi.telegram.handlers.enrichment.get_db", return_value=mock_db),
            patch("alibi.telegram.handlers.enrichment.enrichment_review") as mock_svc,
        ):
            mock_svc.reject_enrichment.return_value = True

            await handle_reject(mock_callback, callback_data)

        assert mock_callback.answer.called
