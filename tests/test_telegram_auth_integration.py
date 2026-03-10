"""Integration tests for Telegram auth and document provenance.

Covers:
- _resolve_telegram_user: unlinked and linked users
- /link command handler: valid and invalid mnemonics
- /unlink command handler: linked and unlinked users
- /whoami command handler: linked and unlinked users
- /setname command handler: name updates for linked users
- Upload provenance: FolderContext source and user_id population
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alibi.services.auth import create_api_key, create_user, find_user_by_telegram
from alibi.telegram.handlers.upload import _resolve_telegram_user


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(
    telegram_user_id: int | None = None,
    text: str | None = None,
    chat_id: int = 9000,
) -> MagicMock:
    """Build a minimal mock Message.

    reply() and answer() are AsyncMock so they can be awaited in handlers.
    """
    msg = MagicMock()
    msg.reply = AsyncMock()
    msg.answer = AsyncMock()
    msg.chat = MagicMock()
    msg.chat.do = AsyncMock()
    msg.chat.id = chat_id
    msg.text = text

    if telegram_user_id is not None:
        msg.from_user = MagicMock()
        msg.from_user.id = telegram_user_id
    else:
        msg.from_user = None

    # No photo / document by default
    msg.photo = None
    msg.document = None
    msg.reply_to_message = None
    msg.bot = None

    return msg


# ---------------------------------------------------------------------------
# _resolve_telegram_user
# ---------------------------------------------------------------------------


class TestResolveTelegramUserIntegration:
    """Integration tests for _resolve_telegram_user using a real DB."""

    def test_unlinked_user_returns_system(self, db):
        """An unrecognised Telegram user ID resolves to 'system'."""
        msg = _make_message(telegram_user_id=111111)
        result = _resolve_telegram_user(db, msg)
        assert result == "system"

    def test_no_from_user_returns_system(self, db):
        """A message with no from_user field resolves to 'system'."""
        msg = _make_message(telegram_user_id=None)
        result = _resolve_telegram_user(db, msg)
        assert result == "system"

    def test_linked_user_returns_user_id(self, db):
        """A Telegram ID that was linked to a user resolves to that user's ID."""
        from alibi.services.auth import link_telegram

        user = create_user(db, "Bob")
        link_telegram(db, user["id"], "222222")

        msg = _make_message(telegram_user_id=222222)
        result = _resolve_telegram_user(db, msg)
        assert result == user["id"]

    def test_different_telegram_id_still_returns_system(self, db):
        """A different Telegram ID from the linked one still returns 'system'."""
        from alibi.services.auth import link_telegram

        user = create_user(db, "Carol")
        link_telegram(db, user["id"], "333333")

        # Different Telegram ID
        msg = _make_message(telegram_user_id=999999)
        result = _resolve_telegram_user(db, msg)
        assert result == "system"

    def test_multiple_users_resolve_independently(self, db):
        """Two independently linked users resolve to their own IDs."""
        from alibi.services.auth import link_telegram

        user_a = create_user(db, "Alice")
        user_b = create_user(db, "Dave")
        link_telegram(db, user_a["id"], "444444")
        link_telegram(db, user_b["id"], "555555")

        msg_a = _make_message(telegram_user_id=444444)
        msg_b = _make_message(telegram_user_id=555555)

        assert _resolve_telegram_user(db, msg_a) == user_a["id"]
        assert _resolve_telegram_user(db, msg_b) == user_b["id"]


# ---------------------------------------------------------------------------
# /link handler
# ---------------------------------------------------------------------------


class TestLinkHandler:
    """Integration tests for link_handler using a real DB and mocked get_db."""

    @pytest.mark.asyncio
    async def test_valid_mnemonic_links_account(self, db):
        """/link with a valid mnemonic links the Telegram account to the user."""
        user = create_user(db, "Eve")
        key = create_api_key(db, user["id"])
        mnemonic = key["mnemonic"]
        tg_id = 777777

        msg = _make_message(
            telegram_user_id=tg_id,
            text=f"/link {mnemonic}",
        )

        with patch("alibi.telegram.handlers.get_db", return_value=db):
            from alibi.telegram.handlers.upload import link_handler

            await link_handler(msg)

        # Verify the Telegram account is now linked
        linked = find_user_by_telegram(db, str(tg_id))
        assert linked is not None
        assert linked["id"] == user["id"]

        # Handler replied with the user's name
        msg.reply.assert_called_once()
        reply_text = msg.reply.call_args[0][0]
        assert "Eve" in reply_text

    @pytest.mark.asyncio
    async def test_valid_mnemonic_reply_contains_welcome(self, db):
        """/link success reply contains 'Linked' confirmation."""
        user = create_user(db, "Frank")
        key = create_api_key(db, user["id"])
        mnemonic = key["mnemonic"]

        msg = _make_message(
            telegram_user_id=888888,
            text=f"/link {mnemonic}",
        )

        with patch("alibi.telegram.handlers.get_db", return_value=db):
            from alibi.telegram.handlers.upload import link_handler

            await link_handler(msg)

        reply_text = msg.reply.call_args[0][0]
        assert "Linked" in reply_text

    @pytest.mark.asyncio
    async def test_invalid_mnemonic_does_not_link(self, db):
        """/link with a bad mnemonic leaves the user unlinked."""
        tg_id = 123456
        msg = _make_message(
            telegram_user_id=tg_id,
            text="/link invalid words that are not a key",
        )

        with patch("alibi.telegram.handlers.get_db", return_value=db):
            from alibi.telegram.handlers.upload import link_handler

            await link_handler(msg)

        # No link should exist
        linked = find_user_by_telegram(db, str(tg_id))
        assert linked is None

    @pytest.mark.asyncio
    async def test_invalid_mnemonic_reply_indicates_failure(self, db):
        """/link with a bad mnemonic sends an error reply."""
        msg = _make_message(
            telegram_user_id=234567,
            text="/link totally wrong mnemonic here",
        )

        with patch("alibi.telegram.handlers.get_db", return_value=db):
            from alibi.telegram.handlers.upload import link_handler

            await link_handler(msg)

        msg.reply.assert_called_once()
        reply_text = msg.reply.call_args[0][0]
        # Handler says "Invalid key" or similar
        assert "Invalid" in reply_text or "invalid" in reply_text

    @pytest.mark.asyncio
    async def test_link_command_with_no_mnemonic_arg(self, db):
        """/link with no argument sends a usage hint."""
        msg = _make_message(
            telegram_user_id=345678,
            text="/link",
        )

        with patch("alibi.telegram.handlers.get_db", return_value=db):
            from alibi.telegram.handlers.upload import link_handler

            await link_handler(msg)

        msg.reply.assert_called_once()
        reply_text = msg.reply.call_args[0][0]
        assert "Usage" in reply_text or "usage" in reply_text

    @pytest.mark.asyncio
    async def test_link_requires_from_user(self, db):
        """/link silently exits when message has no from_user."""
        msg = _make_message(telegram_user_id=None, text="/link somewords here now")

        with patch("alibi.telegram.handlers.get_db", return_value=db):
            from alibi.telegram.handlers.upload import link_handler

            await link_handler(msg)

        # No reply sent — handler returned early
        msg.reply.assert_not_called()


# ---------------------------------------------------------------------------
# /whoami handler
# ---------------------------------------------------------------------------


class TestWhoamiHandler:
    """Integration tests for whoami_handler using a real DB and mocked get_db."""

    @pytest.mark.asyncio
    async def test_linked_user_sees_their_name(self, db):
        """/whoami response contains the linked user's name."""
        from alibi.services.auth import link_telegram

        user = create_user(db, "Grace")
        link_telegram(db, user["id"], "654321")

        msg = _make_message(telegram_user_id=654321, text="/whoami")

        with patch("alibi.telegram.handlers.get_db", return_value=db):
            from alibi.telegram.handlers.upload import whoami_handler

            await whoami_handler(msg)

        msg.reply.assert_called_once()
        reply_text = msg.reply.call_args[0][0]
        assert "Grace" in reply_text

    @pytest.mark.asyncio
    async def test_linked_user_reply_contains_partial_id(self, db):
        """/whoami response contains a truncated form of the user ID."""
        from alibi.services.auth import link_telegram

        user = create_user(db, "Heidi")
        link_telegram(db, user["id"], "765432")

        msg = _make_message(telegram_user_id=765432, text="/whoami")

        with patch("alibi.telegram.handlers.get_db", return_value=db):
            from alibi.telegram.handlers.upload import whoami_handler

            await whoami_handler(msg)

        reply_text = msg.reply.call_args[0][0]
        # Handler shows first 8 chars of the UUID
        assert user["id"][:8] in reply_text

    @pytest.mark.asyncio
    async def test_unlinked_user_sees_not_linked_message(self, db):
        """/whoami for an unlinked Telegram user mentions 'Not linked'."""
        msg = _make_message(telegram_user_id=876543, text="/whoami")

        with patch("alibi.telegram.handlers.get_db", return_value=db):
            from alibi.telegram.handlers.upload import whoami_handler

            await whoami_handler(msg)

        msg.reply.assert_called_once()
        reply_text = msg.reply.call_args[0][0]
        assert "Not linked" in reply_text or "not linked" in reply_text

    @pytest.mark.asyncio
    async def test_unlinked_user_reply_mentions_link_command(self, db):
        """/whoami for an unlinked user hints at the /link command."""
        msg = _make_message(telegram_user_id=987654, text="/whoami")

        with patch("alibi.telegram.handlers.get_db", return_value=db):
            from alibi.telegram.handlers.upload import whoami_handler

            await whoami_handler(msg)

        reply_text = msg.reply.call_args[0][0]
        assert "/link" in reply_text

    @pytest.mark.asyncio
    async def test_whoami_no_from_user_replies_not_linked(self, db):
        """/whoami when message has no from_user sends a 'Not linked' reply."""
        msg = _make_message(telegram_user_id=None, text="/whoami")

        with patch("alibi.telegram.handlers.get_db", return_value=db):
            from alibi.telegram.handlers.upload import whoami_handler

            await whoami_handler(msg)

        msg.reply.assert_called_once()
        reply_text = msg.reply.call_args[0][0]
        assert "Not linked" in reply_text or "not linked" in reply_text


# ---------------------------------------------------------------------------
# Upload provenance — FolderContext population
# ---------------------------------------------------------------------------


class TestUploadProvenance:
    """Tests that _process_attachment sets the correct source and user_id
    on the FolderContext it passes to the processing pipeline.

    We intercept process_file to capture the FolderContext without running
    the actual OCR/LLM pipeline.
    """

    @pytest.mark.asyncio
    async def test_linked_user_provenance(self, db):
        """FolderContext receives source='telegram' and the linked user_id."""
        from alibi.services.auth import link_telegram

        user = create_user(db, "Ivan")
        link_telegram(db, user["id"], "11111111")

        msg = _make_message(telegram_user_id=11111111)

        captured_context: list = []

        def fake_process_file(db_arg, path, folder_context=None, **kwargs):
            captured_context.append(folder_context)
            # Return a minimal success-like object
            result = MagicMock()
            result.success = True
            result.is_duplicate = False
            result.extracted_data = {}
            result.line_items = []
            result.document_id = "fake-doc-id"
            result.record_type = None
            return result

        with (
            patch("alibi.telegram.handlers.get_db", return_value=db),
            patch(
                "alibi.telegram.handlers.upload.process_file",
                side_effect=fake_process_file,
            ),
        ):
            from alibi.telegram.handlers.upload import _process_attachment

            await _process_attachment(
                message=msg,
                data=b"fake-image",
                filename="receipt.jpg",
                doc_type=None,
                vendor_hint=None,
            )

        assert len(captured_context) == 1
        ctx = captured_context[0]
        assert ctx.source == "telegram"
        assert ctx.user_id == user["id"]

    @pytest.mark.asyncio
    async def test_unlinked_user_provenance(self, db):
        """FolderContext receives source='telegram' and user_id='system'."""
        msg = _make_message(telegram_user_id=22222222)

        captured_context: list = []

        def fake_process_file(db_arg, path, folder_context=None, **kwargs):
            captured_context.append(folder_context)
            result = MagicMock()
            result.success = True
            result.is_duplicate = False
            result.extracted_data = {}
            result.line_items = []
            result.document_id = "fake-doc-id"
            result.record_type = None
            return result

        with (
            patch("alibi.telegram.handlers.get_db", return_value=db),
            patch(
                "alibi.telegram.handlers.upload.process_file",
                side_effect=fake_process_file,
            ),
        ):
            from alibi.telegram.handlers.upload import _process_attachment

            await _process_attachment(
                message=msg,
                data=b"fake-image",
                filename="invoice.pdf",
                doc_type=None,
                vendor_hint=None,
            )

        assert len(captured_context) == 1
        ctx = captured_context[0]
        assert ctx.source == "telegram"
        assert ctx.user_id == "system"

    @pytest.mark.asyncio
    async def test_provenance_source_is_always_telegram(self, db):
        """source is 'telegram' regardless of whether the user is linked."""
        msg = _make_message(telegram_user_id=None)  # no from_user

        captured_context: list = []

        def fake_process_file(db_arg, path, folder_context=None, **kwargs):
            captured_context.append(folder_context)
            result = MagicMock()
            result.success = True
            result.is_duplicate = False
            result.extracted_data = {}
            result.line_items = []
            result.document_id = "fake-doc-id"
            result.record_type = None
            return result

        with (
            patch("alibi.telegram.handlers.get_db", return_value=db),
            patch(
                "alibi.telegram.handlers.upload.process_file",
                side_effect=fake_process_file,
            ),
        ):
            from alibi.telegram.handlers.upload import _process_attachment

            await _process_attachment(
                message=msg,
                data=b"data",
                filename="doc.jpg",
                doc_type=None,
                vendor_hint=None,
            )

        assert captured_context[0].source == "telegram"

    @pytest.mark.asyncio
    async def test_provenance_doc_type_forwarded(self, db):
        """FolderContext receives the doc_type passed by the caller."""
        from alibi.db.models import DocumentType

        msg = _make_message(telegram_user_id=33333333)

        captured_context: list = []

        def fake_process_file(db_arg, path, folder_context=None, **kwargs):
            captured_context.append(folder_context)
            result = MagicMock()
            result.success = True
            result.is_duplicate = False
            result.extracted_data = {}
            result.line_items = []
            result.document_id = "fake-doc-id"
            result.record_type = None
            return result

        with (
            patch("alibi.telegram.handlers.get_db", return_value=db),
            patch(
                "alibi.telegram.handlers.upload.process_file",
                side_effect=fake_process_file,
            ),
        ):
            from alibi.telegram.handlers.upload import _process_attachment

            await _process_attachment(
                message=msg,
                data=b"data",
                filename="r.jpg",
                doc_type=DocumentType.RECEIPT,
                vendor_hint="fresko",
            )

        ctx = captured_context[0]
        assert ctx.doc_type == DocumentType.RECEIPT
        assert ctx.vendor_hint == "fresko"


# ---------------------------------------------------------------------------
# /unlink handler
# ---------------------------------------------------------------------------


class TestUnlinkHandler:
    """Integration tests for unlink_handler using a real DB and mocked get_db."""

    @pytest.mark.asyncio
    async def test_unlink_linked_account(self, db):
        """/unlink removes the Telegram link and confirms."""
        from alibi.services.auth import link_telegram

        user = create_user(db, "Nora")
        link_telegram(db, user["id"], "501501")

        msg = _make_message(telegram_user_id=501501, text="/unlink")

        with patch("alibi.telegram.handlers.get_db", return_value=db):
            from alibi.telegram.handlers.upload import unlink_handler

            await unlink_handler(msg)

        # Link should be gone
        linked = find_user_by_telegram(db, "501501")
        assert linked is None

        msg.reply.assert_called_once()
        reply_text = msg.reply.call_args[0][0]
        assert "Unlinked" in reply_text

    @pytest.mark.asyncio
    async def test_unlink_not_linked_account(self, db):
        """/unlink for an unlinked Telegram user reports not linked."""
        msg = _make_message(telegram_user_id=502502, text="/unlink")

        with patch("alibi.telegram.handlers.get_db", return_value=db):
            from alibi.telegram.handlers.upload import unlink_handler

            await unlink_handler(msg)

        msg.reply.assert_called_once()
        reply_text = msg.reply.call_args[0][0]
        assert "not linked" in reply_text.lower()

    @pytest.mark.asyncio
    async def test_unlink_requires_from_user(self, db):
        """/unlink silently exits when message has no from_user."""
        msg = _make_message(telegram_user_id=None, text="/unlink")

        with patch("alibi.telegram.handlers.get_db", return_value=db):
            from alibi.telegram.handlers.upload import unlink_handler

            await unlink_handler(msg)

        msg.reply.assert_not_called()

    @pytest.mark.asyncio
    async def test_unlink_then_whoami_shows_not_linked(self, db):
        """/whoami after /unlink shows 'not linked'."""
        from alibi.services.auth import link_telegram

        user = create_user(db, "Otto")
        link_telegram(db, user["id"], "503503")

        unlink_msg = _make_message(telegram_user_id=503503, text="/unlink")
        whoami_msg = _make_message(telegram_user_id=503503, text="/whoami")

        with patch("alibi.telegram.handlers.get_db", return_value=db):
            from alibi.telegram.handlers.upload import unlink_handler, whoami_handler

            await unlink_handler(unlink_msg)
            await whoami_handler(whoami_msg)

        reply_text = whoami_msg.reply.call_args[0][0]
        assert "Not linked" in reply_text or "not linked" in reply_text


# ---------------------------------------------------------------------------
# /setname handler
# ---------------------------------------------------------------------------


class TestSetnameHandler:
    """Integration tests for setname_handler using a real DB and mocked get_db."""

    @pytest.mark.asyncio
    async def test_setname_updates_display_name(self, db):
        """/setname updates the user's name in the DB."""
        from alibi.services.auth import get_user, link_telegram

        user = create_user(db, "Petra")
        link_telegram(db, user["id"], "601601")

        msg = _make_message(telegram_user_id=601601, text="/setname Quinn")

        with patch("alibi.telegram.handlers.get_db", return_value=db):
            from alibi.telegram.handlers.upload import setname_handler

            await setname_handler(msg)

        msg.reply.assert_called_once()
        reply_text = msg.reply.call_args[0][0]
        assert "Quinn" in reply_text

        updated = get_user(db, user["id"])
        assert updated["name"] == "Quinn"

    @pytest.mark.asyncio
    async def test_setname_no_args_shows_usage(self, db):
        """/setname with no argument sends a usage hint."""
        from alibi.services.auth import link_telegram

        user = create_user(db, "Ruth")
        link_telegram(db, user["id"], "602602")

        msg = _make_message(telegram_user_id=602602, text="/setname")

        with patch("alibi.telegram.handlers.get_db", return_value=db):
            from alibi.telegram.handlers.upload import setname_handler

            await setname_handler(msg)

        msg.reply.assert_called_once()
        reply_text = msg.reply.call_args[0][0]
        assert "Usage" in reply_text or "usage" in reply_text

    @pytest.mark.asyncio
    async def test_setname_unlinked_user_fails(self, db):
        """/setname for an unlinked Telegram user reports not linked."""
        msg = _make_message(telegram_user_id=603603, text="/setname NewName")

        with patch("alibi.telegram.handlers.get_db", return_value=db):
            from alibi.telegram.handlers.upload import setname_handler

            await setname_handler(msg)

        msg.reply.assert_called_once()
        reply_text = msg.reply.call_args[0][0]
        assert "Not linked" in reply_text or "not linked" in reply_text

    @pytest.mark.asyncio
    async def test_setname_requires_from_user(self, db):
        """/setname silently exits when message has no from_user."""
        msg = _make_message(telegram_user_id=None, text="/setname Bob")

        with patch("alibi.telegram.handlers.get_db", return_value=db):
            from alibi.telegram.handlers.upload import setname_handler

            await setname_handler(msg)

        msg.reply.assert_not_called()

    @pytest.mark.asyncio
    async def test_setname_with_spaces(self, db):
        """/setname handles multi-word names."""
        from alibi.services.auth import get_user, link_telegram

        user = create_user(db, "Sam")
        link_telegram(db, user["id"], "604604")

        msg = _make_message(telegram_user_id=604604, text="/setname Sam Smith Jr")

        with patch("alibi.telegram.handlers.get_db", return_value=db):
            from alibi.telegram.handlers.upload import setname_handler

            await setname_handler(msg)

        reply_text = msg.reply.call_args[0][0]
        assert "Sam Smith Jr" in reply_text

        updated = get_user(db, user["id"])
        assert updated["name"] == "Sam Smith Jr"


# ---------------------------------------------------------------------------
# Full link-then-upload round-trip
# ---------------------------------------------------------------------------


class TestLinkThenUploadRoundTrip:
    """End-to-end: link via /link, then verify provenance on upload."""

    @pytest.mark.asyncio
    async def test_link_then_upload_carries_user_id(self, db):
        """After /link, a subsequent upload carries the correct user_id."""
        user = create_user(db, "Judy")
        key = create_api_key(db, user["id"])
        tg_id = 44444444

        # Step 1: link via /link handler
        link_msg = _make_message(
            telegram_user_id=tg_id,
            text=f"/link {key['mnemonic']}",
        )
        with patch("alibi.telegram.handlers.get_db", return_value=db):
            from alibi.telegram.handlers.upload import link_handler

            await link_handler(link_msg)

        # Verify linked
        assert find_user_by_telegram(db, str(tg_id)) is not None

        # Step 2: simulate upload
        upload_msg = _make_message(telegram_user_id=tg_id)
        captured_context: list = []

        def fake_process_file(db_arg, path, folder_context=None, **kwargs):
            captured_context.append(folder_context)
            result = MagicMock()
            result.success = True
            result.is_duplicate = False
            result.extracted_data = {}
            result.line_items = []
            result.document_id = "fake-doc-id"
            result.record_type = None
            return result

        with (
            patch("alibi.telegram.handlers.get_db", return_value=db),
            patch(
                "alibi.telegram.handlers.upload.process_file",
                side_effect=fake_process_file,
            ),
        ):
            from alibi.telegram.handlers.upload import _process_attachment

            await _process_attachment(
                message=upload_msg,
                data=b"image",
                filename="r.jpg",
                doc_type=None,
                vendor_hint=None,
            )

        ctx = captured_context[0]
        assert ctx.source == "telegram"
        assert ctx.user_id == user["id"]
