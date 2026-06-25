"""Annotation command handlers — thin client over the host ``/annotations``."""

from __future__ import annotations

import logging
import re

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from alibi.telegram.api_client import AlibiAPIError
from alibi.telegram.handlers._common import api_key_for, client

router = Router()
logger = logging.getLogger(__name__)

_FACT_ID_RE = re.compile(r"Fact ID: `([^`]+)`")

# Match a UUID or any string that looks like an ID (contains hyphen or
# is long enough to be a UUID).
_UUID_MIN_LEN = 36


def _looks_like_id(token: str) -> bool:
    """Return True if token looks like a UUID or similar identifier."""
    return "-" in token or len(token) >= _UUID_MIN_LEN


def _extract_fact_id_from_reply(message: Message) -> str | None:
    """Pull the fact_id out of the replied-to message's text, if present."""
    reply = message.reply_to_message
    if not reply or not reply.text:
        return None
    m = _FACT_ID_RE.search(reply.text)
    return m.group(1) if m else None


def _parse_tag_args(raw: str) -> tuple[str | None, str | None, str | None]:
    """Parse the argument string after '/tag' into (fact_id, key, value).

    Handles four forms:
        <fact_id> <key> <value>
        <fact_id> <key> "quoted value"
        <key> <value>
        <key> "quoted value"

    Returns (fact_id_or_None, key_or_None, value_or_None).
    """
    token_re = re.compile(r'"([^"]*)"|\S+')
    tokens = [
        m.group(1) if m.group(1) is not None else m.group(0)
        for m in token_re.finditer(raw)
    ]

    if len(tokens) < 2:
        return None, None, None

    if _looks_like_id(tokens[0]):
        fact_id = tokens[0]
        rest = tokens[1:]
    else:
        fact_id = None
        rest = tokens

    if len(rest) < 2:
        return fact_id, None, None

    key = rest[0]
    value = " ".join(rest[1:])
    return fact_id, key, value


@router.message(Command("tag"))
async def tag_handler(message: Message) -> None:
    """Handle /tag command - annotate a fact with a key/value tag.

    Usage:
        /tag <fact_id> <key> <value>
        /tag <fact_id> <key> "quoted value"

    When replying to a bot message that contains "Fact ID: `xxx`", the
    fact_id may be omitted:
        /tag project "kitchen renovation"
        /tag person Maria
    """
    if not message.text:
        await message.answer("Usage: /tag [fact_id] <key> <value>")
        return

    raw_parts = message.text.split(maxsplit=1)
    if len(raw_parts) < 2:
        await message.answer("Usage: /tag [fact_id] <key> <value>")
        return

    fact_id, key, value = _parse_tag_args(raw_parts[1])

    if not fact_id:
        fact_id = _extract_fact_id_from_reply(message)

    if not fact_id:
        await message.answer(
            "No fact ID provided and no replied-to message with a Fact ID."
        )
        return

    if not key or not value:
        await message.answer("Usage: /tag [fact_id] <key> <value>")
        return

    try:
        annotation_id = await client.annotate_fact(
            fact_id,
            annotation_type="user_tag",
            key=key,
            value=value,
            api_key=api_key_for(message),
        )
    except AlibiAPIError as exc:
        logger.exception("tag_handler: failed to annotate fact %s", fact_id)
        await message.answer(f"Failed to add tag: {exc}")
        return

    await message.answer(
        f"Tag added.\n"
        f"Fact ID: `{fact_id}`\n"
        f"Key: {key}\n"
        f"Value: {value}\n"
        f"Annotation ID: `{annotation_id}`",
        parse_mode="Markdown",
    )


@router.message(Command("untag"))
async def untag_handler(message: Message) -> None:
    """Handle /untag command - delete an annotation by ID.

    Usage: /untag <annotation_id>
    """
    if not message.text:
        await message.answer("Usage: /untag <annotation_id>")
        return

    parts = message.text.split()
    if len(parts) != 2:
        await message.answer("Usage: /untag <annotation_id>")
        return

    annotation_id = parts[1]

    try:
        ok = await client.delete_annotation(annotation_id, api_key=api_key_for(message))
    except AlibiAPIError:
        logger.exception("untag_handler: failed to delete %s", annotation_id)
        await message.answer("Could not delete annotation. Please try again.")
        return

    if ok:
        await message.answer(
            f"Annotation deleted.\nAnnotation ID: `{annotation_id}`",
            parse_mode="Markdown",
        )
    else:
        await message.answer(
            f"Annotation not found: `{annotation_id}`",
            parse_mode="Markdown",
        )
