"""Find command handler — thin client over the host ``/search`` endpoint."""

import logging

from aiogram import Router
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import Message

from alibi.telegram.api_client import AlibiAPIError
from alibi.telegram.handlers._common import api_key_for, client

router = Router()
logger = logging.getLogger(__name__)


@router.message(Command("find"))
async def find_handler(message: Message) -> None:
    """Handle /find command - search transactions, items and documents.

    Forwards the query to the host API, which uses vector similarity search
    when LanceDB is configured and falls back to SQL otherwise.

    Usage: /find <query>
    Example: /find amazon
    """
    if not message.text:
        await message.answer("Usage: /find <query>")
        return

    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Please provide a search query. Usage: /find <query>")
        return

    query = parts[1].strip()

    await message.chat.do(ChatAction.TYPING)

    try:
        body = await client.search(query, api_key=api_key_for(message), limit=15)
    except AlibiAPIError:
        logger.exception("Search failed for %r", query)
        await message.answer("Search encountered an error. Please try again.")
        return

    results = body.get("results", [])
    if not results:
        await message.answer(f"No results found for '{query}'")
        return

    facts = [r for r in results if r.get("result_type") == "fact"]
    items = [r for r in results if r.get("result_type") == "item"]
    documents = [r for r in results if r.get("result_type") == "document"]

    response = f"*Search Results for '{query}'*\n\n"

    if facts:
        response += "*Transactions:*\n"
        for r in facts[:5]:
            title = (r.get("title") or "Unknown")[:20]
            date_val = r.get("document_date") or ""
            amount = r.get("amount")
            line = f"{date_val} - {title}"
            if amount is not None:
                line += f": {float(amount):.2f}"
            response += line + "\n"
            if r.get("subtype"):
                response += f"  _{r['subtype']}_\n"
        response += "\n"

    if items:
        response += "*Items:*\n"
        for r in items[:5]:
            title = (r.get("title") or "Unknown")[:30]
            response += f"*{title}*\n"
            if r.get("subtype"):
                response += f"  Category: {r['subtype']}\n"
            if r.get("document_date"):
                response += f"  Purchased: {r['document_date']}\n"
            if r.get("amount") is not None:
                response += f"  Price: {float(r['amount']):.2f}\n"
        response += "\n"

    if documents:
        response += "*Documents:*\n"
        for r in documents[:5]:
            title = (r.get("title") or "Unknown")[:30]
            response += f"{title}"
            if r.get("document_date"):
                response += f" ({r['document_date']})"
            response += "\n"

    if len(response) > 4000:
        response = response[:3900] + "\n\n_...truncated_"

    await message.answer(response, parse_mode="Markdown")
