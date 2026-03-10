"""Find command handler with vector search support."""

import logging
from typing import TYPE_CHECKING

from aiogram import Router
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import Message

from alibi.services.query import search_facts
from alibi.telegram.handlers import require_db

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager

router = Router()
logger = logging.getLogger(__name__)


@router.message(Command("find"))
async def find_handler(message: Message) -> None:
    """Handle /find command - search transactions and items.

    Uses vector search if available, falls back to SQL search via the
    service layer.

    Usage: /find <query>
    Example: /find amazon
    Example: /find grocery shopping
    """
    if not message.text:
        await message.answer("Usage: /find <query>")
        return

    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Please provide a search query. Usage: /find <query>")
        return

    query = parts[1]

    db = await require_db(message)
    if db is None:
        return

    await message.chat.do(ChatAction.TYPING)

    # Try unified search (vector + SQL fallback)
    try:
        from alibi.vectordb.index import VectorIndex
        from alibi.vectordb.search import unified_search_async

        index = VectorIndex()
        use_vector = index.is_initialized()

        results = await unified_search_async(
            db=db,
            index=index if use_vector else None,
            query=query,
            limit=10,
            use_vector=use_vector,
            use_sql=True,
        )

        if not results:
            await message.answer(f"No results found for '{query}'")
            return

        # Format response
        response = f"*Search Results for '{query}'*\n"
        if use_vector:
            response += "_Using semantic search_\n"
        response += "\n"

        # Group results by entity type
        transactions = [r for r in results if r.entity_type == "transaction"]
        items = [r for r in results if r.entity_type == "item"]
        artifacts = [r for r in results if r.entity_type == "artifact"]

        if transactions:
            response += "*Transactions:*\n"
            for r in transactions[:5]:
                vendor = r.vendor or "Unknown"
                amount_str = f"{r.amount:.2f} {r.currency}" if r.amount else ""
                desc = r.description[:25] if r.description else ""

                response += f"{r.date or ''} - {vendor[:20]}"
                if amount_str:
                    response += f": {amount_str}"
                response += "\n"
                if desc and desc != vendor:
                    response += f"  _{desc}_\n"
            response += "\n"

        if items:
            response += "*Items:*\n"
            for r in items[:5]:
                metadata = r.metadata or {}
                name = metadata.get("name", r.description or "Unknown")
                category = metadata.get("category", "")

                response += f"*{name[:30]}*\n"
                if category:
                    response += f"  Category: {category}\n"
                if r.date:
                    response += f"  Purchased: {r.date}\n"
                if r.amount:
                    response += f"  Price: {r.amount:.2f} {r.currency}\n"
                response += "\n"

        if artifacts:
            response += "*Documents:*\n"
            for r in artifacts[:5]:
                vendor = r.vendor or "Unknown"
                metadata = r.metadata or {}
                doc_type = metadata.get("artifact_type", "document")

                response += f"{doc_type.title()}: {vendor[:25]}"
                if r.amount:
                    response += f" - {r.amount:.2f} {r.currency}"
                if r.date:
                    response += f" ({r.date})"
                response += "\n"

        if len(response) > 4000:
            response = response[:3900] + "\n\n_...truncated_"

        await message.answer(response, parse_mode="Markdown")

    except Exception:
        logger.exception("Unified search failed, falling back to service layer search")
        try:
            await _service_search_fallback(message, query, db)
        except Exception:
            logger.exception("Service layer search also failed")
            await message.answer("Search encountered an error. Please try again.")


async def _service_search_fallback(
    message: Message, query: str, db: "DatabaseManager"
) -> None:
    """Fallback to service layer search when vector search fails."""
    result = search_facts(db, query, limit=10)
    facts = result["facts"]

    if not facts:
        await message.answer(f"No results found for '{query}'")
        return

    response = f"*Search Results for '{query}'*\n\n"
    response += "*Transactions:*\n"

    for fact in facts:
        vendor = fact.get("vendor") or "Unknown"
        txn_date = fact.get("event_date") or ""
        amount = float(fact["total_amount"]) if fact.get("total_amount") else 0.0
        currency = fact.get("currency") or "EUR"
        fact_type = fact.get("fact_type") or ""

        response += f"{txn_date} - {vendor[:20]}: {amount:.2f} {currency}\n"
        if fact_type:
            response += f"  _{fact_type}_\n"

    if result["total"] > len(facts):
        response += f"\n_Showing {len(facts)} of {result['total']} matches_\n"

    await message.answer(response, parse_mode="Markdown")
