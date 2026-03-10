"""Telegram bot command handlers."""

try:
    from aiogram import Router
    from aiogram.types import Message
except ImportError:
    raise ImportError("Telegram support requires: uv sync --extra telegram")

from alibi.db.connection import DatabaseManager, get_db


async def require_db(message: Message) -> DatabaseManager | None:
    """Get initialized DB or send error to user. Returns None if not ready."""
    db = get_db()
    if not db.is_initialized():
        await message.answer("Database not initialized. Please run `lt init` first.")
        return None
    return db


from .annotation import router as annotation_router
from .barcode_scan import router as barcode_scan_router
from .budget import router as budget_router
from .correction import router as correction_router
from .enrichment import router as enrichment_router
from .expenses import router as expenses_router
from .find import router as find_router
from .help import router as help_router
from .language import router as language_router
from .lineitem import router as lineitem_router
from .summary import router as summary_router
from .upload import router as upload_router
from .warranty import router as warranty_router

# Main router that includes all handlers
# Order matters: command handlers before catch-all photo/document handler
router = Router()
router.include_router(help_router)
router.include_router(barcode_scan_router)  # Before catch-all photo handler
router.include_router(upload_router)  # Before catch-all photo handler
router.include_router(expenses_router)
router.include_router(warranty_router)
router.include_router(find_router)
router.include_router(summary_router)
router.include_router(budget_router)
router.include_router(lineitem_router)
router.include_router(language_router)
router.include_router(enrichment_router)
router.include_router(correction_router)
router.include_router(annotation_router)

__all__ = ["require_db", "router"]
