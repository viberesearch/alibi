"""Telegram bot command handlers.

The bot is a thin HTTP client of the host API: handlers carry no DB or pipeline
dependency. Importing this package pulls in only aiogram + httpx, so it can run
in a slim container (see ``docs/TELEGRAM_THIN_BOT_PLAN.md``).
"""

try:
    from aiogram import Router
except ImportError:
    raise ImportError("Telegram support requires: uv sync --extra telegram")

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

__all__ = ["router"]
