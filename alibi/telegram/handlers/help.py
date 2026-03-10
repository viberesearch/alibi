"""Help command handler."""

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

router = Router()


@router.message(Command("start"))
async def start_handler(message: Message) -> None:
    """Handle /start command - show welcome message."""
    welcome_text = (
        "Welcome to Alibi -- your local-first financial document intelligence system.\n"
        "\n"
        "Send a receipt photo to get started, or use a type command first:\n"
        "/receipt, /invoice, /payment, /warranty\n"
        "\n"
        "Other useful commands:\n"
        "/help -- full command list\n"
        "/expenses -- recent transactions\n"
        "/find <query> -- search facts\n"
        "/summary -- monthly spending overview\n"
        "\n"
        "For setup: /link to connect your Telegram account."
    )
    await message.answer(welcome_text)


@router.message(Command("help"))
async def help_handler(message: Message) -> None:
    """Handle /help command - show full command list."""
    help_text = """*Alibi -- Document Intelligence Bot*

*Document Upload:*
/receipt [vendor] - Upload a receipt
/invoice [vendor] - Upload an invoice
/payment [vendor] - Upload a payment confirmation
/statement - Upload a bank statement
/warranty - Upload a warranty document
/contract - Upload a contract
/upload [type] [vendor] - Generic upload

*Queries:*
/expenses [days] - Recent expenses (default 7 days)
/find <query> - Search transactions and items
/summary - Monthly spending summary
/lineitem [category|search term] - Query line items

*Corrections:*
/fix <fact\\_id> <field> <value> - Correct vendor, amount, or date
/merge <id\\_a> <id\\_b> - Merge vendor identities
/barcode [item\\_id] <code> - Set barcode on a line item

*Annotations:*
/tag <fact\\_id> <key> <value> - Add a tag
/untag <annotation\\_id> - Remove a tag

*Enrichment:*
/enrich - Review pending enrichment queue
/scan - Detect barcodes from a photo

*Account:*
/link <mnemonic> - Connect your Telegram account
/whoami - Show linked account info
/unlink - Disconnect Telegram account
/setname <name> - Set your display name

*Other:*
/budget - Budget management
/language - Display language preference
/map [fact\\_id] <url> - Set location from Google Maps
/skip - Skip pending location prompt

*Tips:*
Send a photo to process it automatically.
Reply to a result message with /fix or /tag to skip the fact ID.
"""
    await message.answer(help_text, parse_mode="Markdown")
