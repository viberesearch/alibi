"""Language preference Telegram commands."""

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from alibi.config import get_config
from alibi.i18n import get_supported_languages

router = Router()


@router.message(Command("language"))
async def language_command(message: Message) -> None:
    """Show or set display language.

    /language - Show current language setting
    /language <code> - Set display language (en, de, el, ru, original)
    """
    parts = (message.text or "").split()
    supported = get_supported_languages()
    supported_codes = [lang["code"] for lang in supported]

    if len(parts) >= 2:
        lang = parts[1].lower()
        if lang in supported_codes:
            await message.answer(
                f"Language preference set to *{lang}*. "
                "This affects display names for extracted items. "
                "Bot responses remain in English.",
                parse_mode="Markdown",
            )
        else:
            await message.answer(
                f"Unsupported language: {lang}\n"
                f"Supported: {', '.join(supported_codes)}"
            )
    else:
        config = get_config()
        await message.answer(
            f"Current language: *{config.display_language}*\n"
            f"Supported: {', '.join(supported_codes)}",
            parse_mode="Markdown",
        )
