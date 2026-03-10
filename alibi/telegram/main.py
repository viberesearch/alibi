"""Alibi Telegram Bot main entry point."""

import asyncio
import logging
import sys

try:
    from aiogram import Bot, Dispatcher
    from aiogram.client.default import DefaultBotProperties
except ImportError:
    raise ImportError("Telegram support requires: uv sync --extra telegram")

from alibi.config import get_config
from alibi.telegram.handlers import router
from alibi.telegram.middleware import AllowedUsersMiddleware


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the bot."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    # Reduce noise from httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


async def main() -> None:
    """Initialize and run the bot."""
    config = get_config()

    if not config.telegram_token:
        print("Error: TELEGRAM_BOT_TOKEN not set in environment", file=sys.stderr)
        print("Please set TELEGRAM_BOT_TOKEN environment variable.", file=sys.stderr)
        sys.exit(1)

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting Alibi Telegram Bot...")

    bot = Bot(
        token=config.telegram_token,
        default=DefaultBotProperties(parse_mode="Markdown"),
    )
    dp = Dispatcher()
    dp.update.outer_middleware(AllowedUsersMiddleware())
    dp.include_router(router)

    # Log startup info
    try:
        me = await bot.get_me()
        logger.info(f"Bot started: @{me.username}")
    except Exception as e:
        logger.error(f"Failed to get bot info: {e}")
        sys.exit(1)

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


def run() -> None:
    """Entry point for running the bot."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    run()
