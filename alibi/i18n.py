"""Display language resolver for multi-language support."""

from __future__ import annotations

from collections.abc import Mapping

from alibi.config import get_config


def get_display_name(
    original: str | None,
    normalized: str | None,
    lang: str | None = None,
) -> str:
    """Return the appropriate display name based on user's language preference.

    Args:
        original: Original language name (e.g., "Gala" in Greek)
        normalized: English/translated name (e.g., "Milk")
        lang: Override language preference (if None, uses config)

    Returns:
        The appropriate name for display
    """
    config = get_config()
    display_lang = lang or config.display_language

    if display_lang == "original":
        return original or normalized or ""
    elif display_lang in ("en", "normalized"):
        return normalized or original or ""
    else:
        # For specific languages, prefer original if it matches, else normalized
        return original or normalized or ""


def format_line_item_name(
    item: Mapping[str, str | None], lang: str | None = None
) -> str:
    """Format a line item's name for display."""
    return get_display_name(
        item.get("name"),
        item.get("name_normalized"),
        lang=lang,
    )


def get_supported_languages() -> list[dict[str, str]]:
    """Return list of supported display languages."""
    return [
        {"code": "original", "name": "Original (as extracted)"},
        {"code": "en", "name": "English"},
        {"code": "de", "name": "German"},
        {"code": "el", "name": "Greek"},
        {"code": "ru", "name": "Russian"},
        {"code": "ar", "name": "Arabic"},
    ]
