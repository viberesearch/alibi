"""Language detection and text normalization functions.

Pure functions using Unicode script analysis (no external APIs).
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


def detect_language(text: str) -> str:
    """Detect language from text using Unicode script analysis.

    Supports:
        - Latin scripts: en, de, fr, es, it (defaults to 'en')
        - Greek: el
        - Cyrillic: ru
        - Arabic: ar
        - CJK: zh (Chinese), ja (Japanese), ko (Korean)

    Returns:
        ISO 639-1 language code, or 'en' if undetermined.
    """
    if not text:
        return "en"

    # Remove whitespace and count character types
    text_stripped = text.strip()
    if not text_stripped:
        return "en"

    # Count characters by script
    script_counts: dict[str, int] = {}

    for char in text_stripped:
        if char.isspace() or char.isdigit():
            continue

        # Get Unicode script
        try:
            script = _get_unicode_script(char)
            script_counts[script] = script_counts.get(script, 0) + 1
        except Exception:
            logger.debug(
                "Unicode script detection failed for char %r", char, exc_info=True
            )

    if not script_counts:
        return "en"

    # Find dominant script
    dominant_script = max(script_counts, key=script_counts.get)  # type: ignore

    # Map script to language
    return _script_to_language(dominant_script)


def normalize_text(text: str) -> str:
    """Normalize text by stripping extra whitespace and normalizing Unicode.

    Operations:
        - Strip leading/trailing whitespace
        - Collapse multiple spaces to single space
        - Normalize Unicode to NFC form (canonical composition)
        - Normalize line endings

    Returns:
        Normalized text string.
    """
    if not text:
        return ""

    # Normalize Unicode to NFC form
    s = unicodedata.normalize("NFC", text)

    # Normalize line endings
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse multiple spaces
    s = re.sub(r" +", " ", s)

    # Collapse multiple newlines (max 2)
    s = re.sub(r"\n{3,}", "\n\n", s)

    # Strip leading/trailing whitespace
    s = s.strip()

    return s


def _get_unicode_script(char: str) -> str:
    """Get Unicode script name for a character.

    Returns:
        Script name (e.g., 'LATIN', 'GREEK', 'CYRILLIC', 'HAN', 'ARABIC').
    """
    try:
        # Get Unicode category
        category = unicodedata.category(char)

        # Get Unicode name to infer script
        name = unicodedata.name(char, "")

        # Heuristics based on Unicode name
        if "GREEK" in name:
            return "GREEK"
        elif "CYRILLIC" in name:
            return "CYRILLIC"
        elif "ARABIC" in name:
            return "ARABIC"
        elif "CJK" in name or "HAN" in name or "HIRAGANA" in name or "KATAKANA" in name:
            return "CJK"
        elif "HANGUL" in name:
            return "HANGUL"
        elif "HEBREW" in name:
            return "HEBREW"
        elif "DEVANAGARI" in name:
            return "DEVANAGARI"
        elif "THAI" in name:
            return "THAI"
        elif category.startswith("L"):  # Letter category
            # Check Unicode block ranges
            code = ord(char)
            if 0x0370 <= code <= 0x03FF or 0x1F00 <= code <= 0x1FFF:
                return "GREEK"
            elif 0x0400 <= code <= 0x04FF or 0x0500 <= code <= 0x052F:
                return "CYRILLIC"
            elif 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F:
                return "ARABIC"
            elif (
                0x4E00 <= code <= 0x9FFF
                or 0x3040 <= code <= 0x309F
                or 0x30A0 <= code <= 0x30FF
            ):
                return "CJK"
            elif 0xAC00 <= code <= 0xD7AF:
                return "HANGUL"
            elif 0x0590 <= code <= 0x05FF:
                return "HEBREW"
            elif 0x0900 <= code <= 0x097F:
                return "DEVANAGARI"
            elif 0x0E00 <= code <= 0x0E7F:
                return "THAI"
            else:
                return "LATIN"
        else:
            return "LATIN"
    except Exception:
        return "LATIN"


def _script_to_language(script: str) -> str:
    """Map Unicode script to ISO 639-1 language code.

    Returns:
        Two-letter language code.
    """
    mapping = {
        "LATIN": "en",  # Default to English for Latin script
        "GREEK": "el",
        "CYRILLIC": "ru",
        "ARABIC": "ar",
        "CJK": "zh",  # Default to Chinese for CJK
        "HANGUL": "ko",
        "HEBREW": "he",
        "DEVANAGARI": "hi",
        "THAI": "th",
    }
    return mapping.get(script, "en")
