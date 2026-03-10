"""Date parsing and normalization functions.

Pure functions for handling dates in various formats.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any


def parse_date(value: Any) -> date | None:
    """Parse a date from various string formats.

    Handles:
        - ISO format: "2024-12-31", "2024-01-15"
        - European format: "31/12/2024", "15.01.2024", "31-12-2024"
        - US format: "12/31/2024", "01-15-2024"
        - Short year: "31/12/24", "12/31/24"
        - Greek format: "31/12/2024", "15.01.2024"
        - German format: "31.12.2024", "15.01.2024"

    Returns:
        date object, or None if unparsable.
    """
    if value is None:
        return None

    if isinstance(value, date):
        return value

    s = str(value).strip()
    if not s:
        return None

    # Try ISO format: YYYY-MM-DD
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None

    # Try DD/MM/YYYY or DD.MM.YYYY or DD-MM-YYYY
    match = re.match(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})$", s)
    if match:
        day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        # Heuristic: if day > 12, it's definitely DD/MM/YYYY
        # Otherwise, check both possibilities
        if day > 12:
            # Must be DD/MM/YYYY
            try:
                return date(year, month, day)
            except ValueError:
                return None
        elif month > 12:
            # Must be MM/DD/YYYY
            try:
                return date(year, day, month)
            except ValueError:
                return None
        else:
            # Ambiguous: try DD/MM/YYYY first (European default)
            try:
                return date(year, month, day)
            except ValueError:
                # Try MM/DD/YYYY
                try:
                    return date(year, day, month)
                except ValueError:
                    return None

    # Try DD/MM/YY or DD.MM.YY or DD-MM-YY (short year)
    match = re.match(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2})$", s)
    if match:
        day, month, year_short = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )
        # Convert 2-digit year to 4-digit (assume 2000-2099)
        year = 2000 + year_short
        # Same heuristic as above
        if day > 12:
            try:
                return date(year, month, day)
            except ValueError:
                return None
        elif month > 12:
            try:
                return date(year, day, month)
            except ValueError:
                return None
        else:
            # Ambiguous: try DD/MM/YY first (European default)
            try:
                return date(year, month, day)
            except ValueError:
                try:
                    return date(year, day, month)
                except ValueError:
                    return None

    # Try YYYY/MM/DD or YYYY.MM.DD
    match = re.match(r"^(\d{4})[/.](\d{2})[/.](\d{2})$", s)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None

    return None


def disambiguate_date(
    parsed: date,
    raw_value: str,
    *,
    reference_date: date | None = None,
    file_date: date | None = None,
    date_format_hint: str | None = None,
) -> date:
    """Resolve ambiguous DD/MM vs MM/DD dates using contextual signals.

    When a date like "03/05/2026" could be March 5 or May 3, uses:
    1. Future-date rejection: if parsed date is in the future but the
       swapped interpretation isn't, prefer the past/present date.
    2. File metadata proximity: prefer the interpretation closer to the
       file creation/modification date.
    3. Reference date proximity: prefer the interpretation closer to the
       reference date (e.g., extraction date, upload date).

    Args:
        parsed: The date as initially parsed (European DD/MM default).
        raw_value: The original date string for swap analysis.
        reference_date: When the document was processed/uploaded.
        file_date: File creation or modification date.

    Returns:
        The best date interpretation.
    """
    # Only attempt disambiguation for ambiguous dates where both
    # day and month are <= 12 (otherwise the parse is unambiguous).
    if parsed.day > 12 or parsed.month > 12:
        return parsed

    # Build the swapped interpretation (DD/MM <-> MM/DD).
    try:
        swapped = date(parsed.year, parsed.day, parsed.month)
    except ValueError:
        return parsed  # swap is invalid

    if swapped == parsed:
        return parsed  # same either way (e.g., 05/05)

    # Rule 0: Template date format hint.
    if date_format_hint == "mdy":
        return swapped
    elif date_format_hint == "dmy":
        return parsed

    today = date.today()
    ref = reference_date or file_date or today

    # Rule 1: Future-date rejection.
    parsed_future = parsed > today
    swapped_future = swapped > today
    if parsed_future and not swapped_future:
        return swapped
    if swapped_future and not parsed_future:
        return parsed

    # Rule 2: Proximity to file creation date.
    if file_date:
        parsed_dist = abs((parsed - file_date).days)
        swapped_dist = abs((swapped - file_date).days)
        if parsed_dist != swapped_dist:
            return parsed if parsed_dist < swapped_dist else swapped

    # Rule 3: Proximity to reference date.
    parsed_dist = abs((parsed - ref).days)
    swapped_dist = abs((swapped - ref).days)
    if parsed_dist != swapped_dist:
        return parsed if parsed_dist < swapped_dist else swapped

    # Default: keep the original European interpretation.
    return parsed


def normalize_date_format(d: date, fmt: str = "iso") -> str:
    """Output a date in the requested format.

    Args:
        d: date object to format
        fmt: format string - "iso" (YYYY-MM-DD), "eu" (DD/MM/YYYY), "us" (MM/DD/YYYY)

    Returns:
        Formatted date string.
    """
    if fmt == "iso":
        return d.isoformat()
    elif fmt == "eu":
        return f"{d.day:02d}/{d.month:02d}/{d.year}"
    elif fmt == "us":
        return f"{d.month:02d}/{d.day:02d}/{d.year}"
    else:
        return d.isoformat()
