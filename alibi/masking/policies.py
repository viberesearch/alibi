"""Disclosure policies defining what to mask at each tier.

Tier levels (from most restrictive to fully visible):
    T0: Fully masked - amounts hidden, vendors replaced with category, dates to month only
    T1: Rounded amounts, vendor categories only, approximate dates
    T2: Exact amounts, vendor names visible, full dates
    T3: Full details including line items
    T4: Full details + provenance + raw extraction data

Pure business logic - no I/O, no database access.
"""

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_CEILING, Decimal
from typing import Optional

from alibi.db.models import DisplayType, Tier


@dataclass(frozen=True)
class DisclosurePolicy:
    """Define what fields are visible/masked at each disclosure tier."""

    tier: Tier

    def mask_amount(self, amount: Optional[Decimal]) -> str | Decimal | None:
        """Apply tier-appropriate masking to a monetary amount.

        Args:
            amount: The exact monetary amount.

        Returns:
            T0: None (hidden)
            T1: Rounded to nearest 10 (e.g., 47.50 -> 50)
            T2-T4: Exact amount
        """
        if amount is None:
            return None

        if self.tier == Tier.T0:
            return None

        if self.tier == Tier.T1:
            # Round up to nearest 10
            return (amount / Decimal("10")).quantize(
                Decimal("1"), rounding=ROUND_CEILING
            ) * Decimal("10")

        # T2, T3, T4: exact amount
        return amount

    def mask_vendor(self, vendor: Optional[str], category: Optional[str]) -> str:
        """Apply tier-appropriate masking to a vendor name.

        Args:
            vendor: The vendor/merchant name.
            category: The vendor category (e.g., "groceries", "electronics").

        Returns:
            T0: Category name or "Unknown" if no category
            T1: Category name or "Unknown" if no category
            T2-T4: Vendor name (falls back to category or "Unknown")
        """
        if self.tier in (Tier.T0, Tier.T1):
            return category if category else "Unknown"

        # T2, T3, T4: show vendor name
        if vendor:
            return vendor
        return category if category else "Unknown"

    def mask_date(self, d: Optional[date]) -> str | date | None:
        """Apply tier-appropriate masking to a date.

        Args:
            d: The exact date.

        Returns:
            T0: Month and year only as string (e.g., "2024-01")
            T1: First of the month (e.g., 2024-01-15 -> 2024-01-01)
            T2-T4: Exact date
        """
        if d is None:
            return None

        if self.tier == Tier.T0:
            return d.strftime("%Y-%m")

        if self.tier == Tier.T1:
            return d.replace(day=1)

        # T2, T3, T4: exact date
        return d

    def should_include_line_items(self) -> bool:
        """Whether line items should be included in output.

        Returns:
            True for T3 and T4, False otherwise.
        """
        return self.tier in (Tier.T3, Tier.T4)

    def should_include_provenance(self) -> bool:
        """Whether provenance data should be included in output.

        Returns:
            True for T4 only.
        """
        return self.tier == Tier.T4

    def amount_display_type(self) -> DisplayType:
        """Get the DisplayType for amounts at this tier."""
        mapping = {
            Tier.T0: DisplayType.HIDDEN,
            Tier.T1: DisplayType.ROUNDED,
            Tier.T2: DisplayType.EXACT,
            Tier.T3: DisplayType.EXACT,
            Tier.T4: DisplayType.EXACT,
        }
        return mapping[self.tier]

    def vendor_display_type(self) -> DisplayType:
        """Get the DisplayType for vendor names at this tier."""
        mapping = {
            Tier.T0: DisplayType.MASKED,
            Tier.T1: DisplayType.MASKED,
            Tier.T2: DisplayType.EXACT,
            Tier.T3: DisplayType.EXACT,
            Tier.T4: DisplayType.EXACT,
        }
        return mapping[self.tier]

    def date_display_type(self) -> DisplayType:
        """Get the DisplayType for dates at this tier."""
        mapping = {
            Tier.T0: DisplayType.MASKED,
            Tier.T1: DisplayType.ROUNDED,
            Tier.T2: DisplayType.EXACT,
            Tier.T3: DisplayType.EXACT,
            Tier.T4: DisplayType.EXACT,
        }
        return mapping[self.tier]


def get_policy(tier: Tier) -> DisclosurePolicy:
    """Get the disclosure policy for a given tier.

    Args:
        tier: The disclosure tier level.

    Returns:
        DisclosurePolicy configured for the given tier.
    """
    return DisclosurePolicy(tier=tier)
