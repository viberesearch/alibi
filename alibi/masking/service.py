"""Masking service for tiered disclosure and cloud AI anonymization.

Provides three core operations:
1. mask_for_tier - Apply tier-based masking for shared space display
2. mask_for_cloud_ai - Replace sensitive entities before sending to cloud AI
3. unmask - Reverse cloud AI masking using stored masking map

Pure business logic - no I/O, no database access.
"""

import copy
from datetime import date
from decimal import ROUND_CEILING, Decimal
from typing import Any, Optional

from alibi.db.models import Tier
from alibi.masking.policies import DisclosurePolicy, get_policy


class MaskingService:
    """Service for applying disclosure masking to records."""

    def mask_for_tier(
        self, records: list[dict[str, Any]], tier: Tier
    ) -> list[dict[str, Any]]:
        """Apply tier-based masking to records for shared space display.

        Args:
            records: List of record dicts with fields like amount, vendor,
                     transaction_date, category, line_items, provenance, etc.
            tier: The disclosure tier to apply.

        Returns:
            New list of masked record dicts (originals are not modified).
        """
        policy = get_policy(tier)
        return [self._mask_record(record, policy) for record in records]

    def _mask_record(
        self, record: dict[str, Any], policy: DisclosurePolicy
    ) -> dict[str, Any]:
        """Apply a disclosure policy to a single record."""
        masked = copy.deepcopy(record)

        # Mask amount fields
        for field in ("amount", "total_price", "unit_price"):
            if field in masked:
                masked[field] = policy.mask_amount(self._to_decimal(masked[field]))

        # Mask vendor
        if "vendor" in masked:
            masked["vendor"] = policy.mask_vendor(
                masked.get("vendor"), masked.get("category")
            )

        # Mask date fields
        for field in ("transaction_date", "document_date", "purchase_date"):
            if field in masked and masked[field] is not None:
                val = masked[field]
                if isinstance(val, str):
                    try:
                        val = date.fromisoformat(val)
                    except (ValueError, TypeError):
                        continue
                if isinstance(val, date):
                    masked[field] = policy.mask_date(val)

        # Strip line items if policy says so
        if not policy.should_include_line_items():
            masked.pop("line_items", None)

        # Strip provenance if policy says so
        if not policy.should_include_provenance():
            masked.pop("provenance", None)
            masked.pop("extracted_data", None)
            masked.pop("raw_text", None)

        # Mask card numbers at all tiers below T4
        if "card_last4" in masked and policy.tier != Tier.T4:
            masked["card_last4"] = None

        # Mask personal identifiers at tiers below T3
        if policy.tier in (Tier.T0, Tier.T1):
            masked.pop("payment_method", None)
            masked.pop("account_reference", None)

        return masked

    def mask_for_cloud_ai(
        self, records: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Replace sensitive entities before sending to cloud AI.

        Replaces vendors, personal names, card numbers, and rounds amounts
        to protect privacy while preserving data structure for AI analysis.

        Args:
            records: List of record dicts with potentially sensitive data.

        Returns:
            Tuple of (masked_records, masking_map) where masking_map allows
            reversal via unmask().
        """
        masking_map: dict[str, Any] = {
            "vendors": {},
            "persons": {},
            "cards": {},
            "amounts": {},
        }
        vendor_counter = 0
        person_counter = 0

        masked_records = []
        for i, record in enumerate(records):
            masked = copy.deepcopy(record)
            record_key = str(i)

            # Mask vendor names
            if "vendor" in masked and masked["vendor"] is not None:
                vendor_name = str(masked["vendor"])
                if vendor_name not in masking_map["vendors"]:
                    vendor_counter += 1
                    placeholder = f"Merchant_{chr(64 + vendor_counter)}"
                    masking_map["vendors"][vendor_name] = placeholder
                masked["vendor"] = masking_map["vendors"][vendor_name]

            # Mask personal names (description field often contains names)
            if "description" in masked and masked["description"] is not None:
                desc = str(masked["description"])
                # Store original
                if desc not in masking_map["persons"]:
                    person_counter += 1
                    placeholder = f"Person_{chr(64 + person_counter)}"
                    masking_map["persons"][desc] = placeholder
                masked["description"] = masking_map["persons"][desc]

            # Mask card numbers
            if "card_last4" in masked and masked["card_last4"] is not None:
                card = str(masked["card_last4"])
                if card not in masking_map["cards"]:
                    masking_map["cards"][card] = "XXXX"
                masked["card_last4"] = masking_map["cards"][card]

            # Round amounts to ranges
            if "amount" in masked and masked["amount"] is not None:
                original = self._to_decimal(masked["amount"])
                if original is not None:
                    rounded = self._round_to_range(original)
                    masking_map["amounts"][record_key] = str(original)
                    masked["amount"] = rounded

            # Mask account references
            if (
                "account_reference" in masked
                and masked["account_reference"] is not None
            ):
                masked["account_reference"] = "ACCT_MASKED"

            # Mask payment method details
            if "payment_method" in masked and masked["payment_method"] is not None:
                # Keep the type (card, cash, transfer) but remove specifics
                method = str(masked["payment_method"]).lower()
                if "card" in method:
                    masked["payment_method"] = "card"
                elif "cash" in method:
                    masked["payment_method"] = "cash"
                elif "transfer" in method:
                    masked["payment_method"] = "transfer"
                else:
                    masked["payment_method"] = "other"

            masked_records.append(masked)

        return masked_records, masking_map

    def unmask(
        self, masked_records: list[dict[str, Any]], masking_map: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Reverse cloud AI masking using stored masking map.

        Args:
            masked_records: Records previously masked by mask_for_cloud_ai().
            masking_map: The masking map returned by mask_for_cloud_ai().

        Returns:
            New list of records with original values restored.
        """
        # Build reverse mappings
        reverse_vendors: dict[str, str] = {
            v: k for k, v in masking_map.get("vendors", {}).items()
        }
        reverse_persons: dict[str, str] = {
            v: k for k, v in masking_map.get("persons", {}).items()
        }
        reverse_cards: dict[str, str] = {
            v: k for k, v in masking_map.get("cards", {}).items()
        }
        amounts_map: dict[str, str] = masking_map.get("amounts", {})

        unmasked_records = []
        for i, record in enumerate(masked_records):
            unmasked = copy.deepcopy(record)
            record_key = str(i)

            # Restore vendor names
            if "vendor" in unmasked and unmasked["vendor"] is not None:
                vendor_placeholder = str(unmasked["vendor"])
                if vendor_placeholder in reverse_vendors:
                    unmasked["vendor"] = reverse_vendors[vendor_placeholder]

            # Restore descriptions
            if "description" in unmasked and unmasked["description"] is not None:
                desc_placeholder = str(unmasked["description"])
                if desc_placeholder in reverse_persons:
                    unmasked["description"] = reverse_persons[desc_placeholder]

            # Restore card numbers
            if "card_last4" in unmasked and unmasked["card_last4"] is not None:
                card_placeholder = str(unmasked["card_last4"])
                if card_placeholder in reverse_cards:
                    unmasked["card_last4"] = reverse_cards[card_placeholder]

            # Restore exact amounts
            if record_key in amounts_map:
                unmasked["amount"] = Decimal(amounts_map[record_key])

            unmasked_records.append(unmasked)

        return unmasked_records

    @staticmethod
    def _to_decimal(value: Any) -> Optional[Decimal]:
        """Safely convert a value to Decimal."""
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except Exception:
            return None

    @staticmethod
    def _round_to_range(amount: Decimal) -> str:
        """Round an amount to a human-readable range string.

        Examples:
            12.50 -> "10-20"
            247.99 -> "200-300"
            3.50 -> "0-10"
            1500.00 -> "1000-2000"
        """
        abs_amount = abs(amount)
        sign = "-" if amount < 0 else ""

        if abs_amount < Decimal("10"):
            return f"{sign}0-10"
        elif abs_amount < Decimal("100"):
            lower = (abs_amount / Decimal("10")).quantize(
                Decimal("1"), rounding=ROUND_CEILING
            ) * Decimal("10") - Decimal("10")
            upper = lower + Decimal("10")
            return f"{sign}{int(lower)}-{int(upper)}"
        elif abs_amount < Decimal("1000"):
            lower = (abs_amount / Decimal("100")).quantize(
                Decimal("1"), rounding=ROUND_CEILING
            ) * Decimal("100") - Decimal("100")
            upper = lower + Decimal("100")
            return f"{sign}{int(lower)}-{int(upper)}"
        else:
            lower = (abs_amount / Decimal("1000")).quantize(
                Decimal("1"), rounding=ROUND_CEILING
            ) * Decimal("1000") - Decimal("1000")
            upper = lower + Decimal("1000")
            return f"{sign}{int(lower)}-{int(upper)}"
