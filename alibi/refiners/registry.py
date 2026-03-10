"""Refiner registry for mapping RecordType to refiner implementations."""

from __future__ import annotations

from typing import Any

from alibi.db.models import RecordType
from alibi.refiners.base import BaseRefiner
from alibi.refiners.contract import ContractRefiner
from alibi.refiners.insurance import InsuranceRefiner
from alibi.refiners.invoice import InvoiceRefiner
from alibi.refiners.payment import PaymentRefiner
from alibi.refiners.purchase import PurchaseRefiner
from alibi.refiners.statement import StatementRefiner
from alibi.refiners.warranty import WarrantyRefiner


class DefaultRefiner(BaseRefiner):
    """Fallback refiner for unmapped record types.

    Applies only base refinement (amount/date/currency normalization)
    without type-specific logic.
    """

    def _refine_specific(
        self, raw: dict[str, Any], artifact_id: str | None
    ) -> dict[str, Any]:
        """No type-specific refinement for default refiner.

        Args:
            raw: Partially normalized data dict
            artifact_id: Source artifact ID

        Returns:
            Data dict unchanged
        """
        return raw


# Registry mapping RecordType to refiner class
REFINER_REGISTRY: dict[RecordType, type[BaseRefiner]] = {
    RecordType.PAYMENT: PaymentRefiner,
    RecordType.PURCHASE: PurchaseRefiner,
    RecordType.INVOICE: InvoiceRefiner,
    RecordType.WARRANTY: WarrantyRefiner,
    RecordType.INSURANCE: InsuranceRefiner,
    RecordType.STATEMENT: StatementRefiner,
    RecordType.CONTRACT: ContractRefiner,
    # Add more mappings as refiners are implemented
    # RecordType.REFUND: RefundRefiner,
    # RecordType.CLAIM: ClaimRefiner,
    # RecordType.SUBSCRIPTION: SubscriptionRefiner,
}


def get_refiner(record_type: RecordType) -> BaseRefiner:
    """Get the appropriate refiner for a record type.

    Args:
        record_type: RecordType enum value

    Returns:
        Instantiated refiner for the record type, or DefaultRefiner if unmapped
    """
    refiner_class = REFINER_REGISTRY.get(record_type, DefaultRefiner)
    return refiner_class()
