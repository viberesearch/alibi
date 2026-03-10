"""Refiners module for transforming raw extraction data into structured records.

Refiners follow the ma-engine BaseRefiner pattern:
- Take raw dict + artifact_id
- Normalize fields (amounts, dates, currency)
- Apply type-specific transformations
- Build provenance chain
- Return enriched dict ready for model creation

Main exports:
- BaseRefiner: Abstract base class
- get_refiner: Get refiner instance for a RecordType
- Individual refiners: PaymentRefiner, PurchaseRefiner, etc.
"""

from alibi.refiners.base import BaseRefiner
from alibi.refiners.contract import ContractRefiner
from alibi.refiners.insurance import InsuranceRefiner
from alibi.refiners.invoice import InvoiceRefiner
from alibi.refiners.payment import PaymentRefiner
from alibi.refiners.purchase import PurchaseRefiner
from alibi.refiners.registry import DefaultRefiner, get_refiner
from alibi.refiners.statement import StatementRefiner
from alibi.refiners.warranty import WarrantyRefiner

__all__ = [
    "BaseRefiner",
    "ContractRefiner",
    "DefaultRefiner",
    "get_refiner",
    "PaymentRefiner",
    "PurchaseRefiner",
    "InvoiceRefiner",
    "WarrantyRefiner",
    "InsuranceRefiner",
    "StatementRefiner",
]
