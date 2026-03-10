"""Pydantic models for the budget system."""

from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field

from alibi.db.models import DataType


class BudgetScenario(BaseModel):
    """A budget scenario (actual, projected, or target).

    Scenarios can be branched from a parent to create "what if" variants.
    The data_type determines the temporal classification:
    - actual: Real spending from historical transactions
    - projected: Inertial projection based on recurring patterns
    - target: User-defined budget targets
    """

    id: str
    space_id: str
    name: str
    description: Optional[str] = None
    data_type: DataType
    parent_id: Optional[str] = None  # For scenario branching
    period_start: Optional[date] = None
    period_end: Optional[date] = None
    created_at: datetime = Field(default_factory=datetime.now)


class BudgetEntry(BaseModel):
    """A single budget entry (amount for a category in a period).

    Entries represent individual line items within a scenario,
    keyed by category and period (YYYY-MM format).
    """

    id: str
    scenario_id: str
    category: str  # e.g., "groceries", "dining", "transport"
    amount: Decimal
    currency: str = "EUR"
    period: str  # "2025-01", "2025-02" etc (YYYY-MM format)
    note: Optional[str] = None


class BudgetComparison(BaseModel):
    """Result of comparing two scenarios.

    Contains per-category, per-period variance between a base
    and comparison scenario.
    """

    category: str
    period: str
    base_amount: Decimal
    compare_amount: Decimal
    variance: Decimal  # compare - base
    variance_pct: Optional[Decimal] = None  # percentage change
