"""Budget scenarios with temporal modeling.

Provides budget scenario management following the ma-engine tree pattern:
- Trunk (actual): Real spending from transactions (immutable historical data)
- Base (projected): Inertial projection based on recurring patterns
- Branches (scenarios): "What if" budgets with delta-only storage
"""

from alibi.budgets.merge import compute_variance, merge_entries
from alibi.budgets.models import BudgetComparison, BudgetEntry, BudgetScenario
from alibi.budgets.service import BudgetService

__all__ = [
    "BudgetComparison",
    "BudgetEntry",
    "BudgetScenario",
    "BudgetService",
    "compute_variance",
    "merge_entries",
]
