"""Masking module for tiered disclosure and cloud AI anonymization.

Pure business logic for applying privacy-preserving transformations to records.
No I/O, no database access, no side effects.
"""

from alibi.masking.policies import (
    DisclosurePolicy,
    get_policy,
)
from alibi.masking.service import (
    MaskingService,
)

__all__ = [
    # policies.py
    "DisclosurePolicy",
    "get_policy",
    # service.py
    "MaskingService",
]
