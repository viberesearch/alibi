"""Privacy-preserving anonymization for cloud AI analysis.

Exports v2 facts at three anonymization levels:
- categories_only: Only categories and aggregates, no names/amounts
- pseudonymized: Consistent fake names, shifted amounts/dates, reversible
- statistical: Only aggregate statistics, no individual records
"""

from alibi.anonymization.exporter import (
    AnonymizationLevel,
    AnonymizationKey,
    anonymize_export,
    restore_import,
    generate_key,
)

__all__ = [
    "AnonymizationLevel",
    "AnonymizationKey",
    "anonymize_export",
    "restore_import",
    "generate_key",
]
