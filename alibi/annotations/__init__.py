"""Annotation layer for user-defined metadata on entities.

Annotations are open-ended key-value pairs on facts, items, vendors, or
identities. They enrich data for reporting without affecting processing.

Common annotation types:
- person: purchase attribution ("bought_for" -> "Maria")
- project: project tagging ("project" -> "Kitchen renovation")
- category: custom categories ("category" -> "groceries")
- split: expense splitting (with metadata for ratios)
- note: free-text notes
"""

from alibi.annotations.store import (
    add_annotation,
    delete_annotation,
    get_annotations,
    update_annotation,
)

__all__ = [
    "add_annotation",
    "delete_annotation",
    "get_annotations",
    "update_annotation",
]
