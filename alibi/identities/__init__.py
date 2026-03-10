"""Entity identity system — shared knowledge service.

Auto-registers vendor/item identities during extraction and provides
resolution for cloud formation matching. Three feedback sources:

- extraction: auto-created when new documents are processed
- correction: auto-created when bundles are moved between clouds
- historical: auto-created when historical enrichment discovers registrations

Two entity types:
- vendor: groups vendor name variants ("FreSko", "FRESKO BUTANOLO LTD")
- item: groups product name variants ("Happy Cow Milk", "Happy Milk")
"""

from alibi.identities.matching import (
    ensure_vendor_identity,
    find_identities_for_fact,
    find_item_identity,
    find_vendor_identity,
    resolve_item,
    resolve_vendor,
)
from alibi.identities.store import (
    add_member,
    create_identity,
    delete_identity,
    get_identity,
    get_members_by_type,
    list_identities,
    remove_member,
    update_identity,
)

__all__ = [
    # Matching/resolution
    "ensure_vendor_identity",
    "find_vendor_identity",
    "find_item_identity",
    "resolve_vendor",
    "resolve_item",
    "find_identities_for_fact",
    # CRUD
    "create_identity",
    "add_member",
    "remove_member",
    "delete_identity",
    "update_identity",
    "get_identity",
    "list_identities",
    "get_members_by_type",
]
