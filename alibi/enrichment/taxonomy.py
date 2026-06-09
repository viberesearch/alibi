"""Controlled category taxonomy for item enrichment.

A normalised, hierarchical tree of allowed categories. The category enrichment
pass (``alibi/enrichment/categorize.py``) constrains the LLM to choose a leaf
from this taxonomy, so paths are consistent and groupable at any depth — unlike
the historic free-text ``category`` field (``Produce`` vs ``Vegetables`` vs
``Fruit`` vs ``Fruits``, ``Beverage`` vs ``Beverages``).

The taxonomy was derived from the actual corpus (Cyprus/Greek grocery-heavy,
with a services tail and non-product receipt lines). Paths are stored lowercase
with " > " separators, e.g. ``food > dairy > milk``. The leaf (last segment)
mirrors into the flat ``category`` column.

``adjustment > *`` is a deliberate bucket for non-product lines that slip past
the pollution filter (tax, tip, service charge, totals, footers). Categorising
them — rather than leaving them blank — lets analytics *exclude* them cleanly
via the category_path prefix filter, and lifts real item coverage.

Bump ``TAXONOMY_VERSION`` when the tree changes so a re-run can recategorise.
"""

from __future__ import annotations

TAXONOMY_VERSION = 1

# The taxonomy as a nested mapping: top-level -> {sub: [leaves]} or top -> [leaves].
# A branch whose value is a list has those leaves directly under it; a branch
# whose value is a dict has named sub-branches. Single-segment top-level
# categories (no children) map to an empty list.
TAXONOMY: dict[str, object] = {
    "food": {
        "produce": ["vegetables", "fruit", "herbs"],
        "dairy": ["milk", "yogurt", "cheese", "butter"],
        "eggs": [],
        "meat": ["red_meat", "poultry", "deli"],
        "seafood": [],
        "bakery": [],
        "pantry": ["grains_pasta", "canned", "condiments_sauces", "oils", "baking"],
        "snacks": ["sweets", "savory"],
        "beverages": [
            "soft_drinks",
            "juice",
            "water",
            "coffee_tea",
            "energy",
            "alcohol",
        ],
        "frozen": [],
        "prepared": [],
    },
    "household": [],
    "personal_care": [],
    "health": ["medicine", "medical_service"],
    "dining": [],
    "transport": ["fuel"],
    "services": ["telecom", "finance", "entertainment"],
    "pet": [],
    "tobacco": [],
    "other": [],
    "adjustment": ["tax", "discount", "service_charge", "tip", "fee", "non_item"],
}

PATH_SEPARATOR = " > "


def _walk(node: object, prefix: list[str]) -> list[str]:
    """Yield the full path for every leaf reachable from ``node``."""
    paths: list[str] = []
    if isinstance(node, dict):
        for key, child in node.items():
            sub = prefix + [key]
            child_paths = _walk(child, sub)
            if child_paths:
                paths.extend(child_paths)
            else:
                # A branch with no leaves is itself a valid (leaf) path.
                paths.append(PATH_SEPARATOR.join(sub))
    elif isinstance(node, list):
        if not node:
            # Top-level category with no children: the prefix is the leaf.
            paths.append(PATH_SEPARATOR.join(prefix))
        else:
            for leaf in node:
                paths.append(PATH_SEPARATOR.join(prefix + [leaf]))
    return paths


def all_paths() -> list[str]:
    """Return every full (leaf) category path, e.g. ``food > dairy > milk``."""
    paths: list[str] = []
    for top, child in TAXONOMY.items():
        paths.extend(_walk(child, [top]))
    return paths


def all_nodes() -> list[str]:
    """Return every valid category path — leaves AND intermediate branches.

    The LLM legitimately picks an intermediate node (e.g. ``food > dairy`` for
    sour cream when no exact leaf fits, or ``food > produce`` for an item it
    can't sub-classify). Every prefix of a leaf path is therefore a valid
    category; the leaf is its last segment (``dairy``, ``produce``).
    """
    nodes: set[str] = set()
    for leaf_path in all_paths():
        parts = leaf_path.split(PATH_SEPARATOR)
        for depth in range(1, len(parts) + 1):
            nodes.add(PATH_SEPARATOR.join(parts[:depth]))
    return sorted(nodes)


# Precomputed for fast membership checks. Accepts any node (leaf or branch).
VALID_PATHS: frozenset[str] = frozenset(all_nodes())


def is_valid_path(path: str | None) -> bool:
    """True if ``path`` is an exact, known taxonomy path (case-insensitive)."""
    if not path:
        return False
    return path.strip().lower() in VALID_PATHS


def normalize_path(path: str | None) -> str | None:
    """Coerce an LLM-returned path to a canonical taxonomy path, or None.

    Accepts minor formatting drift (extra spaces around the separator, mixed
    case, ``/`` or ``>`` without spaces). Returns the canonical path if it
    matches a known leaf, else None.
    """
    if not path:
        return None
    raw = str(path).strip().lower()
    # Normalise separators: "food/dairy/milk" or "food>dairy>milk" -> " > "
    for sep in ("/", "›", "»"):
        raw = raw.replace(sep, ">")
    parts = [p.strip() for p in raw.split(">") if p.strip()]
    candidate = PATH_SEPARATOR.join(parts)
    if candidate in VALID_PATHS:
        return candidate
    return None


def leaf_of(path: str) -> str:
    """Return the leaf (last segment) of a path, for the flat ``category``."""
    return path.split(PATH_SEPARATOR)[-1]


def render_for_prompt() -> str:
    """Render the taxonomy as an indented tree for an LLM prompt."""
    lines: list[str] = []
    for top, child in TAXONOMY.items():
        lines.append(top)
        if isinstance(child, dict):
            for sub, leaves in child.items():
                if leaves:
                    lines.append(f"  {sub}: {', '.join(leaves)}")
                else:
                    lines.append(f"  {sub}")
        elif isinstance(child, list) and child:
            lines.append(f"  {', '.join(child)}")
    return "\n".join(lines)
