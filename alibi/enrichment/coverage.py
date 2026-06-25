"""Read-only enrichment coverage report across the local-LLM fields.

Each local-LLM enrichment pass (``comparable_name``, ``unit_quantity`` /
``unit``, ``category``, ``attributes``, ``state``) writes its result straight to
``fact_items`` and stamps an idempotency marker so a converged row is never
re-sent. That gives three states per field per row, which this module counts:

* **filled**        -- the field has a real value.
* **answered-null** -- the model was asked and returned "no result" (a count
  item with no size, a non-product line, a staple with no state); the marker is
  stamped so it is not re-asked, but no value was written.
* **pending**       -- not yet answered; a future ``lt enrich`` run will pick it
  up.

The classification mirrors each pass's own selection predicate exactly (the same
marker columns the ``enrich_pending_*`` SELECTs use), so this report is a true
dashboard of what the passes have and have not done -- not a separate heuristic
that could drift. ``state`` is scoped to real products (non-empty
``comparable_name``), matching ``enrich_pending_states``.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field

from alibi.db.connection import DatabaseManager
from alibi.enrichment import taxonomy


@dataclass
class FieldCoverage:
    """Per-field coverage tallies plus a sample of pending item names."""

    field: str
    filled: int
    answered_null: int
    pending: int
    eligible: int
    stragglers: list[str] = dc_field(default_factory=list)


@dataclass
class ItemCoverageRow:
    """One under-captured fact: its line items sum short of the fact total."""

    fact_id: str
    vendor: str | None
    event_date: str | None
    total: float
    item_sum: float
    n_items: int
    coverage_pct: float


@dataclass
class ItemCoverageReport:
    """Fact-level item-extraction coverage (sum of item prices vs fact total).

    A low ratio means line items are missing even though the transaction total
    is known — the dominant under-extraction pattern. ``partial`` facts (some
    items, summing short) are the highest-signal re-extract candidates;
    ``no_items`` facts (a total but zero line items) include legitimate
    card-only slips, so they are reported separately rather than as failures.
    """

    threshold_pct: float
    eligible: int
    below: int
    partial: int
    no_items: int
    worst: list[ItemCoverageRow] = dc_field(default_factory=list)


# Per-field predicates, expressed once so the count query and the straggler
# query cannot diverge. ``{tv}`` is substituted with the taxonomy version param
# placeholder. Each is evaluated over the field's eligible base.
_FILLED = {
    "comparable_name": "comparable_name IS NOT NULL AND comparable_name != ''",
    "unit_quantity": "unit_quantity IS NOT NULL",
    "category": "category_path IS NOT NULL AND category_path != ''",
    "attributes": "attributes IS NOT NULL AND attributes != '{}'",
    "state": "json_extract(attributes, '$.state') IS NOT NULL",
}
_PENDING = {
    "comparable_name": (
        "(comparable_name IS NULL OR comparable_name = '') "
        "AND comparable_name_enriched IS NULL"
    ),
    "unit_quantity": "unit_quantity IS NULL AND unit_enriched IS NULL",
    "category": (
        "(category_path IS NULL OR category_path = '') "
        "AND (category_taxonomy_version IS NULL "
        "OR category_taxonomy_version < ?)"
    ),
    "attributes": "attributes IS NULL",
    "state": "json_extract(attributes, '$.state') IS NULL AND state_enriched IS NULL",
}
_ANSWERED_NULL = {
    "comparable_name": (
        "(comparable_name IS NULL OR comparable_name = '') "
        "AND comparable_name_enriched IS NOT NULL"
    ),
    "unit_quantity": "unit_quantity IS NULL AND unit_enriched IS NOT NULL",
    "category": (
        "(category_path IS NULL OR category_path = '') "
        "AND category_taxonomy_version IS NOT NULL "
        "AND category_taxonomy_version >= ?"
    ),
    "attributes": "attributes = '{}'",
    "state": (
        "json_extract(attributes, '$.state') IS NULL AND state_enriched IS NOT NULL"
    ),
}

# Fields whose eligible base is "real products" (non-empty comparable_name),
# matching their pass's pending SELECT. The rest are eligible for any named item.
_REAL_PRODUCT_FIELDS = frozenset({"state"})

# Order fields are reported in (the pipeline order).
_FIELDS = ("comparable_name", "unit_quantity", "category", "attributes", "state")

_BASE = "name IS NOT NULL AND name != ''"
_REAL_PRODUCT_BASE = (
    f"{_BASE} AND comparable_name IS NOT NULL AND comparable_name != ''"
)


def _base_for(field_name: str) -> str:
    return _REAL_PRODUCT_BASE if field_name in _REAL_PRODUCT_FIELDS else _BASE


def coverage_report(
    db: DatabaseManager, straggler_limit: int = 10
) -> list[FieldCoverage]:
    """Tally filled / answered-null / pending per local-LLM field.

    Returns one :class:`FieldCoverage` per field in pipeline order, each with up
    to ``straggler_limit`` pending item names sampled for quick inspection.
    Read-only.
    """
    tv = taxonomy.TAXONOMY_VERSION
    out: list[FieldCoverage] = []
    for name in _FIELDS:
        base = _base_for(name)
        filled_sql = _FILLED[name]
        pending_sql = _PENDING[name]
        answered_sql = _ANSWERED_NULL[name]

        # Category's pending / answered-null predicates carry a "?" for the
        # taxonomy version; bind it the right number of times per clause.
        pending_params = (tv,) if "?" in pending_sql else ()
        answered_params = (tv,) if "?" in answered_sql else ()

        row = db.fetchone(
            f"SELECT "  # noqa: S608 - predicates are trusted internal literals
            f"SUM(CASE WHEN {filled_sql} THEN 1 ELSE 0 END) AS filled, "
            f"SUM(CASE WHEN {answered_sql} THEN 1 ELSE 0 END) AS answered_null, "
            f"SUM(CASE WHEN {pending_sql} THEN 1 ELSE 0 END) AS pending, "
            f"COUNT(*) AS eligible "
            f"FROM fact_items WHERE {base}",
            (*answered_params, *pending_params),
        )
        # An aggregate SELECT with COUNT(*) always yields exactly one row.
        assert row is not None

        straggler_rows = db.fetchall(
            f"SELECT name FROM fact_items "  # noqa: S608
            f"WHERE {base} AND ({pending_sql}) "
            f"ORDER BY id LIMIT ?",
            (*pending_params, straggler_limit),
        )
        out.append(
            FieldCoverage(
                field=name,
                filled=int(row["filled"] or 0),
                answered_null=int(row["answered_null"] or 0),
                pending=int(row["pending"] or 0),
                eligible=int(row["eligible"] or 0),
                stragglers=[r["name"] for r in straggler_rows],
            )
        )
    return out


def item_coverage_report(
    db: DatabaseManager,
    threshold_pct: float = 92.0,
    worst_limit: int = 20,
) -> ItemCoverageReport:
    """Tally fact-level item-extraction coverage (item-price sum vs fact total).

    Scoped to purchase facts with a positive total. A fact is "below" when its
    summed item prices cover less than ``threshold_pct`` of the total — a proxy
    for missing line items. Returns counts plus the worst ``worst_limit`` facts
    (lowest coverage first) as a re-extraction queue. Read-only.
    """
    rows = db.fetchall(
        "SELECT f.id, f.vendor, f.event_date, f.total_amount AS total, "
        "COUNT(fi.id) AS n_items, "
        "COALESCE(SUM(fi.total_price), 0) AS item_sum "
        "FROM facts f LEFT JOIN fact_items fi ON fi.fact_id = f.id "
        "WHERE f.fact_type = 'purchase' "
        "AND f.total_amount IS NOT NULL AND f.total_amount > 0 "
        "GROUP BY f.id"
    )

    eligible = len(rows)
    below_rows: list[ItemCoverageRow] = []
    partial = no_items = 0
    for r in rows:
        total = float(r["total"])
        item_sum = float(r["item_sum"] or 0)
        n_items = int(r["n_items"] or 0)
        cov = (item_sum / total * 100.0) if total else 0.0
        if cov >= threshold_pct:
            continue
        if n_items == 0:
            no_items += 1
        else:
            partial += 1
        below_rows.append(
            ItemCoverageRow(
                fact_id=r["id"],
                vendor=r["vendor"],
                event_date=str(r["event_date"]) if r["event_date"] else None,
                total=round(total, 2),
                item_sum=round(item_sum, 2),
                n_items=n_items,
                coverage_pct=round(cov, 1),
            )
        )

    below_rows.sort(key=lambda c: c.coverage_pct)
    return ItemCoverageReport(
        threshold_pct=threshold_pct,
        eligible=eligible,
        below=len(below_rows),
        partial=partial,
        no_items=no_items,
        worst=below_rows[:worst_limit],
    )
