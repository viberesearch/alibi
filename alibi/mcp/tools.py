"""MCP tool definitions for alibi data access and lifecycle operations.

Each tool uses DatabaseManager directly for data access and returns
structured dicts/lists. Tools accept db as a parameter for testability
and the MCP-decorated wrappers handle database initialization.

Tools are grouped into:
- Query: search, get_fact, inspect_fact, list_unassigned
- Ingestion: ingest_document
- Correction: correct_vendor, move_bundle
- Annotation: annotate_entity
- Analytics: spending_summary, line_items, spending_patterns, budget, recurring
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from alibi.config import get_config
from alibi.db.connection import DatabaseManager


def _get_db() -> DatabaseManager:
    """Get an initialized database manager."""
    config = get_config()
    db = DatabaseManager(config)
    if not db.is_initialized():
        db.initialize()
    return db


# ---------------------------------------------------------------------------
# Query tools
# ---------------------------------------------------------------------------


def search_transactions(
    db: DatabaseManager,
    query: str,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Search facts across vendor names and item names.

    Uses the service layer search_facts() which performs a UNION query
    to match on both vendor and item name fields.

    Args:
        db: Database manager instance.
        query: Text to search in vendor names and item names.
        date_from: Start date filter (YYYY-MM-DD).
        date_to: End date filter (YYYY-MM-DD).
        limit: Maximum results to return (default 20).

    Returns:
        Dict with query, total count, and matching facts.
    """
    from alibi.services.query import search_facts

    result = search_facts(db, query=query, offset=0, limit=limit)
    facts = result["facts"]

    # Apply date filters on top of search results
    if date_from:
        facts = [f for f in facts if str(f.get("event_date") or "") >= date_from]
    if date_to:
        facts = [f for f in facts if str(f.get("event_date") or "") <= date_to]

    return {
        "query": query,
        "total": len(facts),
        "results": facts,
    }


def get_fact_detail(
    db: DatabaseManager,
    fact_id: str,
) -> dict[str, Any]:
    """Get a fact by ID with its line items.

    Args:
        db: Database manager instance.
        fact_id: UUID of the fact to retrieve.

    Returns:
        Dict with fact fields and items, or error if not found.
    """
    from alibi.services.query import get_fact

    result = get_fact(db, fact_id)
    if result is None:
        return {"error": f"Fact '{fact_id}' not found"}
    return result


def inspect_fact_detail(
    db: DatabaseManager,
    fact_id: str,
) -> dict[str, Any]:
    """Full drill-down of a fact with cloud, bundles, atoms, and documents.

    Args:
        db: Database manager instance.
        fact_id: UUID of the fact to inspect.

    Returns:
        Nested dict with fact, cloud, bundles, items, or error if not found.
    """
    from alibi.services.query import inspect_fact

    result = inspect_fact(db, fact_id)
    if result is None:
        return {"error": f"Fact '{fact_id}' not found"}
    return result


def list_unassigned_bundles(
    db: DatabaseManager,
) -> dict[str, Any]:
    """List bundles with no cloud assignment (needing attention).

    Args:
        db: Database manager instance.

    Returns:
        Dict with total count and list of unassigned bundles.
    """
    from alibi.services.query import list_unassigned

    bundles = list_unassigned(db)
    return {
        "total": len(bundles),
        "bundles": bundles,
    }


# ---------------------------------------------------------------------------
# Ingestion tools
# ---------------------------------------------------------------------------


def _resolve_doc_type(doc_type: str | None) -> Any:
    """Resolve a doc_type string to a FolderContext, or None.

    Valid doc_type values: receipt, invoice, statement, payment,
    warranty, contract. Returns None if doc_type is None or empty.
    Raises ValueError for invalid doc_type.
    """
    if not doc_type:
        return None

    from alibi.db.models import DocumentType
    from alibi.processing.folder_router import FolderContext

    _TYPE_MAP: dict[str, DocumentType] = {
        "receipt": DocumentType.RECEIPT,
        "invoice": DocumentType.INVOICE,
        "statement": DocumentType.STATEMENT,
        "payment": DocumentType.PAYMENT_CONFIRMATION,
        "warranty": DocumentType.WARRANTY,
        "contract": DocumentType.CONTRACT,
    }

    dt = _TYPE_MAP.get(doc_type.lower())
    if dt is None:
        raise ValueError(
            f"Unknown doc_type '{doc_type}'. " f"Valid: {sorted(_TYPE_MAP.keys())}"
        )
    return FolderContext(doc_type=dt)


def _format_processing_result(result: Any) -> dict[str, Any]:
    """Convert ProcessingResult to a dict for MCP response."""
    return {
        "success": result.success,
        "file_path": str(result.file_path),
        "document_id": result.document_id,
        "is_duplicate": result.is_duplicate,
        "duplicate_of": result.duplicate_of,
        "error": result.error,
        "record_type": result.record_type.value if result.record_type else None,
        "extracted_data": result.extracted_data,
    }


def ingest_document(
    db: DatabaseManager,
    path: str,
    doc_type: str | None = None,
) -> dict[str, Any]:
    """Process a document file through the extraction pipeline.

    Triggers the full pipeline: duplicate detection, type detection,
    OCR, parsing, LLM correction, atom/cloud/fact creation.

    Args:
        db: Database manager instance.
        path: Absolute path to the document file (JPG, PNG, PDF, etc.).
        doc_type: Optional document type hint (receipt, invoice, statement,
            payment, warranty, contract). When provided, skips LLM vision
            classification. When omitted, the pipeline auto-detects.

    Returns:
        Dict with success status, document_id, and extracted data summary.
    """
    from alibi.services.ingestion import process_file

    from alibi.processing.folder_router import FolderContext

    try:
        folder_context = _resolve_doc_type(doc_type)
    except ValueError as e:
        return {"error": str(e)}

    # Set provenance: MCP entry point, system user
    if folder_context is None:
        folder_context = FolderContext()
    folder_context.source = "mcp"
    folder_context.user_id = "system"

    file_path = Path(path)
    if not file_path.exists():
        return {"error": f"File not found: {path}"}

    result = process_file(db, file_path, folder_context=folder_context)
    return _format_processing_result(result)


def ingest_bytes(
    db: DatabaseManager,
    data: bytes,
    filename: str,
    doc_type: str | None = None,
) -> dict[str, Any]:
    """Process a document from raw bytes.

    Writes bytes to a temp file, processes through the pipeline, then
    cleans up. Use this for API/Telegram uploads where the file isn't
    on disk.

    Args:
        db: Database manager instance.
        data: Raw document bytes.
        filename: Original filename (extension determines processing path).
        doc_type: Optional document type hint (receipt, invoice, statement,
            payment, warranty, contract). When provided, skips LLM vision
            classification.

    Returns:
        Dict with success status, document_id, and extracted data summary.
    """
    from alibi.processing.folder_router import FolderContext as _FC
    from alibi.services.ingestion import process_bytes

    try:
        folder_context = _resolve_doc_type(doc_type)
    except ValueError as e:
        return {"error": str(e)}

    if folder_context is None:
        folder_context = _FC()
    folder_context.source = "mcp"
    folder_context.user_id = "system"

    result = process_bytes(db, data, filename, folder_context=folder_context)
    return _format_processing_result(result)


# ---------------------------------------------------------------------------
# Correction tools
# ---------------------------------------------------------------------------


def correct_fact_vendor(
    db: DatabaseManager,
    fact_id: str,
    new_vendor: str,
) -> dict[str, Any]:
    """Correct the vendor name on a fact and teach the identity system.

    Updates the vendor field and registers the corrected name in the
    identity system for future matching.

    Args:
        db: Database manager instance.
        fact_id: UUID of the fact to correct.
        new_vendor: Corrected vendor name.

    Returns:
        Dict with success status.
    """
    from alibi.services.correction import correct_vendor

    ok = correct_vendor(db, fact_id, new_vendor)
    if not ok:
        return {"success": False, "error": f"Fact '{fact_id}' not found"}
    return {"success": True, "fact_id": fact_id, "vendor": new_vendor}


def move_fact_bundle(
    db: DatabaseManager,
    bundle_id: str,
    target_cloud_id: str | None = None,
) -> dict[str, Any]:
    """Move a bundle to a different cloud and re-collapse both clouds.

    Use this to fix incorrect document grouping. If target_cloud_id is
    not provided, a new cloud is created for the bundle.

    Args:
        db: Database manager instance.
        bundle_id: UUID of the bundle to move.
        target_cloud_id: UUID of the target cloud, or None to create new.

    Returns:
        Dict with correction result including new fact IDs.
    """
    from alibi.services.correction import move_bundle

    result = move_bundle(db, bundle_id, target_cloud_id)
    return {
        "success": result.success,
        "error": result.error,
        "source_cloud_id": result.source_cloud_id,
        "target_cloud_id": result.target_cloud_id,
        "source_fact_id": result.source_fact_id,
        "target_fact_id": result.target_fact_id,
    }


# ---------------------------------------------------------------------------
# Annotation tools
# ---------------------------------------------------------------------------


def annotate_entity(
    db: DatabaseManager,
    target_type: str,
    target_id: str,
    annotation_type: str,
    key: str,
    value: str,
) -> dict[str, Any]:
    """Add an annotation to a fact, item, vendor, or identity.

    Annotations are open-ended key-value metadata. Common uses:
    - "person" annotations: who a purchase was for
    - "project" annotations: linking expenses to a project
    - "category" annotations: custom categorization

    Args:
        db: Database manager instance.
        target_type: Entity type (fact, fact_item, vendor, identity).
        target_id: UUID of the target entity.
        annotation_type: Type of annotation (person, project, category, etc.).
        key: Annotation key (e.g., "bought_for", "project").
        value: Annotation value (e.g., "Maria", "Kitchen renovation").

    Returns:
        Dict with the annotation ID.
    """
    from alibi.services.annotation import annotate

    valid_types = {"fact", "fact_item", "vendor", "identity"}
    if target_type not in valid_types:
        return {
            "error": f"Invalid target_type '{target_type}'. "
            f"Must be one of: {sorted(valid_types)}"
        }

    annotation_id = annotate(
        db,
        target_type=target_type,
        target_id=target_id,
        annotation_type=annotation_type,
        key=key,
        value=value,
        source="mcp",
    )
    return {
        "success": True,
        "annotation_id": annotation_id,
        "target_type": target_type,
        "target_id": target_id,
    }


# ---------------------------------------------------------------------------
# Analytics tools (existing, kept as-is)
# ---------------------------------------------------------------------------


def get_spending_summary(
    db: DatabaseManager,
    period: str = "month",
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Get aggregated spending grouped by period.

    Args:
        db: Database manager instance.
        period: Grouping period - "month", "week", or "day".
        date_from: Start date filter (YYYY-MM-DD).
        date_to: End date filter (YYYY-MM-DD).

    Returns:
        Dict with period grouping and aggregated spending data.
    """
    conditions = ["fact_type IN ('purchase', 'subscription_payment')"]
    params: list[Any] = []

    if date_from:
        conditions.append("event_date >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("event_date <= ?")
        params.append(date_to)

    where = " WHERE " + " AND ".join(conditions)

    if period == "day":
        group_expr = "strftime('%Y-%m-%d', event_date)"
    elif period == "week":
        group_expr = "strftime('%Y-W%W', event_date)"
    else:
        group_expr = "strftime('%Y-%m', event_date)"

    sql = f"""
        SELECT {group_expr} as period,
               COUNT(*) as count,
               SUM(CAST(total_amount AS REAL)) as total,
               AVG(CAST(total_amount AS REAL)) as average,
               MIN(CAST(total_amount AS REAL)) as min_amount,
               MAX(CAST(total_amount AS REAL)) as max_amount
        FROM facts{where}
        GROUP BY {group_expr}
        ORDER BY period DESC
    """  # noqa: S608

    rows = db.fetchall(sql, tuple(params))

    data = []
    for r in rows:
        row_dict = dict(r)
        # Round floating point values
        for key in ("total", "average", "min_amount", "max_amount"):
            if row_dict.get(key) is not None:
                row_dict[key] = round(row_dict[key], 2)
        data.append(row_dict)

    return {
        "group_by": period,
        "data": data,
    }


def get_line_items(
    db: DatabaseManager,
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Query line items with optional filters.

    Args:
        db: Database manager instance.
        category: Filter by category.
        date_from: Start date filter (YYYY-MM-DD), matched via transaction date.
        date_to: End date filter (YYYY-MM-DD), matched via transaction date.
        limit: Maximum results to return (default 50).

    Returns:
        Dict with total count and matching line items.
    """
    conditions: list[str] = []
    params: list[Any] = []

    if category:
        conditions.append("fi.category = ?")
        params.append(category)
    if date_from:
        conditions.append("f.event_date >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("f.event_date <= ?")
        params.append(date_to)

    where = ""
    if conditions:
        where = " WHERE " + " AND ".join(conditions)

    sql = f"""
        SELECT fi.id, fi.name, fi.name_normalized, fi.category,
               fi.quantity, fi.unit, fi.unit_price,
               fi.total_price, fi.brand,
               f.vendor, f.event_date
        FROM fact_items fi
        LEFT JOIN facts f ON fi.fact_id = f.id
        {where}
        ORDER BY f.event_date DESC, fi.name
        LIMIT ?
    """  # noqa: S608
    params.append(limit)

    rows = db.fetchall(sql, tuple(params))
    results = [dict(r) for r in rows]

    return {
        "total": len(results),
        "results": results,
    }


def analyze_spending_patterns(
    db: DatabaseManager,
    months: int = 3,
) -> dict[str, Any]:
    """Analyze spending patterns: top vendors, categories, averages.

    Args:
        db: Database manager instance.
        months: Number of months to analyze (default 3).

    Returns:
        Dict with top vendors, top categories, average transaction size,
        and total spending over the period.
    """
    today = date.today()
    period_start = today - timedelta(days=months * 31)

    # Top vendors by total spending
    vendor_rows = db.fetchall(
        """
        SELECT vendor, COUNT(*) as count,
               SUM(CAST(total_amount AS REAL)) as total,
               AVG(CAST(total_amount AS REAL)) as average
        FROM facts
        WHERE fact_type IN ('purchase', 'subscription_payment')
              AND vendor IS NOT NULL
              AND event_date >= ?
        GROUP BY vendor
        ORDER BY total DESC
        LIMIT 15
        """,
        (period_start.isoformat(),),
    )

    # Top categories from fact items
    category_rows = db.fetchall(
        """
        SELECT fi.category, COUNT(*) as count,
               SUM(CAST(fi.total_price AS REAL)) as total
        FROM fact_items fi
        LEFT JOIN facts f ON fi.fact_id = f.id
        WHERE fi.category IS NOT NULL
              AND f.event_date >= ?
        GROUP BY fi.category
        ORDER BY total DESC
        LIMIT 10
        """,
        (period_start.isoformat(),),
    )

    # Overall stats
    stats_row = db.fetchone(
        """
        SELECT COUNT(*) as count,
               SUM(CAST(total_amount AS REAL)) as total,
               AVG(CAST(total_amount AS REAL)) as average
        FROM facts
        WHERE fact_type IN ('purchase', 'subscription_payment')
              AND event_date >= ?
        """,
        (period_start.isoformat(),),
    )

    top_vendors = []
    for r in vendor_rows:
        row_dict = dict(r)
        if row_dict.get("total") is not None:
            row_dict["total"] = round(row_dict["total"], 2)
        if row_dict.get("average") is not None:
            row_dict["average"] = round(row_dict["average"], 2)
        top_vendors.append(row_dict)

    top_categories = []
    for r in category_rows:
        row_dict = dict(r)
        if row_dict.get("total") is not None:
            row_dict["total"] = round(row_dict["total"], 2)
        top_categories.append(row_dict)

    stats = {}
    if stats_row:
        stats = {
            "transaction_count": stats_row["count"],
            "total_spending": (
                round(stats_row["total"], 2) if stats_row["total"] else 0
            ),
            "average_transaction": (
                round(stats_row["average"], 2) if stats_row["average"] else 0
            ),
        }

    return {
        "period_months": months,
        "period_start": period_start.isoformat(),
        "top_vendors": top_vendors,
        "top_categories": top_categories,
        "summary": stats,
    }


def get_budget_comparison(
    db: DatabaseManager,
    scenario_id: str,
) -> dict[str, Any]:
    """Compare budget scenario entries against actual spending.

    Args:
        db: Database manager instance.
        scenario_id: The budget scenario ID to compare.

    Returns:
        Dict with scenario info, budget entries, actual spending,
        and per-category variance.
    """
    from alibi.budgets.service import BudgetService

    service = BudgetService(db)
    scenario = service.get_scenario(scenario_id)

    if scenario is None:
        return {"error": f"Scenario '{scenario_id}' not found"}

    entries = service.get_entries(scenario_id)
    entry_dicts = []
    for entry in entries:
        entry_dicts.append(
            {
                "category": entry.category,
                "amount": float(entry.amount),
                "currency": entry.currency,
                "period": entry.period,
                "note": entry.note,
            }
        )

    # Group entries by period and get actuals
    periods = {e.period for e in entries}
    comparisons: list[dict[str, Any]] = []

    for period in sorted(periods):
        period_entries = [e for e in entries if e.period == period]
        actuals = service.get_actual_spending(scenario.space_id, period)

        actual_by_cat: dict[str, float] = {}
        for a in actuals:
            actual_by_cat[a.category] = float(a.amount)

        for entry in period_entries:
            budget_amt = float(entry.amount)
            actual_amt = actual_by_cat.get(entry.category, 0.0)
            variance = actual_amt - budget_amt
            comparisons.append(
                {
                    "period": period,
                    "category": entry.category,
                    "budgeted": budget_amt,
                    "actual": actual_amt,
                    "variance": round(variance, 2),
                    "over_budget": variance > 0,
                }
            )

    return {
        "scenario": {
            "id": scenario.id,
            "name": scenario.name,
            "description": scenario.description,
        },
        "entries": entry_dicts,
        "comparisons": comparisons,
    }


def get_recurring_expenses(
    db: DatabaseManager,
    min_occurrences: int = 3,
) -> dict[str, Any]:
    """Find recurring transactions (same vendor, similar amounts).

    Args:
        db: Database manager instance.
        min_occurrences: Minimum occurrences to consider recurring.

    Returns:
        Dict with detected recurring patterns.
    """
    from alibi.services.analytics import detect_subscriptions

    patterns = detect_subscriptions(
        db,
        min_occurrences=min_occurrences,
    )

    results = []
    for p in patterns:
        results.append(
            {
                "vendor": p.vendor,
                "normalized_vendor": p.vendor_normalized,
                "avg_amount": float(p.avg_amount),
                "frequency_days": p.frequency_days,
                "period_type": p.period_type,
                "confidence": p.confidence,
                "last_date": p.last_date.isoformat(),
                "next_expected": p.next_expected.isoformat(),
                "occurrences": p.occurrences,
                "amount_variance": p.amount_variance,
                "fact_ids": p.fact_ids,
            }
        )

    return {
        "min_occurrences": min_occurrences,
        "total": len(results),
        "patterns": results,
    }


# ---------------------------------------------------------------------------
# Correction tools (line item)
# ---------------------------------------------------------------------------


def update_line_item(
    db: DatabaseManager,
    item_id: str,
    barcode: str | None = None,
    brand: str | None = None,
    category: str | None = None,
    name: str | None = None,
    unit_quantity: float | None = None,
    unit: str | None = None,
    product_variant: str | None = None,
) -> dict[str, Any]:
    """Update fields on a fact item (line item).

    Supported fields: barcode, brand, category, name, unit_quantity, unit,
    product_variant.

    When unit_quantity is set, the value is also propagated to the item
    identity metadata so future ingestions can apply it automatically.

    Args:
        db: Database manager instance.
        item_id: UUID of the fact item to update.
        barcode: Product barcode (EAN/UPC).
        brand: Product brand name.
        category: Product category (title-cased on write).
        name: Product name.
        unit_quantity: Weight or volume per unit (e.g. 0.5 for 500g).
        unit: Unit label (e.g. "kg", "L", "ml").
        product_variant: Product subcategory (e.g. "3%" for milk fat, "L" for egg size).

    Returns:
        Dict with success status and updated field names, or error.
    """
    fields: dict[str, object] = {}
    if barcode is not None:
        fields["barcode"] = barcode
    if brand is not None:
        fields["brand"] = brand
    if category is not None:
        fields["category"] = category
    if name is not None:
        fields["name"] = name
    if unit_quantity is not None:
        fields["unit_quantity"] = unit_quantity
    if unit is not None:
        fields["unit"] = unit
    if product_variant is not None:
        fields["product_variant"] = product_variant

    if not fields:
        return {"error": "No fields to update"}

    from alibi.services.correction import update_fact_item

    try:
        ok = update_fact_item(db, item_id, fields)
    except ValueError as exc:
        return {"error": str(exc)}

    if not ok:
        return {"error": "Item not found"}

    return {"success": True, "item_id": item_id, "updated_fields": list(fields.keys())}


# ---------------------------------------------------------------------------
# Location tools
# ---------------------------------------------------------------------------


def set_fact_location_tool(
    db: DatabaseManager,
    fact_id: str,
    map_url: str,
) -> dict[str, Any]:
    """Set or update the location for a fact from a Google Maps URL.

    Parses the URL to extract coordinates and stores them as an
    annotation on the fact.

    Args:
        db: Database manager instance.
        fact_id: UUID of the fact.
        map_url: Google Maps URL (full or short link).

    Returns:
        Dict with lat, lng, clean_url, place_name — or error.
    """
    from alibi.services.correction import set_fact_location

    result = set_fact_location(db, fact_id, map_url)
    if not result:
        return {"error": "Could not parse map URL or fact not found"}
    return {"success": True, "fact_id": fact_id, **result}


def get_recent_vendor_locations_tool(
    db: DatabaseManager,
    limit: int = 20,
) -> dict[str, Any]:
    """Get recent unique vendor+location pairs.

    Useful for location picker UIs — shows vendors the user has
    recently associated with a physical location.

    Args:
        db: Database manager instance.
        limit: Maximum number of locations to return (default 20).

    Returns:
        Dict with locations list.
    """
    from alibi.services.correction import get_recent_vendor_locations

    locations = get_recent_vendor_locations(db, limit=limit)
    return {"locations": locations, "count": len(locations)}


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def register_tools(mcp: Any) -> None:
    """Register all tools on the MCP server instance.

    Args:
        mcp: The MCPServer instance to register tools on.
    """

    # -- Query tools --

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_search_transactions(
        query: str,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Search facts by vendor name or item name.

        Searches across both vendor names and receipt line-item names.
        Optionally filter by date range. Returns up to `limit` results
        sorted by date descending.
        """
        db = _get_db()
        return search_transactions(
            db,
            query=query,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_get_fact(
        fact_id: str,
    ) -> dict[str, Any]:
        """Get a fact by ID with its line items.

        Returns fact metadata (vendor, amount, date, type) and all
        associated line items with names, quantities, and prices.
        """
        db = _get_db()
        return get_fact_detail(db, fact_id=fact_id)

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_inspect_fact(
        fact_id: str,
    ) -> dict[str, Any]:
        """Full drill-down of a fact.

        Returns the fact, its cloud metadata, all bundles with their
        atoms and source documents, and the fact items. Use this for
        debugging or understanding how a fact was assembled.
        """
        db = _get_db()
        return inspect_fact_detail(db, fact_id=fact_id)

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_list_unassigned() -> dict[str, Any]:
        """List bundles with no cloud assignment.

        These are document extractions that couldn't be matched to any
        existing cloud or fact. They need manual review or correction.
        """
        db = _get_db()
        return list_unassigned_bundles(db)

    # -- Ingestion tools --

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_ingest_document(
        path: str,
        doc_type: str | None = None,
    ) -> dict[str, Any]:
        """Process a document through the extraction pipeline.

        Accepts a file path to a receipt, invoice, statement, or other
        document. Runs OCR, parsing, LLM correction, and creates atoms,
        bundles, clouds, and facts in the database.

        Supported formats: JPG, JPEG, PNG, GIF, PDF.

        Optional doc_type skips LLM vision classification:
        receipt, invoice, statement, payment, warranty, contract.
        Omit for auto-detection.
        """
        db = _get_db()
        return ingest_document(db, path=path, doc_type=doc_type)

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_ingest_bytes(
        data: str,
        filename: str,
        doc_type: str | None = None,
    ) -> dict[str, Any]:
        """Process a document from base64-encoded bytes.

        For uploading documents that aren't on disk (e.g., from an API
        or chat bot). The data should be base64-encoded. The filename
        extension determines the processing path (image vs PDF).

        Optional doc_type skips LLM vision classification:
        receipt, invoice, statement, payment, warranty, contract.
        Omit for auto-detection.
        """
        import base64

        db = _get_db()
        try:
            raw = base64.b64decode(data)
        except Exception:
            return {"error": "Invalid base64 data"}
        return ingest_bytes(db, data=raw, filename=filename, doc_type=doc_type)

    # -- Correction tools --

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_correct_vendor(
        fact_id: str,
        new_vendor: str,
    ) -> dict[str, Any]:
        """Correct the vendor name on a fact.

        Updates the vendor field and registers the corrected name in
        the identity system so future documents from this vendor are
        matched correctly.
        """
        db = _get_db()
        return correct_fact_vendor(db, fact_id=fact_id, new_vendor=new_vendor)

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_move_bundle(
        bundle_id: str,
        target_cloud_id: str | None = None,
    ) -> dict[str, Any]:
        """Move a bundle to a different cloud.

        Use this to fix incorrect document grouping. A bundle is one
        document's extraction; a cloud groups related bundles into a
        single fact. Moving a bundle re-collapses both the source and
        target clouds.

        If target_cloud_id is not given, a new cloud is created.
        """
        db = _get_db()
        return move_fact_bundle(
            db, bundle_id=bundle_id, target_cloud_id=target_cloud_id
        )

    # -- Annotation tools --

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_annotate(
        target_type: str,
        target_id: str,
        annotation_type: str,
        key: str,
        value: str,
    ) -> dict[str, Any]:
        """Add an annotation to a fact, item, vendor, or identity.

        Annotations are open-ended metadata. Examples:
        - type="person", key="bought_for", value="Maria"
        - type="project", key="project", value="Kitchen renovation"
        - type="category", key="custom_category", value="home office"

        Valid target_type values: fact, fact_item, vendor, identity.
        """
        db = _get_db()
        return annotate_entity(
            db,
            target_type=target_type,
            target_id=target_id,
            annotation_type=annotation_type,
            key=key,
            value=value,
        )

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_update_line_item(
        item_id: str,
        barcode: str | None = None,
        brand: str | None = None,
        category: str | None = None,
        name: str | None = None,
        unit_quantity: float | None = None,
        unit: str | None = None,
        product_variant: str | None = None,
    ) -> dict[str, Any]:
        """Update fields on a receipt line item (fact item).

        Supported fields: barcode, brand, category, name, unit_quantity, unit,
        product_variant.

        At least one field must be provided. When unit_quantity is set,
        the value is stored in the item identity so future ingestions
        of the same product apply the correct unit automatically.

        Args:
            item_id: UUID of the fact item to update.
            barcode: Product barcode (EAN/UPC).
            brand: Product brand name.
            category: Product category.
            name: Product name.
            unit_quantity: Weight or volume per unit (e.g. 0.5 for 500g).
            unit: Unit label (e.g. "kg", "L", "ml").
            product_variant: Product subcategory (e.g. "3%" for milk fat, "L" for egg size).
        """
        db = _get_db()
        return update_line_item(
            db,
            item_id=item_id,
            barcode=barcode,
            brand=brand,
            category=category,
            name=name,
            unit_quantity=unit_quantity,
            unit=unit,
            product_variant=product_variant,
        )

    # -- Location tools --

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_set_fact_location(
        fact_id: str,
        map_url: str,
    ) -> dict[str, Any]:
        """Set or update the location for a fact from a Google Maps URL.

        Parses the URL to extract coordinates and stores them as a
        location annotation. Supports full Google Maps URLs and short links.

        Args:
            fact_id: UUID of the fact.
            map_url: Google Maps URL.
        """
        db = _get_db()
        return set_fact_location_tool(db, fact_id=fact_id, map_url=map_url)

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_get_recent_vendor_locations(
        limit: int = 20,
    ) -> dict[str, Any]:
        """Get recent unique vendor+location pairs for location picker.

        Returns vendors the user has recently associated with a physical
        location, useful for quick-select in UIs.

        Args:
            limit: Maximum locations to return (default 20).
        """
        db = _get_db()
        return get_recent_vendor_locations_tool(db, limit=limit)

    # -- Analytics tools --

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_get_spending_summary(
        period: str = "month",
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        """Get aggregated spending grouped by time period.

        Groups expense transactions by day, week, or month and returns
        count, total, average, min, and max for each period.
        """
        db = _get_db()
        return get_spending_summary(
            db,
            period=period,
            date_from=date_from,
            date_to=date_to,
        )

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_get_line_items(
        category: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Query individual line items from receipts.

        Returns itemized receipt data with product names, quantities,
        prices, and categories. Optionally filter by category or date.
        """
        db = _get_db()
        return get_line_items(
            db,
            category=category,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_analyze_spending_patterns(
        months: int = 3,
    ) -> dict[str, Any]:
        """Analyze spending patterns over the past N months.

        Returns top vendors by spending, top line item categories,
        and overall statistics including transaction count, total,
        and average transaction size.
        """
        db = _get_db()
        return analyze_spending_patterns(db, months=months)

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_get_budget_comparison(
        scenario_id: str,
    ) -> dict[str, Any]:
        """Compare a budget scenario against actual spending.

        Returns the budget entries, actual spending per category,
        and variance (actual - budgeted) for each period and category.
        """
        db = _get_db()
        return get_budget_comparison(db, scenario_id=scenario_id)

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_get_recurring_expenses(
        min_occurrences: int = 3,
    ) -> dict[str, Any]:
        """Find recurring expenses like subscriptions and regular bills.

        Detects transactions with the same vendor and similar amounts
        occurring at regular intervals. Returns frequency, confidence,
        and next expected date for each pattern.
        """
        db = _get_db()
        return get_recurring_expenses(db, min_occurrences=min_occurrences)

    # -- YAML template + correction tools --

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_get_yaml_template(
        document_type: str,
    ) -> dict[str, Any]:
        """Get a blank .alibi.yaml template for a document type.

        Returns a YAML-ready dict with all expected fields set to empty
        defaults. Useful for manual YAML creation without running OCR.

        Supported types: receipt, payment_confirmation, invoice,
        statement, contract, warranty.

        Args:
            document_type: The document type to generate a template for.

        Returns:
            Template dict ready for yaml.dump(), or error dict.
        """
        from alibi.extraction.yaml_cache import (
            SUPPORTED_DOCUMENT_TYPES,
            generate_blank_template,
        )

        if document_type not in SUPPORTED_DOCUMENT_TYPES:
            return {
                "error": (
                    f"Unknown document type '{document_type}'. "
                    f"Supported: {', '.join(SUPPORTED_DOCUMENT_TYPES)}"
                )
            }

        return generate_blank_template(document_type)

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_reingest_from_yaml(
        source_path: str,
        is_group: bool = False,
    ) -> dict[str, Any]:
        """Re-ingest a document from its edited .alibi.yaml file.

        Use this after manually editing a .alibi.yaml to correct OCR errors,
        fix vendor names, adjust amounts, etc. The tool cleans up existing
        DB records and re-creates them from the updated YAML.

        Args:
            source_path: Path to the source document file or folder.
            is_group: True if source_path is a folder (multi-page document).

        Returns:
            Dict with success status, document_id, and any error message.
        """
        from pathlib import Path

        from alibi.services.ingestion import reingest_from_yaml

        db = _get_db()
        result = reingest_from_yaml(db, Path(source_path), is_group=is_group)
        return {
            "success": result.success,
            "document_id": result.document_id,
            "is_duplicate": result.is_duplicate,
            "error": result.error,
            "file_path": str(result.file_path),
        }

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_verify_extractions(
        doc_ids: list[str] | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Cross-validate extracted receipts via Gemini batch verification.

        Sends N documents to Gemini in one call to check for extraction
        errors (wrong vendor, mismatched totals, garbled item names).
        """
        db = _get_db()
        from alibi.services import verify_extractions

        results = verify_extractions(db, doc_ids=doc_ids, limit=limit)
        return {
            "verified": len(results),
            "with_issues": sum(1 for r in results if not r.all_ok),
            "results": [
                {"doc_id": r.doc_id, "all_ok": r.all_ok, "issues": r.issues}
                for r in results
            ],
        }

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_find_product_matches(
        category: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        """Find cross-vendor product matches via Gemini.

        Identifies products sold at different vendors under different names
        (e.g., 'Sourdough Baguette' at A == 'Artisan Bread' at B).
        """
        db = _get_db()
        from alibi.services import find_product_matches

        groups = find_product_matches(db, category=category, limit=limit)
        return {
            "match_groups": len(groups),
            "matches": [
                {
                    "canonical": g.canonical_name,
                    "confidence": g.confidence,
                    "reasoning": g.reasoning,
                    "products": [
                        {
                            "name": p.name,
                            "vendor": p.vendor_name,
                            "item_id": p.item_id,
                        }
                        for p in g.products
                    ],
                }
                for g in groups
            ],
        }

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_correction_matrix(
        limit: int = 1000,
        min_count: int = 2,
    ) -> dict[str, Any]:
        """Build confusion matrix from user corrections.

        Analyzes correction history to detect systematic extraction errors
        and identify categories/vendors needing cloud refinement.
        """
        db = _get_db()
        from alibi.services import correction_confusion_matrix

        matrix = correction_confusion_matrix(db, limit=limit, min_count=min_count)
        return {
            "total_corrections": matrix.total_corrections,
            "top_corrected_fields": matrix.top_corrected_fields,
            "category_confusions": [
                {
                    "original": c.original,
                    "corrected": c.corrected,
                    "count": c.count,
                }
                for c in matrix.category_confusions
            ],
            "vendor_stats": [
                {"vendor": v.vendor_name, "corrections": v.total_corrections}
                for v in matrix.vendor_stats[:10]
            ],
            "refinement_candidates": matrix.refinement_candidates,
        }

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_location_spending(
        cluster_radius: float = 100.0,
    ) -> dict[str, Any]:
        """Spending aggregated by location (heatmap data).

        Groups nearby transactions and shows total spending per location.
        """
        db = _get_db()
        from alibi.services import location_spending

        results = location_spending(db, cluster_radius_m=cluster_radius)
        return {
            "location_count": len(results),
            "locations": [
                {
                    "lat": r.lat,
                    "lng": r.lng,
                    "place_name": r.place_name,
                    "total": r.total_amount,
                    "visits": r.visit_count,
                    "vendors": r.vendors,
                }
                for r in results[:20]
            ],
        }

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_vendor_branches(
        vendor_key: str | None = None,
    ) -> dict[str, Any]:
        """Compare vendor branches across locations.

        Shows which branch you visit most, spend most at, etc.
        """
        db = _get_db()
        from alibi.services import vendor_branches

        results = vendor_branches(db, vendor_key=vendor_key)
        return {
            "vendor_count": len(results),
            "vendors": [
                {
                    "vendor": r.vendor_name,
                    "branches": r.branch_count,
                    "total_spent": r.total_spent,
                    "details": [
                        {
                            "place": b.place_name,
                            "visits": b.visit_count,
                            "avg": b.avg_basket,
                        }
                        for b in r.branches
                    ],
                }
                for r in results[:10]
            ],
        }

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_nearby_vendors(
        lat: float,
        lng: float,
        radius: float = 2000.0,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Suggest vendors near a location based on visit history."""
        db = _get_db()
        from alibi.services import nearby_vendors

        results = nearby_vendors(db, lat, lng, radius_m=radius, limit=limit)
        return {
            "suggestions": [
                {
                    "vendor": s.vendor_name,
                    "distance_m": s.distance_meters,
                    "visits": s.visit_count,
                    "avg_basket": s.avg_basket,
                    "reason": s.reason,
                }
                for s in results
            ],
        }

    # -- Correction history tools --

    @mcp.tool()  # type: ignore[misc,untyped-decorator]
    def mcp_get_correction_history(
        entity_type: str | None = None,
        entity_id: str | None = None,
        vendor_key: str | None = None,
        field: str | None = None,
        limit: int = 30,
    ) -> dict[str, Any]:
        """Query correction event history.

        Filter by entity (type+id), vendor_key, or field name.
        Returns correction events newest first.
        """
        db = _get_db()
        from alibi.services import correction_log

        if vendor_key:
            corrections = correction_log.get_vendor_corrections(db, vendor_key, limit)
            rate = correction_log.get_vendor_correction_rate(db, vendor_key)
            return {"corrections": corrections, "rate": rate}
        corrections = correction_log.list_corrections(
            db, entity_type=entity_type, entity_id=entity_id, field=field, limit=limit
        )
        return {"corrections": corrections}
