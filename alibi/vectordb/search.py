"""Semantic search functions for Alibi."""

import logging
from dataclasses import dataclass
from typing import Any

from alibi.db.connection import DatabaseManager
from alibi.vectordb.index import IndexType, VectorIndex

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A search result combining SQL and vector search data."""

    id: str
    source: str  # "sql" or "vector"
    entity_type: str  # "transaction", "artifact", or "item"
    score: float  # Relevance score (0-1, higher is better)

    # Common fields
    vendor: str | None = None
    description: str | None = None
    amount: float | None = None
    currency: str = "EUR"
    date: str | None = None

    # Extra data
    metadata: dict[str, Any] | None = None

    def __lt__(self, other: "SearchResult") -> bool:
        """Sort by score descending."""
        return self.score > other.score  # Reversed for descending sort


def sql_search(
    db: DatabaseManager,
    query: str,
    limit: int = 10,
    space_id: str = "default",
) -> list[SearchResult]:
    """Search using SQL LIKE queries.

    Args:
        db: Database manager
        query: Search query
        limit: Maximum results
        space_id: Space ID to search in

    Returns:
        List of search results
    """
    results: list[SearchResult] = []
    query_lower = query.lower()
    query_pattern = f"%{query_lower}%"

    # Search facts (v2 transactions)
    txn_rows = db.fetchall(
        """
        SELECT id, vendor, fact_type, total_amount, currency, event_date
        FROM facts
        WHERE LOWER(vendor) LIKE ?
        ORDER BY event_date DESC
        LIMIT ?
        """,
        (query_pattern, limit),
    )

    for row in txn_rows:
        vendor = row[1] or ""
        fact_type = row[2] or ""
        match_score = 0.0

        if query_lower in vendor.lower():
            match_score = 0.8
        else:
            match_score = 0.4

        results.append(
            SearchResult(
                id=str(row[0]),
                source="sql",
                entity_type="transaction",
                score=match_score,
                vendor=vendor,
                description=fact_type,
                amount=float(row[3]) if row[3] else None,
                currency=row[4] or "EUR",
                date=str(row[5]) if row[5] else None,
            )
        )

    # Search documents (v2 artifacts)
    artifact_rows = db.fetchall(
        """
        SELECT d.id, d.file_path, d.raw_extraction,
               f.vendor, f.total_amount, f.currency, f.event_date, f.fact_type
        FROM documents d
        LEFT JOIN bundles b ON b.document_id = d.id
        LEFT JOIN cloud_bundles cb ON cb.bundle_id = b.id
        LEFT JOIN facts f ON f.cloud_id = cb.cloud_id
        WHERE LOWER(COALESCE(f.vendor, '')) LIKE ?
           OR LOWER(COALESCE(d.raw_extraction, '')) LIKE ?
        GROUP BY d.id
        ORDER BY d.created_at DESC
        LIMIT ?
        """,
        (query_pattern, query_pattern, limit),
    )

    for row in artifact_rows:
        vendor = row[3] or ""
        raw_text = row[2] or ""
        match_score = 0.0

        if query_lower in vendor.lower():
            match_score = 0.7
        elif query_lower in raw_text.lower():
            match_score = 0.5
        else:
            match_score = 0.3

        results.append(
            SearchResult(
                id=row[0],
                source="sql",
                entity_type="artifact",
                score=match_score,
                vendor=vendor,
                amount=float(row[4]) if row[4] else None,
                currency=row[5] or "EUR",
                date=str(row[6]) if row[6] else None,
                metadata={"artifact_type": row[7] or "document"},
            )
        )

    # Search fact items (v2 line items)
    li_rows = db.fetchall(
        """
        SELECT fi.id, fi.name, fi.quantity, fi.total_price,
               fi.category, fi.brand, f.vendor, f.event_date
        FROM fact_items fi
        LEFT JOIN facts f ON fi.fact_id = f.id
        WHERE (LOWER(fi.name) LIKE ? OR LOWER(COALESCE(fi.brand, '')) LIKE ?
               OR LOWER(COALESCE(fi.category, '')) LIKE ?)
        ORDER BY f.event_date DESC
        LIMIT ?
        """,
        (query_pattern, query_pattern, query_pattern, limit),
    )

    for row in li_rows:
        name = row[1] or ""
        category = row[4] or ""
        brand = row[5] or ""
        vendor = row[6] or ""
        match_score = 0.0

        if query_lower in name.lower():
            match_score = 0.75
        elif query_lower in brand.lower():
            match_score = 0.65
        elif query_lower in category.lower():
            match_score = 0.55
        else:
            match_score = 0.35

        desc = name
        if brand:
            desc = f"{brand} {name}"

        results.append(
            SearchResult(
                id=str(row[0]),
                source="sql",
                entity_type="line_item",
                score=match_score,
                vendor=vendor,
                description=desc,
                amount=float(row[3]) if row[3] else None,
                currency="EUR",
                date=str(row[7]) if row[7] else None,
                metadata={"category": category, "quantity": str(row[2])},
            )
        )

    return sorted(results)[:limit]


def semantic_search(
    index: VectorIndex,
    query: str,
    limit: int = 10,
    index_types: list[IndexType] | None = None,
) -> list[SearchResult]:
    """Search using vector similarity.

    Args:
        index: Vector index
        query: Search query
        limit: Maximum results
        index_types: Filter by type (default: all)

    Returns:
        List of search results
    """
    if not index.is_initialized():
        return []

    vector_results = index.search(query, limit=limit, index_types=index_types)

    results = []
    for r in vector_results:
        # Convert LanceDB distance to similarity score (0-1)
        # LanceDB returns L2 distance, smaller is better
        # We convert to similarity: 1 / (1 + distance)
        distance = r.get("score", 0.0)
        similarity = 1.0 / (1.0 + distance)

        entity_type = r["type"]
        metadata = r.get("metadata", {})

        # Get description based on entity type
        description = None
        if entity_type == IndexType.ITEM.value:
            description = metadata.get("name")
        else:
            description = r.get("text", "")[:100]  # Truncate for display

        results.append(
            SearchResult(
                id=r["id"],
                source="vector",
                entity_type=entity_type,
                score=similarity,
                vendor=r.get("vendor"),
                description=description,
                amount=r.get("amount"),
                currency=r.get("currency", "EUR"),
                date=r.get("date"),
                metadata=metadata,
            )
        )

    return sorted(results)[:limit]


def unified_search(
    db: DatabaseManager,
    index: VectorIndex | None,
    query: str,
    limit: int = 10,
    space_id: str = "default",
    use_vector: bool = True,
    use_sql: bool = True,
) -> list[SearchResult]:
    """Combined SQL and vector search with deduplication.

    Args:
        db: Database manager
        index: Vector index (optional)
        query: Search query
        limit: Maximum results
        space_id: Space ID to search in
        use_vector: Include vector search results
        use_sql: Include SQL search results

    Returns:
        Deduplicated and merged search results
    """
    results: list[SearchResult] = []
    seen_ids: set[str] = set()

    # Vector search first (usually better for semantic queries)
    if use_vector and index is not None and index.is_initialized():
        try:
            vector_results = semantic_search(index, query, limit=limit)
            for r in vector_results:
                if r.id not in seen_ids:
                    results.append(r)
                    seen_ids.add(r.id)
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")

    # SQL search
    if use_sql:
        sql_results = sql_search(db, query, limit=limit, space_id=space_id)
        for r in sql_results:
            if r.id not in seen_ids:
                # Slightly penalize SQL results when vector search is also used
                if use_vector and index is not None and index.is_initialized():
                    r.score *= 0.9
                results.append(r)
                seen_ids.add(r.id)

    # Sort by score and limit
    return sorted(results)[:limit]


async def semantic_search_async(
    index: VectorIndex,
    query: str,
    limit: int = 10,
    index_types: list[IndexType] | None = None,
) -> list[SearchResult]:
    """Async wrapper for semantic search.

    Note: LanceDB operations are synchronous, so this is a thin wrapper
    that allows integration with async code.
    """
    # LanceDB is synchronous, but we provide async interface for consistency
    return semantic_search(index, query, limit=limit, index_types=index_types)


async def unified_search_async(
    db: DatabaseManager,
    index: VectorIndex | None,
    query: str,
    limit: int = 10,
    space_id: str = "default",
    use_vector: bool = True,
    use_sql: bool = True,
) -> list[SearchResult]:
    """Async wrapper for unified search."""
    return unified_search(
        db=db,
        index=index,
        query=query,
        limit=limit,
        space_id=space_id,
        use_vector=use_vector,
        use_sql=use_sql,
    )
