"""Search endpoint wrapping service-layer search."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, Query

from alibi.api.deps import get_database, require_user
from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("")
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    db: Annotated[DatabaseManager, Depends(get_database)] = None,  # type: ignore[assignment]
    user: Annotated[dict[str, Any], Depends(require_user)] = None,  # type: ignore[assignment]
    type: Optional[str] = Query(
        None, description="Filter by type: document, fact, item"
    ),
    limit: int = Query(20, ge=1, le=100),
    semantic: bool = Query(
        False, description="Use vector similarity search (requires LanceDB)"
    ),
) -> dict[str, Any]:
    """Search across documents, facts, and items.

    When semantic=true and LanceDB is configured, uses vector similarity
    search for better results on natural language queries. Falls back to
    SQL LIKE search otherwise.
    """
    # Try semantic search if requested
    if semantic:
        try:
            from alibi.config import get_config
            from alibi.vectordb.index import VectorIndex
            from alibi.vectordb.search import unified_search

            config = get_config()
            lance_path = config.get_lance_path()
            if lance_path:
                index = VectorIndex(db_path=lance_path)
                search_results = unified_search(db, index, q, limit=limit)
                vector_results = [
                    {
                        "id": r.id,
                        "result_type": r.entity_type,
                        "title": r.vendor or r.description,
                        "subtype": r.entity_type,
                        "document_date": str(r.date) if r.date else None,
                        "amount": float(r.amount) if r.amount else None,
                        "score": r.score,
                    }
                    for r in search_results
                ]
                return {
                    "query": q,
                    "total": len(vector_results),
                    "results": vector_results,
                }
        except Exception:
            logger.debug(
                "Vector search unavailable, falling back to SQL", exc_info=True
            )

    # SQL-based search fallback
    results: list[dict[str, Any]] = []

    if type is None or type == "document":
        doc_rows = db.fetchall(
            """SELECT id, 'document' as result_type, file_path as title,
                      NULL as subtype, created_at as document_date, NULL as amount
               FROM documents
               WHERE file_path LIKE ? OR raw_extraction LIKE ?
               ORDER BY created_at DESC LIMIT ?""",
            (f"%{q}%", f"%{q}%", limit),
        )
        results.extend(dict(r) for r in doc_rows)

    if type is None or type == "fact":
        from alibi.services import query as svc_query

        fact_result = svc_query.search_facts(db, q, offset=0, limit=limit)
        for f in fact_result["facts"]:
            results.append(
                {
                    "id": f["id"],
                    "result_type": "fact",
                    "title": f.get("vendor"),
                    "subtype": f.get("fact_type"),
                    "document_date": (
                        str(f["event_date"]) if f.get("event_date") else None
                    ),
                    "amount": (
                        float(f["total_amount"]) if f.get("total_amount") else None
                    ),
                }
            )

    if type is None or type == "item":
        item_rows = db.fetchall(
            """SELECT id, 'item' as result_type, name as title,
                      category as subtype, purchase_date as document_date,
                      purchase_price as amount
               FROM items
               WHERE name LIKE ? OR category LIKE ? OR model LIKE ?
               ORDER BY created_at DESC LIMIT ?""",
            (f"%{q}%", f"%{q}%", f"%{q}%", limit),
        )
        results.extend(dict(r) for r in item_rows)

    return {
        "query": q,
        "total": len(results),
        "results": results[:limit],
    }
