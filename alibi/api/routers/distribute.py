"""Distribution endpoints for audience-specific output."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse

from alibi.api.deps import get_database, require_user
from alibi.db.connection import DatabaseManager
from alibi.distribution import DistributionForm, OutputFormat, DistributionRenderer
from alibi.distribution.forms import distribute

router = APIRouter()


@router.get("/{form}")
async def get_distribution(
    form: DistributionForm,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    format: OutputFormat = Query(OutputFormat.JSON),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
) -> Any:
    """Generate distribution in requested form and format.

    Args:
        form: Distribution form (summary, detailed, analytical, tabular)
        db: Database manager (injected)
        user: Authenticated user (injected)
        format: Output format (json, md, csv, html)
        date_from: Start date filter (YYYY-MM-DD)
        date_to: End date filter (YYYY-MM-DD)

    Returns:
        JSON response or PlainTextResponse depending on format
    """
    if format == OutputFormat.JSON:
        result = distribute(db, form, date_from, date_to)
        return {
            "form": form.value,
            "format": format.value,
            "data": result.data,
            "metadata": result.metadata,
        }

    renderer = DistributionRenderer(db)
    content = renderer.render(form, format, date_from, date_to)
    media_types = {
        "md": "text/markdown",
        "csv": "text/csv",
        "html": "text/html",
    }
    return PlainTextResponse(
        content=content,
        media_type=media_types.get(format.value, "text/plain"),
    )
