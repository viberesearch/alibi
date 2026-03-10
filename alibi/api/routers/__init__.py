"""API routers for Alibi."""

from alibi.api.routers.analytics import router as analytics_router
from alibi.api.routers.annotations import router as annotations_router
from alibi.api.routers.budgets import router as budgets_router
from alibi.api.routers.corrections import router as corrections_router
from alibi.api.routers.artifacts import router as artifacts_router
from alibi.api.routers.distribute import router as distribute_router
from alibi.api.routers.enrichment import router as enrichment_router
from alibi.api.routers.export import router as export_router
from alibi.api.routers.facts import router as facts_router
from alibi.api.routers.health import router as health_router
from alibi.api.routers.identities import router as identities_router
from alibi.api.routers.items import router as items_router
from alibi.api.routers.line_items import router as line_items_router
from alibi.api.routers.nutrition import router as nutrition_router
from alibi.api.routers.predictions import router as predictions_router
from alibi.api.routers.process import router as process_router
from alibi.api.routers.reports import router as reports_router
from alibi.api.routers.search import router as search_router
from alibi.api.routers.users import router as users_router

__all__ = [
    "analytics_router",
    "annotations_router",
    "budgets_router",
    "corrections_router",
    "artifacts_router",
    "distribute_router",
    "enrichment_router",
    "export_router",
    "facts_router",
    "health_router",
    "identities_router",
    "items_router",
    "line_items_router",
    "nutrition_router",
    "predictions_router",
    "process_router",
    "reports_router",
    "search_router",
    "users_router",
]
