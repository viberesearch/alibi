"""HTTP client for the Alibi API used by the (thin) Telegram bot.

The Telegram bot is migrating from an in-process pipeline runner to a thin HTTP
client of the host API (``lt serve``). This module is the foundation of that
migration: it owns the bot's only outbound dependency on the host, so the bot
container itself never touches the DB, Ollama, or the pipeline.

Design notes (see ``docs/TELEGRAM_THIN_BOT_PLAN.md``):

* Base URL comes from ``ALIBI_API_URL`` (default ``http://host.docker.internal``
  ``:3100`` for OrbStack; override to a Tailscale address to run the bot on a
  remote VPS).
* Per-Telegram-user attribution is preserved by sending that user's mnemonic
  ``X-API-Key``; the API's ``validate_api_key`` resolves it to the right user.
* Connection failures (host API briefly down, e.g. the host reboot) are retried
  with backoff so server-buffered Telegram updates aren't dropped.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "http://host.docker.internal:3100"
API_PREFIX = "/api/v1"
# Process can be slow (OCR + local LLM structuring); give it room.
PROCESS_TIMEOUT_SECONDS = 180.0
QUERY_TIMEOUT_SECONDS = 30.0
_MAX_RETRIES = 3
_BACKOFF_BASE_SECONDS = 1.5


class AlibiAPIError(Exception):
    """Raised when the API returns an error or is unreachable after retries."""


class AlibiAPIConnectionError(AlibiAPIError):
    """Raised specifically when the API is unreachable after all retries.

    Distinct from a plain :class:`AlibiAPIError` (which also covers HTTP 4xx/5xx
    responses) so callers can spool-and-retry connection failures while still
    surfacing genuine server errors immediately. Only connection failures should
    be spooled: a 4xx means the request itself is bad and retrying won't help.
    """


@dataclass
class ProcessResult:
    """Mirror of the API ``ProcessResponse`` model."""

    success: bool
    document_id: Optional[str] = None
    fact_id: Optional[str] = None
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    vendor: Optional[str] = None
    amount: Optional[str] = None
    currency: Optional[str] = None
    date: Optional[str] = None
    document_type: Optional[str] = None
    items_count: int = 0
    error: Optional[str] = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ProcessResult":
        return cls(
            success=bool(data.get("success")),
            document_id=data.get("document_id"),
            fact_id=data.get("fact_id"),
            is_duplicate=bool(data.get("is_duplicate")),
            duplicate_of=data.get("duplicate_of"),
            vendor=data.get("vendor"),
            amount=data.get("amount"),
            currency=data.get("currency"),
            date=data.get("date"),
            document_type=data.get("document_type"),
            items_count=int(data.get("items_count") or 0),
            error=data.get("error"),
        )


def _base_url() -> str:
    return os.environ.get("ALIBI_API_URL", DEFAULT_API_URL).rstrip("/")


class AlibiAPIClient:
    """Thin async client for the Alibi API.

    One instance is shared by the bot; a per-request ``api_key`` selects the
    acting user, so a single client serves all Telegram users.
    """

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = (base_url or _base_url()).rstrip("/")

    def _headers(self, api_key: Optional[str]) -> dict[str, str]:
        return {"X-API-Key": api_key} if api_key else {}

    async def _request(
        self,
        method: str,
        path: str,
        *,
        api_key: Optional[str],
        timeout: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Issue a request with retry/backoff on connection errors.

        Connection errors are retried (host API may be briefly down). HTTP error
        responses (4xx/5xx) are not retried -- they're surfaced as
        ``AlibiAPIError`` with the server detail.
        """
        url = f"{self.base_url}{path}"
        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.request(
                        method, url, headers=self._headers(api_key), **kwargs
                    )
                if resp.status_code >= 400:
                    detail = _error_detail(resp)
                    raise AlibiAPIError(f"{resp.status_code}: {detail}")
                return resp.json()
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_exc = exc
                wait = _BACKOFF_BASE_SECONDS * (2**attempt)
                logger.warning(
                    "API unreachable (%s), retry %d/%d in %.1fs",
                    exc,
                    attempt + 1,
                    _MAX_RETRIES,
                    wait,
                )
                await asyncio.sleep(wait)
        raise AlibiAPIConnectionError(
            f"API unreachable after {_MAX_RETRIES} attempts: {last_exc}"
        )

    async def health(self) -> dict[str, Any]:
        """GET /health (no auth, no prefix)."""
        return await self._request(
            "GET", "/health", api_key=None, timeout=QUERY_TIMEOUT_SECONDS
        )

    async def whoami(self, api_key: str) -> Optional[dict[str, Any]]:
        """GET /users/me. Returns the user dict, or None if the key is invalid.

        Doubles as mnemonic validation for ``/link``: an invalid key yields a
        401 -> None rather than raising.
        """
        try:
            return await self._request(
                "GET",
                f"{API_PREFIX}/users/me",
                api_key=api_key,
                timeout=QUERY_TIMEOUT_SECONDS,
            )
        except AlibiAPIError as exc:
            if str(exc).startswith("401"):
                return None
            raise

    async def update_user_name(self, user_id: str, name: str, *, api_key: str) -> None:
        """PATCH /users/{user_id} to set a display name."""
        await self._request(
            "PATCH",
            f"{API_PREFIX}/users/{user_id}",
            api_key=api_key,
            timeout=QUERY_TIMEOUT_SECONDS,
            json={"name": name},
        )

    async def set_fact_location(
        self, fact_id: str, map_url: str, *, api_key: Optional[str] = None
    ) -> dict[str, Any]:
        """POST /facts/{fact_id}/location to attach a Google Maps location."""
        return await self._request(
            "POST",
            f"{API_PREFIX}/facts/{fact_id}/location",
            api_key=api_key,
            timeout=QUERY_TIMEOUT_SECONDS,
            json={"map_url": map_url},
        )

    async def process_document(
        self,
        data: bytes,
        filename: str,
        *,
        api_key: Optional[str] = None,
        doc_type: Optional[str] = None,
        map_url: Optional[str] = None,
        vendor_hint: Optional[str] = None,
    ) -> ProcessResult:
        """POST a single document to /process and return the result."""
        params: dict[str, str] = {}
        if doc_type:
            params["type"] = doc_type
        if map_url:
            params["map_url"] = map_url
        if vendor_hint:
            params["vendor"] = vendor_hint
        files = {"file": (filename, data)}
        body = await self._request(
            "POST",
            f"{API_PREFIX}/process",
            api_key=api_key,
            timeout=PROCESS_TIMEOUT_SECONDS,
            params=params,
            files=files,
        )
        return ProcessResult.from_json(body)

    # -- Query helpers (read-only) ----------------------------------------

    async def search(
        self,
        query: str,
        *,
        api_key: Optional[str] = None,
        type_: Optional[str] = None,
        limit: int = 20,
        semantic: bool = True,
    ) -> dict[str, Any]:
        """GET /search. Returns ``{query, total, results}``."""
        params: dict[str, Any] = {"q": query, "limit": limit, "semantic": semantic}
        if type_:
            params["type"] = type_
        return await self._request(
            "GET",
            f"{API_PREFIX}/search",
            api_key=api_key,
            timeout=QUERY_TIMEOUT_SECONDS,
            params=params,
        )

    async def spending_summary(
        self,
        *,
        api_key: Optional[str] = None,
        period: str = "month",
        date_from: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """GET /analytics/spending grouped by month or vendor."""
        params: dict[str, Any] = {"period": period, "limit": limit}
        if date_from:
            params["date_from"] = date_from
        body = await self._request(
            "GET",
            f"{API_PREFIX}/analytics/spending",
            api_key=api_key,
            timeout=QUERY_TIMEOUT_SECONDS,
            params=params,
        )
        return body if isinstance(body, list) else []

    async def detect_subscriptions(
        self, *, api_key: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """GET /analytics/subscriptions."""
        body = await self._request(
            "GET",
            f"{API_PREFIX}/analytics/subscriptions",
            api_key=api_key,
            timeout=QUERY_TIMEOUT_SECONDS,
        )
        return body if isinstance(body, list) else []

    async def list_facts(
        self,
        *,
        api_key: Optional[str] = None,
        vendor: Optional[str] = None,
        fact_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        page: int = 1,
        per_page: int = 50,
    ) -> dict[str, Any]:
        """GET /facts with filters. Returns a paginated ``{items, total, ...}``."""
        params: dict[str, Any] = {"page": page, "per_page": per_page}
        if vendor:
            params["vendor"] = vendor
        if fact_type:
            params["fact_type"] = fact_type
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        return await self._request(
            "GET",
            f"{API_PREFIX}/facts",
            api_key=api_key,
            timeout=QUERY_TIMEOUT_SECONDS,
            params=params,
        )

    async def get_fact(
        self, fact_id: str, *, api_key: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """GET /facts/{fact_id}. Returns the fact dict, or None on 404."""
        try:
            return await self._request(
                "GET",
                f"{API_PREFIX}/facts/{fact_id}",
                api_key=api_key,
                timeout=QUERY_TIMEOUT_SECONDS,
            )
        except AlibiAPIError as exc:
            if str(exc).startswith("404"):
                return None
            raise

    async def list_line_items(
        self,
        *,
        api_key: Optional[str] = None,
        category: Optional[str] = None,
        name: Optional[str] = None,
        page: int = 1,
        per_page: int = 50,
    ) -> dict[str, Any]:
        """GET /line-items with optional category/name filters."""
        params: dict[str, Any] = {"page": page, "per_page": per_page}
        if category:
            params["category"] = category
        if name:
            params["name"] = name
        return await self._request(
            "GET",
            f"{API_PREFIX}/line-items",
            api_key=api_key,
            timeout=QUERY_TIMEOUT_SECONDS,
            params=params,
        )

    async def list_warranty_expiring(
        self,
        *,
        api_key: Optional[str] = None,
        ahead_days: int = 90,
        expired_days: int = 30,
    ) -> list[dict[str, Any]]:
        """GET /items/warranty/expiring."""
        body = await self._request(
            "GET",
            f"{API_PREFIX}/items/warranty/expiring",
            api_key=api_key,
            timeout=QUERY_TIMEOUT_SECONDS,
            params={"ahead_days": ahead_days, "expired_days": expired_days},
        )
        return body if isinstance(body, list) else []

    async def list_budget_scenarios(
        self,
        *,
        api_key: Optional[str] = None,
        space_id: str = "default",
    ) -> list[dict[str, Any]]:
        """GET /budgets/scenarios."""
        body = await self._request(
            "GET",
            f"{API_PREFIX}/budgets/scenarios",
            api_key=api_key,
            timeout=QUERY_TIMEOUT_SECONDS,
            params={"space_id": space_id},
        )
        return body if isinstance(body, list) else []

    async def compare_budgets(
        self,
        base_id: str,
        compare_id: str,
        *,
        api_key: Optional[str] = None,
        period: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """GET /budgets/compare."""
        params: dict[str, Any] = {"base_id": base_id, "compare_id": compare_id}
        if period:
            params["period"] = period
        body = await self._request(
            "GET",
            f"{API_PREFIX}/budgets/compare",
            api_key=api_key,
            timeout=QUERY_TIMEOUT_SECONDS,
            params=params,
        )
        return body if isinstance(body, list) else []

    async def enrichment_review(
        self,
        *,
        api_key: Optional[str] = None,
        threshold: float = 0.8,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """GET /enrichment/review (items below the confidence threshold)."""
        body = await self._request(
            "GET",
            f"{API_PREFIX}/enrichment/review",
            api_key=api_key,
            timeout=QUERY_TIMEOUT_SECONDS,
            params={"threshold": threshold, "limit": limit},
        )
        return body if isinstance(body, list) else []

    async def enrichment_stats(
        self, *, api_key: Optional[str] = None
    ) -> dict[str, Any]:
        """GET /enrichment/stats."""
        return await self._request(
            "GET",
            f"{API_PREFIX}/enrichment/stats",
            api_key=api_key,
            timeout=QUERY_TIMEOUT_SECONDS,
        )

    async def get_identity(
        self, identity_id: str, *, api_key: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """GET /identities/{identity_id}. Returns None on 404."""
        try:
            return await self._request(
                "GET",
                f"{API_PREFIX}/identities/{identity_id}",
                api_key=api_key,
                timeout=QUERY_TIMEOUT_SECONDS,
            )
        except AlibiAPIError as exc:
            if str(exc).startswith("404"):
                return None
            raise

    # -- Mutation helpers -------------------------------------------------

    async def confirm_enrichment(
        self,
        fact_item_id: str,
        *,
        api_key: Optional[str] = None,
        brand: Optional[str] = None,
        category: Optional[str] = None,
    ) -> bool:
        """POST /enrichment/review/{id}/confirm. Returns False on 404."""
        try:
            await self._request(
                "POST",
                f"{API_PREFIX}/enrichment/review/{fact_item_id}/confirm",
                api_key=api_key,
                timeout=QUERY_TIMEOUT_SECONDS,
                json={"brand": brand, "category": category},
            )
            return True
        except AlibiAPIError as exc:
            if str(exc).startswith("404"):
                return False
            raise

    async def reject_enrichment(
        self, fact_item_id: str, *, api_key: Optional[str] = None
    ) -> bool:
        """POST /enrichment/review/{id}/reject. Returns False on 404."""
        try:
            await self._request(
                "POST",
                f"{API_PREFIX}/enrichment/review/{fact_item_id}/reject",
                api_key=api_key,
                timeout=QUERY_TIMEOUT_SECONDS,
            )
            return True
        except AlibiAPIError as exc:
            if str(exc).startswith("404"):
                return False
            raise

    async def correct_vendor(
        self, fact_id: str, vendor: str, *, api_key: Optional[str] = None
    ) -> bool:
        """POST /facts/{fact_id}/correct-vendor. Returns False on 404."""
        try:
            await self._request(
                "POST",
                f"{API_PREFIX}/facts/{fact_id}/correct-vendor",
                api_key=api_key,
                timeout=QUERY_TIMEOUT_SECONDS,
                json={"vendor": vendor},
            )
            return True
        except AlibiAPIError as exc:
            if str(exc).startswith("404"):
                return False
            raise

    async def update_fact(
        self,
        fact_id: str,
        fields: dict[str, Any],
        *,
        api_key: Optional[str] = None,
    ) -> bool:
        """PATCH /facts/{fact_id}. Returns False on 404."""
        try:
            await self._request(
                "PATCH",
                f"{API_PREFIX}/facts/{fact_id}",
                api_key=api_key,
                timeout=QUERY_TIMEOUT_SECONDS,
                json=fields,
            )
            return True
        except AlibiAPIError as exc:
            if str(exc).startswith("404"):
                return False
            raise

    async def update_line_item(
        self,
        line_item_id: str,
        fields: dict[str, Any],
        *,
        api_key: Optional[str] = None,
    ) -> bool:
        """PATCH /line-items/{id}. Returns False on 404."""
        try:
            await self._request(
                "PATCH",
                f"{API_PREFIX}/line-items/{line_item_id}",
                api_key=api_key,
                timeout=QUERY_TIMEOUT_SECONDS,
                json=fields,
            )
            return True
        except AlibiAPIError as exc:
            if str(exc).startswith("404"):
                return False
            raise

    async def merge_identities(
        self,
        identity_id_a: str,
        identity_id_b: str,
        *,
        api_key: Optional[str] = None,
    ) -> bool:
        """POST /identities/merge. Returns False on 404 (one/both missing)."""
        try:
            await self._request(
                "POST",
                f"{API_PREFIX}/identities/merge",
                api_key=api_key,
                timeout=QUERY_TIMEOUT_SECONDS,
                json={
                    "identity_id_a": identity_id_a,
                    "identity_id_b": identity_id_b,
                },
            )
            return True
        except AlibiAPIError as exc:
            if str(exc).startswith("404"):
                return False
            raise

    async def annotate_fact(
        self,
        fact_id: str,
        *,
        annotation_type: str,
        key: str,
        value: str,
        api_key: Optional[str] = None,
    ) -> str:
        """POST /annotations/facts/{fact_id}. Returns the new annotation id."""
        body = await self._request(
            "POST",
            f"{API_PREFIX}/annotations/facts/{fact_id}",
            api_key=api_key,
            timeout=QUERY_TIMEOUT_SECONDS,
            json={
                "annotation_type": annotation_type,
                "key": key,
                "value": value,
            },
        )
        return str(body.get("id", ""))

    async def delete_annotation(
        self, annotation_id: str, *, api_key: Optional[str] = None
    ) -> bool:
        """DELETE /annotations/{annotation_id}. Returns False on 404."""
        try:
            await self._request(
                "DELETE",
                f"{API_PREFIX}/annotations/{annotation_id}",
                api_key=api_key,
                timeout=QUERY_TIMEOUT_SECONDS,
            )
            return True
        except AlibiAPIError as exc:
            if str(exc).startswith("404"):
                return False
            raise

    # -- Barcode helpers --------------------------------------------------

    async def scan_barcode(
        self,
        data: bytes,
        filename: str,
        *,
        api_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """POST /items/barcode/scan. Returns ``{count, barcodes}``."""
        files = {"file": (filename, data)}
        return await self._request(
            "POST",
            f"{API_PREFIX}/items/barcode/scan",
            api_key=api_key,
            timeout=PROCESS_TIMEOUT_SECONDS,
            files=files,
        )

    async def barcode_lookup(
        self, barcode: str, *, api_key: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """GET /items/barcode/{barcode} (Open Food Facts). None on 404."""
        try:
            return await self._request(
                "GET",
                f"{API_PREFIX}/items/barcode/{barcode}",
                api_key=api_key,
                timeout=QUERY_TIMEOUT_SECONDS,
            )
        except AlibiAPIError as exc:
            if str(exc).startswith("404"):
                return None
            raise

    async def enrich_by_barcode(
        self, barcode: str, *, api_key: Optional[str] = None
    ) -> dict[str, Any]:
        """POST /items/barcode/{barcode}/enrich. Returns ``{matched, enriched}``."""
        return await self._request(
            "POST",
            f"{API_PREFIX}/items/barcode/{barcode}/enrich",
            api_key=api_key,
            timeout=PROCESS_TIMEOUT_SECONDS,
        )

    async def process_document_group(
        self,
        pages: list[tuple[bytes, str]],
        *,
        api_key: Optional[str] = None,
        doc_type: Optional[str] = None,
        map_url: Optional[str] = None,
        vendor_hint: Optional[str] = None,
    ) -> ProcessResult:
        """POST multiple pages to /process/group as one multi-page document."""
        params: dict[str, str] = {}
        if doc_type:
            params["type"] = doc_type
        if map_url:
            params["map_url"] = map_url
        if vendor_hint:
            params["vendor"] = vendor_hint
        files = [("files", (name, data)) for data, name in pages]
        body = await self._request(
            "POST",
            f"{API_PREFIX}/process/group",
            api_key=api_key,
            timeout=PROCESS_TIMEOUT_SECONDS,
            params=params,
            files=files,
        )
        return ProcessResult.from_json(body)


def _error_detail(resp: httpx.Response) -> str:
    try:
        payload = resp.json()
        if isinstance(payload, dict) and "detail" in payload:
            return str(payload["detail"])
    except Exception:  # noqa: BLE001 -- best-effort error extraction
        pass
    return resp.text[:200]
