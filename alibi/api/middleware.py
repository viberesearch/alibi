"""API middleware for rate limiting, security headers, and request logging."""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Cleanup stale rate limiter entries every 5 minutes
_CLEANUP_INTERVAL = 300


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-API-key rate limiting using sliding window.

    Default: 100 requests per 60 seconds per API key.
    Unauthenticated requests use IP-based limiting.
    Periodically cleans up stale entries to prevent memory growth.
    """

    def __init__(
        self,
        app: object,
        requests_per_window: int = 100,
        window_seconds: int = 60,
    ):
        super().__init__(app)  # type: ignore[arg-type]
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._last_cleanup = time.monotonic()

    def _get_key(self, request: Request) -> str:
        """Get rate limit key: API key or client IP."""
        api_key = request.headers.get("x-api-key")
        if api_key:
            return f"key:{api_key[:16]}"
        client = request.client
        return f"ip:{client.host if client else 'unknown'}"

    def _cleanup_stale(self) -> None:
        """Remove keys with no recent requests to prevent memory growth."""
        now = time.monotonic()
        if now - self._last_cleanup < _CLEANUP_INTERVAL:
            return
        self._last_cleanup = now
        cutoff = now - self.window_seconds
        stale_keys = [k for k, v in self._requests.items() if not v or v[-1] < cutoff]
        for k in stale_keys:
            del self._requests[k]

    def _is_rate_limited(self, key: str) -> tuple[bool, int]:
        """Check if key is rate limited. Returns (limited, remaining)."""
        now = time.monotonic()
        cutoff = now - self.window_seconds

        # Remove expired entries
        self._requests[key] = [t for t in self._requests[key] if t > cutoff]

        count = len(self._requests[key])
        remaining = max(0, self.requests_per_window - count)

        if count >= self.requests_per_window:
            return True, 0

        self._requests[key].append(now)
        return False, remaining - 1

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        self._cleanup_stale()

        # Skip rate limiting for health checks and docs
        if request.url.path in ("/health", "/openapi.json", "/docs", "/redoc"):
            return await call_next(request)
        # Skip for testclient
        client = request.client
        if client and client.host == "testclient":
            return await call_next(request)
        # Skip for internal calls (no socket)
        if client is None:
            return await call_next(request)

        key = self._get_key(request)
        limited, remaining = self._is_rate_limited(key)

        if limited:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={
                    "Retry-After": str(self.window_seconds),
                    "X-RateLimit-Limit": str(self.requests_per_window),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_window)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log API requests with timing."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000

        logger.info(
            "%s %s -> %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
