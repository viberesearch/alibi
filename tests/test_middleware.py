"""Tests for API middleware (rate limiting and request logging)."""

import logging
import time

import pytest
from starlette.requests import Request
from starlette.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

from alibi.api.middleware import RateLimitMiddleware, RequestLoggingMiddleware


def _make_app(requests_per_window: int = 5, window_seconds: int = 60) -> Starlette:
    """Create a minimal Starlette app with rate limiting middleware."""

    async def homepage(request: Request) -> PlainTextResponse:
        return PlainTextResponse("OK")

    async def health(request: Request) -> PlainTextResponse:
        return PlainTextResponse("healthy")

    app = Starlette(
        routes=[
            Route("/", homepage),
            Route("/health", health),
        ],
    )
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_window=requests_per_window,
        window_seconds=window_seconds,
    )
    return app


class TestRateLimitInternal:
    """Test internal rate limiting logic directly."""

    def test_is_rate_limited_under_threshold(self) -> None:
        """Requests under limit are not rate limited."""
        middleware = RateLimitMiddleware.__new__(RateLimitMiddleware)
        middleware.requests_per_window = 5
        middleware.window_seconds = 60
        middleware._requests = {}

        from collections import defaultdict

        middleware._requests = defaultdict(list)

        limited, remaining = middleware._is_rate_limited("key:test")
        assert limited is False
        assert remaining == 4  # 5 - 1 (current request)

    def test_is_rate_limited_at_threshold(self) -> None:
        """Reaching the limit returns rate limited."""
        middleware = RateLimitMiddleware.__new__(RateLimitMiddleware)
        middleware.requests_per_window = 3
        middleware.window_seconds = 60

        from collections import defaultdict

        middleware._requests = defaultdict(list)

        # Make 3 requests to fill the window
        for _ in range(3):
            middleware._is_rate_limited("key:test")

        limited, remaining = middleware._is_rate_limited("key:test")
        assert limited is True
        assert remaining == 0

    def test_sliding_window_expires_old_requests(self) -> None:
        """Old requests outside the window are removed."""
        middleware = RateLimitMiddleware.__new__(RateLimitMiddleware)
        middleware.requests_per_window = 3
        middleware.window_seconds = 1  # 1-second window

        from collections import defaultdict

        middleware._requests = defaultdict(list)

        # Fill the window
        for _ in range(3):
            middleware._is_rate_limited("key:test")

        # Should be limited now
        limited, _ = middleware._is_rate_limited("key:test")
        assert limited is True

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again
        limited, remaining = middleware._is_rate_limited("key:test")
        assert limited is False
        assert remaining >= 0

    def test_get_key_with_api_key(self) -> None:
        """API key in header is used for rate limit key."""
        middleware = RateLimitMiddleware.__new__(RateLimitMiddleware)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"x-api-key", b"my-secret-api-key-12345")],
            "query_string": b"",
        }
        request = Request(scope)

        key = middleware._get_key(request)
        assert key.startswith("key:")
        # API key is truncated to first 16 chars
        assert key == "key:my-secret-api-ke"

    def test_get_key_without_api_key(self) -> None:
        """IP address is used when no API key is present."""
        middleware = RateLimitMiddleware.__new__(RateLimitMiddleware)

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "query_string": b"",
            "client": ("192.168.1.1", 12345),
        }
        request = Request(scope)

        key = middleware._get_key(request)
        assert key == "ip:192.168.1.1"


class TestRateLimitMiddlewareIntegration:
    """Integration tests using TestClient.

    Note: RateLimitMiddleware skips testclient host, so these test
    the bypass behavior for health endpoints and testclient.
    """

    def test_health_endpoint_bypasses_rate_limit(self) -> None:
        """Health check endpoint is never rate limited."""
        app = _make_app(requests_per_window=1)
        client = TestClient(app)

        # Health should always work, even with very low limit
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200

    def test_testclient_bypasses_rate_limit(self) -> None:
        """TestClient host is skipped by rate limiter."""
        app = _make_app(requests_per_window=1)
        client = TestClient(app)

        # TestClient is always allowed through
        for _ in range(5):
            response = client.get("/")
            assert response.status_code == 200


class TestRequestLoggingMiddleware:
    """Test request logging middleware."""

    def test_request_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        """Requests produce log entries with method, path, status, timing."""

        async def homepage(request: Request) -> PlainTextResponse:
            return PlainTextResponse("OK")

        app = Starlette(routes=[Route("/test-path", homepage)])
        app.add_middleware(RequestLoggingMiddleware)
        client = TestClient(app)

        with caplog.at_level(logging.INFO, logger="alibi.api.middleware"):
            response = client.get("/test-path")

        assert response.status_code == 200
        assert any(
            "GET" in record.message and "/test-path" in record.message
            for record in caplog.records
        )
        assert any("200" in record.message for record in caplog.records)
