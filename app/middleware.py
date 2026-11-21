"""Outermost middleware: request id, API-key auth by default, per-key rate limit.

Every path requires a valid X-API-Key unless it is in OPEN_PATHS, so a newly
added route is protected without any per-route code. The raw key is never
logged; requests are identified by key_id = sha256(key)[:8].
"""

import hashlib
import re
import time
import uuid
from collections import deque
from contextvars import ContextVar
from logging import getLogger

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.auth import API_KEY_HEADER, configured_key, is_valid_key
from app.config import settings
from app.metrics import ERRORS, REQUESTS

request_id_context: ContextVar[str] = ContextVar("request_id", default="-")
key_id_context: ContextVar[str] = ContextVar("key_id", default="-")
logger = getLogger(__name__)
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")

# The only paths reachable without an API key.
OPEN_PATHS = frozenset({"/health", "/ready", "/docs", "/openapi.json"})
# Probes are never rate limited.
RATE_LIMIT_EXEMPT = frozenset({"/health", "/ready"})
RATE_LIMIT_WINDOW_SECONDS = 60


def key_id_for(key: str) -> str:
    """Stable, non-reversible identifier for logs and rate limiting."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]


class RateLimiter:
    """Sliding-window limiter, in-process (per worker)."""

    def __init__(self, window_seconds: int = RATE_LIMIT_WINDOW_SECONDS) -> None:
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}

    def allow(self, bucket: str, limit: int) -> bool:
        now = time.monotonic()
        hits = self._hits.setdefault(bucket, deque())
        cutoff = now - self.window_seconds
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= limit:
            return False
        hits.append(now)
        return True

    def reset(self) -> None:
        self._hits.clear()


rate_limiter = RateLimiter()


def _json_error(status_code: int, detail: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail}, headers=headers)


class QueryBodyLimitMiddleware:
    """Bound the query body before JSON parsing, including chunked requests."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"].rstrip("/") != "/api/v1/query"
        ):
            await self.app(scope, receive, send)
            return

        limit = settings.max_query_body_bytes
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > limit:
                ERRORS.labels(type="body_too_large").inc()
                response = _json_error(413, "Request body exceeds the byte limit.")
                await response(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, replay, send)


class ProductionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        supplied_id = request.headers.get("X-Request-ID", "")
        provided = request.headers.get(API_KEY_HEADER)
        key = configured_key()
        id_contains_key = (provided and provided in supplied_id) or (
            key and key in supplied_id.encode()
        )
        request_id = (
            supplied_id
            if _REQUEST_ID_PATTERN.fullmatch(supplied_id) and not id_contains_key
            else uuid.uuid4().hex
        )
        request_token = request_id_context.set(request_id)
        key_token = key_id_context.set("-")
        started_at = time.perf_counter()
        status_code = 500
        key_id = "-"
        path = request.url.path

        try:
            cors_preflight = (
                request.method == "OPTIONS"
                and "origin" in request.headers
                and "access-control-request-method" in request.headers
            )
            auth_required = not cors_preflight and path not in OPEN_PATHS
            authenticated = provided is not None and is_valid_key(provided, key)
            if authenticated:
                key_id = key_id_for(provided)
                # Every log line of this request (incl. the access log) carries it.
                key_id_context.set(key_id)

            response = None
            if settings.rate_limit_per_min > 0 and path not in RATE_LIMIT_EXEMPT:
                client_host = request.client.host if request.client else "unknown"
                bucket = f"key:{key_id}" if authenticated else f"ip:{client_host}"
                if not rate_limiter.allow(bucket, settings.rate_limit_per_min):
                    ERRORS.labels(type="rate_limited").inc()
                    response = _json_error(
                        429, "Rate limit exceeded.",
                        {"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
                    )

            if response is None and auth_required and not authenticated:
                ERRORS.labels(type="unauthorized").inc()
                response = _json_error(
                    401, "Invalid or missing API key.", {"WWW-Authenticate": "ApiKey"}
                )

            if response is None:
                response = await call_next(request)

            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            REQUESTS.labels(status=str(status_code)).inc()
            logger.info(
                "request_complete",
                extra={
                    "method": request.method,
                    "path": getattr(request.scope.get("route"), "path", "-"),
                    "status_code": status_code,
                    "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
                },
            )
            key_id_context.reset(key_token)
            request_id_context.reset(request_token)
