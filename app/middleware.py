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

request_id_context: ContextVar[str] = ContextVar("request_id", default="-")
logger = getLogger(__name__)
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


class ProductionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int, window_seconds: int) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = {}

    async def dispatch(self, request: Request, call_next) -> Response:
        supplied_id = request.headers.get("X-Request-ID", "")
        request_id = (
            supplied_id
            if _REQUEST_ID_PATTERN.fullmatch(supplied_id)
            else uuid.uuid4().hex
        )
        token = request_id_context.set(request_id)
        started_at = time.perf_counter()
        status_code = 500

        try:
            if self._rate_limited(request):
                status_code = 429
                response = JSONResponse(
                    status_code=status_code,
                    content={"detail": "Rate limit exceeded."},
                    headers={"Retry-After": str(self.window_seconds)},
                )
            else:
                response = await call_next(request)
                status_code = response.status_code

            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            logger.info(
                "request_complete",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "latency_ms": round(
                        (time.perf_counter() - started_at) * 1000,
                        2,
                    ),
                },
            )
            request_id_context.reset(token)

    def _rate_limited(self, request: Request) -> bool:
        if self.max_requests <= 0 or request.url.path in {"/health", "/ready"}:
            return False

        client_host = request.client.host if request.client else "unknown"
        now = time.monotonic()
        requests = self._requests.setdefault(client_host, deque())
        cutoff = now - self.window_seconds
        while requests and requests[0] <= cutoff:
            requests.popleft()

        if len(requests) >= self.max_requests:
            return True

        requests.append(now)
        return False