"""Prometheus metrics (served at /metrics, which requires an API key)."""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()

STAGES = ("retrieve", "rerank", "generate")
ERROR_TYPES = (
    "unauthorized",
    "rate_limited",
    "service_unavailable",
    "invalid_request",
    "generation_failed",
    "groq_timeout",
    "pipeline_error",
)

STAGE_LATENCY = Histogram(
    "rag_stage_latency_seconds",
    "Latency of each RAG pipeline stage.",
    ["stage"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=REGISTRY,
)
ERRORS = Counter(
    "rag_errors_total",
    "Errors by type (auth, rate limit, unavailable artifacts, LLM failure, ...).",
    ["type"],
    registry=REGISTRY,
)
REQUESTS = Counter(
    "http_requests_total",
    "HTTP requests by status code.",
    ["status"],
    registry=REGISTRY,
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Time to produce an HTTP response.",
    registry=REGISTRY,
)

# Pre-create label values so every series is exported from the first scrape.
for _stage in STAGES:
    STAGE_LATENCY.labels(stage=_stage)
for _error in ERROR_TYPES:
    ERRORS.labels(type=_error)


def render() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
