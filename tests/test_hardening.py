"""Hardening: auth-by-default allowlist, per-key rate limit, query cap,
LLM failure fallback, key_id logging, /metrics. No network, no Groq calls."""

import asyncio
import hashlib
import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
from groq import APIStatusError, APITimeoutError
from pydantic import ValidationError

from app.config import Settings, settings
from app.generation.groq_generator import GroqGenerator
from app.main import _JsonFormatter, app
from app.metrics import REGISTRY
from app.middleware import OPEN_PATHS, QueryBodyLimitMiddleware
from app.rag_service import RAGService
from tests.conftest import TEST_API_KEY


def _passage(pid):
    return {"passage_id": pid, "passage_text": f"passage {pid}", "score": float(pid)}


@pytest.fixture
def mock_service(monkeypatch):
    service = Mock()
    service.answer.return_value = {"answer": "ok", "citations": [], "latency_ms": 1.0}
    monkeypatch.setattr(app.state, "rag_service", service)
    return service


def test_settings_hide_credentials_and_still_load_them(monkeypatch):
    credentials = {"API_KEY": "synthetic-api-value", "GROQ_API_KEY": "synthetic-groq-value"}
    for name, value in credentials.items():
        monkeypatch.setenv(name, value)
    configured = Settings(_env_file=None)
    assert configured.api_key == credentials["API_KEY"]
    assert configured.groq_api_key == credentials["GROQ_API_KEY"]
    for diagnostic in (repr(configured), str(configured)):
        assert all(value not in diagnostic for value in credentials.values())

    monkeypatch.setenv("MAX_QUERY_BODY_BYTES", credentials["GROQ_API_KEY"])
    with pytest.raises(ValidationError) as invalid:
        Settings(_env_file=None)
    assert credentials["GROQ_API_KEY"] not in str(invalid.value)


@pytest.mark.parametrize(("delta", "status"), [(-1, 200), (0, 200), (1, 413)])
def test_query_body_limit_boundary(client, mock_service, monkeypatch, delta, status):
    monkeypatch.setattr(settings, "max_query_body_bytes", 32)
    body = b'{"query":"ok"}'.ljust(32 + delta)
    response = client.post("/api/v1/query", content=body, headers={
        "Content-Type": "application/json", "X-Request-ID": "body-limit-test",
    })
    assert response.status_code == status
    assert response.headers["X-Request-ID"] == "body-limit-test"
    if status == 413:
        assert response.json() == {"detail": "Request body exceeds the byte limit."}
    assert mock_service.answer.call_count == int(status == 200)


@pytest.mark.parametrize("content_length", [None, "1", "invalid", "100"])
@pytest.mark.parametrize("path", ["/api/v1/query", "/api/v1/query/"])
def test_actual_stream_is_bounded_before_json_parsing(
    monkeypatch, mock_service, content_length, path,
):
    monkeypatch.setattr(settings, "max_query_body_bytes", 8)

    async def chunks():
        yield b'{"'
        yield b"invalid and oversized"

    async def post():
        headers = {"X-API-Key": TEST_API_KEY, "Content-Type": "application/json"}
        if content_length is not None:
            headers["Content-Length"] = content_length
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, root_path="/staging"),
            base_url="http://testserver",
        ) as http:
            return await http.post(f"/staging{path}", content=chunks(), headers=headers)

    assert asyncio.run(post()).status_code == 413
    mock_service.answer.assert_not_called()


def test_query_body_limit_preserves_auth_and_malformed_json(client, anon_client, monkeypatch):
    monkeypatch.setattr(settings, "max_query_body_bytes", 8)
    headers = {"Content-Type": "application/json"}
    assert anon_client.post("/api/v1/query", content=b"x" * 9, headers=headers).status_code == 401
    malformed = client.post("/api/v1/query", content=b"{", headers=headers)
    assert malformed.status_code == 422
    assert malformed.json()["detail"][0]["type"] == "json_invalid"


def test_query_body_limit_handles_disconnect_without_replaying():
    endpoint = Mock(side_effect=AssertionError("Endpoint reached after disconnect"))
    send = Mock(side_effect=AssertionError("Response started after disconnect"))
    messages = iter([
        {"type": "http.request", "body": b"{", "more_body": True},
        {"type": "http.disconnect"},
    ])

    async def receive():
        return next(messages)

    scope = {"type": "http", "method": "POST", "path": "/api/v1/query"}
    asyncio.run(QueryBodyLimitMiddleware(endpoint)(scope, receive, send))
    endpoint.assert_not_called()
    send.assert_not_called()


def test_query_body_limit_replays_multiple_chunks_and_forwards_later_disconnect(monkeypatch):
    monkeypatch.setattr(settings, "max_query_body_bytes", 4)
    messages = iter([
        {"type": "http.request", "body": b"ab", "more_body": True},
        {"type": "http.request", "body": b"cd"},
        {"type": "http.disconnect"},
    ])
    received = []

    async def receive():
        return next(messages)

    async def endpoint(scope, read, send):
        received.extend([await read(), await read()])

    scope = {"type": "http", "method": "POST", "path": "/api/v1/query"}
    asyncio.run(QueryBodyLimitMiddleware(endpoint)(scope, receive, Mock()))
    assert received == [
        {"type": "http.request", "body": b"abcd", "more_body": False},
        {"type": "http.disconnect"},
    ]


# --- auth by default + allowlist ------------------------------------------

def test_open_allowlist_is_exactly_the_four_paths():
    assert {"/health", "/ready", "/docs", "/openapi.json"} == OPEN_PATHS


@pytest.mark.parametrize("path", sorted(OPEN_PATHS))
def test_allowlisted_paths_need_no_key(anon_client, path):
    assert anon_client.get(path).status_code in {200, 503}  # /ready may be 503


@pytest.mark.parametrize("path", ["/metadata", "/metrics", "/redoc", "/no-such-route"])
def test_everything_else_needs_a_key(anon_client, path):
    response = anon_client.get(path)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "ApiKey"


def test_unlisted_new_route_requires_key_without_any_route_code(anon_client, client):
    """A route added with no auth code of its own is still protected."""
    app.add_api_route("/new-unlisted-route", lambda: {"ok": True}, methods=["GET"])
    try:
        assert anon_client.get("/new-unlisted-route").status_code == 401
        assert anon_client.get(
            "/new-unlisted-route", headers={"X-API-Key": "wrong"}
        ).status_code == 401
        assert client.get("/new-unlisted-route").json() == {"ok": True}
    finally:
        app.router.routes[:] = [
            r for r in app.router.routes if getattr(r, "path", None) != "/new-unlisted-route"
        ]


def test_cors_preflight_is_not_blocked_by_auth(anon_client):
    response = anon_client.options(
        "/api/v1/query",
        headers={"Origin": "https://example.org", "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code != 401


# --- per-key rate limit ---------------------------------------------------

def test_rate_limit_is_per_key(anon_client, mock_service, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "key-a")
    monkeypatch.setattr(settings, "rate_limit_per_min", 3)
    labels = {"type": "rate_limited"}
    before = REGISTRY.get_sample_value("rag_errors_total", labels)

    def get(key):
        return anon_client.get("/metadata", headers={"X-API-Key": key})

    assert [get("key-a").status_code for _ in range(3)] == [200, 200, 200]
    limited = get("key-a")
    assert limited.status_code == 429
    assert limited.headers["Retry-After"] == "60"
    assert REGISTRY.get_sample_value("rag_errors_total", labels) == before + 1
    # Rotating the key starts a fresh rate-limit bucket.
    monkeypatch.setattr(settings, "api_key", "key-b")
    assert get("key-a").status_code == 401
    assert get("key-b").status_code == 200
    # Probes are never limited.
    assert anon_client.get("/health").status_code == 200


def test_unauthenticated_requests_are_limited_per_ip(anon_client, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_min", 2)
    codes = [anon_client.get("/metadata", headers={"X-API-Key": f"guess-{i}"}).status_code
             for i in range(3)]
    assert codes == [401, 401, 429], "key guessing is rate limited"


def test_rate_limit_zero_disables(client, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_min", 0)
    assert all(client.get("/metadata").status_code == 200 for _ in range(70))


# --- query length cap ------------------------------------------------------

def test_query_length_cap_default_2000(client, mock_service):
    assert client.post("/api/v1/query", json={"query": "x" * 2000}).status_code == 200
    assert client.post("/api/v1/query", json={"query": "x" * 2001}).status_code == 422


def test_query_length_cap_is_configurable(client, mock_service, monkeypatch):
    monkeypatch.setattr(settings, "max_query_chars", 10)
    assert client.post("/api/v1/query", json={"query": "x" * 10}).status_code == 200
    response = client.post("/api/v1/query", json={"query": "x" * 11})
    assert response.status_code == 422
    mock_service.answer.assert_called_once()


# --- LLM failure: timeout + retries, then passages with answer=null --------

def _service_with_groq(monkeypatch, side_effect):
    created = {}

    def fake_groq(**kwargs):
        created.update(kwargs)
        client = Mock()
        client.chat.completions.create.side_effect = side_effect
        created["client"] = client
        return client

    monkeypatch.setattr("app.generation.groq_generator.Groq", fake_groq)
    retriever, reranker = Mock(), Mock()
    retriever.retrieve.return_value = [_passage(1), _passage(2), _passage(3)]
    reranker.rerank.return_value = [_passage(3), _passage(1)]
    generator = GroqGenerator(api_key="test-groq-key", timeout_seconds=7.5, max_retries=2)
    service = RAGService(retriever, reranker, generator, rerank_depth=20, rerank_top_k=2)
    monkeypatch.setattr(app.state, "rag_service", service)
    return created


def _timeout():
    return APITimeoutError(request=httpx.Request("POST", "https://api.groq.test"))


def test_groq_timeouts_retry_twice_then_return_passages(client, monkeypatch):
    labels = {"type": "groq_timeout"}
    before = REGISTRY.get_sample_value("rag_errors_total", labels)
    failures_before = REGISTRY.get_sample_value("rag_errors_total", {"type": "generation_failed"})
    created = _service_with_groq(monkeypatch, [_timeout(), _timeout(), _timeout()])
    with patch("app.generation.groq_generator.time.sleep") as sleep:
        response = client.post("/api/v1/query", json={"query": "q", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] is None
    assert body["error"] == "generation_failed"
    assert [c["passage_id"] for c in body["citations"]] == [3, 1]
    assert created["client"].chat.completions.create.call_count == 3  # 1 + 2 retries
    assert [c.args[0] for c in sleep.call_args_list] == [0.2, 0.4]  # exponential backoff
    assert created["timeout"] == 7.5  # per-attempt timeout reaches the client
    assert created["max_retries"] == 0  # SDK retries off; ours are bounded

    metrics = client.get("/metrics").text
    assert f'rag_errors_total{{type="generation_failed"}} {failures_before + 1}' in metrics
    assert f'rag_errors_total{{type="groq_timeout"}} {before + 3}' in metrics


def test_non_transient_groq_error_is_not_retried_and_degrades(client, monkeypatch, caplog):
    query = "private-query-123"
    passage_text = _passage(3)["passage_text"]
    bad_request = APIStatusError(
        f"provider exception with {query}, {passage_text}, and test-groq-key",
        response=httpx.Response(400, request=httpx.Request("POST", "https://api.groq.test")),
        body=None,
    )
    created = _service_with_groq(monkeypatch, [bad_request])
    with patch("app.generation.groq_generator.time.sleep") as sleep:
        response = client.post("/api/v1/query", json={"query": query})

    assert response.status_code == 200
    assert response.json()["answer"] is None
    assert created["client"].chat.completions.create.call_count == 1
    sleep.assert_not_called()
    logs = "\n".join(_JsonFormatter().format(record) for record in caplog.records)
    assert "generation_failed" in logs
    assert all(value not in logs for value in (query, passage_text, "test-groq-key", TEST_API_KEY))


def test_transient_failure_then_success_returns_answer(client, monkeypatch):
    ok = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="fine [pid:3]"))])
    _service_with_groq(monkeypatch, [_timeout(), ok])
    with patch("app.generation.groq_generator.time.sleep"):
        body = client.post("/api/v1/query", json={"query": "q"}).json()
    assert body["answer"] == "fine [pid:3]"
    assert body["error"] is None


# --- logs: request_id passthrough, key_id, never the raw key ----------------

def test_logs_have_request_id_and_key_id_but_never_the_raw_key(client, anon_client,
                                                                monkeypatch, caplog):
    ok = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="fine"))])
    _service_with_groq(monkeypatch, [ok])
    wrong_key = "wrong-secret-key-123"
    query = "private-query-123"
    formatter = _JsonFormatter()

    with caplog.at_level(logging.INFO):
        response = client.post("/api/v1/query", json={"query": query},
                               headers={"X-Request-ID": "req-abc-123"})
        rejected_response = anon_client.get(
            f"/{query}?query={query}",
            headers={"X-API-Key": wrong_key, "X-Request-ID": wrong_key},
        )
        probe = anon_client.get("/health", headers={"X-Request-ID": TEST_API_KEY})

    assert response.headers["X-Request-ID"] == "req-abc-123"
    assert rejected_response.headers["X-Request-ID"] != wrong_key
    assert probe.headers["X-Request-ID"] != TEST_API_KEY
    lines = [formatter.format(record) for record in caplog.records]
    output = "\n".join(lines)
    assert TEST_API_KEY not in output
    assert wrong_key not in output
    assert query not in output
    assert _passage(3)["passage_text"] not in output

    expected_key_id = hashlib.sha256(TEST_API_KEY.encode()).hexdigest()[:8]
    entries = [json.loads(line) for line in lines]
    request_logs = [e for e in entries if e["request_id"] == "req-abc-123"]
    assert {e["message"] for e in request_logs} >= {
        "retrieve_complete", "rerank_complete", "generate_complete", "request_complete"
    }
    assert all(e["key_id"] == expected_key_id for e in request_logs)
    rejected = [e for e in entries
                if e["message"] == "request_complete" and e.get("status_code") == 401]
    assert rejected and rejected[0]["key_id"] == "-"


# --- /metrics ---------------------------------------------------------------

def test_metrics_requires_key_and_exports_stage_histograms(client, anon_client, monkeypatch):
    ok = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="fine"))])
    _service_with_groq(monkeypatch, [ok])
    client.post("/api/v1/query", json={"query": "q"})

    assert anon_client.get("/metrics").status_code == 401
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    for stage in ("retrieve", "rerank", "generate"):
        assert f'rag_stage_latency_seconds_count{{stage="{stage}"}}' in text
        assert f'rag_stage_latency_seconds_bucket{{le="0.01",stage="{stage}"}}' in text
    for error_type in ("unauthorized", "rate_limited", "generation_failed"):
        assert f'rag_errors_total{{type="{error_type}"}}' in text
    assert "http_requests_total" in text
    assert "http_request_duration_seconds_count" in text


# --- removed settings warn at startup ---------------------------------------

def test_legacy_rate_limit_settings_warn(monkeypatch, tmp_path, caplog):
    from app.config import legacy_settings_present, warn_legacy_settings

    for name in ("RATE_LIMIT_REQUESTS", "RATE_LIMIT_WINDOW_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    no_file = tmp_path / "missing.env"
    assert legacy_settings_present(no_file) == []

    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "10")
    env_file = tmp_path / ".env"
    env_file.write_text("RATE_LIMIT_WINDOW_SECONDS=30\n", encoding="utf-8")
    assert legacy_settings_present(env_file) == ["RATE_LIMIT_REQUESTS",
                                                 "RATE_LIMIT_WINDOW_SECONDS"]

    with caplog.at_level(logging.WARNING, logger="app.config"):
        warn_legacy_settings(env_file)
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 2
    assert all("RATE_LIMIT_PER_MIN" in m for m in messages)


# --- per-IP limit uses the TCP peer, never X-Forwarded-For --------------------

def _guess(client, i, spoof):
    return client.get("/metadata", headers={"X-API-Key": f"guess-{i}",
                                            "X-Forwarded-For": spoof}).status_code


def test_spoofed_xff_does_not_reset_ip_limit(anon_client, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_min", 2)
    codes = [_guess(anon_client, i, f"10.0.0.{i}") for i in range(4)]
    assert codes == [401, 401, 429, 429]


def test_spoofed_xff_does_not_reset_ip_limit_under_uvicorn(monkeypatch):
    """End to end through uvicorn configured as we deploy it (no proxy headers)."""
    import socket
    import threading
    import time as _time

    import uvicorn

    monkeypatch.setattr(settings, "rate_limit_per_min", 2)
    monkeypatch.setattr(app.state, "get_rag_service", Mock(return_value=Mock()))
    monkeypatch.setattr(app.state, "draining", False)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="error", proxy_headers=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = _time.monotonic() + 30
    while not server.started and _time.monotonic() < deadline:
        _time.sleep(0.05)
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}") as client:
            codes = [client.get("/metadata", headers={"X-API-Key": f"guess-{i}",
                                                      "X-Forwarded-For": f"10.0.0.{i}"}
                                ).status_code for i in range(4)]
    finally:
        server.should_exit = True
        thread.join(10)
    assert codes == [401, 401, 429, 429]


def test_every_uvicorn_launch_disables_proxy_headers():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert '"--no-proxy-headers"' in (root / "Dockerfile").read_text(encoding="utf-8")
    assert "proxy_headers=False" in (root / "scripts/manual/serve_no_llm.py").read_text(
        encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert all("--no-proxy-headers" in line
               for line in readme.splitlines() if line.startswith("uvicorn app.main:app"))
