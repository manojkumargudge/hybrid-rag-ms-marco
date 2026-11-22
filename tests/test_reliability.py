import asyncio
import json
import socket
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import httpx
import pytest
import uvicorn
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from app.api.routes import query_rag
from app.config import settings
from app.main import app, lifespan
from app.schemas.query import QueryRequest
from tests.conftest import TEST_API_KEY


def test_cancelled_http_task_keeps_slot_and_defers_groq_close(monkeypatch):
    entered, release = Event(), Event()
    service = Mock()
    monkeypatch.setattr(settings, "max_concurrent_queries", 1)
    monkeypatch.setattr(app.state, "draining", False)
    monkeypatch.setattr(app.state, "get_rag_service", Mock(return_value=service))

    def answer(**_):
        entered.set()
        assert release.wait(10)
        return {"answer": "ok", "citations": [], "latency_ms": 0}

    service.answer.side_effect = answer

    async def run():
        request = Request({"type": "http", "app": app})
        payload = QueryRequest(query="question")
        lifecycle = lifespan(app)
        await lifecycle.__aenter__()
        try:
            active = asyncio.create_task(query_rag(request, payload))
            assert await asyncio.to_thread(entered.wait, 5)
            active.cancel()
            with pytest.raises(asyncio.CancelledError):
                await active
            with pytest.raises(HTTPException) as rejected:
                await query_rag(request, payload)
            assert rejected.value.status_code == 429
        finally:
            stopping = asyncio.create_task(lifecycle.__aexit__(None, None, None))
            try:
                await asyncio.sleep(0)
                assert app.state.draining
                service.generator.close.assert_not_called()
            finally:
                release.set()
                await stopping
        service.generator.close.assert_called_once_with()
        assert not app.state.query_tasks

    asyncio.run(run())


def test_uvicorn_finishes_in_flight_query_before_closing_groq(monkeypatch):
    monkeypatch.setattr(app.state, "draining", False)
    service = Mock()
    monkeypatch.setattr(app.state, "get_rag_service", Mock(return_value=service))

    async def run():
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        release = Event()

        def answer(**_):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(10)
            return {"answer": "ok", "citations": [], "latency_ms": 0}

        service.answer.side_effect = answer
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", proxy_headers=False))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            host, port = sock.getsockname()
            serving = asyncio.create_task(server.serve(sockets=[sock]))
            try:
                async with asyncio.timeout(10):
                    while not server.started:
                        await asyncio.sleep(0.01)
                async with httpx.AsyncClient(
                    base_url=f"http://{host}:{port}", headers={"X-API-Key": TEST_API_KEY},
                    trust_env=False, timeout=10,
                ) as client:
                    request = asyncio.create_task(client.post("/api/v1/query", json={"query": "q"}))
                    try:
                        await asyncio.wait_for(entered.wait(), timeout=5)
                        server.should_exit = True
                        async with asyncio.timeout(5):
                            while server.servers[0].is_serving():
                                await asyncio.sleep(0.01)
                        service.generator.close.assert_not_called()
                    finally:
                        release.set()
                        response = await request
                    assert response.status_code == 200
            finally:
                server.should_exit = True
                await asyncio.wait_for(serving, timeout=10)
        service.generator.close.assert_called_once_with()

    asyncio.run(run())


def test_query_admission_rejects_excess_work_and_reuses_slot(monkeypatch):
    monkeypatch.setattr(settings, "max_concurrent_queries", 1)
    monkeypatch.setattr(app.state, "draining", False)
    entered, release = Event(), Event()
    service = Mock()

    def answer(**_):
        entered.set()
        assert release.wait(10)
        return {"answer": "ok", "citations": [], "latency_ms": 0}

    service.answer.side_effect = answer
    monkeypatch.setattr(app.state, "get_rag_service", Mock(return_value=service))
    payload = {"query": "question"}
    with TestClient(app, headers={"X-API-Key": TEST_API_KEY}) as client:
        with ThreadPoolExecutor(max_workers=1) as pool:
            active = pool.submit(client.post, "/api/v1/query", json=payload)
            try:
                assert entered.wait(10)
                response = client.post("/api/v1/query", json=payload)
                assert response.status_code == 429
                assert response.headers["Retry-After"] == "1"
                assert client.get("/health").status_code == 200
                assert service.answer.call_count == 1
            finally:
                release.set()
            assert active.result(timeout=10).status_code == 200
        assert client.post("/api/v1/query", json=payload).status_code == 200
        app.state.draining = True
        assert client.post("/api/v1/query", json=payload).status_code == 503
        assert client.get("/ready").status_code == 503
        assert service.answer.call_count == 2


@pytest.mark.parametrize("body", [
    b"",
    b"{",
    b'{"query":"x","top_k":NaN}',
    b'{"query":"x","top_k":Infinity}',
    b'{"query":"\\ud800"}',
    b'{"query":"\xff"}',
    b"[" * 1500 + b"]" * 1500,
    *[json.dumps(value).encode() for value in (
        None, True, 42, [], {}, {"query": []}, {"query": {"nested": True}},
        {"query": "x" * 2001}, {"query": "x", "top_k": True},
        {"query": "x", "top_k": 0}, {"query": "x", "top_k": 21},
    )],
])
def test_malformed_query_inputs_return_client_errors(client, body):
    response = client.post(
        "/api/v1/query", content=body, headers={"Content-Type": "application/json"},
    )
    assert response.status_code in {400, 422}
