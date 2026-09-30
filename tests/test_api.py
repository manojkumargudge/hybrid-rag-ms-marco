from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "hybrid-rag"


def test_request_id_is_returned_and_metadata_is_non_sensitive():
    response = client.get(
        "/metadata",
        headers={"X-Request-ID": "test-request-123"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request-123"
    data = response.json()
    assert data["embedding_model"]
    assert data["reranker_model"]
    assert data["index_version"]
    assert "GROQ_API_KEY" not in response.text


def test_trusted_host_rejects_unknown_host():
    response = client.get("/health", headers={"host": "malicious.example"})

    assert response.status_code == 400


def test_query_endpoint():
    original_service = app.state.rag_service
    mock_service = Mock()
    mock_service.answer.return_value = {
        "answer": "Diabetes can cause increased thirst and urination.",
        "citations": [
            {
                "passage_id": 1,
                "passage_text": "Diabetes symptoms include thirst.",
                "score": 0.9,
            }
        ],
        "latency_ms": 12.3,
    }
    app.state.rag_service = mock_service

    response = client.post(
        "/api/v1/query",
        json={
            "query": "what are the symptoms of diabetes?",
            "top_k": 1,
        },
    )

    try:
        assert response.status_code == 200
        data = response.json()
        assert data["answer"]
        assert len(data["citations"]) == 1
        assert data["latency_ms"] == 12.3
        mock_service.answer.assert_called_once_with(
            query="what are the symptoms of diabetes?",
            top_k=1,
        )
    finally:
        app.state.rag_service = original_service


def test_empty_query():
    response = client.post(
        "/api/v1/query",
        json={
            "query": ""
        },
    )

    assert response.status_code == 422


def test_missing_query():
    response = client.post(
        "/api/v1/query",
        json={},
    )

    assert response.status_code == 422


def test_invalid_query_type():
    response = client.post(
        "/api/v1/query",
        json={
            "query": 123
        },
    )

    assert response.status_code == 422


def test_invalid_top_k():
    response = client.post(
        "/api/v1/query",
        json={"query": "test query", "top_k": 0},
    )

    assert response.status_code == 422


def test_query_rejects_long_query():
    response = client.post(
        "/api/v1/query",
        json={"query": "x" * 2001},
    )

    assert response.status_code == 422


def test_query_rejects_string_top_k():
    response = client.post(
        "/api/v1/query",
        json={"query": "test query", "top_k": "5"},
    )

    assert response.status_code == 422


def test_readiness_endpoint_does_not_expose_configuration():
    response = client.get("/ready")

    assert response.status_code in {200, 503}
    assert "GROQ_API_KEY" not in response.text
    assert "indices/" not in response.text


def test_query_service_unavailable():
    original_getter = app.state.get_rag_service
    app.state.get_rag_service = Mock(side_effect=FileNotFoundError("missing index"))

    try:
        response = client.post("/api/v1/query", json={"query": "test query"})
        assert response.status_code == 503
        assert response.json()["detail"] == "RAG service is not ready."
    finally:
        app.state.get_rag_service = original_getter


def test_query_service_value_error():
    original_service = app.state.rag_service

    mock_service = Mock()
    mock_service.answer.side_effect = ValueError(
        "Query cannot be empty."
    )

    app.state.rag_service = mock_service

    try:
        response = client.post(
            "/api/v1/query",
            json={
                "query": "test query"
            },
        )

        assert response.status_code == 400
        assert response.json()["detail"] == "Query cannot be empty."

    finally:
        app.state.rag_service = original_service


def test_query_service_internal_error():
    original_service = app.state.rag_service

    mock_service = Mock()
    mock_service.answer.side_effect = RuntimeError(
        "Unexpected pipeline failure."
    )

    app.state.rag_service = mock_service

    try:
        response = client.post(
            "/api/v1/query",
            json={
                "query": "test query"
            },
        )

        assert response.status_code == 502
        assert response.json()["detail"] == "Answer generation failed."

    finally:
        app.state.rag_service = original_service


def test_rate_limit_returns_429_after_configured_limit():
    responses = [client.get("/metadata") for _ in range(61)]

    assert responses[-1].status_code == 429
    assert responses[-1].headers["Retry-After"] == "60"
