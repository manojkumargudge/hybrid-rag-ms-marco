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


def test_query_endpoint():
    response = client.post(
        "/api/v1/query",
        json={
            "query": "what are the symptoms of diabetes?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "what are the symptoms of diabetes?"
    assert isinstance(data["answer"], str)
    assert data["answer"].strip()

    assert "sources" in data
    assert len(data["sources"]) == 5

    passage_ids = [
        source["passage_id"]
        for source in data["sources"]
    ]

    assert len(passage_ids) == len(set(passage_ids))


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

        assert response.status_code == 500
        assert response.json()["detail"] == "RAG pipeline failed."

    finally:
        app.state.rag_service = original_service
