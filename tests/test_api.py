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
    