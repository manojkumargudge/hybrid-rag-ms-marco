import json

from app.main import app
from scripts.export_openapi import SNAPSHOT


def test_openapi_matches_committed_snapshot(client):
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json() == app.openapi() == expected
