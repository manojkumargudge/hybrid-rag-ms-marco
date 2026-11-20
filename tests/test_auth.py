"""X-API-Key authentication."""

import logging
from unittest.mock import Mock

import pytest

from app.auth import is_valid_key, log_auth_configuration
from app.config import settings
from app.main import app

PROTECTED = [("post", "/api/v1/query", {"query": "test query"}), ("get", "/metadata", None)]


@pytest.fixture
def mock_service(monkeypatch):
    service = Mock()
    service.answer.return_value = {"answer": "ok", "citations": [], "latency_ms": 1.0}
    monkeypatch.setattr(app.state, "rag_service", service)
    return service


def _call(client, method, path, body, headers=None):
    if method == "post":
        return client.post(path, json=body, headers=headers or {})
    return client.get(path, headers=headers or {})


@pytest.mark.parametrize(("method", "path", "body"), PROTECTED)
def test_missing_key_is_401(anon_client, mock_service, method, path, body):
    response = _call(anon_client, method, path, body)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "ApiKey"
    mock_service.answer.assert_not_called()


@pytest.mark.parametrize(("method", "path", "body"), PROTECTED)
def test_wrong_key_is_401(anon_client, mock_service, method, path, body):
    response = _call(anon_client, method, path, body, {"X-API-Key": "wrong"})
    assert response.status_code == 401


@pytest.mark.parametrize(("method", "path", "body"), PROTECTED)
def test_valid_key_is_accepted(client, mock_service, method, path, body):
    assert _call(client, method, path, body).status_code == 200


def test_any_of_several_comma_separated_keys_works(anon_client, mock_service, monkeypatch):
    monkeypatch.setattr(settings, "api_keys", " key-a , key-b ,, ")
    for key in ("key-a", "key-b"):
        response = anon_client.post("/api/v1/query", json={"query": "q"},
                                    headers={"X-API-Key": key})
        assert response.status_code == 200, key
    response = anon_client.post("/api/v1/query", json={"query": "q"},
                                headers={"X-API-Key": "key-a,key-b"})
    assert response.status_code == 401


def test_health_and_ready_are_open(anon_client):
    assert anon_client.get("/health").status_code == 200
    assert anon_client.get("/ready").status_code in {200, 503}


def test_no_keys_configured_fails_closed(anon_client, mock_service, monkeypatch):
    monkeypatch.setattr(settings, "api_keys", "")
    for header in ({}, {"X-API-Key": ""}, {"X-API-Key": "anything"}):
        response = anon_client.post("/api/v1/query", json={"query": "q"}, headers=header)
        assert response.status_code == 401
    assert anon_client.get("/ready").status_code == 503


def test_auth_disabled_skips_check_and_warns(anon_client, mock_service, monkeypatch, caplog):
    monkeypatch.setattr(settings, "auth_disabled", True)
    monkeypatch.setattr(settings, "api_keys", "")

    assert anon_client.post("/api/v1/query", json={"query": "q"}).status_code == 200
    with caplog.at_level(logging.WARNING, logger="app.auth"):
        log_auth_configuration()
    assert any("auth_disabled" in record.getMessage() for record in caplog.records)


def test_auth_is_on_by_default():
    from app.config import Settings

    assert Settings.model_fields["auth_disabled"].default is False
    assert Settings.model_fields["api_keys"].default == ""


def test_key_comparison():
    keys = [b"alpha", b"beta"]
    assert is_valid_key("beta", keys)
    assert not is_valid_key("bet", keys)
    assert not is_valid_key("", keys)
    assert not is_valid_key(None, keys)
    assert not is_valid_key("café", keys)  # non-ASCII must not raise
