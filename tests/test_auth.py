"""X-API-Key authentication."""

import logging
from unittest.mock import Mock

import pytest

from app.auth import is_valid_key, log_auth_configuration
from app.config import settings, warn_legacy_settings
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


def test_health_and_ready_are_open(anon_client):
    assert anon_client.get("/health").status_code == 200
    assert anon_client.get("/ready").status_code in {200, 503}


def test_no_key_configured_fails_closed(anon_client, mock_service, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "")
    for header in ({}, {"X-API-Key": ""}, {"X-API-Key": "anything"}):
        response = anon_client.post("/api/v1/query", json={"query": "q"}, headers=header)
        assert response.status_code == 401
    assert anon_client.get("/ready").status_code == 503


def test_api_key_environment(monkeypatch, caplog):
    from app.config import Settings

    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("API_KEYS", "legacy")
    monkeypatch.setenv("AUTH_DISABLED", "true")
    assert Settings(_env_file=None).api_key == ""

    monkeypatch.setenv("API_KEY", "configured")
    configured = Settings(_env_file=None)
    assert configured.api_key == "configured"
    assert "configured" not in repr(configured)
    warn_legacy_settings()
    assert "legacy_setting_ignored: AUTH_DISABLED" in caplog.text
    assert "legacy_setting_ignored: API_KEYS" in caplog.text


def test_unconfigured_key_logs_warning(monkeypatch, caplog):
    monkeypatch.setattr(settings, "api_key", "")
    with caplog.at_level(logging.WARNING, logger="app.auth"):
        log_auth_configuration()
    assert any("auth_no_key" in record.getMessage() for record in caplog.records)


def test_non_preflight_options_requires_key(anon_client):
    assert anon_client.options("/api/v1/query").status_code == 401


def test_key_comparison():
    key = b"beta"
    assert is_valid_key("beta", key)
    assert not is_valid_key("bet", key)
    assert not is_valid_key("", key)
    assert not is_valid_key(None, key)
    assert not is_valid_key("caf\u00e9", key)
    assert not is_valid_key("beta", b"")
