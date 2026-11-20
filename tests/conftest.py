from itertools import count

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

TEST_API_KEY = "test-key"
_client_ids = count()


@pytest.fixture(autouse=True)
def _auth_settings(monkeypatch):
    """Every test runs with auth on and one known key (independent of .env)."""
    monkeypatch.setattr(settings, "api_keys", TEST_API_KEY)
    monkeypatch.setattr(settings, "auth_disabled", False)


def _new_client(**headers) -> TestClient:
    # The rate limiter keys on client host, so a fresh address isolates each
    # test from requests made by earlier tests (no order dependence).
    return TestClient(
        app, client=(f"test-client-{next(_client_ids)}", 50000), headers=headers
    )


@pytest.fixture
def client() -> TestClient:
    """Authenticated client (sends a valid X-API-Key)."""
    return _new_client(**{"X-API-Key": TEST_API_KEY})


@pytest.fixture
def anon_client() -> TestClient:
    """Client that sends no X-API-Key."""
    return _new_client()
