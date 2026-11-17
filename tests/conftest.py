from itertools import count

import pytest
from fastapi.testclient import TestClient

from app.main import app

_client_ids = count()


@pytest.fixture
def client() -> TestClient:
    """A TestClient with a unique client address per test.

    The rate limiter keys on client host, so a fresh address isolates each
    test from requests made by earlier tests (no order dependence).
    """
    return TestClient(app, client=(f"test-client-{next(_client_ids)}", 50000))
