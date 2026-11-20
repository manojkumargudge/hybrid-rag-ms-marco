"""API-key helpers (X-API-Key header).

Keys come from API_KEYS (comma-separated). Enforcement happens for every
path except an explicit allowlist in app.middleware.ProductionMiddleware.
Authentication is on unless AUTH_DISABLED=true. With auth on and no keys
configured, every protected request is rejected (fail closed).
"""

from hmac import compare_digest
from logging import getLogger

from app.config import settings

API_KEY_HEADER = "X-API-Key"
logger = getLogger(__name__)


def configured_keys() -> list[bytes]:
    return [key.strip().encode() for key in settings.api_keys.split(",") if key.strip()]


def is_valid_key(provided: str | None, keys: list[bytes]) -> bool:
    """Constant-time check against every configured key (no early exit)."""
    if not provided:
        return False
    candidate = provided.encode()
    matched = False
    for key in keys:
        matched |= compare_digest(candidate, key)
    return matched


def log_auth_configuration() -> None:
    if settings.auth_disabled:
        logger.warning("auth_disabled: AUTH_DISABLED=true, API keys are not checked")
    elif not configured_keys():
        logger.warning("auth_no_keys: API_KEYS is empty, all protected requests get 401")
