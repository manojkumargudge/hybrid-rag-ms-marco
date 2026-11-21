"""API-key helpers for the X-API-Key header."""

from hmac import compare_digest
from logging import getLogger

from app.config import settings

API_KEY_HEADER = "X-API-Key"
logger = getLogger(__name__)


def configured_key() -> bytes:
    return settings.api_key.strip().encode()


def is_valid_key(provided: str | None, key: bytes) -> bool:
    return bool(provided and key and compare_digest(provided.encode(), key))


def log_auth_configuration() -> None:
    if not configured_key():
        logger.warning("auth_no_key: API_KEY is empty, protected requests require a key")
