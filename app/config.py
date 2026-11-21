
import os
from logging import getLogger
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = getLogger(__name__)


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    groq_api_key: str = ""
    api_key: str = Field(default="", repr=False)
    groq_model: str = "openai/gpt-oss-120b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Pinned Hugging Face commit SHAs; the Docker image bakes exactly these.
    embedding_model_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    reranker_model_revision: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    application_version: str = "1.0.0"
    git_version: str = "unknown"

    # Built by scripts/prepare_data.py + scripts/build_indices.py.
    manifest_path: str = "artifacts/v3/manifest.json"
    passages_path: str = "artifacts/v3/passages.parquet"
    dense_index_path: str = "artifacts/v3/dense_index.faiss"
    bm25_index_path: str = "artifacts/v3/bm25/bm25_index.pkl"
    bm25_corpus_path: str = "artifacts/v3/bm25/tokenized_corpus.pkl"
    index_version: str = "v3"
    data_version: str = "v3"

    # Served default chosen on the tune split only
    # (eval/results/20261117-v3-tune-c858705): dense retrieval, cross-encoder
    # rerank of the top 20 candidates.
    retrieval_mode: Literal["dense", "hybrid"] = "dense"
    rerank_depth: int = Field(default=20, gt=0)
    rerank_top_k: int = Field(default=5, gt=0)
    # Hybrid mode only: candidates fetched from each retriever before RRF,
    # RRF constant, and dense weight (0.5 = standard RRF).
    retrieval_k: int = Field(default=20, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    hybrid_dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    # Per attempt. Timeouts, 429 and 5xx are retried GROQ_MAX_RETRIES times
    # with exponential backoff, all within GENERATION_DEADLINE_SECONDS; then
    # the API returns the passages with answer=null.
    groq_timeout_seconds: float = Field(default=10.0, gt=0)
    groq_max_retries: int = Field(default=2, ge=0)
    generation_deadline_seconds: float = Field(default=20.0, gt=0)
    cors_origins: str = ""
    trusted_hosts: str = "localhost,127.0.0.1,testserver"
    # Requests per minute per API key (per client IP for unauthenticated
    # requests); 0 disables. /health and /ready are never limited.
    rate_limit_per_min: int = Field(default=60, ge=0)
    max_query_chars: int = Field(default=2000, gt=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()

# Removed settings that are now silently ignored (extra="ignore"), mapped to
# their replacement.
LEGACY_SETTINGS = {
    "RATE_LIMIT_REQUESTS": "RATE_LIMIT_PER_MIN",
    "RATE_LIMIT_WINDOW_SECONDS": "RATE_LIMIT_PER_MIN",
}


def legacy_settings_present(env_file: str | Path = ".env") -> list[str]:
    """Names of removed settings found in the environment or the .env file."""
    names = {name.upper() for name in os.environ}
    if Path(env_file).is_file():
        names |= {name.upper() for name in dotenv_values(env_file)}
    return sorted(name for name in LEGACY_SETTINGS if name in names)


def warn_legacy_settings(env_file: str | Path = ".env") -> None:
    for name in legacy_settings_present(env_file):
        logger.warning(
            f"legacy_setting_ignored: {name} is no longer used; "
            f"set {LEGACY_SETTINGS[name]} (requests per minute per API key) instead"
        )
