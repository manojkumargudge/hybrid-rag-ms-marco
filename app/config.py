
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    groq_api_key: str = ""
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
    groq_timeout_seconds: float = 30.0
    groq_max_retries: int = 2
    cors_origins: str = ""
    trusted_hosts: str = "localhost,127.0.0.1,testserver"
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
