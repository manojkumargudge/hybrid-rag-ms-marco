
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    application_version: str = "1.0.0"
    git_version: str = "unknown"

    passages_path: str = "data/passages_subset.parquet"
    dense_index_path: str = "indices/dense_index.faiss"
    bm25_index_path: str = "indices/bm25/bm25_index.pkl"
    bm25_corpus_path: str = "indices/bm25/tokenized_corpus.pkl"
    index_version: str = "v1"
    data_version: str = "v1"

    retrieval_k: int = 20
    hybrid_top_k: int = 10
    rerank_top_k: int = 5
    rrf_k: int = 60
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
