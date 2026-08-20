from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"

    passages_path: str = "data/passages_subset.parquet"
    dense_index_path: str = "indices/dense_index.faiss"
    bm25_index_path: str = "indices/bm25/bm25_index.pkl"
    bm25_corpus_path: str = "indices/bm25/tokenized_corpus.pkl"

    retrieval_k: int = 20
    hybrid_top_k: int = 10
    rerank_top_k: int = 5
    rrf_k: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
