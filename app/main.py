import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.routes import router
from app.config import settings
from app.generation.groq_generator import GroqGenerator
from app.middleware import ProductionMiddleware, request_id_context
from app.rag_service import RAGService
from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.hybrid_retriever import HybridRetriever


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", request_id_context.get()),
        }
        for field in (
            "method",
            "path",
            "status_code",
            "latency_ms",
            "stage",
            "stage_ms",
            "result_count",
            "error_type",
        ):
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        return json.dumps(payload, separators=(",", ":"))


def _configure_logging() -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        root_logger.addHandler(handler)


_configure_logging()


def create_rag_service() -> RAGService:
    """Initialize the complete Hybrid RAG pipeline."""

    dense_retriever = DenseRetriever(
        index_path=settings.dense_index_path,
        passages_path=settings.passages_path,
        model_name=settings.embedding_model,
    )

    bm25_retriever = BM25Retriever(
        index_path=settings.bm25_index_path,
        corpus_path=settings.bm25_corpus_path,
        passages_path=settings.passages_path,
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        rrf_k=settings.rrf_k,
    )

    reranker = CrossEncoderReranker(model_name=settings.reranker_model)

    generator = GroqGenerator(
        model_name=settings.groq_model,
    )

    return RAGService(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        generator=generator,
        retrieval_k=settings.retrieval_k,
        hybrid_top_k=settings.hybrid_top_k,
        rerank_top_k=settings.rerank_top_k,
    )


app = FastAPI(
    title="Hybrid RAG over MS MARCO",
    description=(
        "Production-oriented hybrid retrieval-augmented generation "
        "system using Dense Retrieval, BM25, RRF, Cross-Encoder "
        "Reranking, and Groq-hosted LLM generation."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in settings.cors_origins.split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Request-ID"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        host.strip()
        for host in settings.trusted_hosts.split(",")
        if host.strip()
    ],
)
app.add_middleware(
    ProductionMiddleware,
    max_requests=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
)


def get_rag_service() -> RAGService:
    if app.state.rag_service is None:
        app.state.rag_service = create_rag_service()
    return app.state.rag_service


app.state.rag_service = None
app.state.get_rag_service = get_rag_service

app.include_router(router)


@app.get("/health")
def health_check() -> dict:
    """Return service health status."""

    return {
        "status": "healthy",
        "service": "hybrid-rag",
    }


@app.get("/metadata")
def metadata() -> dict:
    """Return non-sensitive model, artifact, and application versions."""

    return {
        "service": "hybrid-rag",
        "application_version": settings.application_version,
        "git_version": settings.git_version,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "index_version": settings.index_version,
        "data_version": settings.data_version,
    }


@app.get("/ready", response_class=JSONResponse)
def readiness_check() -> JSONResponse:
    """Report whether configured RAG artifacts and generation are available."""

    required_paths = {
        "passages": settings.passages_path,
        "dense_index": settings.dense_index_path,
        "bm25_index": settings.bm25_index_path,
        "bm25_corpus": settings.bm25_corpus_path,
    }
    missing = [
        name
        for name, path in required_paths.items()
        if not Path(path).is_file()
    ]
    if not settings.groq_api_key:
        missing.append("groq_api_key")

    if missing:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "service": "hybrid-rag"},
        )

    return JSONResponse(
        status_code=200,
        content={"status": "ready", "service": "hybrid-rag"},
    )
