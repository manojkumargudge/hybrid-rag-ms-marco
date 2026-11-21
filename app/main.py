import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.routes import router
from app.artifacts import verify_artifacts
from app.auth import API_KEY_HEADER, configured_key, log_auth_configuration
from app.config import settings, warn_legacy_settings
from app.generation.groq_generator import GroqGenerator
from app.metrics import render as render_metrics
from app.middleware import ProductionMiddleware, key_id_context, request_id_context
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
            # Captured when the record is created (see _configure_logging), so
            # they stay correct even if formatting happens later.
            "request_id": getattr(record, "request_id", "-"),
            # key_id = sha256(API key)[:8]; the raw key is never logged.
            "key_id": getattr(record, "key_id", "-"),
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


def _install_context_record_factory() -> None:
    """Stamp request_id and key_id onto every log record at creation time."""
    base_factory = logging.getLogRecordFactory()
    if getattr(base_factory, "_adds_request_context", False):
        return

    def factory(*args, **kwargs):
        record = base_factory(*args, **kwargs)
        record.request_id = request_id_context.get()
        record.key_id = key_id_context.get()
        return record

    factory._adds_request_context = True
    logging.setLogRecordFactory(factory)


def _configure_logging() -> None:
    _install_context_record_factory()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        root_logger.addHandler(handler)


_configure_logging()
log_auth_configuration()
warn_legacy_settings()


def create_rag_service() -> RAGService:
    """Initialize the RAG pipeline for the configured retrieval mode."""

    # Served corpus must be exactly the evaluated one (BM25 pickles are
    # verified inside BM25Retriever before unpickling).
    verify_artifacts(
        settings.manifest_path,
        settings.passages_path,
        settings.dense_index_path,
    )

    retriever = DenseRetriever(
        index_path=settings.dense_index_path,
        passages_path=settings.passages_path,
        model_name=settings.embedding_model,
        model_revision=settings.embedding_model_revision,
    )

    if settings.retrieval_mode == "hybrid":
        bm25_retriever = BM25Retriever(
            index_path=settings.bm25_index_path,
            corpus_path=settings.bm25_corpus_path,
            passages_path=settings.passages_path,
            manifest_path=settings.manifest_path,
        )
        retriever = HybridRetriever(
            dense_retriever=retriever,
            bm25_retriever=bm25_retriever,
            rrf_k=settings.rrf_k,
            dense_weight=settings.hybrid_dense_weight,
            retrieval_k=settings.retrieval_k,
        )

    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model,
        model_revision=settings.reranker_model_revision,
    )

    generator = GroqGenerator(
        model_name=settings.groq_model,
    )

    return RAGService(
        retriever=retriever,
        reranker=reranker,
        generator=generator,
        rerank_depth=settings.rerank_depth,
        rerank_top_k=settings.rerank_top_k,
    )


app = FastAPI(
    title="Hybrid RAG over MS MARCO",
    description=(
        "Retrieval-augmented generation over an MS MARCO passage subset: "
        "dense retrieval (or hybrid dense + BM25 with RRF), cross-encoder "
        "reranking, and Groq-hosted LLM generation."
    ),
    version="1.0.0",
    # Only /docs and /openapi.json are on the open allowlist; ReDoc and the
    # OAuth2 redirect page are not needed.
    redoc_url=None,
    swagger_ui_oauth2_redirect_url=None,
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
    allow_headers=["Content-Type", "X-Request-ID", API_KEY_HEADER],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        host.strip()
        for host in settings.trusted_hosts.split(",")
        if host.strip()
    ],
)
# Outermost: request id, auth for every non-allowlisted path, per-key rate
# limit, access log (see app/middleware.py).
app.add_middleware(ProductionMiddleware)


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


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Prometheus metrics (requires an API key like every non-allowlisted path)."""

    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


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
        "retrieval_mode": settings.retrieval_mode,
        "rerank_depth": settings.rerank_depth,
    }


@app.get("/ready", response_class=JSONResponse)
def readiness_check() -> JSONResponse:
    """Report whether configured RAG artifacts and generation are available."""

    required_paths = {
        "manifest": settings.manifest_path,
        "passages": settings.passages_path,
        "dense_index": settings.dense_index_path,
    }
    if settings.retrieval_mode == "hybrid":
        required_paths["bm25_index"] = settings.bm25_index_path
        required_paths["bm25_corpus"] = settings.bm25_corpus_path
    missing = [
        name
        for name, path in required_paths.items()
        if not Path(path).is_file()
    ]
    if not settings.groq_api_key:
        missing.append("groq_api_key")
    if not configured_key():
        missing.append("api_key")

    if missing:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "service": "hybrid-rag"},
        )

    return JSONResponse(
        status_code=200,
        content={"status": "ready", "service": "hybrid-rag"},
    )
