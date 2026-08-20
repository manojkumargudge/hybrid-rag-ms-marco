from fastapi import FastAPI

from app.api.routes import router
from app.config import settings
from app.generation.groq_generator import GroqGenerator
from app.rag_service import RAGService
from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.hybrid_retriever import HybridRetriever


def create_rag_service() -> RAGService:
    """Initialize the complete Hybrid RAG pipeline."""

    dense_retriever = DenseRetriever(
        index_path=settings.dense_index_path,
        passages_path=settings.passages_path,
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

    reranker = CrossEncoderReranker()

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


app.state.rag_service = create_rag_service()

app.include_router(router)


@app.get("/health")
def health_check() -> dict:
    """Return service health status."""

    return {
        "status": "healthy",
        "service": "hybrid-rag",
    }
