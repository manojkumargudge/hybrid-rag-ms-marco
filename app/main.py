
from fastapi import FastAPI

from app.api.routes import router
from app.generation.groq_generator import GroqGenerator
from app.rag_service import RAGService
from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.hybrid_retriever import HybridRetriever


PASSAGES_PATH = "data/passages_subset.parquet"

DENSE_INDEX_PATH = "indices/dense_index.faiss"

BM25_INDEX_PATH = "indices/bm25/bm25_index.pkl"

BM25_CORPUS_PATH = "indices/bm25/tokenized_corpus.pkl"


def create_rag_service() -> RAGService:
    """Initialize the complete Hybrid RAG pipeline."""

    dense_retriever = DenseRetriever(
        index_path=DENSE_INDEX_PATH,
        passages_path=PASSAGES_PATH,
    )

    bm25_retriever = BM25Retriever(
        index_path=BM25_INDEX_PATH,
        corpus_path=BM25_CORPUS_PATH,
        passages_path=PASSAGES_PATH,
    )

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        rrf_k=60,
    )

    reranker = CrossEncoderReranker()

    generator = GroqGenerator()

    return RAGService(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        generator=generator,
        retrieval_k=20,
        hybrid_top_k=10,
        rerank_top_k=5,
    )


app = FastAPI(
    title="Hybrid RAG over MS MARCO",
    description=(
        "Hybrid retrieval-augmented generation system using "
        "Dense Retrieval, BM25, RRF, Cross-Encoder Reranking, "
        "and Groq LLM generation."
    ),
    version="1.0.0",
)


rag_service = create_rag_service()


app.state.rag_service = rag_service

app.include_router(router)


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "healthy",
        "service": "hybrid-rag",
    }