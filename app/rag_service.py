from logging import getLogger
from time import perf_counter

from app.generation.groq_generator import GroqGenerator
from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.retrieval.hybrid_retriever import HybridRetriever

logger = getLogger(__name__)


class RAGService:
    """
    End-to-end Retrieval-Augmented Generation service.

    Pipeline:
        Dense + BM25
            ↓
        RRF Fusion
            ↓
        Cross-Encoder Reranking
            ↓
        Groq LLM Generation
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        generator: GroqGenerator,
        retrieval_k: int = 20,
        hybrid_top_k: int = 10,
        rerank_top_k: int = 5,
    ) -> None:
        if retrieval_k <= 0:
            raise ValueError("retrieval_k must be greater than zero.")

        if hybrid_top_k <= 0:
            raise ValueError("hybrid_top_k must be greater than zero.")

        if rerank_top_k <= 0:
            raise ValueError("rerank_top_k must be greater than zero.")

        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.generator = generator

        self.retrieval_k = retrieval_k
        self.hybrid_top_k = hybrid_top_k
        self.rerank_top_k = rerank_top_k

    def answer(self, query: str, top_k: int | None = None) -> dict:
        """
        Run the complete RAG pipeline.

        Returns:
            Dictionary containing the final answer and
            retrieved source passages.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        final_top_k = self.rerank_top_k if top_k is None else top_k
        if final_top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        started_at = perf_counter()

        retrieval_started_at = perf_counter()
        hybrid_results = self.hybrid_retriever.retrieve(
            query=query,
            top_k=self.hybrid_top_k,
            retrieval_k=self.retrieval_k,
        )
        logger.info(
            "retrieval_complete",
            extra={
                "stage": "retrieval",
                "stage_ms": _elapsed_ms(retrieval_started_at),
                "result_count": len(hybrid_results),
            },
        )

        if not hybrid_results:
            return {
                "answer": (
                    "I could not find relevant information "
                    "in the retrieved documents."
                ),
                "citations": [],
                "latency_ms": _elapsed_ms(started_at),
            }

        rerank_started_at = perf_counter()
        reranked_results = self.reranker.rerank(
            query=query,
            passages=hybrid_results,
            top_k=final_top_k,
        )
        logger.info(
            "reranking_complete",
            extra={
                "stage": "reranking",
                "stage_ms": _elapsed_ms(rerank_started_at),
                "result_count": len(reranked_results),
            },
        )

        if not reranked_results:
            return {
                "answer": (
                    "I could not find relevant information "
                    "in the retrieved documents."
                ),
                "citations": [],
                "latency_ms": _elapsed_ms(started_at),
            }

        generation_started_at = perf_counter()
        answer = self.generator.generate(
            query=query,
            passages=reranked_results,
        )
        logger.info(
            "generation_complete",
            extra={
                "stage": "generation",
                "stage_ms": _elapsed_ms(generation_started_at),
            },
        )

        return {
            "answer": answer,
            "citations": reranked_results,
            "latency_ms": _elapsed_ms(started_at),
        }


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)