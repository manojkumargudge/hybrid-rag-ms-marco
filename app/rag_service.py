from logging import getLogger
from time import perf_counter
from typing import Protocol

from app.generation.groq_generator import GroqGenerator
from app.reranking.cross_encoder_reranker import CrossEncoderReranker

logger = getLogger(__name__)


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[dict]: ...


class RAGService:
    """
    End-to-end Retrieval-Augmented Generation service.

    Pipeline:
        Retriever (dense, or hybrid dense + BM25 with RRF) -> top rerank_depth
            ↓
        Cross-Encoder Reranking -> top rerank_top_k (or the request's top_k)
            ↓
        Groq LLM Generation
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: CrossEncoderReranker,
        generator: GroqGenerator,
        rerank_depth: int = 20,
        rerank_top_k: int = 5,
    ) -> None:
        if rerank_depth <= 0:
            raise ValueError("rerank_depth must be greater than zero.")

        if rerank_top_k <= 0:
            raise ValueError("rerank_top_k must be greater than zero.")

        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

        self.rerank_depth = rerank_depth
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
        candidates = self.retriever.retrieve(
            query=query,
            top_k=self.rerank_depth,
        )
        logger.info(
            "retrieval_complete",
            extra={
                "stage": "retrieval",
                "stage_ms": _elapsed_ms(retrieval_started_at),
                "result_count": len(candidates),
            },
        )

        if not candidates:
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
            passages=candidates,
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