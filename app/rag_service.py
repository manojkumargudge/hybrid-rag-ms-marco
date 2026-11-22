from logging import getLogger
from time import perf_counter
from typing import Protocol

from app.generation.groq_generator import GroqCircuitOpen, GroqGenerator
from app.metrics import ERRORS, STAGE_LATENCY
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

        stage_started_at = perf_counter()
        candidates = self.retriever.retrieve(
            query=query,
            top_k=self.rerank_depth,
        )
        _record_stage("retrieve", stage_started_at, len(candidates))

        if not candidates:
            return {
                "answer": (
                    "I could not find relevant information "
                    "in the retrieved documents."
                ),
                "citations": [],
                "latency_ms": _elapsed_ms(started_at),
            }

        stage_started_at = perf_counter()
        reranked_results = self.reranker.rerank(
            query=query,
            passages=candidates,
            top_k=final_top_k,
        )
        _record_stage("rerank", stage_started_at, len(reranked_results))

        if not reranked_results:
            return {
                "answer": (
                    "I could not find relevant information "
                    "in the retrieved documents."
                ),
                "citations": [],
                "latency_ms": _elapsed_ms(started_at),
            }

        stage_started_at = perf_counter()
        try:
            answer = self.generator.generate(
                query=query,
                passages=reranked_results,
            )
        except RuntimeError as exc:
            _record_stage("generate", stage_started_at)
            error = (
                "generation_circuit_open" if isinstance(exc, GroqCircuitOpen) else "generation_failed"
            )
            ERRORS.labels(type=error).inc()
            logger.warning(
                error,
                extra={"error_type": type(exc.__cause__ or exc).__name__},
            )
            return {
                "answer": None,
                "error": error,
                "citations": reranked_results,
                "latency_ms": _elapsed_ms(started_at),
            }
        _record_stage("generate", stage_started_at)

        return {
            "answer": answer,
            "citations": reranked_results,
            "latency_ms": _elapsed_ms(started_at),
        }


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)


def _record_stage(stage: str, started_at: float, result_count: int | None = None) -> None:
    elapsed = perf_counter() - started_at
    STAGE_LATENCY.labels(stage=stage).observe(elapsed)
    extra = {"stage": stage, "stage_ms": round(elapsed * 1000, 2)}
    if result_count is not None:
        extra["result_count"] = result_count
    logger.info(f"{stage}_complete", extra=extra)
