from app.generation.groq_generator import GroqGenerator
from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.retrieval.hybrid_retriever import HybridRetriever


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

    def answer(self, query: str) -> dict:
        """
        Run the complete RAG pipeline.

        Returns:
            Dictionary containing the final answer and
            retrieved source passages.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        # Step 1: Hybrid retrieval + RRF
        hybrid_results = self.hybrid_retriever.retrieve(
            query=query,
            top_k=self.hybrid_top_k,
            retrieval_k=self.retrieval_k,
        )

        if not hybrid_results:
            return {
                "query": query,
                "answer": (
                    "I could not find relevant information "
                    "in the retrieved documents."
                ),
                "sources": [],
            }

        # Step 2: Cross-encoder reranking
        reranked_results = self.reranker.rerank(
            query=query,
            passages=hybrid_results,
            top_k=self.rerank_top_k,
        )

        if not reranked_results:
            return {
                "query": query,
                "answer": (
                    "I could not find relevant information "
                    "in the retrieved documents."
                ),
                "sources": [],
            }

        # Step 3: LLM generation
        answer = self.generator.generate(
            query=query,
            passages=reranked_results,
        )

        return {
            "query": query,
            "answer": answer,
            "sources": reranked_results,
        }