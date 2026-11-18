from typing import Any


class HybridRetriever:
    """
    Hybrid retriever that combines Dense and BM25 rankings
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        dense_retriever: Any,
        bm25_retriever: Any,
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        retrieval_k: int = 10,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        if not 0.0 <= dense_weight <= 1.0:
            raise ValueError("dense_weight must be between 0 and 1.")

        if retrieval_k <= 0:
            raise ValueError("retrieval_k must be greater than zero.")

        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.retrieval_k = retrieval_k
        # Weighted RRF: 2*w / (k + rank_dense) + 2*(1 - w) / (k + rank_bm25).
        # w = 0.5 is exactly standard (unweighted) RRF.
        self.dense_scale = 2.0 * dense_weight
        self.bm25_scale = 2.0 * (1.0 - dense_weight)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        retrieval_k: int | None = None,
    ) -> list[dict]:
        """
        Retrieve documents using Dense + BM25 and fuse their rankings
        using Reciprocal Rank Fusion.

        Args:
            query: User search query.
            top_k: Number of final hybrid results to return.
            retrieval_k: Number of candidates retrieved from each
                individual retriever before fusion (default: the
                constructor's retrieval_k).

        Returns:
            Ranked list of hybrid retrieval results.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if retrieval_k is None:
            retrieval_k = self.retrieval_k

        if retrieval_k <= 0:
            raise ValueError("retrieval_k must be greater than zero.")

        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=retrieval_k,
        )

        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            top_k=retrieval_k,
        )

        # passage_id -> fused document information
        documents: dict[int, dict] = {}

        # passage_id -> RRF score
        rrf_scores: dict[int, float] = {}

        # Fuse Dense ranking.
        for rank, result in enumerate(dense_results, start=1):
            passage_id = int(result["passage_id"])

            rrf_score = self.dense_scale / (self.rrf_k + rank)

            rrf_scores[passage_id] = (
                rrf_scores.get(passage_id, 0.0) + rrf_score
            )

            if passage_id not in documents:
                documents[passage_id] = {
                    "passage_id": passage_id,
                    "passage_text": result["passage_text"],
                }

        # Fuse BM25 ranking.
        for rank, result in enumerate(bm25_results, start=1):
            passage_id = int(result["passage_id"])

            rrf_score = self.bm25_scale / (self.rrf_k + rank)

            rrf_scores[passage_id] = (
                rrf_scores.get(passage_id, 0.0) + rrf_score
            )

            if passage_id not in documents:
                documents[passage_id] = {
                    "passage_id": passage_id,
                    "passage_text": result["passage_text"],
                }

        # Sort documents by descending RRF score.
        ranked_passage_ids = sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True,
        )

        results = []

        for passage_id in ranked_passage_ids[:top_k]:
            document = documents[passage_id]

            results.append(
                {
                    "passage_id": passage_id,
                    "passage_text": document["passage_text"],
                    "score": rrf_scores[passage_id],
                }
            )

        return results