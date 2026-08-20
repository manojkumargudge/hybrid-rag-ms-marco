from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Reranks retrieved passages using a cross-encoder model.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        passages: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Rerank candidate passages for a query.

        Args:
            query: User query.
            passages: Candidate passages from the hybrid retriever.
            top_k: Number of reranked passages to return.

        Returns:
            Passages sorted by cross-encoder relevance score.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not passages:
            return []

        top_k = min(top_k, len(passages))

        pairs = [
            [query, passage["passage_text"]]
            for passage in passages
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for passage, score in zip(passages, scores):
            reranked.append(
                {
                    "passage_id": int(passage["passage_id"]),
                    "passage_text": str(passage["passage_text"]),
                    "score": float(score),
                }
            )

        reranked.sort(
            key=lambda result: result["score"],
            reverse=True,
        )

        return reranked[:top_k]