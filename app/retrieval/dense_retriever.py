from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(
        self,
        index_path: str,
        passages_path: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.index_path = Path(index_path)
        self.passages_path = Path(passages_path)

        self.model = SentenceTransformer(model_name)

        self.index = faiss.read_index(str(self.index_path))

        self.passages = pd.read_parquet(self.passages_path)

        if self.index.ntotal != len(self.passages):
            raise ValueError(
                f"FAISS index contains {self.index.ntotal:,} vectors, "
                f"but passages contain {len(self.passages):,} rows."
            )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            row = self.passages.iloc[int(index)]

            results.append(
                {
                    "passage_id": int(row["passage_id"]),
                    "passage_text": str(row["passage_text"]),
                    "score": float(score),
                }
            )

        return results