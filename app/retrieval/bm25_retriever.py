from pathlib import Path
import pickle
import string

import pandas as pd


class BM25Retriever:
    def __init__(
        self,
        index_path: str,
        corpus_path: str,
        passages_path: str,
    ) -> None:
        self.index_path = Path(index_path)
        self.corpus_path = Path(corpus_path)
        self.passages_path = Path(passages_path)

        # Load pre-built BM25 index
        with open(self.index_path, "rb") as file:
            self.bm25 = pickle.load(file)

        # Load pre-tokenized corpus
        with open(self.corpus_path, "rb") as file:
            self.tokenized_corpus = pickle.load(file)

        # Load passages
        self.passages = pd.read_parquet(self.passages_path)

        # Verify BM25 index and passage alignment
        if len(self.bm25.doc_freqs) != len(self.passages):
            raise ValueError(
                f"BM25 index contains {len(self.bm25.doc_freqs):,} documents, "
                f"but passages contain {len(self.passages):,} rows."
            )

        if len(self.tokenized_corpus) != len(self.passages):
            raise ValueError(
                f"Tokenized corpus contains {len(self.tokenized_corpus):,} documents, "
                f"but passages contain {len(self.passages):,} rows."
            )

    @staticmethod
    def _tokenize(query: str) -> list[str]:
        """
        Tokenize a user query for BM25 retrieval.

        Steps:
        1. Convert query to lowercase.
        2. Remove punctuation.
        3. Split into individual tokens.
        4. Remove common stopwords.
        """

        stopwords = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "how",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "that",
            "the",
            "to",
            "was",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "with",
        }

        # Remove punctuation such as:
        # ? ! , . : ; ( ) [ ] { }
        translator = str.maketrans(
            "",
            "",
            string.punctuation,
        )

        cleaned_query = query.lower().translate(translator)

        tokens = [
            token
            for token in cleaned_query.split()
            if token not in stopwords
        ]

        return tokens

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Retrieve the top-k passages using BM25.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # Tokenize query
        query_tokens = self._tokenize(query)

        if not query_tokens:
            raise ValueError(
                "Query contains no meaningful terms after "
                "stopword filtering."
            )

        # Calculate BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get indices of top-k documents
        top_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results = []

        for index in top_indices:
            row = self.passages.iloc[index]

            results.append(
                {
                    "passage_id": int(row["passage_id"]),
                    "passage_text": str(row["passage_text"]),
                    "score": float(scores[index]),
                }
            )

        return results