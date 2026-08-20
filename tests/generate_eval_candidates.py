import json

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.hybrid_retriever import HybridRetriever


PASSAGES_PATH = "data/passages_subset.parquet"

DENSE_INDEX_PATH = "indices/dense_index.faiss"

BM25_INDEX_PATH = "indices/bm25/bm25_index.pkl"

BM25_CORPUS_PATH = "indices/bm25/tokenized_corpus.pkl"

QUERIES_PATH = "eval/queries.json"


def safe_text(text: str, max_length: int = 500) -> str:
    """
    Convert passage text to Windows-console-safe ASCII.

    Some MS MARCO passages contain Unicode characters that
    Windows CP1252 cannot encode.
    """
    return (
        str(text)[:max_length]
        .encode("ascii", errors="replace")
        .decode("ascii")
    )


def main() -> None:
    print("Loading Dense Retriever...")

    dense = DenseRetriever(
        index_path=DENSE_INDEX_PATH,
        passages_path=PASSAGES_PATH,
    )

    print("Loading BM25 Retriever...")

    bm25 = BM25Retriever(
        index_path=BM25_INDEX_PATH,
        corpus_path=BM25_CORPUS_PATH,
        passages_path=PASSAGES_PATH,
    )

    print("Creating Hybrid Retriever...")

    hybrid = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        rrf_k=60,
    )

    with open(QUERIES_PATH, "r", encoding="utf-8") as file:
        queries = json.load(file)

    for item in queries:
        query_id = item["query_id"]
        query = item["query"]

        print("\n" + "=" * 80)
        print(f"{query_id}: {query}")
        print("=" * 80)

        results = hybrid.retrieve(
            query=query,
            top_k=5,
            retrieval_k=20,
        )

        for rank, result in enumerate(results, start=1):
            print(f"\nRank {rank}")
            print(f"Passage ID: {result['passage_id']}")
            print(f"RRF Score: {result['score']:.6f}")

            text = safe_text(result["passage_text"])

            print(f"Text: {text}")


if __name__ == "__main__":
    main()