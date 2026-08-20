from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.hybrid_retriever import HybridRetriever


PASSAGES_PATH = "data/passages_subset.parquet"

DENSE_INDEX_PATH = "indices/dense_index.faiss"

BM25_INDEX_PATH = "indices/bm25/bm25_index.pkl"

BM25_CORPUS_PATH = "indices/bm25/tokenized_corpus.pkl"


def main() -> None:
    print("Loading Dense Retriever...")

    dense_retriever = DenseRetriever(
        index_path=DENSE_INDEX_PATH,
        passages_path=PASSAGES_PATH,
    )

    print("Loading BM25 Retriever...")

    bm25_retriever = BM25Retriever(
        index_path=BM25_INDEX_PATH,
        corpus_path=BM25_CORPUS_PATH,
        passages_path=PASSAGES_PATH,
    )

    print("Creating Hybrid Retriever...")

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        rrf_k=60,
    )

    query = "what are the symptoms of diabetes?"

    print()
    print(f"Query: {query}")
    print()

    results = hybrid_retriever.retrieve(
        query=query,
        top_k=5,
        retrieval_k=10,
    )

    print("=== HYBRID RRF RESULTS ===")

    for rank, result in enumerate(results, start=1):
        print()
        print(f"Rank {rank}")
        print(f"Passage ID: {result['passage_id']}")
        print(f"RRF Score: {result['score']:.6f}")
        print(f"Text: {result['passage_text'][:300]}")

    print()
    print("=== VERIFICATION ===")

    assert len(results) == 5

    passage_ids = [
        result["passage_id"]
        for result in results
    ]

    assert len(passage_ids) == len(set(passage_ids))

    scores = [
        result["score"]
        for result in results
    ]

    assert scores == sorted(scores, reverse=True)

    print("✓ Returned exactly 5 results")
    print("✓ No duplicate passage IDs")
    print("✓ Results sorted by descending RRF score")
    print("✓ Hybrid RRF retrieval working correctly")


if __name__ == "__main__":
    main()