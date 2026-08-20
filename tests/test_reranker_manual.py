from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.reranking.cross_encoder_reranker import CrossEncoderReranker


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

    print("Loading Cross-Encoder Reranker...")

    reranker = CrossEncoderReranker()

    query = "what are the symptoms of diabetes?"

    print()
    print(f"Query: {query}")

    print()
    print("Retrieving hybrid candidates...")

    candidates = hybrid_retriever.retrieve(
        query=query,
        top_k=10,
        retrieval_k=20,
    )

    print(f"Candidates retrieved: {len(candidates)}")

    print()
    print("Reranking candidates...")

    results = reranker.rerank(
        query=query,
        passages=candidates,
        top_k=5,
    )

    print()
    print("=== RERANKED RESULTS ===")

    for rank, result in enumerate(results, start=1):
        print()
        print(f"Rank {rank}")
        print(f"Passage ID: {result['passage_id']}")
        print(f"Reranker Score: {result['score']:.6f}")
        print(f"Text: {result['passage_text'][:300]}")

    print()
    print("=== VERIFICATION ===")

    assert len(results) == 5

    scores = [
        result["score"]
        for result in results
    ]

    assert scores == sorted(scores, reverse=True)

    passage_ids = [
        result["passage_id"]
        for result in results
    ]

    assert len(passage_ids) == len(set(passage_ids))

    print("✓ Returned exactly 5 reranked results")
    print("✓ Results sorted by reranker score")
    print("✓ No duplicate passage IDs")
    print("✓ Cross-encoder reranking working correctly")


if __name__ == "__main__":
    main()