from pathlib import Path

from app.retrieval.bm25_retriever import BM25Retriever


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    retriever = BM25Retriever(
        index_path=ROOT / "indices" / "bm25" / "bm25_index.pkl",
        corpus_path=ROOT / "indices" / "bm25" / "tokenized_corpus.pkl",
        passages_path=ROOT / "data" / "passages_subset.parquet",
    )

    query = "what are the symptoms of diabetes?"

    results = retriever.retrieve(
        query=query,
        top_k=5,
    )

    print(f"\nQuery: {query}")
    print(f"Retrieved: {len(results)} passages\n")

    for rank, result in enumerate(results, start=1):
        print(f"--- Rank {rank} ---")
        print(f"Passage ID: {result['passage_id']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['passage_text'][:500]}")
        print()


if __name__ == "__main__":
    main()