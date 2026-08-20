from app.generation.groq_generator import GroqGenerator
from app.rag_service import RAGService
from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.hybrid_retriever import HybridRetriever


PASSAGES_PATH = "data/passages_subset.parquet"

DENSE_INDEX_PATH = "indices/dense_index.faiss"

BM25_INDEX_PATH = "indices/bm25/bm25_index.pkl"

BM25_CORPUS_PATH = "indices/bm25/tokenized_corpus.pkl"


def main() -> None:
    print("=== INITIALIZING HYBRID RAG ===")

    print("\nLoading Dense Retriever...")

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

    print("Loading Groq Generator...")

    generator = GroqGenerator()

    print("Creating RAG Service...")

    rag = RAGService(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        generator=generator,
        retrieval_k=20,
        hybrid_top_k=10,
        rerank_top_k=5,
    )

    query = "what are the symptoms of diabetes?"

    print("\n" + "=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)

    print("\nRunning complete RAG pipeline...")

    result = rag.answer(query)

    print("\n=== FINAL ANSWER ===")
    print(result["answer"])

    print("\n=== SOURCES ===")

    for rank, source in enumerate(result["sources"], start=1):
        print(f"\nSource {rank}")
        print(f"Passage ID: {source['passage_id']}")
        print(f"Score: {source['score']:.6f}")
        print(f"Text: {source['passage_text'][:250]}...")

    print("\n=== VERIFICATION ===")

    assert result["query"] == query
    assert result["answer"]
    assert isinstance(result["answer"], str)
    assert len(result["sources"]) == 5

    source_ids = [
        source["passage_id"]
        for source in result["sources"]
    ]

    assert len(source_ids) == len(set(source_ids))

    print("✓ Query processed successfully")
    print("✓ Final answer generated")
    print("✓ Exactly 5 sources returned")
    print("✓ No duplicate source IDs")
    print("✓ END-TO-END RAG PIPELINE WORKING")


if __name__ == "__main__":
    main()