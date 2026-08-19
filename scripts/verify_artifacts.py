from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pickle


ROOT = Path(__file__).resolve().parents[1]

PASSAGES_PATH = ROOT / "data" / "passages_subset.parquet"
EMBEDDINGS_PATH = ROOT / "embeddings" / "all_embeddings.npy"
DENSE_INDEX_PATH = ROOT / "indices" / "dense_index.faiss"
BM25_INDEX_PATH = ROOT / "indices" / "bm25" / "bm25_index.pkl"
TOKENIZED_CORPUS_PATH = ROOT / "indices" / "bm25" / "tokenized_corpus.pkl"


def verify_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path}")

    print(f"✓ {path.relative_to(ROOT)}")


def main() -> None:
    print("=== HYBRID RAG ARTIFACT VERIFICATION ===\n")

    paths = [
        PASSAGES_PATH,
        EMBEDDINGS_PATH,
        DENSE_INDEX_PATH,
        BM25_INDEX_PATH,
        TOKENIZED_CORPUS_PATH,
    ]

    for path in paths:
        verify_file(path)

    print("\n=== Loading passages ===")
    passages = pd.read_parquet(PASSAGES_PATH)

    print(f"Rows: {len(passages):,}")
    print(f"Columns: {list(passages.columns)}")

    print("\n=== Loading embeddings ===")
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")

    print(f"Shape: {embeddings.shape}")
    print(f"Dtype: {embeddings.dtype}")

    print("\n=== Loading FAISS index ===")
    dense_index = faiss.read_index(str(DENSE_INDEX_PATH))

    print(f"Vectors: {dense_index.ntotal:,}")
    print(f"Dimension: {dense_index.d}")

    print("\n=== Loading BM25 index ===")
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_index = pickle.load(f)

    print(f"BM25 type: {type(bm25_index).__name__}")

    print("\n=== Loading tokenized corpus ===")
    with open(TOKENIZED_CORPUS_PATH, "rb") as f:
        tokenized_corpus = pickle.load(f)

    print(f"Documents: {len(tokenized_corpus):,}")

    print("\n=== Consistency checks ===")

    assert len(passages) == embeddings.shape[0], (
        f"Passage/embedding mismatch: "
        f"{len(passages)} vs {embeddings.shape[0]}"
    )

    assert embeddings.shape[0] == dense_index.ntotal, (
        f"Embedding/FAISS mismatch: "
        f"{embeddings.shape[0]} vs {dense_index.ntotal}"
    )

    assert len(passages) == len(tokenized_corpus), (
        f"Passage/BM25 mismatch: "
        f"{len(passages)} vs {len(tokenized_corpus)}"
    )

    print("✓ Passage count matches embeddings")
    print("✓ Embedding count matches FAISS")
    print("✓ Passage count matches BM25 corpus")

    print("\n=== VERIFICATION SUCCESSFUL ===")


if __name__ == "__main__":
    main()