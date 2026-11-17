"""Verify an artifacts directory: sha256 of every manifest file + alignment.

Usage: python -m scripts.verify_artifacts [--dir artifacts/v3]
"""

import argparse
import json
import pickle
from pathlib import Path

import faiss
import pandas as pd

from app.artifacts import verify_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("artifacts/v3"))
    args = parser.parse_args()

    manifest_path = args.dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = sorted(manifest["files"])
    verify_artifacts(manifest_path, *(args.dir / name for name in names))
    for name in names:
        print(f"sha256 OK  {name}")

    passages = pd.read_parquet(args.dir / "passages.parquet")
    counts = {"passages": len(passages)}
    if (args.dir / "dense_index.faiss").exists():
        counts["faiss_vectors"] = faiss.read_index(
            str(args.dir / "dense_index.faiss")
        ).ntotal
    corpus_path = args.dir / "bm25" / "tokenized_corpus.pkl"
    if corpus_path.exists():
        with corpus_path.open("rb") as handle:
            counts["bm25_documents"] = len(pickle.load(handle))

    print(json.dumps(counts))
    if len(set(counts.values())) != 1:
        raise SystemExit(f"Row count mismatch: {counts}")
    if counts["passages"] != manifest["counts"]["passages"]:
        raise SystemExit("passages.parquet row count differs from manifest.")
    print("VERIFICATION SUCCESSFUL")


if __name__ == "__main__":
    main()
