"""Build BM25 and dense (FAISS) indices for a prepared artifacts directory.

Reads <dir>/passages.parquet and <dir>/manifest.json (from prepare_data.py),
writes:
    bm25/bm25_index.pkl, bm25/tokenized_corpus.pkl   (--only bm25)
    dense_index.faiss                                 (--only dense)
and records their sha256, model name/revision, tokenizer and timings in
manifest.json. Row i of passages.parquet == FAISS id i == BM25 doc i.

Refuses to overwrite existing index files unless --force.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from app.artifacts import sha256_file, verify_artifacts
from app.retrieval.bm25_retriever import BM25Retriever

BM25_FILES = ("bm25/bm25_index.pkl", "bm25/tokenized_corpus.pkl")
DENSE_FILES = ("dense_index.faiss",)


def _check_overwrite(out_dir: Path, names: tuple[str, ...], force: bool) -> None:
    existing = [name for name in names if (out_dir / name).exists()]
    if existing and not force:
        raise FileExistsError(
            f"{out_dir} already has {existing}; refusing to overwrite (use --force)."
        )


def _load(out_dir: Path) -> tuple[dict, pd.DataFrame]:
    manifest_path = out_dir / "manifest.json"
    verify_artifacts(manifest_path, out_dir / "passages.parquet")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest, pd.read_parquet(out_dir / "passages.parquet")


def _save_manifest(out_dir: Path, manifest: dict, names: tuple[str, ...]) -> None:
    for name in names:
        manifest["files"][name] = sha256_file(out_dir / name)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def build_bm25(out_dir: Path, force: bool = False) -> dict:
    _check_overwrite(out_dir, BM25_FILES, force)
    manifest, passages = _load(out_dir)
    started = time.perf_counter()
    tokenized = [BM25Retriever._tokenize(text) for text in passages["passage_text"]]
    bm25 = BM25Okapi(tokenized)
    (out_dir / "bm25").mkdir(exist_ok=True)
    with (out_dir / BM25_FILES[0]).open("wb") as handle:
        pickle.dump(bm25, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (out_dir / BM25_FILES[1]).open("wb") as handle:
        pickle.dump(tokenized, handle, protocol=pickle.HIGHEST_PROTOCOL)
    manifest.setdefault("indices", {})["bm25"] = {
        "library": "rank_bm25.BM25Okapi (defaults k1=1.5 b=0.75 epsilon=0.25)",
        "tokenizer": "app.retrieval.bm25_retriever.BM25Retriever._tokenize",
        "documents": len(tokenized),
        "build_seconds": round(time.perf_counter() - started, 1),
    }
    _save_manifest(out_dir, manifest, BM25_FILES)
    return manifest


def build_dense(
    out_dir: Path,
    encoder,
    model_name: str,
    model_revision: str | None,
    batch_size: int = 128,
    force: bool = False,
) -> dict:
    """Encode passages with `encoder.encode(list[str], ...)` into IndexFlatIP."""
    _check_overwrite(out_dir, DENSE_FILES, force)
    manifest, passages = _load(out_dir)
    started = time.perf_counter()
    embeddings = encoder.encode(
        passages["passage_text"].tolist(),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(out_dir / DENSE_FILES[0]))
    manifest.setdefault("indices", {})["dense"] = {
        "model": model_name,
        "model_revision": model_revision,
        "index": "faiss.IndexFlatIP over L2-normalized embeddings (cosine)",
        "dimension": int(embeddings.shape[1]),
        "vectors": int(index.ntotal),
        "batch_size": batch_size,
        "build_seconds": round(time.perf_counter() - started, 1),
    }
    _save_manifest(out_dir, manifest, DENSE_FILES)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("artifacts/v3"))
    parser.add_argument("--only", choices=["bm25", "dense"], required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        help="Only time encoding of the first N passages; writes nothing.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.only == "bm25":
        manifest = build_bm25(args.dir, force=args.force)
        print(json.dumps(manifest["indices"]["bm25"], indent=2))
        return

    from sentence_transformers import SentenceTransformer

    from app.config import settings

    encoder = SentenceTransformer(
        settings.embedding_model, revision=settings.embedding_model_revision
    )
    if args.benchmark:
        texts = pd.read_parquet(args.dir / "passages.parquet")["passage_text"]
        sample = texts.head(args.benchmark).tolist()
        started = time.perf_counter()
        encoder.encode(sample, batch_size=args.batch_size, normalize_embeddings=True)
        rate = len(sample) / (time.perf_counter() - started)
        print(
            f"{rate:.0f} passages/s -> est. {len(texts) / rate / 60:.1f} min "
            f"for {len(texts):,} passages"
        )
        return

    manifest = build_dense(
        args.dir,
        encoder,
        settings.embedding_model,
        settings.embedding_model_revision,
        batch_size=args.batch_size,
        force=args.force,
    )
    print(json.dumps(manifest["indices"]["dense"], indent=2))


if __name__ == "__main__":
    main()
