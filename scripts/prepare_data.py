"""Build the v2 MS MARCO passage subset and its eval/tune query splits.

Corpus = relevant passages of the sampled queries (qrels.dev.small)
       + top-N hard negatives per sampled query from top1000.dev
         (pool re-ranked with BM25, because top1000.dev has no scores)
       + seeded random passages from the full collection up to --target-size.

Every passage is labelled with its source (relevant / hard / random), and
all inputs, parameters, counts and output hashes go into manifest.json.

Outputs (in --out-dir):
    passages.parquet          passage_id (int64), passage_text, source
    queries_{eval,tune}.tsv   qid <TAB> query
    qrels_{eval,tune}.tsv     TREC format: qid 0 pid 1
    manifest.json

Refuses to overwrite an existing output directory unless --force.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.retrieval.bm25_retriever import BM25Retriever
from app.utils.text import repair_mojibake

OUTPUT_FILES = (
    "passages.parquet",
    "queries_eval.tsv",
    "queries_tune.tsv",
    "qrels_eval.tsv",
    "qrels_tune.tsv",
    "manifest.json",
)

# Anserini's MS MARCO passage defaults.
BM25_K1 = 0.9
BM25_B = 0.4

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)


def tokenize(text: str) -> list[str]:
    return BM25Retriever._tokenize(text)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)


def load_qrels(path: Path) -> dict[int, set[int]]:
    qrels: dict[int, set[int]] = defaultdict(set)
    for row in read_tsv(path):
        qid, _, pid, rel = row[:4]
        if int(rel) > 0:
            qrels[int(qid)].add(int(pid))
    return dict(qrels)


def load_queries(path: Path, wanted: set[int]) -> dict[int, str]:
    return {int(qid): text for qid, text in read_tsv(path) if int(qid) in wanted}


def sample_queries(
    qids: list[int], n_eval: int, n_tune: int, seed: int
) -> tuple[list[int], list[int]]:
    if n_eval + n_tune > len(qids):
        raise ValueError(
            f"Requested {n_eval + n_tune} queries but only {len(qids)} available."
        )
    rng = random.Random(seed)
    sampled = rng.sample(sorted(qids), n_eval + n_tune)
    return sorted(sampled[:n_eval]), sorted(sampled[n_eval:])


def iter_pool_lines(path: Path, wanted: set[int]):
    for qid, pid, _query, passage in read_tsv(path):
        qid_i = int(qid)
        if qid_i in wanted:
            yield qid_i, int(pid), passage


def bm25_hard_negatives(
    path: Path,
    queries: dict[int, str],
    wanted: set[int],
    qrels: dict[int, set[int]],
    per_query: int,
) -> tuple[dict[int, list[int]], dict]:
    """Rank each wanted query's top1000 pool with BM25; keep top non-relevant.

    Streams the file twice (memory stays small):
      1. document frequencies / lengths over the union of wanted pools, and
         where relevant passages sit in file order (if the file were
         rank-ordered they would cluster near position 0);
      2. score every candidate line, keeping a per-query heap.
    Ties are broken by lower passage_id.
    """
    seen: set[int] = set()
    df: Counter[str] = Counter()
    total_len = 0
    lines_per_query: Counter[int] = Counter()
    relevant_hits: list[tuple[int, int]] = []
    for qid, pid, passage in iter_pool_lines(path, wanted):
        position = lines_per_query[qid]
        lines_per_query[qid] += 1
        if pid in qrels.get(qid, ()):
            relevant_hits.append((qid, position))
        if pid not in seen:
            seen.add(pid)
            tokens = tokenize(repair_mojibake(passage))
            df.update(set(tokens))
            total_len += len(tokens)

    n_docs = len(seen)
    del seen
    avgdl = total_len / max(1, n_docs)
    idf_cache: dict[str, float] = {}

    def idf(term: str) -> float:
        if term not in idf_cache:
            n = df.get(term, 0)
            idf_cache[term] = math.log((n_docs - n + 0.5) / (n + 0.5) + 1.0)
        return idf_cache[term]

    query_terms = {qid: tokenize(queries[qid]) for qid in wanted}
    heaps: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for qid, pid, passage in iter_pool_lines(path, wanted):
        if pid in qrels.get(qid, ()):
            continue
        heap = heaps[qid]
        if any(entry[1] == -pid for entry in heap):
            continue
        tokens = tokenize(repair_mojibake(passage))
        tf = Counter(tokens)
        norm = BM25_K1 * (1 - BM25_B + BM25_B * len(tokens) / avgdl)
        score = sum(
            idf(term) * tf[term] * (BM25_K1 + 1) / (tf[term] + norm)
            for term in query_terms[qid]
            if term in tf
        )
        # Min-heap root = worst kept: lowest score, then highest pid.
        entry = (score, -pid)
        if len(heap) < per_query:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)

    hard = {
        qid: [-neg_pid for _, neg_pid in sorted(heap, key=lambda e: (-e[0], -e[1]))]
        for qid, heap in heaps.items()
    }
    positions = [
        position / max(1, lines_per_query[qid] - 1) for qid, position in relevant_hits
    ]
    order_check = {
        "queries_with_pool": len(lines_per_query),
        "pool_union_size": n_docs,
        "relevant_found_in_pool": len(positions),
        "mean_relative_position_of_relevant": (
            round(sum(positions) / len(positions), 4) if positions else None
        ),
        "note": "0.0 = always first line, ~0.5 = no rank order in file order",
    }
    return hard, order_check


def count_collection(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def collect_texts(path: Path, wanted: set[int]) -> dict[int, str]:
    found = {}
    for pid, text in read_tsv(path):
        pid_i = int(pid)
        if pid_i in wanted:
            found[pid_i] = repair_mojibake(text)
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"{len(missing)} passage ids not found in collection.")
    return found


def write_queries(path: Path, qids: list[int], queries: dict[int, str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for qid in qids:
            handle.write(f"{qid}\t{queries[qid]}\n")


def write_qrels(path: Path, qids: list[int], qrels: dict[int, set[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for qid in qids:
            for pid in sorted(qrels[qid]):
                handle.write(f"{qid}\t0\t{pid}\t1\n")


def prepare(
    raw_dir: Path,
    out_dir: Path,
    *,
    seed: int = 42,
    n_eval: int = 1000,
    n_tune: int = 500,
    hard_per_query: int = 50,
    target_size: int = 150_000,
    collection_name: str = "collection.tsv",
    queries_name: str = "queries.dev.small.tsv",
    qrels_name: str = "qrels.dev.small.tsv",
    top1000_name: str = "top1000.dev",
    force: bool = False,
    created_at: str | None = None,
) -> dict:
    existing = [name for name in OUTPUT_FILES if (out_dir / name).exists()]
    if existing and not force:
        raise FileExistsError(
            f"{out_dir} already has {existing}; refusing to overwrite (use --force)."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    collection = raw_dir / collection_name
    qrels_all = load_qrels(raw_dir / qrels_name)
    queries_all = load_queries(raw_dir / queries_name, set(qrels_all))
    usable = sorted(set(qrels_all) & set(queries_all))
    eval_qids, tune_qids = sample_queries(usable, n_eval, n_tune, seed)
    selected = set(eval_qids) | set(tune_qids)

    hard, order_check = bm25_hard_negatives(
        raw_dir / top1000_name, queries_all, selected, qrels_all, hard_per_query
    )

    source: dict[int, str] = {}
    for qid in sorted(selected):
        for pid in qrels_all[qid]:
            source[pid] = "relevant"
    for qid in sorted(hard):
        for pid in hard[qid]:
            source.setdefault(pid, "hard")

    collection_size = count_collection(collection)
    n_random = target_size - len(source)
    if n_random < 0:
        raise ValueError(
            f"target_size {target_size} < {len(source)} relevant+hard passages."
        )
    # Draw enough ids that, after skipping already-selected ones, n_random
    # remain; keep draw order (deterministic, no 8.8M-element list).
    rng = random.Random(seed + 1)
    draws = rng.sample(range(collection_size), min(collection_size, n_random + len(source)))
    random_pids = [pid for pid in draws if pid not in source][:n_random]
    for pid in random_pids:
        source[pid] = "random"

    texts = collect_texts(collection, set(source))
    passage_ids = sorted(source)
    frame = pd.DataFrame(
        {
            "passage_id": pd.array(passage_ids, dtype="int64"),
            "passage_text": [texts[pid] for pid in passage_ids],
            "source": [source[pid] for pid in passage_ids],
        }
    )
    frame.to_parquet(out_dir / "passages.parquet", index=False)

    write_queries(out_dir / "queries_eval.tsv", eval_qids, queries_all)
    write_queries(out_dir / "queries_tune.tsv", tune_qids, queries_all)
    write_qrels(out_dir / "qrels_eval.tsv", eval_qids, qrels_all)
    write_qrels(out_dir / "qrels_tune.tsv", tune_qids, qrels_all)

    downloads_path = raw_dir / "downloads.json"
    manifest = {
        "version": out_dir.name,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "builder": "scripts/prepare_data.py",
        "sources": (
            json.loads(downloads_path.read_text(encoding="utf-8"))
            if downloads_path.exists()
            else None
        ),
        "inputs": {
            name: sha256_file(raw_dir / name)
            for name in (collection_name, queries_name, qrels_name, top1000_name)
        },
        "params": {
            "seed_queries": seed,
            "seed_random_passages": seed + 1,
            "n_eval_queries": n_eval,
            "n_tune_queries": n_tune,
            "hard_negatives_per_query": hard_per_query,
            "hard_negative_ranking": f"BM25 k1={BM25_K1} b={BM25_B}, IDF over union "
            "of selected top1000.dev pools, app tokenizer, relevant excluded, "
            "ties by passage_id",
            "target_size": target_size,
            "text_normalization": "app.utils.text.repair_mojibake",
        },
        "top1000_order_check": order_check,
        "counts": {
            "collection_size": collection_size,
            "dev_small_queries_usable": len(usable),
            "eval_queries": len(eval_qids),
            "tune_queries": len(tune_qids),
            "eval_qrels": sum(len(qrels_all[q]) for q in eval_qids),
            "tune_qrels": sum(len(qrels_all[q]) for q in tune_qids),
            "passages": len(frame),
            "by_source": dict(Counter(frame["source"])),
        },
        "files": {
            name: sha256_file(out_dir / name)
            for name in OUTPUT_FILES
            if name != "manifest.json"
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("artifacts/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/v2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-eval", type=int, default=1000)
    parser.add_argument("--n-tune", type=int, default=500)
    parser.add_argument("--hard-per-query", type=int, default=50)
    parser.add_argument("--target-size", type=int, default=150_000)
    parser.add_argument("--top1000-name", default="top1000.dev")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = prepare(
        args.raw_dir,
        args.out_dir,
        seed=args.seed,
        n_eval=args.n_eval,
        n_tune=args.n_tune,
        hard_per_query=args.hard_per_query,
        target_size=args.target_size,
        top1000_name=args.top1000_name,
        force=args.force,
    )
    print(json.dumps({k: manifest[k] for k in ("counts", "top1000_order_check")}, indent=2))


if __name__ == "__main__":
    main()
