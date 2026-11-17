"""Retrieval evaluation on the served corpus (official MS MARCO dev-small qrels).

Systems (all on the same eval queries):
    bm25            BM25Retriever, top --depth
    dense           DenseRetriever (pinned model revision), top --depth
    hybrid          RRF (app HybridRetriever code) over the two lists above
    hybrid+rerank   cross-encoder reranks hybrid top --rerank-depth (served
                    default: HYBRID_TOP_K); ranks below that keep hybrid order

Metrics: MRR@10, Recall@10, Recall@100, nDCG@10 (binary qrels), mean with
95% bootstrap CI; paired bootstrap for hybrid vs bm25, hybrid vs dense,
hybrid+rerank vs hybrid. Per-stage latency p50/p95 (single process,
sequential, CPU). Never reads the tune split. No LLM calls.

Writes eval/results/<run>/metrics.json and per_query.jsonl.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from app.artifacts import sha256_file, verify_artifacts
from app.config import settings
from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from evaluation.metrics import (
    METRICS,
    bootstrap_mean_ci,
    paired_bootstrap,
    percentile,
    resample_indices,
)

SYSTEMS = ("bm25", "dense", "hybrid", "hybrid+rerank")
COMPARISONS = (("hybrid", "bm25"), ("hybrid", "dense"), ("hybrid+rerank", "hybrid"))
WARMUP_QUERIES = ("what is a warmup query", "how long does it take to boil an egg")


class _Fixed:
    """Feeds a precomputed ranking into HybridRetriever (same fusion code)."""

    def __init__(self, results: list[dict]) -> None:
        self.results = results

    def retrieve(self, query: str, top_k: int) -> list[dict]:
        return self.results[:top_k]


def load_split(artifacts: Path) -> tuple[list[tuple[int, str]], dict[int, set[int]]]:
    queries = pd.read_csv(
        artifacts / "queries_eval.tsv", sep="\t", header=None, quoting=3,
        names=["qid", "query"], dtype={"qid": int, "query": str},
    )
    qrels = pd.read_csv(artifacts / "qrels_eval.tsv", sep="\t", header=None)
    relevant = {int(q): set(map(int, g[2])) for q, g in qrels.groupby(0)}
    tune = set(
        pd.read_csv(artifacts / "queries_tune.tsv", sep="\t", header=None, quoting=3)[0]
    )
    if tune & set(queries["qid"]):
        raise ValueError("Eval and tune query sets overlap.")
    missing = set(queries["qid"]) - relevant.keys()
    if missing:
        raise ValueError(f"{len(missing)} eval queries have no qrels.")
    return list(zip(queries["qid"], queries["query"])), relevant


def hardware() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": sys.version.split()[0],
        "packages": {
            name: version(name)
            for name in ("torch", "sentence-transformers", "faiss-cpu", "rank-bm25", "numpy")
        },
        "execution": "single process, sequential queries, CPU only",
    }
    try:
        import torch

        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    if sys.platform == "win32":
        try:
            name = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=30, check=True,
            ).stdout.strip()
            info["cpu_name"] = name

            class _Mem(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong)] + [
                    (f"_{i}", ctypes.c_ulonglong) for i in range(6)]

            mem = _Mem()
            mem.dwLength = ctypes.sizeof(_Mem)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            info["ram_gb"] = round(mem.ullTotalPhys / 2**30, 1)
        except (OSError, subprocess.SubprocessError, AttributeError):
            pass
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu_name"] = line.split(":", 1)[1].strip()
                break
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                info["ram_gb"] = round(int(line.split()[1]) / 2**20, 1)
    return info


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def evaluate_query(qid, query, relevant, bm25, dense, reranker, depth, rerank_depth, rrf_k):
    timings = {}
    started = time.perf_counter()
    try:
        bm25_results = bm25.retrieve(query, top_k=depth)
    except ValueError:  # query is all stopwords after tokenization
        bm25_results = []
    timings["bm25_ms"] = _ms(started)

    started = time.perf_counter()
    dense_results = dense.retrieve(query, top_k=depth)
    timings["dense_ms"] = _ms(started)

    started = time.perf_counter()
    hybrid_results = HybridRetriever(
        _Fixed(dense_results), _Fixed(bm25_results), rrf_k=rrf_k
    ).retrieve(query, top_k=depth, retrieval_k=depth)
    timings["fusion_ms"] = _ms(started)

    started = time.perf_counter()
    reranked = reranker.rerank(query, hybrid_results[:rerank_depth], top_k=rerank_depth)
    timings["rerank_ms"] = _ms(started)

    rankings = {
        "bm25": [r["passage_id"] for r in bm25_results],
        "dense": [r["passage_id"] for r in dense_results],
        "hybrid": [r["passage_id"] for r in hybrid_results],
    }
    rankings["hybrid+rerank"] = [r["passage_id"] for r in reranked] + rankings["hybrid"][
        rerank_depth:
    ]
    latency = {
        "bm25": timings["bm25_ms"],
        "dense": timings["dense_ms"],
        "hybrid": timings["bm25_ms"] + timings["dense_ms"] + timings["fusion_ms"],
    }
    latency["hybrid+rerank"] = latency["hybrid"] + timings["rerank_ms"]

    record = {"qid": int(qid), "query": query, "relevant": sorted(relevant),
              "bm25_empty_query": not bm25_results, "stage_ms": timings, "systems": {}}
    for system, ranking in rankings.items():
        first = next((i for i, pid in enumerate(ranking, 1) if pid in relevant), None)
        record["systems"][system] = {
            **{name: fn(ranking, relevant) for name, fn in METRICS.items()},
            "first_relevant_rank": first,
            "latency_ms": latency[system],
            "top10": ranking[:10],
        }
    return record


def summarize(records: list[dict], n_resamples: int, seed: int) -> dict:
    indices = resample_indices(len(records), n_resamples, seed)
    systems = {}
    for system in SYSTEMS:
        entry = {}
        for metric in METRICS:
            values = [r["systems"][system][metric] for r in records]
            mean, low, high = bootstrap_mean_ci(values, indices=indices)
            entry[metric] = {"mean": mean, "ci95": [low, high]}
        lat = [r["systems"][system]["latency_ms"] for r in records]
        entry["latency_ms"] = {"p50": percentile(lat, 50), "p95": percentile(lat, 95)}
        systems[system] = entry

    stages = {
        stage: {
            "p50": percentile([r["stage_ms"][stage] for r in records], 50),
            "p95": percentile([r["stage_ms"][stage] for r in records], 95),
        }
        for stage in records[0]["stage_ms"]
    }
    comparisons = {
        f"{a} vs {b}": {
            metric: paired_bootstrap(
                [r["systems"][a][metric] for r in records],
                [r["systems"][b][metric] for r in records],
                indices=indices,
            )
            for metric in METRICS
        }
        for a, b in COMPARISONS
    }
    return {"systems": systems, "stage_latency_ms": stages, "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts-dir", type=Path,
                        default=Path(settings.manifest_path).parent)
    parser.add_argument("--out-root", type=Path, default=Path("eval/results"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--rerank-depth", type=int, default=settings.hybrid_top_k)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--estimate", type=int, default=0,
                        help="Time the first N eval queries and print an estimate; "
                             "writes nothing.")
    args = parser.parse_args()

    artifacts = args.artifacts_dir
    manifest_path = artifacts / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = {name: artifacts / name for name in (
        "passages.parquet", "dense_index.faiss", "bm25/bm25_index.pkl",
        "bm25/tokenized_corpus.pkl", "queries_eval.tsv", "qrels_eval.tsv",
        "queries_tune.tsv")}
    verify_artifacts(manifest_path, *files.values())
    dense_meta = manifest["indices"]["dense"]
    if (dense_meta["model"], dense_meta["model_revision"]) != (
        settings.embedding_model, settings.embedding_model_revision
    ):
        raise ValueError("Configured embedding model differs from the one used to build the index.")

    queries, relevant = load_split(artifacts)
    run_name = args.run_name or (
        f"{datetime.now(timezone.utc):%Y%m%d}-{manifest['version']}-"
        f"{(git_state()['commit'] or 'nogit')[:7]}"
    )
    out_dir = args.out_root / run_name
    if not args.estimate and out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; refusing to overwrite.")

    load_started = time.perf_counter()
    bm25 = BM25Retriever(files["bm25/bm25_index.pkl"], files["bm25/tokenized_corpus.pkl"],
                         files["passages.parquet"], manifest_path)
    dense = DenseRetriever(files["dense_index.faiss"], files["passages.parquet"],
                           settings.embedding_model, settings.embedding_model_revision)
    reranker = CrossEncoderReranker(settings.reranker_model, settings.reranker_model_revision)
    load_seconds = time.perf_counter() - load_started

    for warm in WARMUP_QUERIES:  # not reported
        evaluate_query(0, warm, {-1}, bm25, dense, reranker, args.depth,
                       args.rerank_depth, settings.rrf_k)

    subset = queries[: args.estimate] if args.estimate else queries
    started = time.perf_counter()
    records = []
    for i, (qid, query) in enumerate(subset, 1):
        records.append(evaluate_query(qid, query, relevant[qid], bm25, dense, reranker,
                                      args.depth, args.rerank_depth, settings.rrf_k))
        if i % 100 == 0:
            print(f"{i}/{len(subset)} queries, {time.perf_counter() - started:.0f}s",
                  flush=True)
    elapsed = time.perf_counter() - started

    if args.estimate:
        per_query = elapsed / len(subset)
        print(f"load {load_seconds:.0f}s; {per_query:.2f}s/query -> "
              f"est. {(load_seconds + per_query * len(queries)) / 60:.1f} min "
              f"for {len(queries)} queries (+ bootstrap)")
        return

    summary = summarize(records, args.n_resamples, args.seed)
    metrics = {
        "run": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "corpus": {
            "version": manifest["version"],
            "manifest_sha256": sha256_file(manifest_path),
            "passages": manifest["counts"]["passages"],
            "files_sha256": {name: manifest["files"][name] for name in files},
        },
        "split": "eval (official MS MARCO dev-small qrels); tune split not used",
        "n_queries": len(records),
        "n_bm25_empty_queries": sum(r["bm25_empty_query"] for r in records),
        "config": {
            "depth": args.depth,
            "rrf_k": settings.rrf_k,
            "rerank_depth": args.rerank_depth,
            "served_retrieval_k": settings.retrieval_k,
            "note": "Eval fuses top-depth from each retriever (needed for Recall@100); "
                    "the served API fuses top RETRIEVAL_K. hybrid+rerank reranks the "
                    "hybrid top rerank_depth, as served; lower ranks keep hybrid order.",
            "embedding_model": f"{settings.embedding_model}@{settings.embedding_model_revision}",
            "reranker_model": f"{settings.reranker_model}@{settings.reranker_model_revision}",
            "bootstrap": {"n_resamples": args.n_resamples, "seed": args.seed,
                          "paired": "same resampled query sets for all systems"},
        },
        "hardware": hardware(),
        "timing_seconds": {"load": round(load_seconds, 1), "queries": round(elapsed, 1)},
        **summary,
    }

    out_dir.mkdir(parents=True)
    with (out_dir / "per_query.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
