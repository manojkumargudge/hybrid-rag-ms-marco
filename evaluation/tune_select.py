"""Select the served retrieval config on the TUNE split only.

Families (rerank = app CrossEncoderReranker on the top `depth` candidates;
ranks below `depth` keep the base order):
    dense+rerank          depth in DEPTHS
    wrrf+rerank           weighted RRF (app HybridRetriever, dense_weight=w),
                          w in WEIGHTS, depth in DEPTHS

Per query, retriever top-100 lists and cross-encoder scores for the union of
all candidates any config can rerank are computed ONCE and cached
(eval/cache/, not committed); the grid reuses them.

Decision rule (fixed before running): pick the best config of each family by
mean MRR@10 on tune (ties -> smaller depth, then larger w). Serve the best
wrrf+rerank only if it beats the best dense+rerank on tune with the paired
95% bootstrap CI of the MRR@10 difference entirely above 0; otherwise serve
the best dense+rerank, and hybrid stays a config option.

Latency rule (added 2026-11-17, before the eval split was run with these
systems): within the chosen family (same w for wrrf), serve the SMALLEST
rerank depth whose tune MRR@10 is within LATENCY_TOLERANCE of the chosen
config's. Rationale: depth 50 costs ~1.6 s CPU rerank per query (50 pairs x
~33 ms) for +0.008 tune MRR@10 over depth 20.

Never reads qrels_eval.tsv. Writes eval/results/<run>/metrics.json and
per_query.jsonl.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
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
    resample_indices,
)
from evaluation.run_eval import _Fixed, git_state, hardware

WEIGHTS = (0.5, 0.6, 0.7, 0.8, 0.9)
DEPTHS = (10, 20, 50)
DEPTH = 100
DECISION_RULE = (
    "Best config per family by mean MRR@10 on tune (ties: smaller depth, then larger "
    "w). Serve best wrrf+rerank only if it beats best dense+rerank on tune with the "
    "paired 95% bootstrap CI of the MRR@10 difference entirely above 0; else serve "
    "best dense+rerank (hybrid remains a config option)."
)
LATENCY_TOLERANCE = 0.01
LATENCY_RULE = (
    "Within the chosen family (same w for wrrf), serve the smallest rerank depth whose "
    f"tune MRR@10 mean is >= chosen MRR@10 - {LATENCY_TOLERANCE}. Rationale: depth 50 "
    "costs ~1.6 s CPU rerank/query (50 pairs x ~33 ms) for +0.008 tune MRR@10 vs depth 20."
)


def apply_latency_rule(grid: dict, chosen: str, tolerance: float = LATENCY_TOLERANCE) -> str:
    target = grid[chosen]
    floor = target["mrr@10"]["mean"] - tolerance
    candidates = [
        name for name, g in grid.items()
        if g["family"] == target["family"]
        and g["dense_weight"] == target["dense_weight"]
        and g["mrr@10"]["mean"] >= floor
    ]
    return min(candidates, key=lambda name: grid[name]["rerank_depth"])


def load_tune(artifacts: Path) -> tuple[list[tuple[int, str]], dict[int, set[int]]]:
    queries = pd.read_csv(artifacts / "queries_tune.tsv", sep="\t", header=None, quoting=3,
                          names=["qid", "query"], dtype={"qid": int, "query": str})
    qrels = pd.read_csv(artifacts / "qrels_tune.tsv", sep="\t", header=None)
    relevant = {int(q): set(map(int, g[2])) for q, g in qrels.groupby(0)}
    eval_qids = set(pd.read_csv(artifacts / "queries_eval.tsv", sep="\t", header=None,
                                quoting=3)[0])
    if eval_qids & set(queries["qid"]):
        raise ValueError("Eval and tune query sets overlap.")
    return list(zip(queries["qid"], queries["query"])), relevant


def fuse(dense: list[dict], bm25: list[dict], weight: float) -> list[int]:
    fused = HybridRetriever(_Fixed(dense), _Fixed(bm25), rrf_k=settings.rrf_k,
                            dense_weight=weight).retrieve("q", top_k=DEPTH, retrieval_k=DEPTH)
    return [r["passage_id"] for r in fused]


def rerank(base: list[int], scores: dict[int, float], depth: int) -> list[int]:
    head = sorted(base[:depth], key=lambda pid: scores[pid], reverse=True)  # stable
    return head + base[depth:]


def build_cache(artifacts: Path, queries, cache_path: Path) -> None:
    manifest_path = artifacts / "manifest.json"
    files = [artifacts / n for n in ("passages.parquet", "dense_index.faiss",
                                     "bm25/bm25_index.pkl", "bm25/tokenized_corpus.pkl")]
    verify_artifacts(manifest_path, *files)
    bm25 = BM25Retriever(files[2], files[3], files[0], manifest_path)
    dense = DenseRetriever(files[1], files[0], settings.embedding_model,
                           settings.embedding_model_revision)
    reranker = CrossEncoderReranker(settings.reranker_model, settings.reranker_model_revision)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with cache_path.open("w", encoding="utf-8", newline="\n") as fh:
        for i, (qid, query) in enumerate(queries, 1):
            try:
                bm25_res = bm25.retrieve(query, top_k=DEPTH)
            except ValueError:
                bm25_res = []
            dense_res = dense.retrieve(query, top_k=DEPTH)
            texts = {r["passage_id"]: r["passage_text"] for r in bm25_res + dense_res}
            union = dict.fromkeys([r["passage_id"] for r in dense_res][: max(DEPTHS)])
            for w in WEIGHTS:
                union.update(dict.fromkeys(fuse(dense_res, bm25_res, w)[: max(DEPTHS)]))
            scored = reranker.rerank(
                query, [{"passage_id": p, "passage_text": texts[p]} for p in union],
                top_k=len(union),
            )
            fh.write(json.dumps({
                "qid": int(qid),
                "bm25": [[r["passage_id"], r["score"]] for r in bm25_res],
                "dense": [[r["passage_id"], r["score"]] for r in dense_res],
                "rerank_scores": {str(r["passage_id"]): r["score"] for r in scored},
            }) + "\n")
            if i % 50 == 0:
                print(f"{i}/{len(queries)} cached, {time.perf_counter() - started:.0f}s",
                      flush=True)


def select(cache_path: Path, relevant: dict[int, set[int]], n_resamples: int, seed: int):
    rows = [json.loads(line) for line in cache_path.open(encoding="utf-8")]
    configs = [("dense+rerank", None, d) for d in DEPTHS] + [
        ("wrrf+rerank", w, d) for w in WEIGHTS for d in DEPTHS]
    per_query = []
    scores: dict[str, list[dict]] = {}
    for row in rows:
        rel = relevant[row["qid"]]
        dense = [{"passage_id": p, "passage_text": "", "score": s} for p, s in row["dense"]]
        bm25 = [{"passage_id": p, "passage_text": "", "score": s} for p, s in row["bm25"]]
        ce = {int(k): v for k, v in row["rerank_scores"].items()}
        record = {"qid": row["qid"], "configs": {}}
        for family, w, d in configs:
            base = [r["passage_id"] for r in dense] if w is None else fuse(dense, bm25, w)
            ranking = rerank(base, ce, d)
            name = f"{family}(depth={d})" if w is None else f"{family}(w={w},depth={d})"
            values = {m: fn(ranking, rel) for m, fn in METRICS.items()}
            scores.setdefault(name, []).append(values)
            record["configs"][name] = {"mrr@10": values["mrr@10"], "top10": ranking[:10]}
        per_query.append(record)

    indices = resample_indices(len(rows), n_resamples, seed)
    grid = {}
    for (family, w, d) in configs:
        name = f"{family}(depth={d})" if w is None else f"{family}(w={w},depth={d})"
        grid[name] = {"family": family, "dense_weight": w, "rerank_depth": d}
        for m in METRICS:
            mean, low, high = bootstrap_mean_ci([v[m] for v in scores[name]], indices=indices)
            grid[name][m] = {"mean": mean, "ci95": [low, high]}

    def best(family: str) -> str:
        names = [n for n, g in grid.items() if g["family"] == family]
        return max(names, key=lambda n: (grid[n]["mrr@10"]["mean"], -grid[n]["rerank_depth"],
                                         grid[n]["dense_weight"] or 0))

    best_dense, best_wrrf = best("dense+rerank"), best("wrrf+rerank")
    comparison = {
        m: paired_bootstrap([v[m] for v in scores[best_wrrf]],
                            [v[m] for v in scores[best_dense]], indices=indices)
        for m in METRICS
    }
    wrrf_wins = comparison["mrr@10"]["ci_low"] > 0
    chosen = best_wrrf if wrrf_wins else best_dense
    return grid, best_dense, best_wrrf, comparison, chosen, per_query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(settings.manifest_path).parent)
    parser.add_argument("--cache", type=Path, default=Path("eval/cache/tune-v3.jsonl"))
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--out-root", type=Path, default=Path("eval/results"))
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--estimate", type=int, default=0)
    args = parser.parse_args()

    artifacts = args.artifacts_dir
    manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    queries, relevant = load_tune(artifacts)
    run = (f"{datetime.now(timezone.utc):%Y%m%d}-{manifest['version']}-tune-"
           f"{(git_state()['commit'] or 'nogit')[:7]}")
    out_dir = args.out_root / run
    if not args.estimate and out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; refusing to overwrite.")

    if args.estimate:
        started = time.perf_counter()
        build_cache(artifacts, queries[: args.estimate],
                    args.cache.with_name("estimate.jsonl"))
        per_query = (time.perf_counter() - started) / args.estimate
        print(f"{per_query:.2f}s/query (incl. load) -> est. "
              f"{per_query * len(queries) / 60:.1f} min for {len(queries)} tune queries")
        return

    started = time.perf_counter()
    if not args.reuse_cache:
        build_cache(artifacts, queries, args.cache)
    cache_seconds = time.perf_counter() - started
    grid, best_dense, best_wrrf, comparison, chosen_by_mrr, per_query = select(
        args.cache, relevant, args.n_resamples, args.seed)
    chosen = apply_latency_rule(grid, chosen_by_mrr)

    metrics = {
        "run": run,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "split": "tune (official MS MARCO dev-small qrels); eval split not read",
        "n_queries": len(per_query),
        "corpus": {"version": manifest["version"],
                   "manifest_sha256": sha256_file(artifacts / "manifest.json")},
        "config": {
            "depth": DEPTH, "rrf_k": settings.rrf_k, "weights": WEIGHTS, "rerank_depths": DEPTHS,
            "weighted_rrf": "2w/(k+rank_dense) + 2(1-w)/(k+rank_bm25); w=0.5 == standard RRF",
            "embedding_model": f"{settings.embedding_model}@{settings.embedding_model_revision}",
            "reranker_model": f"{settings.reranker_model}@{settings.reranker_model_revision}",
            "bootstrap": {"n_resamples": args.n_resamples, "seed": args.seed},
            "cache_sha256": sha256_file(args.cache),
        },
        "decision_rule": DECISION_RULE,
        "latency_rule": LATENCY_RULE,
        "chosen_by_mrr": chosen_by_mrr,
        "best": {"dense+rerank": best_dense, "wrrf+rerank": best_wrrf},
        "best_wrrf_minus_best_dense": comparison,
        "chosen": {"name": chosen, **{k: grid[chosen][k] for k in
                                      ("family", "dense_weight", "rerank_depth")}},
        "grid": grid,
        "hardware": hardware(),
        "timing_seconds": {"cache_build": round(cache_seconds, 1)},
    }
    out_dir.mkdir(parents=True)
    with (out_dir / "per_query.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for record in per_query:
            fh.write(json.dumps(record) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"best": metrics["best"], "chosen": metrics["chosen"],
                      "mrr_diff": comparison["mrr@10"]}, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
