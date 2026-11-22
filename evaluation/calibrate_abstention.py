"""Calibrate a top reranker score cutoff on the official v3 TUNE split only."""

import argparse
import csv
import json
from pathlib import Path

from evaluation.metrics import reciprocal_rank_at_k

MAX_MRR_LOSS = 0.01
MANIFEST = Path("manifests/v3.json")
ARTIFACTS = Path("artifacts/v3")


def select_threshold(rows: list[dict]) -> dict:
    """Maximize TUNE abstentions within a one-point effective MRR@10 budget."""
    n = len(rows)
    baseline = sum(row["rr_at_10"] for row in rows) / n
    best = {
        "threshold": None, "abstained": 0, "abstention_rate": 0.0,
        "effective_mrr_at_10": baseline, "answered_only_mrr_at_10": baseline,
    }
    for threshold in sorted({row["top_score"] for row in rows}):
        answered = [row for row in rows if row["top_score"] >= threshold]
        total_rr = sum(row["rr_at_10"] for row in answered)
        effective = total_rr / n
        abstained = n - len(answered)
        if baseline - effective <= MAX_MRR_LOSS and abstained > best["abstained"]:
            best = {
                "threshold": threshold, "abstained": abstained, "abstention_rate": abstained / n,
                "effective_mrr_at_10": effective, "answered_only_mrr_at_10": total_rr / len(answered),
            }
    return {
        "n_queries": n, "baseline_mrr_at_10": baseline, "max_mrr_loss": MAX_MRR_LOSS,
        "mrr_at_10_delta": best["effective_mrr_at_10"] - baseline, **best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")

    import faiss
    import torch

    from app.artifacts import sha256_file, verify_artifacts
    from app.config import Settings
    from app.reranking.cross_encoder_reranker import CrossEncoderReranker
    from app.retrieval.dense_retriever import DenseRetriever

    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)
    paths = [ARTIFACTS / name for name in (
        "passages.parquet", "dense_index.faiss", "queries_tune.tsv", "qrels_tune.tsv"
    )]
    verify_artifacts(ARTIFACTS / "manifest.json", *paths, trusted_manifest_path=MANIFEST)
    passages, index, queries, qrels = paths
    defaults = {key: field.default for key, field in Settings.model_fields.items()}
    retriever = DenseRetriever(
        index, passages, defaults["embedding_model"], defaults["embedding_model_revision"]
    )
    reranker = CrossEncoderReranker(defaults["reranker_model"], defaults["reranker_model_revision"])
    relevance: dict[int, set[int]] = {}
    with qrels.open(encoding="utf-8", newline="") as handle:
        for qid, _, pid, grade in csv.reader(handle, delimiter="\t"):
            if int(grade) > 0:
                relevance.setdefault(int(qid), set()).add(int(pid))
    rows = []
    with queries.open(encoding="utf-8", newline="") as handle:
        for i, (qid, query) in enumerate(csv.reader(handle, delimiter="\t"), 1):
            ranked = reranker.rerank(query, retriever.retrieve(query, top_k=20), top_k=10)
            ids = [passage["passage_id"] for passage in ranked]
            rows.append({
                "qid": int(qid), "top_score": ranked[0]["score"], "top10": ids,
                "rr_at_10": reciprocal_rank_at_k(ids, relevance[int(qid)], 10),
            })
            if i % 50 == 0:
                print(f"Scored {i} TUNE queries", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "split": "TUNE", "retrieval_mode": "dense", "rerank_depth": 20, "threads": args.threads,
        "models": {key: defaults[key] for key in (
            "embedding_model", "embedding_model_revision", "reranker_model", "reranker_model_revision"
        )},
        "input_sha256": {path.name: sha256_file(path) for path in paths},
        **select_threshold(rows),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.output / "per_query.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
