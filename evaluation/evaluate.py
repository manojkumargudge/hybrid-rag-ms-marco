"""Evaluate retrieval candidates against manually reviewed qrels."""

from __future__ import annotations

import json
import math
from pathlib import Path

QRELS_PATH = Path("eval/qrels.json")
CANDIDATES_PATH = Path("eval/candidate_results.txt")


def load_qrels(path: Path = QRELS_PATH) -> dict[str, set[int]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return {
        item["query_id"]: set(item["relevant_passage_ids"])
        for item in data
    }


def parse_candidates(path: Path = CANDIDATES_PATH) -> dict[str, list[int]]:
    candidates: dict[str, list[int]] = {}
    query_id: str | None = None

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith("q") and ":" in line:
            query_id = line.split(":", 1)[0]
            candidates[query_id] = []
        elif line.startswith("Passage ID:") and query_id:
            candidates[query_id].append(int(line.split(":", 1)[1].strip()))

    return candidates


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(retrieved: list[int], relevant: set[int]) -> float:
    for rank, passage_id in enumerate(retrieved, start=1):
        if passage_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0

    def dcg(ids: list[int]) -> float:
        return sum(
            (1.0 if passage_id in relevant else 0.0) / math.log2(rank + 1)
            for rank, passage_id in enumerate(ids[:k], start=1)
        )

    ideal = dcg(list(relevant))
    return dcg(retrieved) / ideal if ideal else 0.0


def evaluate(
    qrels: dict[str, set[int]],
    candidates: dict[str, list[int]],
) -> dict[str, float]:
    if not qrels:
        raise ValueError("No relevance judgments were found.")

    rows = [
        (relevant, candidates.get(query_id, []))
        for query_id, relevant in qrels.items()
    ]
    count = len(rows)
    return {
        "recall_at_5": sum(recall_at_k(ids, rel, 5) for rel, ids in rows) / count,
        "recall_at_10": sum(recall_at_k(ids, rel, 10) for rel, ids in rows) / count,
        "mrr": sum(reciprocal_rank(ids, rel) for rel, ids in rows) / count,
        "ndcg_at_5": sum(ndcg_at_k(ids, rel, 5) for rel, ids in rows) / count,
    }


def main() -> None:
    metrics = evaluate(load_qrels(), parse_candidates())
    print("HYBRID RAG RETRIEVAL EVALUATION")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
