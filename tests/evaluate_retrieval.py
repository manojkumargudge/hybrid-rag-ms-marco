import json
import math
from pathlib import Path


QRELS_PATH = Path("eval/qrels.json")
CANDIDATES_PATH = Path("eval/candidate_results.txt")


def load_qrels() -> dict[str, set[int]]:
    with QRELS_PATH.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    return {
        item["query_id"]: set(item["relevant_passage_ids"])
        for item in data
    }


def parse_candidates() -> dict[str, list[int]]:
    candidates: dict[str, list[int]] = {}

    current_query_id = None

    for line in CANDIDATES_PATH.read_text(
        encoding="utf-8-sig"
    ).splitlines():

        line = line.strip()

        if line.startswith("q") and ":" in line:
            current_query_id = line.split(":", 1)[0]
            candidates[current_query_id] = []

        elif line.startswith("Passage ID:") and current_query_id:
            passage_id = int(
                line.split(":", 1)[1].strip()
            )
            candidates[current_query_id].append(passage_id)

    return candidates


def recall_at_k(
    retrieved: list[int],
    relevant: set[int],
    k: int,
) -> float:
    if not relevant:
        return 0.0

    retrieved_k = set(retrieved[:k])

    return len(retrieved_k & relevant) / len(relevant)


def reciprocal_rank(
    retrieved: list[int],
    relevant: set[int],
) -> float:
    for rank, passage_id in enumerate(retrieved, start=1):
        if passage_id in relevant:
            return 1.0 / rank

    return 0.0


def dcg_at_k(
    retrieved: list[int],
    relevant: set[int],
    k: int,
) -> float:
    score = 0.0

    for rank, passage_id in enumerate(retrieved[:k], start=1):
        if passage_id in relevant:
            relevance = 1.0
        else:
            relevance = 0.0

        score += relevance / math.log2(rank + 1)

    return score


def ndcg_at_k(
    retrieved: list[int],
    relevant: set[int],
    k: int,
) -> float:
    if not relevant:
        return 0.0

    actual_dcg = dcg_at_k(
        retrieved,
        relevant,
        k,
    )

    ideal_retrieval = list(relevant)

    ideal_dcg = dcg_at_k(
        ideal_retrieval,
        relevant,
        k,
    )

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def main() -> None:
    qrels = load_qrels()
    candidates = parse_candidates()

    print("=" * 70)
    print("HYBRID RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    recall_scores = []
    mrr_scores = []
    ndcg_scores = []

    for query_id in sorted(qrels):
        relevant = qrels[query_id]
        retrieved = candidates.get(query_id, [])

        recall = recall_at_k(
            retrieved,
            relevant,
            k=5,
        )

        rr = reciprocal_rank(
            retrieved,
            relevant,
        )

        ndcg = ndcg_at_k(
            retrieved,
            relevant,
            k=5,
        )

        recall_scores.append(recall)
        mrr_scores.append(rr)
        ndcg_scores.append(ndcg)

        print(
            f"{query_id}: "
            f"Recall@5={recall:.4f} | "
            f"RR={rr:.4f} | "
            f"nDCG@5={ndcg:.4f}"
        )

    query_count = len(qrels)

    mean_recall = sum(recall_scores) / query_count
    mrr = sum(mrr_scores) / query_count
    mean_ndcg = sum(ndcg_scores) / query_count

    print()
    print("=" * 70)
    print("FINAL METRICS")
    print("=" * 70)

    print(f"Queries evaluated : {query_count}")
    print(f"Recall@5          : {mean_recall:.4f}")
    print(f"MRR               : {mrr:.4f}")
    print(f"nDCG@5            : {mean_ndcg:.4f}")

    print("=" * 70)


if __name__ == "__main__":
    main()