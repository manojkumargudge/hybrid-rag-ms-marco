"""run_eval: incremental dense+rerank scoring, depth parsing, comparisons."""

import pytest

from evaluation.run_eval import (
    incremental_rerank,
    parse_depths,
    systems_and_comparisons,
)


class RecordingReranker:
    """Scores passage p as SCORES[p]; records every call's passage ids."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def rerank(self, query, passages, top_k):
        self.calls.append([p["passage_id"] for p in passages])
        ranked = sorted(
            ({"passage_id": p["passage_id"], "score": self.scores[p["passage_id"]]}
             for p in passages),
            key=lambda r: r["score"], reverse=True,
        )
        return ranked[:top_k]


def _candidates(n):
    return [{"passage_id": pid, "passage_text": str(pid)} for pid in range(n)]


def test_each_passage_is_scored_once_in_depth_segments():
    scores = {pid: (pid * 37) % 101 for pid in range(100)}
    reranker = RecordingReranker(scores)

    out = incremental_rerank(reranker, "q", _candidates(100), [20, 50])

    assert reranker.calls == [list(range(20)), list(range(20, 50))]
    ranking20, ms20 = out[20]
    ranking50, ms50 = out[50]
    assert ranking20[:20] == sorted(range(20), key=lambda p: scores[p], reverse=True)
    assert ranking20[20:] == list(range(20, 100)), "ranks below depth keep dense order"
    assert ranking50[:50] == sorted(range(50), key=lambda p: scores[p], reverse=True)
    assert ranking50[50:] == list(range(50, 100))
    assert 0 <= ms20 <= ms50, "latency is cumulative"


def test_incremental_equals_single_call_including_ties():
    scores = {pid: pid % 3 for pid in range(60)}  # many ties
    incremental = incremental_rerank(RecordingReranker(scores), "q", _candidates(60), [20, 50])
    single = incremental_rerank(RecordingReranker(scores), "q", _candidates(60), [50])
    assert incremental[50][0] == single[50][0]


def test_depth_beyond_candidates_is_safe():
    out = incremental_rerank(RecordingReranker({0: 1.0, 1: 2.0}), "q", _candidates(2), [20, 50])
    assert out[20][0] == out[50][0] == [1, 0]


def test_parse_depths_requires_served_depth_and_bounds():
    assert parse_depths("50,20", 20, 100) == [20, 50]
    with pytest.raises(ValueError, match="served depth"):
        parse_depths("10,50", 20, 100)
    with pytest.raises(ValueError, match="exceeds"):
        parse_depths("20,200", 20, 100)
    with pytest.raises(ValueError):
        parse_depths("0,20", 20, 100)


def test_comparisons_include_served_vs_dense_and_hybrid_rerank():
    systems, comparisons = systems_and_comparisons([20, 50], 20)
    served = "dense+rerank(depth=20)"
    assert systems[-2:] == [served, "dense+rerank(depth=50)"]
    assert (served, "dense") in comparisons
    assert (served, "hybrid+rerank") in comparisons
    assert ("dense+rerank(depth=50)", served) in comparisons


class FakeRetriever:
    def __init__(self, order):
        self.order = order

    def retrieve(self, query, top_k):
        return [{"passage_id": p, "passage_text": str(p), "score": 0.0}
                for p in self.order[:top_k]]


def test_evaluate_and_summarize_wire_all_systems():
    from evaluation.run_eval import evaluate_query, summarize

    dense = FakeRetriever(list(range(100)))
    bm25 = FakeRetriever(list(range(99, -1, -1)))
    reranker = RecordingReranker({pid: -pid if pid != 30 else 1000 for pid in range(100)})
    systems, comparisons = systems_and_comparisons([20, 50], 20)

    records = [
        evaluate_query(qid, "q", {30}, bm25, dense, reranker, 100, 10, 60, [20, 50])
        for qid in range(5)
    ]
    summary = summarize(records, 200, 42, systems, comparisons)

    assert set(summary["systems"]) == set(systems)
    # pid 30 is at dense rank 31: reranking depth 50 lifts it to rank 1,
    # depth 20 cannot reach it.
    assert summary["systems"]["dense+rerank(depth=50)"]["mrr@10"]["mean"] == 1.0
    assert summary["systems"]["dense+rerank(depth=20)"]["mrr@10"]["mean"] == 0.0
    assert "dense+rerank(depth=20) vs dense" in summary["comparisons"]
    stage = records[0]["stage_ms"]
    assert stage["dense_rerank_cumulative_ms@20"] <= stage["dense_rerank_cumulative_ms@50"]


def test_served_default_matches_tune_selection():
    """Serving default must be the config selected on tune."""
    import json
    from pathlib import Path

    from app.config import Settings
    from evaluation.run_eval import SERVED_SOURCE

    chosen = json.loads(Path(SERVED_SOURCE).read_text(encoding="utf-8"))["chosen"]
    fields = Settings.model_fields
    assert chosen["family"] == "dense+rerank" and fields["retrieval_mode"].default == "dense"
    assert chosen["rerank_depth"] == fields["rerank_depth"].default == 20
