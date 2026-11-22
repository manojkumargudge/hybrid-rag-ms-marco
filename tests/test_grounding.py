from unittest.mock import Mock

import pytest

from app.generation.citations import has_returned_citations
from app.rag_service import RAGService
from evaluation.calibrate_abstention import select_threshold

PASSAGES = [
    {"passage_id": 123, "passage_text": "First source.", "score": 3.0},
    {"passage_id": 456, "passage_text": "Second source.", "score": 2.0},
]


@pytest.mark.parametrize(("answer", "valid"), [
    ("First claim. [pid:123]\nSecond claim. [pid:456] [pid:123]", True),
    ("Cost is 2.5 units. [pid:123]", True),
    ("Uncited claim.\nCited claim. [pid:123]", False),
    ("Uncited sentence. Cited sentence. [pid:123]", False),
    ("Unknown reference. [pid:999] [pid:123]", False),
    ("Unknown [pid:999] hidden in text. [pid:123]", False),
    ("Malformed reference. [pid:0123]", False),
    ("[pid:123] [pid:456]", False),
    ("", False),
])
def test_citation_check_requires_each_line_to_reference_returned_passages(answer, valid):
    assert has_returned_citations(answer, PASSAGES) is valid


@pytest.mark.parametrize(("generated", "cutoff", "expected_error", "provider_calls"), [
    ("Claim. [pid:123]", None, None, 1),
    ("Claim. [pid:123]", 3.0, None, 1),
    ("Claim. [pid:123]", 3.1, "low_retrieval_confidence", 0),
    ("Claim. [pid:999]", None, "invalid_answer_citations", 1),
    ("INSUFFICIENT_EVIDENCE", None, "insufficient_evidence", 1),
])
def test_grounding_keeps_citations_and_skips_provider_below_cutoff(
    generated, cutoff, expected_error, provider_calls,
):
    retriever, reranker, generator = Mock(), Mock(), Mock()
    retriever.retrieve.return_value = PASSAGES
    reranker.rerank.return_value = PASSAGES
    generator.generate.return_value = generated
    service = RAGService(retriever, reranker, generator, min_rerank_score=cutoff)
    result = service.answer("question")
    assert result.get("error") == expected_error
    assert result["answer"] == (None if expected_error else generated)
    assert result["citations"] == PASSAGES
    assert generator.generate.call_count == provider_calls


def test_calibration_maximizes_abstention_within_mrr_loss_budget():
    rows = [
        {"top_score": -3, "rr_at_10": 0.0},
        {"top_score": -1, "rr_at_10": 0.1},
        {"top_score": 1, "rr_at_10": 1.0},
        {"top_score": 2, "rr_at_10": 1.0},
    ]
    result = select_threshold(rows)
    assert result["threshold"] == -1
    assert result["abstention_rate"] == 0.25
    assert result["mrr_at_10_delta"] == 0
    assert result["answered_only_mrr_at_10"] == pytest.approx(0.7)
