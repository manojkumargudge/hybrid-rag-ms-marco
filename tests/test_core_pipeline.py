from unittest.mock import Mock

import numpy as np
import pytest

from app.rag_service import RAGService
from app.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.utils.text import repair_mojibake


def passage(passage_id: int, text: str = "text", score: float = 1.0) -> dict:
    return {"passage_id": passage_id, "passage_text": text, "score": score}


def test_rrf_combines_rankings_and_deduplicates():
    dense = Mock()
    dense.retrieve.return_value = [passage(1), passage(2)]
    bm25 = Mock()
    bm25.retrieve.return_value = [passage(2), passage(3)]

    results = HybridRetriever(dense, bm25, rrf_k=60).retrieve("query", top_k=3, retrieval_k=2)

    assert [item["passage_id"] for item in results] == [2, 1, 3]
    assert len({item["passage_id"] for item in results}) == 3


def test_bm25_tokenization_removes_stopwords_and_punctuation():
    assert BM25Retriever._tokenize("What are the symptoms, of diabetes?") == ["symptoms", "diabetes"]


def test_reranker_sorts_by_cross_encoder_score():
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model = Mock()
    reranker.model.predict.return_value = np.array([0.1, 0.9])

    results = reranker.rerank("query", [passage(1), passage(2)], top_k=1)

    assert [item["passage_id"] for item in results] == [2]


def _service(rerank_depth=20, rerank_top_k=5, candidates=None):
    retriever, reranker, generator = Mock(), Mock(), Mock()
    retriever.retrieve.return_value = (
        [passage(1), passage(2)] if candidates is None else candidates
    )
    reranker.rerank.return_value = [passage(2)]
    generator.generate.return_value = "grounded answer"
    service = RAGService(retriever, reranker, generator,
                         rerank_depth=rerank_depth, rerank_top_k=rerank_top_k)
    return service, retriever, reranker, generator


def test_rag_service_forwards_top_k_and_returns_latency():
    service, retriever, reranker, generator = _service()

    result = service.answer("query", top_k=1)

    retriever.retrieve.assert_called_once_with(query="query", top_k=20)
    reranker.rerank.assert_called_once_with(
        query="query", passages=retriever.retrieve.return_value, top_k=1
    )
    generator.generate.assert_called_once_with(query="query", passages=[passage(2)])
    assert result["answer"] == "grounded answer"
    assert result["citations"] == [passage(2)]
    assert result["latency_ms"] >= 0


def test_rag_service_retrieves_rerank_depth_and_defaults_to_rerank_top_k():
    service, retriever, reranker, _ = _service(rerank_depth=50, rerank_top_k=3)

    service.answer("query")

    retriever.retrieve.assert_called_once_with(query="query", top_k=50)
    assert reranker.rerank.call_args.kwargs["top_k"] == 3


def test_rag_service_skips_rerank_and_generation_without_candidates():
    service, _, reranker, generator = _service(candidates=[])

    result = service.answer("query")

    assert result["citations"] == []
    reranker.rerank.assert_not_called()
    generator.generate.assert_not_called()


@pytest.mark.parametrize("kwargs", [{"rerank_depth": 0}, {"rerank_top_k": 0}])
def test_rag_service_rejects_non_positive_depths(kwargs):
    with pytest.raises(ValueError):
        RAGService(Mock(), Mock(), Mock(), **kwargs)


def test_hybrid_uses_constructor_retrieval_k_by_default():
    dense, bm25 = Mock(), Mock()
    dense.retrieve.return_value = _ranked([1, 2])
    bm25.retrieve.return_value = _ranked([2, 3])

    HybridRetriever(dense, bm25, retrieval_k=7).retrieve("q", top_k=3)
    dense.retrieve.assert_called_once_with(query="q", top_k=7)
    bm25.retrieve.assert_called_once_with(query="q", top_k=7)

    dense.reset_mock()
    HybridRetriever(dense, bm25, retrieval_k=7).retrieve("q", top_k=3, retrieval_k=4)
    dense.retrieve.assert_called_once_with(query="q", top_k=4)

    with pytest.raises(ValueError):
        HybridRetriever(dense, bm25, retrieval_k=0)


def test_rag_service_rejects_empty_query():
    service = RAGService(Mock(), Mock(), Mock())

    with pytest.raises(ValueError, match="Query cannot be empty"):
        service.answer(" ")


def test_rag_service_rejects_non_positive_top_k():
    service = RAGService(Mock(), Mock(), Mock())

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        service.answer("query", top_k=0)


def test_repair_mojibake_preserves_valid_text():
    assert repair_mojibake("chÃ¢teaux") == "châteaux"
    assert repair_mojibake("plain text") == "plain text"


def _ranked(ids):
    return [{"passage_id": i, "passage_text": str(i), "score": 0.0} for i in ids]


def test_weighted_rrf_half_weight_equals_standard_rrf():
    dense, bm25 = Mock(), Mock()
    dense.retrieve.return_value = _ranked([1, 2, 3])
    bm25.retrieve.return_value = _ranked([3, 4, 1])
    standard = HybridRetriever(dense, bm25, rrf_k=60).retrieve("q", top_k=4)
    weighted = HybridRetriever(dense, bm25, rrf_k=60, dense_weight=0.5).retrieve("q", top_k=4)
    assert standard == weighted
    assert standard[0]["score"] == pytest.approx(1 / 61 + 1 / 63)


def test_weighted_rrf_extremes_follow_one_retriever():
    dense, bm25 = Mock(), Mock()
    dense.retrieve.return_value = _ranked([1, 2, 3])
    bm25.retrieve.return_value = _ranked([3, 2, 1])
    only_dense = HybridRetriever(dense, bm25, dense_weight=1.0).retrieve("q", top_k=3)
    only_bm25 = HybridRetriever(dense, bm25, dense_weight=0.0).retrieve("q", top_k=3)
    assert [r["passage_id"] for r in only_dense] == [1, 2, 3]
    assert [r["passage_id"] for r in only_bm25] == [3, 2, 1]
    with pytest.raises(ValueError):
        HybridRetriever(dense, bm25, dense_weight=1.5)
