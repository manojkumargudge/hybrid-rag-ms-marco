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


def test_rag_service_forwards_top_k_and_returns_latency():
    hybrid = Mock()
    hybrid.retrieve.return_value = [passage(1), passage(2)]
    reranker = Mock()
    reranker.rerank.return_value = [passage(2)]
    generator = Mock()
    generator.generate.return_value = "grounded answer"
    service = RAGService(hybrid, reranker, generator)

    result = service.answer("query", top_k=1)

    reranker.rerank.assert_called_once_with(query="query", passages=hybrid.retrieve.return_value, top_k=1)
    assert result["answer"] == "grounded answer"
    assert result["citations"] == [passage(2)]
    assert result["latency_ms"] >= 0


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
