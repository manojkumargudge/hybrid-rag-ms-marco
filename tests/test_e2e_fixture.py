"""End-to-end: HTTP request -> real create_rag_service wiring -> answer.

Real: manifest verification, BM25 (rank_bm25 pickle), FAISS, RRF fusion,
reranker sorting, prompt building, API schema. Stubbed (no network/models):
the sentence-transformer (HashEncoder), the cross-encoder scorer, and the
Groq client.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from app.config import settings
from app.main import app
from scripts.build_indices import build_bm25, build_dense
from scripts.prepare_data import prepare
from tests.msmarco_fixture import TOPICS, HashEncoder, write_raw_fixture


class OverlapCrossEncoder:
    """Scores (query, passage) pairs by shared-token count."""

    def __init__(self, *_, **__):
        pass

    def predict(self, pairs):
        return np.array(
            [len(set(q.lower().split()) & set(p.lower().split())) for q, p in pairs],
            dtype=float,
        )


@pytest.fixture
def fixture_service(tmp_path, monkeypatch):
    raw, out = tmp_path / "raw", tmp_path / "v3"
    write_raw_fixture(raw)
    prepare(raw, out, seed=42, n_eval=4, n_tune=2, hard_per_query=3,
            target_size=40, created_at="2026-01-01T00:00:00+00:00")
    build_bm25(out)
    build_dense(out, HashEncoder(), "hash-encoder", None, batch_size=8)

    groq_client = Mock()
    groq_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Grounded answer."))]
    )
    monkeypatch.setattr(
        "app.retrieval.dense_retriever.SentenceTransformer",
        lambda *a, **k: HashEncoder(),
    )
    monkeypatch.setattr(
        "app.reranking.cross_encoder_reranker.CrossEncoder", OverlapCrossEncoder
    )
    monkeypatch.setattr("app.generation.groq_generator.Groq", lambda **_: groq_client)

    for name, value in {
        "manifest_path": out / "manifest.json",
        "passages_path": out / "passages.parquet",
        "dense_index_path": out / "dense_index.faiss",
        "bm25_index_path": out / "bm25/bm25_index.pkl",
        "bm25_corpus_path": out / "bm25/tokenized_corpus.pkl",
        "groq_api_key": "test-key",
    }.items():
        monkeypatch.setattr(settings, name, str(value))
    monkeypatch.setattr(app.state, "rag_service", None)

    yield out, groq_client
    app.state.rag_service = None


def test_query_end_to_end_on_fixture(client, fixture_service):
    out, groq_client = fixture_service
    qrels = pd.read_csv(out / "qrels_eval.tsv", sep="\t", header=None)
    qid, relevant_pid = int(qrels.iloc[0, 0]), int(qrels.iloc[0, 2])
    query = TOPICS[qid - 1000]

    response = client.post("/api/v1/query", json={"query": query, "top_k": 3})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == "Grounded answer."
    citation_ids = [c["passage_id"] for c in body["citations"]]
    assert len(citation_ids) == 3
    assert citation_ids[0] == relevant_pid
    corpus_ids = set(pd.read_parquet(out / "passages.parquet")["passage_id"])
    assert set(citation_ids) <= corpus_ids, "citations must exist in served corpus"

    prompt = groq_client.chat.completions.create.call_args.kwargs["messages"][1]
    assert body["citations"][0]["passage_text"] in prompt["content"]


def test_default_serving_paths_are_v3():
    """Served corpus must be the evaluated one (artifacts/v3)."""
    from app.config import Settings

    fields = Settings.model_fields
    for name in (
        "manifest_path",
        "passages_path",
        "dense_index_path",
        "bm25_index_path",
        "bm25_corpus_path",
    ):
        assert fields[name].default.startswith("artifacts/v3/"), name
    assert fields["index_version"].default == fields["data_version"].default == "v3"


def test_ready_is_200_with_fixture_artifacts(client, fixture_service):
    assert client.get("/ready").status_code == 200


def test_tampered_artifact_makes_query_503(client, fixture_service):
    out, _ = fixture_service
    (out / "bm25/bm25_index.pkl").write_bytes(b"tampered")

    response = client.post("/api/v1/query", json={"query": TOPICS[0]})

    assert response.status_code == 503
    assert response.json()["detail"] == "RAG service is not ready."
