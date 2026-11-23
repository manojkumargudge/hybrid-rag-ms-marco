"""End-to-end: HTTP request -> real create_rag_service wiring -> answer.

Runs in both retrieval modes. Real: manifest verification, FAISS, BM25
(rank_bm25 pickle, hybrid mode), RRF fusion, reranker sorting, prompt
building, API schema. Stubbed (no network/models): the sentence-transformer
(HashEncoder), the cross-encoder scorer, and the Groq client.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.artifacts import sha256_file
from app.config import Settings, settings
from app.main import app
from scripts.build_indices import build_bm25, build_dense
from scripts.prepare_data import prepare
from tests.msmarco_fixture import TOPICS, HashEncoder, write_raw_fixture

MODES = ["dense", "hybrid"]


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
def built(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "v3"
    write_raw_fixture(raw)
    prepare(raw, out, seed=42, n_eval=4, n_tune=2, hard_per_query=3,
            target_size=40, created_at="2026-01-01T00:00:00+00:00")
    build_bm25(out)
    build_dense(out, HashEncoder(), "hash-encoder", None, batch_size=8)
    return out


@pytest.fixture
def service(built, monkeypatch, request):
    """Fixture artifacts wired into the app; mode from parametrize (default dense)."""
    mode = getattr(request, "param", "dense")
    out = built
    groq_client = Mock()
    def completion(**kwargs):
        tag = next(line for line in kwargs["messages"][1]["content"].splitlines()
                   if line.startswith("[pid:"))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=f"Grounded answer. {tag}"))]
        )

    groq_client.chat.completions.create.side_effect = completion
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
        "trusted_manifest_path": out.parent / "trusted.json",
        "passages_path": out / "passages.parquet",
        "dense_index_path": out / "dense_index.faiss",
        "bm25_index_path": out / "bm25/bm25_index.pkl",
        "bm25_corpus_path": out / "bm25/tokenized_corpus.pkl",
        "groq_api_key": "test-key",
    }.items():
        monkeypatch.setattr(settings, name, str(value))
    (out.parent / "trusted.json").write_bytes((out / "manifest.json").read_bytes())
    monkeypatch.setattr(settings, "retrieval_mode", mode)
    monkeypatch.setattr(app.state, "rag_service", None)
    monkeypatch.setattr(app.state, "draining", False)

    yield out, groq_client, mode
    app.state.rag_service = None


def _first_eval_query(out):
    qrels = pd.read_csv(out / "qrels_eval.tsv", sep="\t", header=None)
    qid, relevant_pid = int(qrels.iloc[0, 0]), int(qrels.iloc[0, 2])
    return TOPICS[qid - 1000], relevant_pid


@pytest.mark.parametrize("service", MODES, indirect=True)
def test_query_end_to_end_on_fixture(client, service):
    out, groq_client, mode = service
    query, relevant_pid = _first_eval_query(out)

    response = client.post("/api/v1/query", json={"query": query, "top_k": 3})

    assert response.status_code == 200, response.text
    body = response.json()
    citation_ids = [c["passage_id"] for c in body["citations"]]
    assert body["answer"] == f"Grounded answer. [pid:{citation_ids[0]}]"
    assert len(citation_ids) == 3
    assert citation_ids[0] == relevant_pid
    corpus_ids = set(pd.read_parquet(out / "passages.parquet")["passage_id"])
    assert set(citation_ids) <= corpus_ids, "citations must exist in served corpus"

    prompt = groq_client.chat.completions.create.call_args.kwargs["messages"][1]
    assert body["citations"][0]["passage_text"] in prompt["content"]
    assert f"[pid:{citation_ids[0]}]" in prompt["content"]

    meta = client.get("/metadata").json()
    assert meta["retrieval_mode"] == mode
    assert meta["rerank_depth"] == settings.rerank_depth


def test_defaults_are_dense_depth_20_on_v3():
    """Served default = config chosen on tune; corpus = the evaluated v3."""
    fields = Settings.model_fields
    assert fields["retrieval_mode"].default == "dense"
    assert fields["rerank_depth"].default == 20
    assert "hybrid_top_k" not in fields
    for name in ("manifest_path", "passages_path", "dense_index_path",
                 "bm25_index_path", "bm25_corpus_path"):
        assert fields[name].default.startswith("artifacts/v3/"), name
    assert fields["index_version"].default == fields["data_version"].default == "v3"


@pytest.mark.parametrize("service", MODES, indirect=True)
def test_ready_is_200_with_fixture_artifacts(anon_client, service):
    _, groq_client, _ = service
    assert anon_client.get("/ready").status_code == 200
    initialized = app.state.rag_service
    assert initialized is not None
    assert anon_client.get("/ready").status_code == 200
    assert app.state.rag_service is initialized
    groq_client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize(("service", "filename"), [
    ("dense", "dense_index.faiss"),
    ("dense", "passages.parquet"),
    ("hybrid", "bm25/bm25_index.pkl"),
    ("hybrid", "bm25/tokenized_corpus.pkl"),
], indirect=["service"])
def test_startup_rejects_tampering_even_with_updated_mounted_manifest(service, filename):
    out, groq_client, _ = service
    file = out / filename
    file.write_bytes(b"tampered")
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][filename] = sha256_file(file)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="RAG service initialization failed"), TestClient(app):
        pytest.fail("Startup accepted a modified artifact.")
    assert app.state.rag_service is None
    groq_client.chat.completions.create.assert_not_called()


def test_lifespan_loads_and_releases_service(service):
    _, groq_client, _ = service
    with TestClient(app) as running:
        assert app.state.rag_service is not None
        assert running.get("/ready").status_code == 200
        groq_client.close.assert_not_called()
    groq_client.close.assert_called_once_with()
    assert app.state.rag_service is None


@pytest.mark.parametrize("service", ["hybrid"], indirect=True)
def test_bm25_deserializes_verified_snapshot_if_path_changes(service, monkeypatch):
    from app.retrieval import bm25_retriever

    read_verified = bm25_retriever.read_verified_artifact

    def read_then_replace(manifest_path, path, **kwargs):
        verified = read_verified(manifest_path, path, **kwargs)
        path.write_bytes(b"replaced after verification")
        return verified

    monkeypatch.setattr(bm25_retriever, "read_verified_artifact", read_then_replace)
    with TestClient(app) as running:
        assert running.get("/ready").status_code == 200


def test_open_circuit_returns_same_ranked_passages_without_calling_groq(client, service, monkeypatch):
    out, groq_client, _ = service
    monkeypatch.setattr(settings, "groq_breaker_failure_threshold", 1)
    monkeypatch.setattr(settings, "groq_max_retries", 0)
    query, _ = _first_eval_query(out)
    payload = {"query": query, "top_k": 3}
    healthy = client.post("/api/v1/query", json=payload)
    assert healthy.status_code == 200
    groq_client.chat.completions.create.side_effect = TimeoutError()
    failed = client.post("/api/v1/query", json=payload)
    skipped = client.post("/api/v1/query", json=payload)
    assert failed.status_code == skipped.status_code == 200
    assert failed.json()["error"] == "generation_failed"
    assert skipped.json()["error"] == "generation_circuit_open"
    assert skipped.json()["answer"] is None
    assert skipped.json()["citations"] == failed.json()["citations"] == healthy.json()["citations"]
    assert groq_client.chat.completions.create.call_count == 2


@pytest.mark.parametrize("service", MODES, indirect=True)
def test_tampered_dense_index_makes_query_503(client, service):
    out, _, _ = service
    (out / "dense_index.faiss").write_bytes(b"tampered")

    assert client.get("/ready").status_code == 503
    response = client.post("/api/v1/query", json={"query": TOPICS[0]})

    assert response.status_code == 503
    assert response.json()["detail"] == "RAG service is not ready."


@pytest.mark.parametrize(("service", "expected"), [("hybrid", 503), ("dense", 200)],
                         indirect=["service"])
def test_tampered_bm25_only_matters_in_hybrid_mode(client, service, expected):
    out, _, _ = service
    (out / "bm25/bm25_index.pkl").write_bytes(b"tampered")

    assert client.get("/ready").status_code == expected
    response = client.post("/api/v1/query", json={"query": TOPICS[0]})

    assert response.status_code == expected, response.text


def test_dense_mode_works_without_bm25_files(client, service):
    out, _, mode = service
    assert mode == "dense"
    (out / "bm25/bm25_index.pkl").unlink()
    (out / "bm25/tokenized_corpus.pkl").unlink()

    assert client.get("/ready").status_code == 200
    response = client.post("/api/v1/query", json={"query": TOPICS[0]})
    assert response.status_code == 200, response.text


def test_opt_in_reranker_batch_size_reaches_model(client, service, monkeypatch):
    out, _, _ = service
    original_predict = OverlapCrossEncoder().predict
    predict = Mock(side_effect=lambda pairs, **_: original_predict(pairs))
    monkeypatch.setattr(OverlapCrossEncoder, "predict", predict)
    monkeypatch.setattr(settings, "rerank_batch_size", 8)
    query, relevant_pid = _first_eval_query(out)
    response = client.post("/api/v1/query", json={"query": query, "top_k": 3})
    assert response.status_code == 200, response.text
    assert response.json()["citations"][0]["passage_id"] == relevant_pid
    assert predict.call_args.kwargs == {"batch_size": 8}
    assert len(predict.call_args.args[0]) == settings.rerank_depth


@pytest.mark.parametrize("service", MODES, indirect=True)
def test_tuned_confidence_gate_is_dense_only(client, service, monkeypatch):
    out, groq_client, mode = service
    monkeypatch.setattr(settings, "dense_min_rerank_score", 1000)
    query, _ = _first_eval_query(out)
    response = client.post("/api/v1/query", json={"query": query, "top_k": 3})
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["citations"]) == 3
    if mode == "dense":
        assert body["answer"] is None
        assert body["error"] == "low_retrieval_confidence"
        groq_client.chat.completions.create.assert_not_called()
    else:
        assert body["answer"] is not None
        assert body["error"] is None
        groq_client.chat.completions.create.assert_called_once()


@pytest.mark.parametrize("service", ["hybrid"], indirect=True)
def test_hybrid_ready_requires_bm25_files(client, service):
    out, _, _ = service
    (out / "bm25/bm25_index.pkl").unlink()
    assert client.get("/ready").status_code == 503


@pytest.mark.parametrize("loader", [
    "app.retrieval.dense_retriever.SentenceTransformer",
    "app.reranking.cross_encoder_reranker.CrossEncoder",
])
def test_ready_rejects_unavailable_model_then_recovers(anon_client, service, loader, monkeypatch):
    with monkeypatch.context() as unavailable:
        unavailable.setattr(loader, Mock(side_effect=TypeError("model unavailable")))
        assert anon_client.get("/ready").status_code == 503
        assert app.state.rag_service is None

    assert anon_client.get("/ready").status_code == 200


def test_hybrid_dense_weight_default_and_wiring(service, monkeypatch):
    """HYBRID_DENSE_WEIGHT defaults to 0.5 (standard RRF) and reaches the
    HybridRetriever that create_rag_service builds; out-of-range is rejected."""
    from pydantic import ValidationError

    from app.main import create_rag_service
    from app.retrieval.hybrid_retriever import HybridRetriever

    assert Settings.model_fields["hybrid_dense_weight"].default == 0.5
    with pytest.raises(ValidationError):
        Settings(hybrid_dense_weight=1.5)

    monkeypatch.setattr(settings, "retrieval_mode", "hybrid")
    monkeypatch.setattr(settings, "hybrid_dense_weight", 0.8)
    retriever = create_rag_service().retriever
    assert isinstance(retriever, HybridRetriever)
    assert retriever.dense_scale == pytest.approx(1.6)
    assert retriever.bm25_scale == pytest.approx(0.4)
    assert retriever.retrieval_k == settings.retrieval_k
