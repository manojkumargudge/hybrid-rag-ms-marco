"""prepare_data.py + build_indices.py on a tiny MS MARCO-format fixture."""

import json

import pandas as pd
import pytest

from app.artifacts import ArtifactIntegrityError, verify_artifacts
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from scripts.build_indices import build_bm25, build_dense
from scripts.prepare_data import prepare
from tests.msmarco_fixture import TOPICS, HashEncoder, write_raw_fixture

PARAMS = {"seed": 42, "n_eval": 4, "n_tune": 2, "hard_per_query": 3, "target_size": 40}
FIXED_TIME = "2026-01-01T00:00:00+00:00"


def _prepare(raw, out, **overrides):
    return prepare(raw, out, created_at=FIXED_TIME, **{**PARAMS, **overrides})


@pytest.fixture
def built(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "v2"
    write_raw_fixture(raw)
    _prepare(raw, out)
    build_bm25(out)
    build_dense(out, HashEncoder(), "hash-encoder", None, batch_size=8)
    return raw, out


def test_prepare_is_deterministic(tmp_path):
    raw = tmp_path / "raw"
    write_raw_fixture(raw)
    first = _prepare(raw, tmp_path / "a")
    second = _prepare(raw, tmp_path / "b")
    assert first["files"] == second["files"]
    assert first["counts"] == second["counts"]


def test_splits_sources_and_counts(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "v2"
    write_raw_fixture(raw)
    manifest = _prepare(raw, out)

    eval_q = pd.read_csv(out / "queries_eval.tsv", sep="\t", header=None)[0]
    tune_q = pd.read_csv(out / "queries_tune.tsv", sep="\t", header=None)[0]
    assert len(eval_q) == 4 and len(tune_q) == 2
    assert not set(eval_q) & set(tune_q), "eval and tune must be disjoint"

    passages = pd.read_parquet(out / "passages.parquet")
    assert len(passages) == 40
    assert passages["passage_id"].is_monotonic_increasing
    qrels = pd.read_csv(out / "qrels_eval.tsv", sep="\t", header=None)
    relevant = set(passages.loc[passages["source"] == "relevant", "passage_id"])
    assert set(qrels[2]) <= relevant
    assert manifest["counts"]["by_source"]["hard"] > 0
    assert sum(manifest["counts"]["by_source"].values()) == 40
    assert manifest["params"]["seed_queries"] == 42


def test_hard_negatives_exclude_relevant_and_are_bm25_ranked(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "v2"
    write_raw_fixture(raw)
    manifest = _prepare(raw, out, hard_per_query=1)
    passages = pd.read_parquet(out / "passages.parquet")
    hard = set(passages.loc[passages["source"] == "hard", "passage_id"])
    sampled = set()
    for name in ("queries_eval.tsv", "queries_tune.tsv"):
        sampled |= set(pd.read_csv(out / name, sep="\t", header=None)[0])
    # In the fixture, pid 40+i is the only pool passage sharing query i's
    # topic words, so BM25 must rank it first; the relevant pid i is excluded.
    assert hard == {40 + (qid - 1000) for qid in sampled}
    # Relevant passage is the LAST line of every fixture pool.
    assert manifest["top1000_order_check"]["mean_relative_position_of_relevant"] == 1.0


def test_text_is_normalized(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "v2"
    write_raw_fixture(raw)
    _prepare(raw, out, target_size=60)
    text = pd.read_parquet(out / "passages.parquet").set_index("passage_id").loc[59]
    assert text["passage_text"].startswith("café")


def test_refuses_to_overwrite(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "v2"
    write_raw_fixture(raw)
    _prepare(raw, out)
    with pytest.raises(FileExistsError):
        _prepare(raw, out)


def test_build_indices_records_hashes_and_refuses_overwrite(built):
    _, out = built
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert {"bm25/bm25_index.pkl", "dense_index.faiss"} <= manifest["files"].keys()
    assert manifest["indices"]["dense"]["vectors"] == 40
    assert manifest["indices"]["bm25"]["documents"] == 40
    verify_artifacts(out / "manifest.json", *(out / n for n in manifest["files"]))
    with pytest.raises(FileExistsError):
        build_bm25(out)


def test_retrievers_load_built_artifacts(built):
    _, out = built
    bm25 = BM25Retriever(
        out / "bm25/bm25_index.pkl",
        out / "bm25/tokenized_corpus.pkl",
        out / "passages.parquet",
        out / "manifest.json",
    )
    dense = DenseRetriever(
        out / "dense_index.faiss", out / "passages.parquet", model=HashEncoder()
    )
    qrels = pd.read_csv(out / "qrels_eval.tsv", sep="\t", header=None)
    qid, pid = int(qrels.iloc[0, 0]), int(qrels.iloc[0, 2])
    topic = TOPICS[qid - 1000]
    assert bm25.retrieve(topic, top_k=1)[0]["passage_id"] == pid
    assert dense.retrieve(topic, top_k=1)[0]["passage_id"] == pid


def test_tampered_pickle_is_rejected_before_unpickling(built):
    _, out = built
    (out / "bm25/bm25_index.pkl").write_bytes(b"not the pickle we built")
    with pytest.raises(ArtifactIntegrityError, match="sha256 mismatch"):
        BM25Retriever(
            out / "bm25/bm25_index.pkl",
            out / "bm25/tokenized_corpus.pkl",
            out / "passages.parquet",
            out / "manifest.json",
        )


def test_missing_manifest_is_rejected(built):
    _, out = built
    (out / "manifest.json").unlink()
    with pytest.raises(ArtifactIntegrityError, match="Manifest not found"):
        verify_artifacts(out / "manifest.json", out / "passages.parquet")
