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


def _hard_by_query(out):
    hard = pd.read_csv(out / "hard_negatives.tsv", sep="\t", header=None)
    return {qid: set(group[1]) for qid, group in hard.groupby(0)}


def _pools_and_qrels(raw):
    top = pd.read_csv(raw / "top1000.dev", sep="\t", header=None)
    qrels = pd.read_csv(raw / "qrels.dev.small.tsv", sep="\t", header=None)
    pools = {qid: set(group[1]) for qid, group in top.groupby(0)}
    relevant = {qid: set(group[2]) for qid, group in qrels.groupby(0)}
    return pools, relevant


def test_hard_negatives_are_seeded_sample_of_own_pool(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "v3"
    write_raw_fixture(raw)
    manifest = _prepare(raw, out)
    pools, relevant = _pools_and_qrels(raw)
    hard = _hard_by_query(out)

    sampled = set()
    for name in ("queries_eval.tsv", "queries_tune.tsv"):
        sampled |= set(pd.read_csv(out / name, sep="\t", header=None)[0])
    assert set(hard) == sampled
    for qid, pids in hard.items():
        assert len(pids) == 3
        assert pids <= pools[qid] - relevant[qid]

    source = pd.read_parquet(out / "passages.parquet").set_index("passage_id")["source"]
    all_hard = set().union(*hard.values())
    assert set(source[source == "hard"].index) <= all_hard
    assert "top1000.dev" in manifest["params"]["hard_negative_rule"]
    # Relevant passage is the LAST line of every fixture pool.
    assert manifest["top1000_order_check"]["mean_relative_position_of_relevant"] == 1.0


def test_hard_negatives_ignore_file_order_and_depend_on_seed(tmp_path):
    raw = tmp_path / "raw"
    write_raw_fixture(raw)
    _prepare(raw, tmp_path / "a")
    baseline = _hard_by_query(tmp_path / "a")

    lines = (raw / "top1000.dev").read_text(encoding="utf-8").splitlines(keepends=True)
    (raw / "top1000.dev").write_text("".join(reversed(lines)), encoding="utf-8")
    _prepare(raw, tmp_path / "b")
    assert _hard_by_query(tmp_path / "b") == baseline

    # Another seed also changes which queries are sampled; compare only the
    # queries both runs share.
    _prepare(raw, tmp_path / "c", seed=7)
    other = _hard_by_query(tmp_path / "c")
    shared = baseline.keys() & other.keys()
    assert shared
    assert any(baseline[qid] != other[qid] for qid in shared)


def test_every_qrel_passage_is_in_corpus(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "v3"
    write_raw_fixture(raw)
    manifest = _prepare(raw, out)
    counts = manifest["counts"]
    assert counts["qrel_passages_in_corpus"] == counts["qrel_passages_unique"] > 0


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
