"""Prepare an offline, extractive 30-query spot check for manual labeling."""

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd

from app.artifacts import verify_artifacts
from app.generation.citations import has_returned_citations
from app.utils.text import repair_mojibake

REFERENCE = Path("eval/results/20261118-v3-0d90c58/per_query.jsonl")
ARTIFACTS = Path("artifacts/v3")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing review")
    corpus_path = ARTIFACTS / "passages.parquet"
    qrels_path = ARTIFACTS / "qrels_eval.tsv"
    verify_artifacts(
        ARTIFACTS / "manifest.json", corpus_path, qrels_path,
        trusted_manifest_path="manifests/v3.json",
    )
    relevance: dict[int, set[int]] = {}
    with qrels_path.open(encoding="utf-8", newline="") as handle:
        for qid, _, pid, grade in csv.reader(handle, delimiter="\t"):
            if int(grade) > 0:
                relevance.setdefault(int(qid), set()).add(int(pid))
    with REFERENCE.open(encoding="utf-8") as handle:
        records = [json.loads(next(handle)) for _ in range(30)]
    rankings = [record["systems"]["dense+rerank(depth=20)"]["top10"][:5] for record in records]
    corpus = pd.read_parquet(
        corpus_path, columns=["passage_id", "passage_text"],
        filters=[("passage_id", "in", [ranking[0] for ranking in rankings])],
    ).set_index("passage_id")["passage_text"]
    rows = []
    for record, ranking in zip(records, rankings, strict=True):
        qid, pid = record["qid"], ranking[0]
        source = repair_mojibake(str(corpus[pid]))
        source = " ".join(source.split())
        excerpt = re.split(r"(?<=[.!?])\s+", source, maxsplit=1)[0]
        answer = f"{excerpt} [pid:{pid}]"
        rows.append({
            "qid": qid, "query": record["query"], "returned_top5": ranking,
            "official_relevant_ids": sorted(relevance[qid]),
            "cited_passage_id": pid, "cited_source": source, "offline_candidate": answer,
            "citation_check_passed": has_returned_citations(answer, [
                {"passage_id": returned} for returned in ranking
            ]),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
