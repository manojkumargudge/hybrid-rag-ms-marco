"""Compare API top-10 citations with the eval harness's served-system top-10.

    python -m scripts.manual.parity_check <results_dir> [n] [base_url]

Sends the first n eval-split queries (default 50) to POST /api/v1/query with
top_k=10 and compares passage ids, in order, to per_query.jsonl. Sends the
API key from the API_KEY environment variable (matching the server's API_KEY).
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

HEADERS = {"X-API-Key": os.environ.get("API_KEY", "")}


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **HEADERS},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def main() -> None:
    results = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    base = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8010"

    metrics = json.loads((results / "metrics.json").read_text(encoding="utf-8"))
    served = metrics["served"]["system"]
    meta_request = urllib.request.Request(f"{base}/metadata", headers=HEADERS)
    with urllib.request.urlopen(meta_request, timeout=30) as response:
        meta = json.loads(response.read())
    print(f"API: retrieval_mode={meta['retrieval_mode']} rerank_depth={meta['rerank_depth']}; "
          f"harness served system: {served}")

    rows = [json.loads(line) for line in (results / "per_query.jsonl").open(encoding="utf-8")]
    matches, mismatches = 0, []
    for row in rows[:n]:
        body = post(f"{base}/api/v1/query", {"query": row["query"], "top_k": 10})
        api_ids = [c["passage_id"] for c in body["citations"]]
        expected = row["systems"][served]["top10"]
        if api_ids == expected:
            matches += 1
        else:
            first = next((i for i, (a, b) in enumerate(zip(api_ids, expected)) if a != b),
                         min(len(api_ids), len(expected)))
            mismatches.append({"qid": row["qid"], "first_diff_rank": first + 1,
                               "api": api_ids, "harness": expected})
    print(f"exact top-10 match: {matches}/{n}")
    for item in mismatches:
        print(json.dumps(item))


if __name__ == "__main__":
    main()
