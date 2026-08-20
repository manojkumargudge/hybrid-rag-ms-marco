import json
import re
from pathlib import Path

import pandas as pd


CANDIDATE_RESULTS_PATH = Path("eval/candidate_results.txt")
PASSAGES_PATH = Path("data/passages_subset.parquet")
OUTPUT_PATH = Path("eval/candidate_review.json")


def main() -> None:
    text = CANDIDATE_RESULTS_PATH.read_text(encoding="utf-8-sig")

    df = pd.read_parquet(PASSAGES_PATH)

    passage_lookup = dict(
        zip(
            df["passage_id"].astype(int),
            df["passage_text"],
        )
    )

    blocks = re.split(r"(?=q\d{2}:)", text)

    output = []

    for block in blocks:
        query_match = re.match(r"(q\d{2}): (.*)", block)

        if not query_match:
            continue

        query_id = query_match.group(1)
        query = query_match.group(2)

        passage_ids = [
            int(pid)
            for pid in re.findall(r"Passage ID: (\d+)", block)
        ]

        candidates = []

        for passage_id in passage_ids:
            candidates.append(
                {
                    "passage_id": passage_id,
                    "text": passage_lookup.get(passage_id, ""),
                }
            )

        output.append(
            {
                "query_id": query_id,
                "query": query,
                "candidates": candidates,
            }
        )

    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Created {OUTPUT_PATH}")
    print(f"Queries: {len(output)}")
    print(
        f"Candidates: "
        f"{sum(len(item['candidates']) for item in output)}"
    )


if __name__ == "__main__":
    main()