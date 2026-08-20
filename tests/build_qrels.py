import json
from pathlib import Path


REVIEW_PATH = Path("eval/candidate_review.json")
QRELS_PATH = Path("eval/qrels.json")


# Manually judged relevant passages from the top-5 candidates.
# Only passages that directly answer or substantially support
# the query are included.
RELEVANT = {
    "q01": [24998, 24999, 191982, 210796],
    "q02": [162991],
    "q03": [43371, 261730],
    "q04": [132184, 132183, 132182, 188054],
    "q05": [7, 2, 3, 8, 1],
    "q06": [50736, 184098, 78060, 312785, 289991],
    "q07": [7370, 7373],
    "q08": [109899, 326129, 326135, 20996, 100660],
    "q09": [182549, 182546, 182545, 182550, 182552],
    "q10": [250032, 59329, 129129],
    "q11": [76362, 295633, 235657],
    "q12": [38872, 172902, 38871],
    "q13": [75274, 164459, 164451],
    "q14": [215319, 155417, 253391],
    "q15": [162216, 292780],
    "q16": [286397, 296100],
    "q17": [19778, 19779, 19780],
    "q18": [221959, 221965, 221960],
    "q19": [308505, 308499],
    "q20": [118210, 191056, 217419],
}


def main() -> None:
    review = json.loads(
        REVIEW_PATH.read_text(encoding="utf-8")
    )

    query_ids = {item["query_id"] for item in review}

    missing = query_ids - RELEVANT.keys()

    if missing:
        raise ValueError(
            f"Missing relevance judgments for: {sorted(missing)}"
        )

    qrels = []

    for item in review:
        query_id = item["query_id"]
        candidate_ids = {
            candidate["passage_id"]
            for candidate in item["candidates"]
        }

        relevant_ids = RELEVANT[query_id]

        invalid = set(relevant_ids) - candidate_ids

        if invalid:
            raise ValueError(
                f"{query_id}: relevant IDs not in candidates: {invalid}"
            )

        qrels.append(
            {
                "query_id": query_id,
                "relevant_passage_ids": relevant_ids,
            }
        )

    QRELS_PATH.write_text(
        json.dumps(qrels, indent=2),
        encoding="utf-8",
    )

    print("Created eval/qrels.json")
    print(f"Queries: {len(qrels)}")
    print(
        "Relevant judgments:",
        sum(len(x["relevant_passage_ids"]) for x in qrels),
    )


if __name__ == "__main__":
    main()