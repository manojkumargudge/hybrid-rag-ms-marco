"""Manual end-to-end smoke test of the served pipeline.

Uses the exact service the API builds (create_rag_service), so it honours
RETRIEVAL_MODE, RERANK_DEPTH and the other settings. Needs the artifacts,
the models, and GROQ_API_KEY (one real LLM call).

    python -m scripts.manual.smoke_rag_service ["your question"]
"""

import sys

from app.config import settings
from app.main import create_rag_service


def main() -> None:
    # LLM answers and MS MARCO passages contain characters a Windows console
    # code page (cp1252) cannot encode.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    query = sys.argv[1] if len(sys.argv) > 1 else "what are the symptoms of diabetes?"
    print(f"retrieval_mode={settings.retrieval_mode} rerank_depth={settings.rerank_depth} "
          f"rerank_top_k={settings.rerank_top_k}")

    result = create_rag_service().answer(query)

    print(f"\nQUERY: {query}\n\nANSWER:\n{result['answer']}\n")
    for rank, citation in enumerate(result["citations"], start=1):
        print(f"[{rank}] passage {citation['passage_id']} "
              f"(score {citation['score']:.3f}): {citation['passage_text'][:200]}")
    print(f"\nlatency_ms={result['latency_ms']}")

    ids = [c["passage_id"] for c in result["citations"]]
    assert result["answer"], "empty answer"
    assert len(ids) == len(set(ids)), "duplicate citations"
    assert len(ids) <= settings.rerank_top_k


if __name__ == "__main__":
    main()
