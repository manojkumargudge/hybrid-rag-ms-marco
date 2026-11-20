"""Serve the real app (uvicorn + create_rag_service) with LLM generation stubbed.

Retrieval, reranking, artifacts and HTTP are exactly as served; only
GroqGenerator.generate is replaced, so no Groq calls are made. For
retrieval parity checks only.

    python -m scripts.manual.serve_no_llm [port]
"""

import sys

import uvicorn

from app.generation.groq_generator import GroqGenerator
from app.main import app


def _no_llm(self, query, passages):
    return "[generation disabled for parity check]"


if __name__ == "__main__":
    GroqGenerator.generate = _no_llm
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8010)
