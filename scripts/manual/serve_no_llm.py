"""Serve the real app (uvicorn + create_rag_service) with LLM generation stubbed.

The direct entry point serves the standard retrieval pipeline for parity checks.
The benchmark factory also supports isolated experiments with each worker.
Both replace GroqGenerator.generate, so no Groq calls are made.

    python -m scripts.manual.serve_no_llm [port]
"""

import os
import sys
from functools import lru_cache
from pathlib import Path

import uvicorn

from app.generation.groq_generator import GroqGenerator
from app.main import app


def _no_llm(self, query, passages):
    return "[generation disabled for parity check]"


def benchmark_app():
    """Create an offline worker with one explicitly selected experiment."""
    import torch

    GroqGenerator.generate = _no_llm
    get_service = app.state.get_rag_service
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    ready = Path(os.environ["BENCHMARK_READY_DIR"]) / f"{os.getpid()}.ready"
    cache = int(os.environ["BENCHMARK_EMBEDDING_CACHE"])

    def initialize():
        service = get_service()
        if not ready.exists():
            if cache:
                original_encode = service.retriever.model.encode

                @lru_cache(maxsize=cache)
                def encode(sentences: tuple[str, ...], **kwargs):
                    return original_encode(list(sentences), **kwargs)

                def cached_encode(sentences: list[str], **kwargs):
                    return encode(tuple(sentences), **kwargs)

                service.retriever.model.encode = cached_encode
            ready.touch()
        return service

    app.state.get_rag_service = initialize
    return app


if __name__ == "__main__":
    GroqGenerator.generate = _no_llm
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8010,
                proxy_headers=False, access_log=False)
