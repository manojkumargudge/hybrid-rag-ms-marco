# Architecture decisions

## ADR 1. Serve dense retrieval with top-20 reranking

The default retrieves dense FAISS candidates and reranks the top 20 before returning
the requested passages. Hybrid mode additionally loads BM25 and fuses the dense
and lexical rankings with weighted reciprocal rank fusion. The disjoint TUNE
selection did not show a clear matched-depth gain from hybrid. Depth 20 met the
latency rule of staying within 0.01 MRR@10 of the best TUNE setting. The prespecified
served system reached 0.779 MRR@10 on the published 150,000-passage subset. That
result applies to this subset with sparse qrels. [TUNE selection](../eval/results/20261117-v3-tune-c858705/metrics.json)
and [EVAL results](../eval/results/20261118-v3-0d90c58/metrics.json).

## ADR 2. Pin the MiniLM models and decouple generation

The index uses normalized embeddings from `sentence-transformers/all-MiniLM-L6-v2`.
Reranking uses `cross-encoder/ms-marco-MiniLM-L-6-v2`. Pin both Hugging Face
revisions in configuration and bake them into the runtime image. This avoids
model downloads during startup and keeps the embedding/index pairing
reviewable. Sustained inference is CPU-bound on the measured setup. Batch
size 8 remains opt-in because the speedup was measured on one Windows host.
Groq is a separate provider behind bounded retries and a per-worker circuit
breaker. The offline tests replace it deterministically. [Model configuration](../app/config.py)
and [P4 measurements](../eval/results/20261122-p4-windows/metrics.json).

## ADR 3. Verify artifacts against an image-owned manifest

The published v3 bundle represents a reproducible sample of the full MS MARCO
collection. The downloader checks SHA-256 against the committed manifest.
At startup the worker verifies the required local files against the mounted
manifest and the independently trusted manifest baked into the image. Hybrid
mode deserializes the same verified pickle bytes. Keep artifact mounts
read-only and prevent host-side mutation while workers start. An intentional
rebuild needs a distinct reviewed artifact/image pair. [Artifact checks](../app/artifacts.py),
[BM25 loading](../app/retrieval/bm25_retriever.py), and [v3 manifest](../manifests/v3.json).

## ADR 4. Use a single-host Compose release bundle

Deploy the non-root image by recorded identity, mount artifacts read-only, use
a private runtime env file, and bind the operations profile to loopback.
The container's HTTP health check measures liveness. Release validation also
checks readiness and authenticated metadata. Retain the compatible previous
image and artifacts as a unit. The single service can interrupt traffic during
replacement. TLS gateway behavior, production traffic, and registry recovery
remain outside the isolated rehearsal. [Compose profile](../compose.operations.yml),
[deployment](deployment.md), and [rollback](rollback.md).

## ADR 5. Bound per-worker work and degrade generation explicitly

Keep concurrency, rate limiting, latency metrics, and circuit-breaker state
in process. Admission rejects excess work and retains the slot while synchronous
work finishes. Shutdown drains in-flight work within the container stop grace
period. Failed generation and invalid citations can leave passages available
with a null answer. Citation checks verify reference structure and membership.
The opt-in confidence cutoff was selected only on TUNE. Multiple workers
duplicate model memory and require separate metric scraping. A shared quota
would need additional infrastructure. [Admission](../app/api/routes.py),
[generation and grounding](../app/rag_service.py), and [monitoring](operations.md).
