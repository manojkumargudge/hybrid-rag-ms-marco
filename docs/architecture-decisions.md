# Architecture decisions

## ADR 1. Serve dense retrieval by default and offer BM25 hybrid retrieval

The default retrieves dense FAISS candidates and reranks the top 20 before returning
the requested passages. Hybrid mode additionally loads BM25 and fuses the dense
and lexical rankings with weighted reciprocal rank fusion. The disjoint TUNE
selection did not show a clear matched-depth gain from hybrid. Depth 20 met the
latency rule of staying within 0.01 MRR@10 of the best TUNE setting. The prespecified
served system reached 0.779 MRR@10 on the published 150,000-passage subset. That
result applies to this subset with sparse qrels. [TUNE selection](../eval/results/20261117-v3-tune-c858705/metrics.json)
and [EVAL results](../eval/results/20261118-v3-0d90c58/metrics.json).
BM25-only also ran in the comparison. Hybrid retains lexical matching as an
operator option and requires extra artifacts, memory, and startup checks.

## ADR 2. Fuse hybrid rankings with weighted reciprocal rank fusion

Combine the rankings by passage ID using the sum of each available
`2 * weight / (k + rank)` contribution, with rank starting at one. The
default weights of one half give standard RRF. Direct score interpolation
would require calibrating FAISS and BM25 scores. RRF avoids that calibration
and discards score magnitudes. Candidate depth, weights, and the rank
constant determine how much influence either retriever has.
[Fusion implementation](../app/retrieval/hybrid_retriever.py).

## ADR 3. Rerank a bounded candidate pool with a pinned cross-encoder

The index uses normalized embeddings from `sentence-transformers/all-MiniLM-L6-v2`.
Reranking uses `cross-encoder/ms-marco-MiniLM-L-6-v2`. Pin both Hugging Face
revisions in configuration and bake them into the runtime image. This avoids
model downloads during startup and keeps the embedding/index pairing
reviewable. Sustained inference is CPU-bound on the measured setup. Batch
size 8 remains opt-in because the speedup was measured on one Windows host.
Returning the first-stage order would avoid pairwise inference. The bounded
cross-encoder stage can reorder those candidates at a fixed depth cost and
cannot recover passages that the first stage missed. [Model configuration](../app/config.py)
and [P4 measurements](../eval/results/20261122-p4-windows/metrics.json).

## ADR 4. Use Groq for grounded generation with retrieval-only fallback

Send the selected passages to Groq and validate that generated claims cite
returned passage IDs. A retrieval-only API would avoid provider latency and
credentials but would omit synthesized answers. Bounded retries and a
per-worker circuit breaker contain provider failure, and unusable generation
can return passages with a null answer. Citation validation checks the
response structure and referenced IDs. Human review is needed to assess
semantic support. Offline verification uses a deterministic provider stub.
[Generation service](../app/rag_service.py).

## ADR 5. Verify artifacts against an image-owned manifest

The published v3 bundle represents a reproducible sample of the full MS MARCO
collection. The downloader checks SHA-256 against the committed manifest.
At startup the worker verifies the required local files against the mounted
manifest and the independently trusted manifest baked into the image. Hybrid
mode deserializes the same verified pickle bytes. Keep artifact mounts
read-only and prevent host-side mutation while workers start. An intentional
rebuild needs a distinct reviewed artifact/image pair. Baking the corpus
into the image would couple application releases to large index layers.
External mounts keep those inputs separately staged and make coordinating the
image and artifact bundle an operator responsibility. [Artifact checks](../app/artifacts.py),
[BM25 loading](../app/retrieval/bm25_retriever.py), and [v3 manifest](../manifests/v3.json).

## ADR 6. Use a single-host Compose release bundle

Deploy the non-root image by recorded identity, mount artifacts read-only, use
a private runtime env file, and bind the operations profile to loopback.
The worker runs without Linux capabilities or privilege escalation.
The container's HTTP health check measures liveness. Release validation also
checks readiness and authenticated metadata. Retain the compatible previous
image and artifacts as a unit. The manual GHCR workflow validates an exact
main SHA, scans the smoke-tested image, and checks its identity by registry
digest before recording it for operators. An orchestrated
rolling rollout could reduce interruption but would add deployment
infrastructure. The single service can interrupt traffic during replacement.
TLS gateway behavior, production traffic, and registry recovery
remain outside the isolated rehearsal. [Compose profile](../compose.operations.yml),
[deployment](deployment.md), and [rollback](rollback.md).

## ADR 7. Separate liveness, readiness, and authenticated observability

Use public liveness and readiness probes, structured request logs, and
authenticated Prometheus metrics. Readiness reflects local pipeline
initialization. Probing Groq on each readiness check would make provider
outages evict workers that can still retrieve passages. Operators need a
separate authenticated query to verify live generation.
Keep concurrency, rate limiting, latency metrics, and circuit-breaker state
in process. Admission rejects excess work and retains the slot while synchronous
work finishes. Shutdown drains in-flight work within the container stop grace
period. Multiple workers
duplicate model memory and require separate metric scraping. A shared quota
would need additional infrastructure. [Admission](../app/api/routes.py),
[generation and grounding](../app/rag_service.py), and [monitoring](operations.md).
