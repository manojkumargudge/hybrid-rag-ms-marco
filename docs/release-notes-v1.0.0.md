# Hybrid RAG v1.0.0 release candidate

**Draft for GitHub Releases.** Fill in the approved source SHA, CI run, and
verified GHCR digest after release authorization.

- Dense FAISS retrieval and top-20 cross-encoder reranking are the served
  defaults. Hybrid mode adds BM25 and weighted reciprocal rank fusion.
- Groq answers reference retrieved passage IDs. Bounded retries and a circuit
  breaker preserve access to passages when generation fails.
- API-key authentication, request limits, structured logs, Prometheus
  metrics, and independent liveness and readiness probes support single-host
  Compose operation.
- The image runs as a non-root user with pinned model snapshots. Stage the
  matching v3 artifact bundle for the 150,000-passage MS MARCO subset and a
  private runtime configuration before startup.

See the [changelog](../CHANGELOG.md), [deployment prerequisites](deployment.md),
and [rollback procedure](rollback.md). The hosted smoke and rollback checks
use a deterministic Groq stub and the published dense artifacts. Live Groq
access and deployment cutover require separate operational verification.

| Release metadata | Pending verification |
|---|---|
| Approved source SHA | Supply the validated main revision |
| Hosted CI | Link the successful run for that revision |
| Image | Record the verified GHCR repository digest |
| Artifacts | [v3 trusted manifest](../manifests/v3.json) |
