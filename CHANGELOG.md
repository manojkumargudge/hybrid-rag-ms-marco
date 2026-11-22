# Changelog

## Unreleased (v1.0.0 candidate)

- P1 adds API-key enforcement on protected routes and a query-length cap.
  CI runs dependency auditing and scoped Bandit checks.
- P2 adds structured request and stage logs, authenticated Prometheus metrics,
  and readiness checks for the required artifacts and offline models.
- P3 adds per-worker query admission limits, graceful shutdown, a Groq circuit
  breaker with retrieval-only fallback, startup SHA-256 checks, and Compose
  CPU and memory limits.
- P4 records CPU load benchmarks for dense retrieval and top-20 reranking.
  Opt-in reranker batch size 8 preserved the ordered top 10 on 200 queries.
- P5 checks claim-line citations against returned passage IDs and adds a
  TUNE-calibrated, opt-in confidence cutoff for dense retrieval. The 30-query
  faithfulness review uses offline excerpts.
- P6 verifies artifact-backed Docker startup, health and readiness, exact
  retrieval ordering, rejected missing or corrupt artifacts, and SIGTERM
  draining with a deterministic Groq stub. CI enforces application coverage
  and checks OpenAPI drift.
- P7 documents single-host deployment, monitoring, incident response, and
  rollback. Hosted CI rehearses failed readiness and restores a recorded
  image with the verified dense artifacts.

The hosted validation uses offline generation. See the [release checklist](docs/release.md)
for publication prerequisites and remaining production checks.
