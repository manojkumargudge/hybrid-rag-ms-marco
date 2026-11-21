# Production checklist

| Phase | Item | Status | Evidence |
|---|---|---|---|
| Step 0 | Record branch, main SHA, open PRs, PR #6, local branches, main CI and test count | Done | Clean `main` (only local branch before P1) at `b16cef56ac1e09b75d4acc402cbc53029ea92409`. No open PRs. [PR #6](https://github.com/manojkumargudge/hybrid-rag-ms-marco/pull/6) merged with README and MIT license. [CI #45](https://github.com/manojkumargudge/hybrid-rag-ms-marco/actions/runs/44735929786) green, 129 tests passed. |
| Step 0 | Create this tracker and update it each phase | Done | This file |
| P1 | Environment-backed `X-API-Key` auth, constant-time comparison, fail closed, public health/readiness, 401 and no raw keys in logs | Done | `app/auth.py`, `app/middleware.py`, `tests/test_auth.py`, `tests/test_hardening.py` |
| P1 | pip-audit, Bandit, Trivy image scan and Gitleaks in CI | Partial | Main runs pip-audit and Bandit. Trivy and Gitleaks are in PR #12. |
| P1 | Dependabot | Todo | Configuration is in PR #12. |
| P1 | Scoped security exceptions with reason, expiry and failing CI when expired | Done | `.github/security-ignore.json`, `scripts/security_gate.py`, `tests/test_security_gate.py` |
| P1 | Query length and request body byte limits | Partial | Main validates query length. The byte limit is in PR #12. |
| P1 | Untrusted retrieved passages and prompt-injection regression | Partial | `app/generation/groq_generator.py`, mocked provider contract in PR #12 |
| P2 | JSON logs with request ID and no query, key or passage contents | Done | `app/main.py`, `app/middleware.py`, privacy regression in `tests/test_hardening.py` |
| P2 | Prometheus request count, errors, per-stage latency, Groq failures/timeouts and rate-limit hits | Done | `app/metrics.py`, retry and rate-limit assertions in `tests/test_hardening.py` |
| P2 | Readiness verifies index and models, with BM25 only in hybrid mode | Done | `app/main.py`, artifact and model readiness checks in `tests/test_e2e_fixture.py` |
| P2 | Document metric names in README | Done | [README operations](../README.md#operations) |
| P3 | Graceful shutdown and concurrency limit with 429/503 backpressure | Todo | — |
| P3 | Groq circuit breaker and flagged passage-only fallback | Todo | — |
| P3 | Startup refuses artifact SHA-256 mismatch against `manifests/v3.json` | Todo | — |
| P3 | Docker Compose CPU and memory limits | Todo | — |
| P3 | Fuzz bad inputs and assert no 500 | Todo | — |
| P3 | Keep in-memory rate limiter, document Redis for multiple workers and assess optional Redis support | Todo | — |
| P4 | `scripts/load_test.py` exercises served dense + rerank (20) only | Todo | — |
| P4 | Report p50/p95/p99, requests/s at 1, 5 and 20 clients, memory and cold start | Todo | — |
| P4 | Measure query embedding LRU, reranker batching and Uvicorn workers individually | Todo | — |
| P4 | Keep clear gains with identical top 10 on 200 queries | Todo | — |
| P4 | Commit measured results under `eval/results/` without rerunning accuracy evaluation | Todo | — |
| P5 | Strip or flag claims without citations to returned passage IDs | Todo | — |
| P5 | Tune low-confidence abstention on TUNE only and report abstention rate and tune MRR impact | Todo | — |
| P5 | Report faithfulness spot check on 30 queries with manual labels from official qrels | Todo | — |
| P6 | CI coverage gate and measured coverage | Todo | — |
| P6 | Strict mypy and Ruff where passing | Todo | — |
| P6 | CI Docker build, start, readiness and one query with mocked Groq | Todo | — |
| P6 | OpenAPI export and drift test | Todo | — |
| P6 | Document 400, 401, 422, 429 and 503 error shapes | Todo | — |
| P7 | Slim non-root image and HEALTHCHECK | Todo | — |
| P7 | GHCR push on main, `v1.0.0` tag and changelog | Todo | — |
| P7 | Production Compose with limits and complete `.env.example` | Todo | — |
| P7 | Runbook for startup, API and Groq key rotation, index rebuild, rollback and common errors | Todo | — |
| P7 | Mermaid architecture diagram | Todo | — |
| P7 | ADRs for dense + rerank over hybrid, 150k-passage subset and v3 corpus | Todo | — |
| P7 | Document 150k-passage subset versus the 8.8M corpus, CPU latency, in-memory limits and single-node operation | Todo | — |
| P7 | Verify every README step in a separate fresh clone, record times and gaps, and fix README | Todo | — |
| P8 | Map each README claim to code or results in `docs/evidence.md` | Todo | — |
| P8 | Reconcile README with `eval/results/20261118-v3-0d90c58` and report served MRR@10 0.779 on a 150k-passage subset | Todo | — |
| P8 | Final report of completed items, skips and known risks | Todo | — |

## P1 verification

In PR #12, the Trivy scan gates high and critical findings. The temporary Debian
exceptions apply only to the package versions in the ignore file and expire
on 2026-12-21. A reported upstream fix requires a fresh decision or upgrade.
Historical Gitleaks false positives match individual commit fingerprints.
