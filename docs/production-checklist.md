# Production checklist

| Phase | Item | Status | Evidence |
|---|---|---|---|
| Step 0 | Record branch, main SHA, open PRs, PR #6, local branches, main CI and test count | Done | Clean `main` (only local branch before P1) at `b16cef56ac1e09b75d4acc402cbc53029ea92409`. No open PRs. [PR #6](https://github.com/manojkumargudge/hybrid-rag-ms-marco/pull/6) merged with README and MIT license. [CI #45](https://github.com/manojkumargudge/hybrid-rag-ms-marco/actions/runs/44735929786) green, 129 tests passed. |
| Step 0 | Create this tracker and update it each phase | Done | This file |
| P1 | Environment-backed `X-API-Key` auth, constant-time comparison, fail closed, public health/readiness, 401 and no raw keys in logs | Done | `app/auth.py`, `app/middleware.py`, `tests/test_auth.py`, `tests/test_hardening.py` |
| P1/P7 | pip-audit, Bandit, Trivy image scan and Gitleaks in CI | Configured | CI checks both lockfiles, scans the tracked tree and reachable history, and gates high and critical findings on the tested Docker image. |
| P1/P7 | Dependabot | Configured | Weekly Python, GitHub Actions, and Docker update checks. |
| P1 | Scoped security exceptions with reason, expiry and failing CI when expired | Done | `.github/security-ignore.json`, `scripts/security_gate.py`, `tests/test_security_gate.py` |
| P1/P7 | Query length and request body byte limits | Implemented | Authenticated query bodies are capped before JSON parsing. The existing character limit still applies to parsed queries. |
| P1 | Untrusted retrieved passages and prompt-injection regression | Partial | `app/generation/groq_generator.py`, mocked provider contract in PR #12 |
| P2 | JSON logs with request ID and no query, key or passage contents | Done | `app/main.py`, `app/middleware.py`, privacy regression in `tests/test_hardening.py` |
| P2 | Prometheus request count, errors, per-stage latency, Groq failures/timeouts and rate-limit hits | Done | `app/metrics.py`, retry and rate-limit assertions in `tests/test_hardening.py` |
| P2 | Readiness verifies index and models, with BM25 only in hybrid mode | Done | `app/main.py`, artifact and model readiness checks in `tests/test_e2e_fixture.py` |
| P2 | Document metric names in README | Done | [README operations](../README.md#operations) |
| P3 | Graceful shutdown and concurrency limit with 429/503 backpressure | Done | `app/main.py`, `app/api/routes.py`, and `tests/test_reliability.py` |
| P3 | Groq circuit breaker and flagged passage-only fallback | Done | `app/generation/groq_generator.py`, `app/rag_service.py`, and breaker and fixture tests |
| P3 | Startup refuses artifact SHA-256 mismatch against `manifests/v3.json` | Done | Trusted `manifests/v3.json` is packaged in the image. Startup tests reject modified artifacts even if their mounted hashes were updated. |
| P3 | Docker Compose CPU and memory limits | Done | `docker-compose.yml` has 2 CPU and 4 GiB defaults and a 45 s stop grace period. Compose configuration validated. |
| P3 | Fuzz bad inputs and assert no 500 | Done | `tests/test_reliability.py` covers malformed JSON, non-finite numbers, invalid UTF-8, and invalid field values. |
| P3 | Keep in-memory rate limiter, document Redis for multiple workers and assess optional Redis support | Done | [README scaling](../README.md#operations) assesses shared Redis quotas, atomicity, expiration, and outage handling. |
| P4 | `scripts/load_test.py` exercises served dense + rerank (20) only | Done | Real FastAPI, FAISS, and pinned CPU models with offline generation stub |
| P4 | Report p50/p95/p99, requests/s at 1, 5 and 20 clients, memory and cold start | Done | [Raw load results](../eval/results/20261122-p4-windows/metrics.json) and [methodology](../README.md#performance-benchmark) |
| P4 | Measure query embedding LRU, reranker batching and Uvicorn workers individually | Done | Baseline plus three isolated variants, 12 load cases |
| P4 | Keep clear gains with identical top 10 on 200 queries | Done | Opt-in batch size 8, with exact ordered [baseline](../eval/results/20261122-p4-parity-baseline/metrics.json) and [batch-size-8](../eval/results/20261122-p4-parity-batch8/metrics.json) parity on all 200 queries |
| P4 | Commit measured results under `eval/results/` without rerunning accuracy evaluation | Done | Load and parity artifacts linked above. Parity uses the saved v3 evaluation output. |
| P5 | Strip or flag claims without citations to returned passage IDs | Done | Claim lines with invalid citation structure or IDs return passages only with a reason. Covered by unit and mocked-provider tests. |
| P5 | Tune low-confidence abstention on TUNE only and report abstention rate and tune MRR impact | Done | [500-query calibration](../eval/results/20261122-p5-tune/metrics.json). The opt-in cutoff abstained on 2.2% for an effective MRR@10 decrease of 0.00895. |
| P5 | Report faithfulness spot check on 30 queries with manual labels from official qrels | Done (offline) | [Reviewed excerpts](../eval/results/20261122-p5-faithfulness/per_query.jsonl) and [summary](../eval/results/20261122-p5-faithfulness/metrics.json). Official qrels supply retrieval relevance. Manual labels assess source support and responsiveness. Live Groq quality remains unmeasured. |
| P6 | CI coverage gate and measured coverage | Done | 90% application line coverage gate. Local full suite covered 706 of 738 statements (95.7%). |
| P6 | Strict mypy and Ruff where passing | Done | Strict mypy covers artifact verification and citation validation. Ruff covers application, scripts, evaluation, and tests. |
| P6 | CI Docker build, start, readiness and one query with mocked Groq | Done | [Reproducible smoke commands](../README.md#offline-docker-smoke-test). Pinned offline models, verified dense v3 artifacts, four ranking checks, SIGTERM drain, and two startup failure cases. |
| P6 | OpenAPI export and drift test | Done | [Schema snapshot](openapi.json) and live endpoint comparison in pytest. |
| P6 | Document 400, 401, 422, 429 and 503 error shapes | Done | [API response examples](../README.md#api) include error details and response headers. |
| P7 | Slim non-root image and HEALTHCHECK | Done | Existing Dockerfile. Health check exercised by the linked P6 Docker job in the deployment guide. |
| P7 | GHCR publication, `v1.0.0` tag and candidate documentation | Candidate prepared | [Release checklist](release.md), [draft notes](release-notes-v1.0.0.md), and [changelog](../CHANGELOG.md). Registry access, authorization, publication, and tagging are pending. |
| P7 | Production Compose with limits and complete `.env.example` | Done | `compose.operations.yml` uses the existing runtime env template, a selected image, loopback publication, and read-only release artifacts. |
| P7 | Runbook for startup, API and Groq key rotation, index rebuild, rollback and common errors | Done | [Deployment](deployment.md), [rollback](rollback.md), and [incident response](operations.md). The CI rehearsal exercises failed readiness and rollback with a deterministic provider stub. |
| P7 | Mermaid architecture diagram | Done | Existing [architecture diagram](../README.md#architecture). |
| P7 | ADRs for retrieval, fusion, reranking, generation, artifacts, release, and observability | Done | [Architecture decisions](architecture-decisions.md) record alternatives, tradeoffs, and consequences. |
| P7 | Document 150k-passage subset versus the 8.8M corpus, CPU latency, in-memory limits and single-node operation | Done | [Deployment](deployment.md), [monitoring](operations.md), and [CPU benchmark](../README.md#performance-benchmark). |
| P7 | Verify every README step in a separate fresh clone, record times and gaps, and fix README | Partial | Hosted CI covers artifact retrieval, Docker smoke, and the operations rehearsal in a fresh checkout. The full README workflow, source artifact rebuild, and production cutover remain unverified. |
| P8 | Map README claims to code or results | Done | [Evidence map](evidence.md) includes source code, recorded results, and verification boundaries. |
| P8 | Reconcile README with the served EVAL result on the 150k-passage subset | Done | [Saved EVAL](../eval/results/20261118-v3-0d90c58/metrics.json) records 0.779317857 MRR@10 for dense + rerank top 20. |
| P8 | Prepare GHCR publication and v1.0.0 tagging | Prepared | [Release workflow](../.github/workflows/release-image.yml) has read-only PR validation and a separate manual main-only publication job. Refer to the [operator checks](release.md). |
| P8 | Verify the release in a clean checkout | Automated | Hosted CI covers hashed install, lock regeneration, tests, published dense artifacts, Docker startup, and isolated rollback. The release workflow exercises the pinned image build and smoke with registry writes disabled. |
| P8 | Final quality and security audit | Automated | Coverage, Ruff, scoped strict mypy, OpenAPI drift, dependency auditing, scoped Bandit, and a checksummed Gitleaks tracked-tree scan run in hosted CI. |
| P8 | Report completed items, skips and known risks | Done | [Verification boundaries](evidence.md#verification-boundaries) and [release prerequisites](release.md). |

## v1.0.0 operator sign-off

Record the host, approved DNS name, certificate renewal owner, private port,
release source SHA, registry digest, and inspected image ID in the release
record. Record the retrieval mode, artifact manifest, and previous release's
image digest and image ID. Verify and retain the previous image and matching
artifacts before opening ingress for the candidate.

| Required evidence | Status at candidate preparation |
|---|---|
| Exact-SHA CI, release smoke and image scan | Run before publishing. Save both workflow links. |
| TLS, trusted hosts, CORS, and gateway request limits | Verify on the selected target host. |
| Secret injection, API-key rotation, and private metrics scrape | Verify with authorized credentials and the operator responsible for rotation. |
| Health, readiness, authenticated metadata, and one grounded query | Run on loopback before opening ingress, then verify through the authorized gateway. |
| Rollback bundle and monitoring ownership | Name the release and incident owners, stage the known-good bundle, and check alert delivery. |

## Security exception review

Review the current 12 Debian CVEs by **2026-12-21**, the two BM25 Bandit
exceptions by **2026-12-31**, and the exact-history Gitleaks exceptions by
**2027-02-21**. The [register](../.github/security-ignore.json) records the
exact scopes and reasons. Apply the package update when a fix becomes
available, then rebuild and rescan the candidate. Avoid carrying exceptions to
different package versions or to deployments that enable the affected code.

## Historical P1 verification

In PR #12, the Trivy scan gates high and critical findings. The temporary Debian
exceptions apply only to the package versions in the ignore file and expire
on 2026-12-21. A reported upstream fix requires a fresh decision or upgrade.
Historical Gitleaks false positives match individual commit fingerprints.
