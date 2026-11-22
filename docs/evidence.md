# Release evidence

## README claims

| README topic | Evidence and scope |
|---|---|
| Headline result, comparisons, and corpus counts | [Published v3 manifest](../manifests/v3.json) and [saved EVAL metrics](../eval/results/20261118-v3-0d90c58/metrics.json). The served `dense+rerank(depth=20)` row is 0.779 MRR@10 on 1,000 EVAL queries over the 150,000-passage subset. The full-collection leaderboard is outside this comparison. |
| Configuration selection and architectures | [TUNE results](../eval/results/20261117-v3-tune-c858705/metrics.json), [application wiring](../app/main.py), and [architecture decisions](architecture-decisions.md). |
| API, errors, authentication, and readiness | [OpenAPI snapshot](openapi.json), [auth tests](../tests/test_auth.py), [hardening tests](../tests/test_hardening.py), and [HTTP contract tests](../tests/test_api.py). |
| Data, artifact fetch, preparation, and dependency locking | [Manifest](../manifests/v3.json), [verified fetcher](../scripts/fetch_artifacts.py), [preparation scripts](../scripts/), [lock generator](../scripts/lock_deps.sh), and [CI](../.github/workflows/ci.yml). Source corpus rebuild remains outside the hosted smoke. |
| Docker quick start and release operation | [Dockerfile](../Dockerfile), [P6 real-container smoke](https://github.com/manojkumargudge/hybrid-rag-ms-marco/actions/runs/44999677261/job/137138232390), and [P7 Compose rehearsal](https://github.com/manojkumargudge/hybrid-rag-ms-marco/actions/runs/45001629648/job/137144228629). The hosted jobs use the dense subset and a deterministic Groq stub. |
| Performance benchmark | [P4 timings](../eval/results/20261122-p4-windows/metrics.json) and the ordered [baseline](../eval/results/20261122-p4-parity-baseline/metrics.json) and [batch-size-8](../eval/results/20261122-p4-parity-batch8/metrics.json) parity records. |
| Grounding and abstention | [500-query TUNE calibration](../eval/results/20261122-p5-tune/metrics.json) and [offline faithfulness review](../eval/results/20261122-p5-faithfulness/metrics.json). The review checks extracts and responsiveness. Live output quality remains unmeasured. |
| Operations, load, and rollback | [Metrics and limitations](operations.md), [reliability tests](../tests/test_reliability.py), and [Compose rehearsal](../scripts/operations_smoke.sh). |
| Audits, known exceptions, and release workflow permissions | [Security gate](../scripts/security_audit.sh), [scoped exceptions](../.github/security-ignore.json), [tracked-tree secret scan](../scripts/scan_release_secrets.sh), and [release policy tests](../tests/test_release_workflow.py). |

## Verification boundaries

- Hosted Linux CI installs hashed dependencies from a new checkout, verifies lock
  regeneration, runs pytest with coverage and schema drift, exercises the checked
  v3 dense artifacts in Docker, and rehearses rollback with two local image IDs.
- The PR release-image job exercises the release build and Docker smoke with
  contents-read permission and no registry publication. Unit tests cover
  publication gating, existing-tag refusal, ambiguous registry failures, and
  the digest handoff with an offline Docker command stand-in.
- The Dockerfile pins the same Python base digest used in the
  [validated P7 build](https://github.com/manojkumargudge/hybrid-rag-ms-marco/actions/runs/45001629648/job/137144228629).
- Gitleaks scans the tracked tree at the checkout SHA with redacted output.
  Git history, operators' local files, and external secret stores are outside
  that scan. The Bandit pickle exceptions remain scoped and time-limited.
- Outstanding production checks include GHCR access and package creation,
  private pull-by-digest, live Groq, hybrid containers, TLS and gateway setup,
  source artifact rebuilds, sustained resource use, and production traffic.
  The [release runbook](release.md) lists the authorization and digest checks
  before publication.
