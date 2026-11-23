# Verify the v1.0.0 release candidate

## Validate a fresh checkout

Use Linux x86-64, Python 3.11, and Docker Engine with the Compose plugin. Set
`VALIDATED_COMMIT` to the full candidate SHA and use a new checkout. The development
lock targets this platform. The runtime image separately installs the production
lock. The container build downloads the pinned model snapshots, and the smoke
containers use those snapshots offline.

```bash
git clone https://github.com/manojkumargudge/hybrid-rag-ms-marco.git hybrid-rag-release-check
cd hybrid-rag-release-check
git checkout --detach "$VALIDATED_COMMIT"
test -z "$(git status --porcelain)"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --no-deps --require-hashes \
  --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.lock
python -m pip install uv==0.13.8
sh scripts/lock_deps.sh
git diff --exit-code -- requirements.lock requirements-dev.lock
ruff check app evaluation tests scripts
mypy --strict app/artifacts.py app/generation/citations.py
coverage run --source=app -m pytest -q
coverage report --fail-under=90
bash scripts/check_release_image.sh
bash scripts/operations_smoke.sh hybrid-rag:release-check
```

The full test run includes served OpenAPI drift. The release build script
downloads only the required dense inputs, checks them against the committed
v3 manifest, and runs the container with networking disabled and a fake Groq
client. The Compose rehearsal detects failed readiness and restores a retained
image. The [CI workflow](../.github/workflows/ci.yml) also runs the existing
dependency, Bandit, and Gitleaks audits in a separate job. The release build
also runs Trivy on the exact smoke-tested image. See [evidence](evidence.md) for
the distinction between hosted checks and outstanding production verification.

## Publish after the stack is approved and merged

These steps are for a future authorized release operator.

1. Confirm the exact commit is on `main`, the required checks passed, and the
   [draft release notes](release-notes-v1.0.0.md) were reviewed. Configure the
   GitHub `ghcr-release` environment for main only, with required reviewers
   and self-review disabled where supported. Review access to the GHCR
   package. After publication is separately authorized, set
   `GHCR_RELEASE_ENABLED` to `true` as a secret scoped to that environment.
   Leave it unset at repository and organization level. The workflow grants
   package write access only to the manual publication job.
2. Dispatch **Release image** on `main` with publishing disabled for a dry run.
   Provide the full `main` SHA and review the build, smoke, and Trivy results.
   PR runs perform the same validation
   without registry credentials.
3. In a separately approved manual dispatch on `main`, enable `publish` and
   supply the exact full main commit SHA. The workflow runs the full CI jobs
   for that revision first. The publication job then checks the requested SHA
   and the environment opt-in, builds, smoke-tests, and scans the local image, and
   attempts to create
   `ghcr.io/manojkumargudge/hybrid-rag-ms-marco:sha-<full-sha>`. The workflow
   refuses to overwrite this tag or proceed if registry access cannot prove
   its absence. It confirms the pulled digest resolves to the tested image ID
   and OCI labels before recording the digest. If a push succeeded but digest
   verification failed, inspect the package before retrying.
4. Pull the recorded digest and verify its Docker repository digest and
   `org.opencontainers.image.revision` label. The
   `org.opencontainers.image.version` label is `1.0.0`. The registry tag
   identifies the build revision, while the version label identifies the
   application version. Use the digest for deployment.
   The workflow guards against overwrites through this publisher. Other
   registry writers can still move tags, which is why deployment uses digests.
5. After sign-off, create an annotated `v1.0.0` Git tag on that same SHA and
   publish release notes based on [CHANGELOG.md](../CHANGELOG.md). For the
   approved release only, use `git tag -a v1.0.0 "$VALIDATED_COMMIT" -m "v1.0.0"`
   followed by `git push origin refs/tags/v1.0.0`. Include the recorded image
   digest, the compatible artifact manifest, and the validation evidence in the
   notes. A Git tag does not trigger image publication.

```bash
image='ghcr.io/manojkumargudge/hybrid-rag-ms-marco@sha256:<recorded-digest>'
docker pull "$image"
docker image inspect --format '{{json .RepoDigests}} {{index .Config.Labels "org.opencontainers.image.revision"}} {{index .Config.Labels "org.opencontainers.image.version"}}' "$image"
```

For private packages, authenticate using the approved credential mechanism.
Keep credentials out of command arguments, environment files in the checkout,
and shared logs. The current image includes the trusted v3 manifest. Stage the
matching passages and FAISS index for dense mode. Hybrid mode also needs both
BM25 artifacts. Use the image-ID and artifact preflight in the [deployment
runbook](deployment.md). Retain the prior image digest and verified artifact
bundle for [rollback](rollback.md).

No tag, release, or registry publication is performed by a dry run. GHCR
authentication, package creation, and pull-by-digest remain for the authorized
release operator to verify.
