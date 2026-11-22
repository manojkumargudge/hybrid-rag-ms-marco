# Prepare the v1.0.0 release

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
dependency and Bandit audit in a separate job. See [evidence](evidence.md) for
the distinction between hosted checks and outstanding production verification.

## Publish after the stack is approved and merged

These steps are for a future authorized release operator.

1. Confirm the exact commit is on `main`, the required checks passed, and the
   release notes were reviewed. Configure the GitHub `ghcr-release` environment
   for approved main deployments and review access to the GHCR package. The
   workflow grants package write access only to the manual publication job.
2. Dispatch **Release image** on `main` with publishing disabled for a dry run.
   Review the build and smoke results. PR runs perform the same validation
   without registry credentials.
3. In a separately approved manual dispatch on `main`, enable `publish` and
   supply the exact full main commit SHA. The job checks the requested SHA,
   builds and smoke-tests the local image, then attempts to create
   `ghcr.io/manojkumargudge/hybrid-rag-ms-marco:sha-<full-sha>`. The workflow
   refuses to overwrite this tag or proceed if registry access cannot prove
   its absence. The job records the registry digest. If a push succeeded but
   digest verification failed, inspect the package before retrying.
4. Pull the recorded digest and verify its Docker repository digest and
   `org.opencontainers.image.revision` label. Use the digest for deployment.
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
docker image inspect --format '{{json .RepoDigests}} {{index .Config.Labels "org.opencontainers.image.revision"}}' "$image"
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
