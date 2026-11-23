# Single-host deployment

Use a Linux Docker Engine host with Docker Compose supporting `env_file.format: raw`
and `up --wait`, Python 3.11, Git, and an authorized Docker operator. Allocate at least the configured 2 CPUs and
4 GiB for the container, plus headroom for the host. Stage enough disk space for
the candidate and previous images and both artifact directories. The image
build accesses the pinned package and model sources. Artifact retrieval accesses
the GitHub release. The running image uses its baked model snapshots and needs
Groq connectivity for live generation.

The [operations Compose file](../compose.operations.yml) binds only to IPv4
loopback. Handle TLS and any external routing with your existing gateway. Keep
the listener local until the checks below pass. Use the dense profile and the
published v3 corpus for the deployment covered by CI. The release contains a
150,000-passage subset of the 8.8M-passage collection. Review the
[MS MARCO non-commercial research terms](../README.md#data) before use.

## Stage a release

Run from a reviewed repository checkout. Use a new release directory each
time and retain the previous known-good release. These Bash examples assume
ordinary non-whitespace local paths.

```bash
set -euo pipefail
set +x
umask 077
root="$HOME/hybrid-rag-releases"
mkdir -p "$root"
revision=$(git rev-parse HEAD)
release="$root/$(date -u +%Y%m%dT%H%M%SZ)-${revision:0:12}"
mkdir "$release"
artifacts="$release/artifacts-v3"
image="hybrid-rag:release-${release##*/}"
python -m scripts.fetch_artifacts --dense-only --dest "$artifacts"
# The non-root container user needs read access inside the mounted bundle.
find "$artifacts" -type d -exec chmod 0555 {} +
find "$artifacts" -type f -exec chmod 0444 {} +
docker build --tag "$image" .
docker image inspect --format '{{.Id}}' "$image" > "$release/image-id"
printf '%s\n' "$revision" > "$release/source-revision"
printf 'dense\n' > "$release/retrieval-mode"
cp .env.example "$release/runtime.env"
chmod 0600 "$release/runtime.env"
printf 'RAG_IMAGE=%s\nRAG_ARTIFACT_DIR=%s\nRAG_RUNTIME_ENV_FILE=%s\nAPI_PORT=8000\nAPI_CPUS=2\nAPI_MEMORY_LIMIT=4g\n' \
  "$image" "$artifacts" "$release/runtime.env" > "$release/compose.env"
```

The downloader verifies the release manifest against the checked-in manifest and
checks SHA-256 of each required file before placing it in the bundle. For hybrid
mode, fetch without `--dense-only`, record `hybrid` in `retrieval-mode`, and set
`RETRIEVAL_MODE=hybrid` in the runtime file. The hosted smoke job exercises dense.
The image and artifact bundle must match even when rolling back to a different
version. Keep staged files and local image tags under deployer control.

For an intentional index rebuild, follow the [data preparation commands](../README.md#build-corpus-v3-from-source-instead)
in a separate workspace. Review the generated manifest, model revisions, and
ranking evidence, then bake the reviewed manifest into a distinct release
image and record its ID. A rebuilt index can differ byte-for-byte from the
published index. Keep the mounted manifest and the image's independently
trusted copy aligned before preflight. Rebuilding is a new release operation.

Edit `runtime.env` privately. Set `API_KEY` to a strong random value and
`GROQ_API_KEY` to the provider secret. Use the current authorized values from
your secret store. Keep the file owner-readable only and out of source control,
build context, logs, and shared deployment records. Use raw `KEY=value` lines
without shell quoting or expansion. Keep the artifact paths relative to
`/app` as in the example. Set `TRUSTED_HOSTS` to the local and gateway hostnames
you use. Set `CORS_ORIGINS` only if browser clients require it. Leave the
embedded model revisions aligned with the image and the artifact manifest.
Set `GIT_VERSION` to the full recorded source SHA for authenticated metadata.
The template leaves both credential values blank. An empty Groq key causes
startup to fail after local pipeline initialization. The log records
`rag_initialization_failed` with the exception type.
An empty API key leaves protected endpoints unavailable. A nonempty Groq key
permits local initialization, and a controlled authenticated query is needed
to confirm provider access.

Use the same versioned file workflow for key rotation. Stage the current release
with the new private runtime file, coordinate client updates for the API key, and
recreate the container. The service accepts one API key at a time. Keep an
independent way to validate with the new key. Once verified, revoke the old key
at the provider or retire the old API key for clients, as applicable.

## Start and validate

Keep the same Compose project name and host port for this listener. Remove any
inherited shell overrides for values in the release selection file.

```bash
unset RAG_IMAGE RAG_ARTIFACT_DIR RAG_RUNTIME_ENV_FILE API_PORT API_CPUS API_MEMORY_LIMIT
compose() {
  docker compose -p hybrid-rag-prod --env-file "$release/compose.env" \
    -f compose.operations.yml "$@"
}
image_id=$(cat "$release/image-id")
python -m scripts.validate_release preflight --image "$image" --image-id "$image_id" \
  --artifacts "$artifacts" --mode "$(cat "$release/retrieval-mode")"
compose config --quiet
test "$(compose config --images)" = "$image"
compose up -d --no-build --pull never --force-recreate --wait --wait-timeout 150
container=$(compose ps -q hybrid-rag)
test "$(docker inspect --format '{{.Image}}' "$container")" = "$image_id"
docker inspect --format '{{.State.Health.Status}} {{.HostConfig.NanoCpus}} {{.HostConfig.Memory}}' "$container"
(
  read -rsp 'Current API key: ' API_KEY
  printf '\n'
  export API_KEY
  python -m scripts.validate_release probe --port 8000
)
```

Use the configured port if it differs from 8000. The preflight inspects the image
by its recorded ID with networking disabled and compares its bundled trusted
manifest to the required files. Compose health checks `/health`. The probe
separately verifies `/ready` and authenticated `/metadata` on loopback and refuses
redirects. Avoid bare `docker inspect` and `docker compose config` in shared output.
Their full output can expose runtime environment values.

If validation fails, capture only the relevant status and logs and use the
[rollback runbook](rollback.md). Mark the release known-good only after validation.
For a planned production generation check, you can also run the probe with
`--query` and a benign question whose expected source is known. That optional
check invokes the configured generator and consumes its provider quota. Omit it
when credentials or provider access are unavailable. The offline CI rehearsal
uses a deterministic stub for this check. Confirm the first live response
against the returned passages before routing traffic.

## Verification evidence

The P6 Docker job verifies pinned offline models, published artifacts, exact
top-10 ranking for four dense queries, and in-flight SIGTERM drain. The
operations CI rehearsal uses this Compose file with a test-only offline overlay.
It stages two distinct local image IDs, detects a candidate that fails readiness,
restores the recorded previous image, and verifies a stubbed query. The overlay
disables external networking and host port publication. A real TLS gateway,
live Groq, hybrid startup, and a production-host cutover remain outside this
rehearsal. [P6 Docker run](https://github.com/manojkumargudge/hybrid-rag-ms-marco/actions/runs/44999677261/job/137138232390)
