# Roll back a single-host release

Use this procedure when the candidate fails health, readiness, authentication,
the expected release checks, or post-cutover validation. A single service
replacement can interrupt traffic. Pause ingress or announce a maintenance
window before replacing the container.

## Restore the previous release

1. Record the failed release's image ID, artifact version, timestamp, readiness
   result, and filtered logs. Keep its bundle for investigation. Use
   `docker compose logs --tail 150 --no-color hybrid-rag` with the same project and
   release selection file. Keep full Docker configuration and secret values out
   of incident records.
2. Locate the last release that passed validation. Select its retained image
   tag, recorded image ID, artifact directory, retrieval mode, and compatible
   settings. Never retag a current image as the old release. If the image was
   archived or staged by digest, restore that identical image first and check its
   recorded ID. Rebuilding an old revision can resolve to a different base image.
3. Confirm the selected runtime file has current authorized secrets. If secrets
   were revoked or rotated, stage a private copy using the active values and
   keep the old release's compatible settings. Point the selection file at the
   copy. Preserve the existing host port and Compose project name.
4. Run the preflight and startup commands below. A missing image or digest
   mismatch is a stop condition. Find a retained compatible bundle and verify it
   before proceeding.

```bash
set -euo pipefail
set +x
unset RAG_IMAGE RAG_ARTIFACT_DIR RAG_RUNTIME_ENV_FILE API_PORT API_CPUS API_MEMORY_LIMIT
# Set these paths and the tag from the last known-good release record.
previous=/absolute/path/to/known-good-release
image=hybrid-rag:recorded-release-tag
artifacts="$previous/artifacts-v3"
image_id=$(cat "$previous/image-id")
python -m scripts.validate_release preflight --image "$image" --image-id "$image_id" \
  --artifacts "$artifacts" --mode "$(cat "$previous/retrieval-mode")"
compose() {
  docker compose -p hybrid-rag-prod --env-file "$previous/compose.env" \
    -f compose.operations.yml "$@"
}
compose config --quiet
test "$(compose config --images)" = "$image"
compose up -d --no-build --pull never --force-recreate --wait --wait-timeout 150
container=$(compose ps -q hybrid-rag)
test "$(docker inspect --format '{{.Image}}' "$container")" = "$image_id"
(
  read -rsp 'Current API key: ' API_KEY
  printf '\n'
  export API_KEY
  python -m scripts.validate_release probe --port 8000
)
```

Run from the repository root and adjust the probe port to the selected host
port. The Compose file keeps the artifacts mounted read-only. Restore the
previous compatible manifest and artifact set as a unit. Keep the failed bundle
unmodified while gathering evidence. Avoid pruning images or deleting release
directories during recovery.

## Confirm recovery

- Require Docker health, `/health`, `/ready`, and authenticated `/metadata` before
  reopening ingress. Check the inspected image ID and recent logs against
  the selected release.
- If live generation is part of the incident, use a controlled benign query and
  verify that its answer cites the expected sources and has no fallback reason.
  Check request failures, stage latency, and Groq circuit errors after restoring
  traffic. A Groq outage can leave the service ready with retrieval-only responses.
- If the previous bundle also fails validation, keep traffic paused, retain the
  logs, and follow the [incident procedures](operations.md). Repeated restarts
  do not repair a rejected manifest or restore provider access.

CI rehearses a candidate with failed readiness and restoration of a different,
recorded image ID using the same verified dense artifacts. The synthetic
candidate keeps the same application layers. This exercise leaves registry
recovery, secret-store integration, and live traffic cutover unverified.
