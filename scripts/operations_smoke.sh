#!/usr/bin/env bash
# Linux Docker rehearsal of a failed candidate and rollback using the operations Compose file.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

baseline=${1:-hybrid-rag:ci}
staging=$(mktemp -d)
project=rag-operations-${staging##*.}
project=${project,,}
candidate=hybrid-rag:operations-${project}
previous_release=$staging/previous.env
candidate_release=$staging/candidate.env
fixtures=$staging/fixtures
artifacts=$PWD/artifacts/v3
key=offline-operations-validation-key

compose() {
    local release=$1
    shift
    docker compose -p "$project" --env-file "$release" \
        -f compose.operations.yml -f "$staging/offline.yml" "$@"
}

cleanup() {
    if [[ -f "$previous_release" ]]; then
        compose "$previous_release" down --remove-orphans >/dev/null 2>&1 || true
    fi
    docker image rm "$candidate" >/dev/null 2>&1 || true
    rm -f -- "$previous_release" "$candidate_release" "$staging/good.runtime" \
        "$staging/bad.runtime" "$staging/offline.yml" "$fixtures/docker_smoke_provider.py" \
        "$fixtures/validate_release.py" "$fixtures/drain-query"
    rmdir -- "$fixtures" "$staging"
}
mkdir -m 755 "$fixtures"
trap cleanup EXIT
cp scripts/docker_smoke_provider.py scripts/validate_release.py "$fixtures/"
printf 'unused offline drain sentinel\n' > "$fixtures/drain-query"
printf 'API_KEY=%s\nGROQ_API_KEY=offline-placeholder\nOMP_NUM_THREADS=2\nMKL_NUM_THREADS=2\n' \
    "$key" > "$staging/good.runtime"
printf 'API_KEY=\nGROQ_API_KEY=offline-placeholder\nOMP_NUM_THREADS=2\nMKL_NUM_THREADS=2\n' \
    > "$staging/bad.runtime"

cat > "$staging/offline.yml" <<'YAML'
services:
  hybrid-rag:
    network_mode: none
    ports: !reset []
    environment:
      PYTHONPATH: /smoke
    volumes:
      - type: bind
        source: ${RAG_TEST_FIXTURES}
        target: /smoke
        read_only: true
    command: [uvicorn, "docker_smoke_provider:create_app", --factory, --host, "0.0.0.0", --port, "8000", --no-proxy-headers, --no-access-log]
YAML

docker build --network none --build-arg "BASE_IMAGE=$baseline" -t "$candidate" - <<'DOCKERFILE'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.description="Synthetic operations candidate"
DOCKERFILE
previous_id=$(docker image inspect --format '{{.Id}}' "$baseline")
candidate_id=$(docker image inspect --format '{{.Id}}' "$candidate")
test "$previous_id" != "$candidate_id"

printf 'RAG_IMAGE=%s\nRAG_RUNTIME_ENV_FILE=%s\nRAG_ARTIFACT_DIR=%s\nRAG_TEST_FIXTURES=%s\n' \
    "$baseline" "$staging/good.runtime" "$artifacts" "$fixtures" > "$previous_release"
printf 'RAG_IMAGE=%s\nRAG_RUNTIME_ENV_FILE=%s\nRAG_ARTIFACT_DIR=%s\nRAG_TEST_FIXTURES=%s\n' \
    "$candidate" "$staging/bad.runtime" "$artifacts" "$fixtures" > "$candidate_release"

start_release() {
    local release=$1 image=$2 image_id=$3
    python -m scripts.validate_release preflight --image "$image" --image-id "$image_id" \
        --artifacts "$artifacts"
    compose "$release" config --quiet
    test "$(compose "$release" config --images)" = "$image"
    compose "$release" up -d --no-build --pull never --force-recreate --wait --wait-timeout 150
    local container
    container=$(compose "$release" ps -q hybrid-rag)
    test "$(docker inspect --format '{{.Image}}' "$container")" = "$image_id"
}

probe() {
    compose "$1" exec -T -e "API_KEY=$key" hybrid-rag python /smoke/validate_release.py probe \
        --timeout 3 --query "Bethel University was founded in what year"
}

start_release "$previous_release" "$baseline" "$previous_id"
probe "$previous_release"
start_release "$candidate_release" "$candidate" "$candidate_id"
if failure=$(probe "$candidate_release"); then
    echo "The unconfigured candidate unexpectedly passed readiness." >&2
    exit 1
else
    status=$?
    printf '%s\n' "$failure"
    test "$status" -eq 1
    grep -Fq 'Health or readiness did not pass before the deadline.' <<< "$failure"
fi
start_release "$previous_release" "$baseline" "$previous_id"
probe "$previous_release"
echo "Compose rehearsal passed. Candidate failed readiness and the recorded image recovered."
