#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
revision=$(git rev-parse HEAD)
if [[ -n "${GITHUB_SHA:-}" ]]; then
    test "$revision" = "$GITHUB_SHA"
fi
docker build -t hybrid-rag:release-check \
    --label "org.opencontainers.image.source=https://github.com/manojkumargudge/hybrid-rag-ms-marco" \
    --label "org.opencontainers.image.revision=$revision" .
python -m scripts.fetch_artifacts --dense-only
python -m scripts.docker_smoke --image hybrid-rag:release-check
