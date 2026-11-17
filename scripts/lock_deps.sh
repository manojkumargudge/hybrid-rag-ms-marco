#!/usr/bin/env sh
# Regenerate hash-pinned lockfiles for the Linux CI runner and Docker image
# (python 3.11, x86_64 manylinux, CPU-only torch). Requires `uv`.
set -eu
cd "$(git rev-parse --show-toplevel)"
UV="${UV:-uv}"
for name in requirements requirements-dev; do
  $UV pip compile "$name.txt" \
    --generate-hashes \
    --python-version 3.11 \
    --python-platform x86_64-manylinux_2_28 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    --override scripts/torch-cpu-override.txt \
    --no-header \
    -o "$name.lock"
done
