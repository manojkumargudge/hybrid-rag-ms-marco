#!/usr/bin/env bash
# Inspect only the clean tracked tree. No local env files or Git history enter the scan.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
staging=$(mktemp -d)
trap 'rm -rf -- "$staging"' EXIT
mkdir "$staging/tree"
git archive HEAD | tar -xf - -C "$staging/tree"
archive="$staging/gitleaks.tar.gz"
curl --fail --location --silent --show-error --retry 3 --proto '=https' \
    --proto-redir '=https' -o "$archive" \
    https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz
printf '%s  %s\n' 551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb "$archive" \
    | sha256sum --check
tar -xzf "$archive" -C "$staging" gitleaks
"$staging/gitleaks" dir --no-banner --redact "$staging/tree"
