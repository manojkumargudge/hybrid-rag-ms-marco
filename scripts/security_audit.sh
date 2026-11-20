#!/usr/bin/env sh
# Dependency + static security checks (run in CI; see README "Security").
#
# pip-audit: audits the hash-locked lockfiles without installing anything.
#   torch is pinned as 2.13.0+cpu, a local version that is not on PyPI, so it
#   cannot be looked up as-is. We audit a temporary copy with the local
#   suffix removed (same upstream release). --strict fails if any package is
#   skipped, and the copy is checked to be non-empty and to still contain
#   torch, so an empty or broken copy cannot pass silently.
# bandit: medium+ severity (-ll) on application code.
set -eu
cd "$(git rev-parse --show-toplevel)"
PIP_AUDIT="${PIP_AUDIT:-pip-audit}"
BANDIT="${BANDIT:-bandit}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

for lock in requirements.lock requirements-dev.lock; do
  sed 's/==2\.13\.0+cpu/==2.13.0/' "$lock" > "$tmp/$lock"
  packages="$(grep -c '^[a-z0-9]' "$tmp/$lock" || true)"
  if [ "$packages" -lt 10 ] || ! grep -q '^torch==2\.13\.0 ' "$tmp/$lock"; then
    echo "audit copy of $lock looks wrong ($packages packages, torch missing?)" >&2
    exit 1
  fi
  echo "pip-audit: $lock ($packages pinned packages)"
  $PIP_AUDIT -r "$tmp/$lock" --require-hashes --disable-pip --strict --progress-spinner off
done

echo "bandit: app/ (severity >= medium)"
$BANDIT -r app -ll
