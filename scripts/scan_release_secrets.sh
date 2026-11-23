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
if ! "$staging/gitleaks" dir --no-banner --redact \
    --config "$staging/tree/.github/gitleaks-release.toml" \
    --report-format json --report-path "$staging/report.json" "$staging/tree"; then
    if [[ -f "$staging/report.json" ]]; then
        python - "$staging/report.json" "$staging/tree" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[2]).resolve()
for finding in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")):
    path = Path(finding["File"]).resolve().relative_to(root)
    print(f'{path}:{finding["StartLine"]}: {finding["RuleID"]}')
PY
    fi
    exit 1
fi
