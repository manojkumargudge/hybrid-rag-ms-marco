#!/usr/bin/env bash
# Scan the exact tracked tree and every commit reachable from its HEAD.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
staging=$(mktemp -d)
trap 'rm -rf -- "$staging"' EXIT
if [[ "$(git rev-parse --is-shallow-repository)" != false ]]; then
    echo 'A full Git history is required for the reachable-history scan.' >&2
    exit 1
fi
mkdir "$staging/tree"
git archive HEAD | tar -xf - -C "$staging/tree"
archive="$staging/gitleaks.tar.gz"
curl --fail --location --silent --show-error --retry 3 --proto '=https' \
    --proto-redir '=https' -o "$archive" \
    https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz
printf '%s  %s\n' 551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb "$archive" \
    | sha256sum --check
tar -xzf "$archive" -C "$staging" gitleaks
tree_status=0
if ! "$staging/gitleaks" dir --no-banner --redact=100 --ignore-gitleaks-allow \
    --gitleaks-ignore-path /dev/null \
    --config "$staging/tree/.github/gitleaks-release.toml" \
    --report-format json --report-path "$staging/report.json" "$staging/tree"; then
    if [[ -f "$staging/report.json" ]]; then
        python3 - "$staging/report.json" "$staging/tree" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[2]).resolve()
for finding in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")):
    path = Path(finding["File"]).resolve().relative_to(root)
    print(f'{str(path)!r}:{finding["StartLine"]}: {finding["RuleID"]!r}')
PY
    fi
    tree_status=1
fi

printf '[extend]\nuseDefault = true\n' > "$staging/history.toml"
history_status=0
"$staging/gitleaks" git --no-banner --redact=100 --ignore-gitleaks-allow \
    --config "$staging/history.toml" --gitleaks-ignore-path /dev/null \
    --log-opts='--diff-merges=first-parent HEAD' --exit-code 3 \
    --report-format json --report-path "$staging/history.json" . || history_status=$?
if [[ "$history_status" != 0 && "$history_status" != 3 ]]; then
    exit "$history_status"
fi
python3 scripts/security_gate.py gitleaks "$staging/history.json"
exit "$tree_status"
