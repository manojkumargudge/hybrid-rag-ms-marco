#!/usr/bin/env bash
# Run the pinned Trivy scanner against the locally built image.
set -euo pipefail
image=${1:?Usage: scan_image_security.sh IMAGE REPORT}
report=${2:?Usage: scan_image_security.sh IMAGE REPORT}
staging=$(mktemp -d)
trap 'rm -rf -- "$staging"' EXIT
archive="$staging/trivy.tar.gz"
curl --fail --location --silent --show-error --retry 3 --proto '=https' \
    --proto-redir '=https' -o "$archive" \
    https://github.com/aquasecurity/trivy/releases/download/v0.76.2/trivy_0.76.2_Linux-64bit.tar.gz
printf '%s  %s\n' bc0fc726e017b8927a62b2d3531d7a84ee01769daf5c99ed866dced54089fdbf "$archive" \
    | sha256sum --check
tar -xzf "$archive" -C "$staging" trivy
"$staging/trivy" image --timeout 8m --no-progress --image-src docker \
    --scanners vuln --severity HIGH,CRITICAL \
    --ignorefile /dev/null --format json --output "$report" "$image"
python3 scripts/security_gate.py trivy "$report"
