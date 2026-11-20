"""Download the prebuilt corpus v3 artifacts and verify them.

Standard library only, so it runs before any dependencies are installed:

    python scripts/fetch_artifacts.py [--dest artifacts/v3]

Files come from the GitHub release `v3-artifacts`. The only source of truth
for their hashes is the committed manifests/v3.json: every file's sha256
must match it, and the downloaded manifest.json must equal it. A file that
does not match is deleted and the script exits non-zero; nothing is moved
into place unverified. Existing files that already match are kept; existing
files that do not match are never overwritten.

The data is derived from MS MARCO (non-commercial research use only; see
https://microsoft.github.io/msmarco/).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path, PurePosixPath

REPO = "manojkumargudge/hybrid-rag-ms-marco"
TAG = "v3-artifacts"
ROOT = Path(__file__).resolve().parents[1]
CHUNK = 1 << 20


class VerificationError(Exception):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def asset_name(key: str) -> str:
    """Release assets are flat: 'bm25/bm25_index.pkl' -> 'bm25_index.pkl'."""
    return PurePosixPath(key).name


def download(url: str, target: Path) -> None:
    partial = target.with_name(target.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
            while chunk := response.read(CHUNK):
                out.write(chunk)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def fetch_file(base_url: str, key: str, expected_sha: str, dest: Path) -> str:
    """Download one file to dest/key and verify it. Returns 'kept' or 'downloaded'."""
    target = dest / key
    if target.exists():
        if sha256_file(target) == expected_sha:
            return "kept"
        raise VerificationError(
            f"{target} exists but does not match manifests/v3.json; refusing to overwrite. "
            "Move it away and re-run."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + ".download")
    try:
        download(f"{base_url}/{asset_name(key)}", staging)
        actual = sha256_file(staging)
        if actual != expected_sha:
            raise VerificationError(
                f"sha256 mismatch for {key}: got {actual}, manifest says {expected_sha}"
            )
        staging.replace(target)
    finally:
        staging.unlink(missing_ok=True)
    return "downloaded"


def fetch(dest: Path, base_url: str, manifest_path: Path) -> list[tuple[str, str]]:
    committed = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    for key, expected in sorted(committed["files"].items()):
        results.append((key, fetch_file(base_url, key, expected, dest)))

    manifest_target = dest / "manifest.json"
    if manifest_target.exists():
        existing = json.loads(manifest_target.read_text(encoding="utf-8"))
        if existing != committed:
            raise VerificationError(
                f"{manifest_target} differs from manifests/v3.json; refusing to overwrite."
            )
        results.append(("manifest.json", "kept"))
    else:
        staging = dest / "manifest.json.download"
        try:
            download(f"{base_url}/manifest.json", staging)
            if json.loads(staging.read_text(encoding="utf-8")) != committed:
                raise VerificationError("Downloaded manifest.json differs from manifests/v3.json.")
            staging.replace(manifest_target)
        finally:
            staging.unlink(missing_ok=True)
        results.append(("manifest.json", "downloaded"))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=ROOT / "artifacts" / "v3")
    parser.add_argument("--base-url",
                        default=f"https://github.com/{REPO}/releases/download/{TAG}")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests" / "v3.json")
    args = parser.parse_args()

    try:
        for key, status in fetch(args.dest, args.base_url.rstrip("/"), args.manifest):
            print(f"{status:10} {key}")
    except VerificationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"All files verified against {args.manifest}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
