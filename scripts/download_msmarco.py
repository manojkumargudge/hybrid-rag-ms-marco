"""Download the official MS MARCO passage files and verify their checksums.

Expected MD5s are the values recorded independently by ir_datasets
(ir_datasets/etc/downloads.json). A file whose MD5 does not match is
rejected. SHA-256 and MD5 of every file are written to
<raw_dir>/downloads.json, which prepare_data.py copies into manifest.json.

Existing verified files are not downloaded again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

BASE_URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/"

SOURCES = {
    "collectionandqueries.tar.gz": {
        "md5": "31644046b18952c1386cd4564ba2ae69",
        "size": 1057717952,
    },
    "top1000.dev.tar.gz": {
        "md5": "8c140662bdf123a98fbfe3bb174c5831",
        "size": 687414398,
    },
}

CHUNK = 8 * 1024 * 1024


def file_digests(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            sha256.update(chunk)
            md5.update(chunk)
    return sha256.hexdigest(), md5.hexdigest()


def download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    with (
        urllib.request.urlopen(url, timeout=60) as response,
        partial.open("wb") as handle,
    ):
        while chunk := response.read(CHUNK):
            handle.write(chunk)
    partial.replace(destination)


def safe_extract(archive: Path, target: Path) -> list[str]:
    """Extract regular files only, refusing paths that escape target."""
    target = target.resolve()
    names = []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            out = (target / member.name).resolve()
            if target not in out.parents:
                raise ValueError(f"Unsafe path in archive: {member.name}")
            if out.exists() and out.stat().st_size == member.size:
                names.append(member.name)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, out.open("wb") as dst:
                while chunk := src.read(CHUNK):
                    dst.write(chunk)
            names.append(member.name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("artifacts/raw"))
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    record = {}
    for name, expected in SOURCES.items():
        path = args.raw_dir / name
        if not (path.exists() and path.stat().st_size == expected["size"]):
            print(f"Downloading {name} ({expected['size']:,} bytes)...", flush=True)
            download(BASE_URL + name, path)

        sha256, md5 = file_digests(path)
        if md5 != expected["md5"]:
            raise SystemExit(
                f"{name}: MD5 {md5} != expected {expected['md5']}; refusing to use it."
            )
        print(f"{name}: md5 OK, sha256={sha256}", flush=True)

        extracted = safe_extract(path, args.raw_dir)
        record[name] = {
            "url": BASE_URL + name,
            "bytes": path.stat().st_size,
            "md5": md5,
            "md5_reference": "ir_datasets/etc/downloads.json",
            "sha256": sha256,
            "extracted": {
                member: file_digests(args.raw_dir / member)[0] for member in extracted
            },
        }

    (args.raw_dir / "downloads.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.raw_dir / 'downloads.json'}")


if __name__ == "__main__":
    main()
