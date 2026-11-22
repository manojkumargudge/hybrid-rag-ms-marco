"""Artifact integrity checks against a build manifest (sha256 per file)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path


class ArtifactIntegrityError(ValueError):
    """An artifact is missing from the manifest or does not match its hash."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifacts(
    manifest_path: str | Path, *paths: str | Path,
    trusted_manifest_path: str | Path | None = None,
) -> None:
    """Raise unless every path is listed in the manifest with a matching sha256.

    Manifest keys are paths relative to the manifest's directory. Must be
    called before deserializing any untrusted format such as pickle.
    When supplied, the independent trusted manifest must list the same digest
    for each requested path.
    """
    for resolved, key, expected in _artifact_digests(manifest_path, paths, trusted_manifest_path):
        if sha256_file(resolved) != expected:
            raise ArtifactIntegrityError(f"sha256 mismatch for {key}")


def read_verified_artifact(
    manifest_path: str | Path,
    path: str | Path,
    *,
    trusted_manifest_path: str | Path | None = None,
) -> bytes:
    """Return the exact verified bytes for a loader that accepts a snapshot."""
    ((resolved, key, expected),) = _artifact_digests(manifest_path, (path,), trusted_manifest_path)
    data = resolved.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ArtifactIntegrityError(f"sha256 mismatch for {key}")
    return data


def _artifact_digests(
    manifest_path: str | Path,
    paths: tuple[str | Path, ...],
    trusted_manifest_path: str | Path | None,
) -> Iterator[tuple[Path, str, str]]:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise ArtifactIntegrityError(f"Manifest not found: {manifest_path}")

    files = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", {})
    trusted_files = None
    if trusted_manifest_path is not None:
        trusted_files = json.loads(
            Path(trusted_manifest_path).read_text(encoding="utf-8")
        )["files"]
    root = manifest_path.resolve().parent

    for path in paths:
        resolved = Path(path).resolve()
        try:
            key = resolved.relative_to(root).as_posix()
        except ValueError:
            raise ArtifactIntegrityError(
                f"{path} is outside the manifest directory {root}"
            ) from None
        expected = files.get(key)
        if expected is None:
            raise ArtifactIntegrityError(f"{key} is not listed in {manifest_path}")
        if trusted_files is not None and expected != trusted_files.get(key):
            raise ArtifactIntegrityError(f"Manifest SHA-256 differs from trusted digest for {key}")
        yield resolved, key, expected
