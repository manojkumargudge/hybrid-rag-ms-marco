"""Artifact integrity checks against a build manifest (sha256 per file)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ArtifactIntegrityError(ValueError):
    """An artifact is missing from the manifest or does not match its hash."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifacts(manifest_path: str | Path, *paths: str | Path) -> None:
    """Raise unless every path is listed in the manifest with a matching sha256.

    Manifest keys are paths relative to the manifest's directory. Must be
    called before deserializing any untrusted format such as pickle.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise ArtifactIntegrityError(f"Manifest not found: {manifest_path}")

    files = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", {})
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
        if sha256_file(resolved) != expected:
            raise ArtifactIntegrityError(f"sha256 mismatch for {key}")
