"""fetch_artifacts: verified downloads from a local 'release' (file:// URLs)."""

import hashlib
import json

import pytest

from scripts.fetch_artifacts import VerificationError, fetch

FILES = {"passages.parquet": b"passages", "bm25/bm25_index.pkl": b"pickle-bytes"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def release(tmp_path):
    """A fake release dir (flat asset names) + committed manifest."""
    assets = tmp_path / "release"
    assets.mkdir()
    manifest = {"version": "v3", "files": {k: _sha(v) for k, v in FILES.items()}}
    for key, data in FILES.items():
        (assets / key.split("/")[-1]).write_bytes(data)
    (assets / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    committed = tmp_path / "v3.json"
    committed.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return assets.as_uri(), committed, assets, tmp_path / "dest"


def test_downloads_and_verifies_everything(release):
    base, committed, _, dest = release
    results = dict(fetch(dest, base, committed))
    assert results == {"bm25/bm25_index.pkl": "downloaded", "passages.parquet": "downloaded",
                       "manifest.json": "downloaded"}
    assert (dest / "bm25/bm25_index.pkl").read_bytes() == b"pickle-bytes"
    # Re-run is idempotent: verified files are kept, not re-downloaded.
    assert set(dict(fetch(dest, base, committed)).values()) == {"kept"}


def test_tampered_asset_is_refused_and_not_left_behind(release):
    base, committed, assets, dest = release
    (assets / "bm25_index.pkl").write_bytes(b"malicious pickle")
    with pytest.raises(VerificationError, match="sha256 mismatch for bm25/bm25_index.pkl"):
        fetch(dest, base, committed)
    leftovers = [p.name for p in (dest / "bm25").iterdir()]
    assert leftovers == [], "no unverified or partial file may remain"


def test_existing_mismatched_file_is_never_overwritten(release):
    base, committed, _, dest = release
    (dest).mkdir()
    (dest / "passages.parquet").write_bytes(b"someone else's file")
    with pytest.raises(VerificationError, match="refusing to overwrite"):
        fetch(dest, base, committed)
    assert (dest / "passages.parquet").read_bytes() == b"someone else's file"


def test_release_manifest_must_equal_committed_manifest(release):
    base, committed, assets, dest = release
    tampered = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    tampered["version"] = "evil"
    (assets / "manifest.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(VerificationError, match="manifest.json differs"):
        fetch(dest, base, committed)
    assert not (dest / "manifest.json").exists()
