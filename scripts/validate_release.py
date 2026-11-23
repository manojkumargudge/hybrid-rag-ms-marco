"""Validate a staged Compose release and probe its local API without printing secrets."""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

ARTIFACTS = {
    "dense": ("passages.parquet", "dense_index.faiss"),
    "hybrid": (
        "passages.parquet", "dense_index.faiss", "bm25/bm25_index.pkl", "bm25/tokenized_corpus.pkl",
    ),
}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def docker_output(*args: str) -> str:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=True, timeout=30,
    ).stdout.strip()


def preflight(image: str, expected_id: str, artifacts: Path, mode: str) -> None:
    from app.artifacts import verify_artifacts

    image_id = docker_output("image", "inspect", "--format", "{{.Id}}", "--", image)
    if image_id != expected_id:
        raise ValueError("The staged tag does not resolve to the recorded image ID.")
    manifest = docker_output(
        "run", "--rm", "--network", "none", "--entrypoint", "cat", "--pull", "never",
        "--", image_id, "/app/manifests/v3.json",
    )
    with tempfile.TemporaryDirectory(prefix="rag-release-check-") as directory:
        trusted = Path(directory) / "v3.json"
        trusted.write_text(manifest, encoding="utf-8")
        verify_artifacts(
            artifacts / "manifest.json", *(artifacts / key for key in ARTIFACTS[mode]),
            trusted_manifest_path=trusted,
        )


def probe(port: int, api_key: str, timeout: float, query: str | None) -> None:
    if not api_key.strip():
        raise ValueError("Load API_KEY into the environment before probing.")
    opener = build_opener(ProxyHandler({}), NoRedirect())

    def request(path: str, *, protected: bool = False, text: str | None = None):
        headers = {"X-API-Key": api_key} if protected else {}
        payload = None
        if text is not None:
            headers["Content-Type"] = "application/json"
            payload = json.dumps({"query": text}).encode()
        with opener.open(
            Request(f"http://127.0.0.1:{port}{path}", data=payload, headers=headers),
            timeout=30 if text is not None else 5,
        ) as response:
            return json.load(response)

    deadline = time.monotonic() + timeout
    while True:
        try:
            if (
                request("/health") == {"status": "healthy", "service": "hybrid-rag"}
                and request("/ready") == {"status": "ready", "service": "hybrid-rag"}
            ):
                break
        except (URLError, TimeoutError):
            pass
        if time.monotonic() >= deadline:
            raise ValueError("Health or readiness did not pass before the deadline.")
        time.sleep(0.5)
    metadata = request("/metadata", protected=True)
    if metadata.get("service") != "hybrid-rag":
        raise ValueError("Unexpected authenticated metadata response.")
    if query is not None:
        body = request("/api/v1/query", protected=True, text=query)
        if not body.get("answer") or not body.get("citations") or body.get("error") is not None:
            raise ValueError("The query did not return a usable answer and citations.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    staged = commands.add_parser("preflight")
    staged.add_argument("--image", required=True)
    staged.add_argument("--image-id", required=True)
    staged.add_argument("--artifacts", required=True, type=Path)
    staged.add_argument("--mode", choices=ARTIFACTS, default="dense")
    running = commands.add_parser("probe")
    running.add_argument("--port", type=int, default=8000)
    running.add_argument("--timeout", type=float, default=120)
    running.add_argument("--query")
    args = parser.parse_args()
    if args.command == "probe" and (not 1 <= args.port <= 65535 or args.timeout <= 0):
        parser.error("Probe requires a valid port and a positive timeout.")
    try:
        if args.command == "preflight":
            preflight(args.image, args.image_id, args.artifacts, args.mode)
            print("Recorded image and required artifact digests verified.")
        else:
            probe(args.port, os.environ.get("API_KEY", ""), args.timeout, args.query)
            print("Health, readiness and authenticated deployment checks passed.")
    except subprocess.CalledProcessError as error:
        print(f"Docker command failed with exit {error.returncode}.")
        return 1
    except subprocess.TimeoutExpired:
        print("Docker command did not finish before the timeout.")
        return 1
    except URLError as error:
        # Never print a response body or the authenticated request.
        print(f"Deployment probe request failed ({getattr(error, 'code', 'transport error')}).")
        return 1
    except (OSError, ValueError, KeyError) as error:
        # Artifact integrity errors contain file paths or digests, never secret values.
        print(f"Deployment validation failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
