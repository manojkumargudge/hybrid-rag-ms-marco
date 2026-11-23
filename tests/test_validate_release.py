import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError

import pytest

from app.artifacts import ArtifactIntegrityError
from scripts import validate_release


def test_preflight_pins_the_image_and_verifies_its_artifacts(monkeypatch, tmp_path):
    image_id = "sha256:" + "a" * 64
    manifest = {"files": {}}
    for key in validate_release.ARTIFACTS["dense"]:
        (tmp_path / key).write_bytes(key.encode())
        manifest["files"][key] = hashlib.sha256(key.encode()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    calls = []

    def run(*args):
        calls.append(args)
        if args[0] == "image":
            return image_id
        assert args[-2:] == (image_id, "/app/manifests/v3.json")
        return json.dumps(manifest)

    monkeypatch.setattr(validate_release, "docker_output", run)
    with pytest.raises(ValueError, match="recorded image ID"):
        validate_release.preflight("rag:old", "unexpected", tmp_path, "dense")
    assert len(calls) == 1
    validate_release.preflight("rag:old", image_id, tmp_path, "dense")
    (tmp_path / "dense_index.faiss").write_bytes(b"corrupted")
    with pytest.raises(ArtifactIntegrityError, match="sha256 mismatch"):
        validate_release.preflight("rag:old", image_id, tmp_path, "dense")


def test_probe_keeps_credentials_local_and_queries_only_when_requested():
    calls = []
    redirect = False

    class Handler(BaseHTTPRequestHandler):
        def answer(self):
            calls.append((self.path, self.headers.get("X-API-Key")))
            if self.path == "/metadata" and redirect:
                self.send_response(302)
                self.send_header("Location", "/destination")
                self.end_headers()
                return
            payload = {
                "/health": {"service": "hybrid-rag", "status": "healthy"},
                "/ready": {"service": "hybrid-rag", "status": "ready"},
                "/metadata": {"service": "hybrid-rag"},
                "/api/v1/query": {"answer": "Stub.", "citations": [7], "error": None},
            }
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(payload[self.path]).encode())

        do_GET = answer
        do_POST = answer

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        port = server.server_port
        try:
            validate_release.probe(port, "test-key", 1, None)
            assert calls == [("/health", None), ("/ready", None), ("/metadata", "test-key")]
            validate_release.probe(port, "test-key", 1, "offline test")
            assert calls[-1] == ("/api/v1/query", "test-key")
            redirect = True
            with pytest.raises(HTTPError) as error:
                validate_release.probe(port, "test-key", 1, None)
            assert error.value.code == 302
            assert all(path != "/destination" for path, _ in calls)
        finally:
            server.shutdown()
            thread.join()
