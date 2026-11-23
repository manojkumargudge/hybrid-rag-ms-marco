"""Run the P6 Docker smoke checks against the built application image."""

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8000"
API_KEY = "local-container-smoke-key"
REFERENCE = ROOT / "eval/results/20261118-v3-0d90c58/per_query.jsonl"
UVICORN = [
    "uvicorn", "--host", "0.0.0.0", "--port", "8000", "--no-proxy-headers", "--no-access-log",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_response(body: dict, expected_ids: list[int]) -> None:
    require(set(body) == {"answer", "citations", "latency_ms", "error"}, "Query schema changed")
    require(body["error"] is None, "Query returned a degraded answer")
    require(
        body["answer"] == f"Deterministic smoke answer. [pid:{expected_ids[0]}]",
        "Expected deterministic answer citing the first returned passage",
    )
    latency = body["latency_ms"]
    require(type(latency) in (int, float) and math.isfinite(latency) and latency >= 0,
            "Latency must be a finite non-negative number")
    citations = body["citations"]
    require(isinstance(citations, list), "Citations must be a list")
    for citation in citations:
        require(
            isinstance(citation, dict) and set(citation) == {"passage_id", "passage_text", "score"},
            "Citation schema changed",
        )
        require(type(citation["passage_id"]) is int, "Expected integer passage ID")
        require(isinstance(citation["passage_text"], str) and bool(citation["passage_text"].strip()),
                "Expected nonempty passage text")
        score = citation["score"]
        require(type(score) in (int, float) and math.isfinite(score), "Expected finite citation score")
    require([citation["passage_id"] for citation in citations] == expected_ids,
            "Returned citation IDs or their ranking differ from saved v3 evaluation")


def request(path: str, *, text: str | None = None, authenticated: bool = False) -> tuple[int, dict]:
    headers = {"X-API-Key": API_KEY} if authenticated else {}
    data = None
    if text is not None:
        data = json.dumps({"query": text, "top_k": 10}).encode()
        headers["Content-Type"] = "application/json"
    message = urllib.request.Request(BASE_URL + path, data=data, headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(message, timeout=50) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        with error:
            return error.code, json.load(error)


def wait_for(check, message: str, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.25)
    raise RuntimeError(message)


def in_container(phase: str) -> None:
    cases = json.loads(Path("/smoke/cases.json").read_text(encoding="utf-8"))
    if phase == "probe":
        def ready():
            try:
                health, ready_response = request("/health"), request("/ready")
            except urllib.error.URLError:
                return False
            return health == (200, {"status": "healthy", "service": "hybrid-rag"}) and (
                ready_response == (200, {"status": "ready", "service": "hybrid-rag"})
            )

        wait_for(ready, "Container failed /health or /ready checks", 120)
        status, _ = request("/api/v1/query", text=cases[0]["query"])
        require(status == 401, "Query route did not enforce auth")
        status, metadata = request("/metadata", authenticated=True)
        require(status == 200 and metadata["retrieval_mode"] == "dense"
                and metadata["rerank_depth"] == 20, "Expected served dense top-20 configuration")
        selected = cases[:3]
    else:
        selected = cases[3:]

    for case in selected:
        status, body = request("/api/v1/query", text=case["query"], authenticated=True)
        require(status == 200, f"Query failed with HTTP {status}")
        validate_response(body, case["ids"])
    print(f"{phase}: {len(selected)} authenticated query responses matched the saved ranking")


def docker(*args: str, check: bool = True, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], text=True, encoding="utf-8", errors="replace",
        capture_output=True, check=check, timeout=timeout,
    )


def mount(source: Path, target: str) -> str:
    return f"type=bind,source={source.resolve()},target={target},readonly"


def assert_startup_refused(image: str, name: str, artifacts: Path | None = None) -> None:
    bind = ["--mount", mount(artifacts, "/app/artifacts/v3")] if artifacts else []
    try:
        result = docker("run", "--name", name, "--network", "none", *bind, image,
                        check=False, timeout=90)
        logs = result.stdout + result.stderr
        require(result.returncode == 3 and '"error_type":"ArtifactIntegrityError"' in logs
                and "Application startup complete" not in logs,
                f"Expected artifact integrity startup failure of {name}:\n{logs}")
    finally:
        docker("rm", "-f", name, check=False)


def smoke(image: str, artifacts: Path) -> None:
    from app.artifacts import verify_artifacts

    manifest = artifacts / "manifest.json"
    verify_artifacts(manifest, artifacts / "dense_index.faiss", artifacts / "passages.parquet",
                     trusted_manifest_path=ROOT / "manifests/v3.json")
    name = f"rag-p6-{uuid.uuid4().hex[:12]}"
    with tempfile.TemporaryDirectory(prefix="rag-p6-") as temp:
        directory = Path(temp)
        directory.chmod(0o755)
        fixtures = directory / "fixtures"
        fixtures.mkdir(mode=0o755)
        corrupt = directory / "corrupt"
        corrupt.mkdir(mode=0o755)
        shutil.copyfile(manifest, corrupt / "manifest.json")
        (corrupt / "passages.parquet").write_bytes(b"invalid artifact")
        (corrupt / "dense_index.faiss").write_bytes(b"invalid artifact")
        for filename in ("docker_smoke.py", "docker_smoke_provider.py"):
            shutil.copyfile(ROOT / "scripts" / filename, fixtures / filename)
        with REFERENCE.open(encoding="utf-8") as source:
            cases = [
                {"query": row["query"], "ids": row["systems"]["dense+rerank(depth=20)"]["top10"]}
                for row in (json.loads(next(source)) for _ in range(4))
            ]
        (fixtures / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
        (fixtures / "drain-query").write_text(cases[-1]["query"], encoding="utf-8")
        fixture_mount = mount(fixtures, "/smoke")
        assert_startup_refused(image, f"{name}-missing")
        assert_startup_refused(image, f"{name}-corrupt", corrupt)
        print("Missing and corrupted artifacts refused startup", flush=True)

        client_name = f"{name}-client"
        try:
            docker(
                "run", "-d", "--name", name, "--network", "none", "--cpus", "2", "--memory", "4g",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--mount", mount(artifacts, "/app/artifacts/v3"), "--mount", fixture_mount,
                "-e", f"API_KEY={API_KEY}", "-e", "GROQ_API_KEY=offline-smoke-placeholder",
                "-e", "OMP_NUM_THREADS=2", "-e", "MKL_NUM_THREADS=2", "-e", "PYTHONPATH=/smoke",
                image, *UVICORN, "docker_smoke_provider:create_app", "--factory",
            )
            print(docker("exec", name, "python", "/smoke/docker_smoke.py", "probe",
                         timeout=200).stdout.strip(), flush=True)
            wait_for(
                lambda: docker("inspect", "-f", "{{.State.Health.Status}}", name).stdout.strip()
                == "healthy",
                "Docker HEALTHCHECK did not become healthy", 60,
            )
            config = json.loads(docker("inspect", "-f", "{{json .HostConfig}}", name).stdout)
            require(config["Memory"] == 4 * 2**30 and config["NanoCpus"] == 2 * 10**9,
                    "Docker did not record the requested CPU and memory limits")

            def server_logs() -> str:
                result = docker("logs", name)
                return result.stdout + result.stderr

            # The client shares the isolated loopback but outlives the server PID.
            with subprocess.Popen(
                ["docker", "run", "--rm", "--name", client_name, "--network", f"container:{name}",
                 "--mount", fixture_mount, image, "python", "/smoke/docker_smoke.py", "drain"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            ) as client:
                try:
                    wait_for(lambda: "smoke_provider_waiting" in server_logs(),
                             "Drain query did not reach the provider", 25)
                    docker("kill", "--signal", "TERM", name)
                    wait_for(lambda: "Waiting for connections to close" in server_logs(),
                             "Uvicorn did not begin draining the live connection", 15)
                    (fixtures / "release").touch()
                    output, _ = client.communicate(timeout=30)
                    require(client.returncode == 0, f"Drain request did not finish: {output}")
                    print(output.strip(), flush=True)
                    exit_code = int(docker("wait", name, timeout=20).stdout.strip())
                    logs = server_logs()
                    require(exit_code in (0, 143) and "Application shutdown complete" in logs
                            and "smoke_provider_closed" in logs,
                            f"Container did not shut down gracefully (exit {exit_code})")
                finally:
                    if client.poll() is None:
                        docker("rm", "-f", client_name, check=False)
                        client.kill()
        except (RuntimeError, subprocess.SubprocessError) as error:
            if isinstance(error, subprocess.CalledProcessError):
                print(error.stdout, error.stderr, file=sys.stderr)
            result = docker("logs", name, check=False)
            print(result.stdout, result.stderr, file=sys.stderr)
            raise
        finally:
            docker("rm", "-f", name, check=False)
    print("Docker health, readiness, limits, parity and graceful shutdown passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="hybrid-rag:ci")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts/v3")
    args = parser.parse_args()
    smoke(args.image, args.artifacts)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in {"probe", "drain"}:
        in_container(sys.argv[1])
    else:
        main()
