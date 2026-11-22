"""Benchmark the served dense pipeline over loopback with generation stubbed."""

import argparse
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path
from threading import Event, Thread

import httpx

from app.artifacts import sha256_file

REFERENCE = Path("eval/results/20261118-v3-0d90c58/per_query.jsonl")
VARIANTS = {
    "baseline": (1, 0, 0),
    "embedding-lru": (1, 256, 0),
    "reranker-batch-8": (1, 0, 8),
    "two-workers": (2, 0, 0),
}
CLIENT_COUNTS = (1, 5, 20)
BENCHMARK_KEY = "local-benchmark"


def percentile(samples: list[float], fraction: float) -> float | None:
    if not samples:
        return None
    return round(sorted(samples)[math.ceil(fraction * len(samples)) - 1], 3)


def query(client: httpx.Client, text: str) -> tuple[str, float, list[int]]:
    start = time.perf_counter()
    try:
        response = client.post("/api/v1/query", json={"query": text, "top_k": 10})
    except httpx.RequestError as exc:
        return type(exc).__name__, (time.perf_counter() - start) * 1000, []
    elapsed_ms = (time.perf_counter() - start) * 1000
    if response.status_code != 200:
        return str(response.status_code), elapsed_ms, []
    try:
        body = response.json()
        ids = [citation["passage_id"] for citation in body["citations"]]
        if body["error"] is None and len(ids) == 10 and all(type(pid) is int for pid in ids):
            return "200", elapsed_ms, ids
    except (KeyError, TypeError, ValueError):
        pass
    return "unexpected_response", elapsed_ms, []


def measure(base_url: str, queries: list[str], clients: int) -> dict:
    def worker(client: httpx.Client, texts: list[str]) -> list[tuple[str, float, list[int]]]:
        results = []
        for i, text in enumerate(texts):
            observation = query(client, text)
            results.append(observation)
            if observation[0] == "429" and i + 1 < len(texts):
                time.sleep(1)
        return results

    with ExitStack() as stack, ThreadPoolExecutor(max_workers=clients) as pool:
        sessions = [stack.enter_context(httpx.Client(**client_options(base_url))) for _ in range(clients)]
        start = time.perf_counter()
        results = list(pool.map(worker, sessions, (queries[i::clients] for i in range(clients))))
        elapsed = time.perf_counter() - start
    observations = [result for chunk in results for result in chunk]
    statuses = Counter(status for status, _, _ in observations)
    latencies = [latency for status, latency, _ in observations if status == "200"]
    return {
        "clients": clients, "attempts": len(observations), "statuses": dict(statuses),
        "elapsed_seconds": round(elapsed, 4),
        "attempts_per_second": round(len(observations) / elapsed, 3),
        "successful_requests_per_second": round(len(latencies) / elapsed, 3),
        "p50_ms": percentile(latencies, 0.50), "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
    }


def client_options(base_url: str) -> dict:
    return {
        "base_url": base_url, "headers": {"X-API-Key": BENCHMARK_KEY},
        "timeout": 120, "trust_env": False,
    }


class MemorySampler:
    def __init__(self, process):
        self.process = process
        self.stop = Event()
        self.samples: list[int] = []
        self.thread = Thread(target=self.sample, daemon=True)

    def sample(self) -> None:
        import psutil

        while not self.stop.is_set():
            try:
                processes = [self.process, *self.process.children(recursive=True)]
                total = 0
                for process in processes:
                    try:
                        total += process.memory_info().rss
                    except psutil.NoSuchProcess:
                        continue
                if total:
                    self.samples.append(total)
            except psutil.NoSuchProcess:
                return
            self.stop.wait(0.1)


def run_variant(name: str, workloads: dict[int, list[str]], rows: list[dict], threads: int) -> dict:
    import psutil

    workers, cache, batch = VARIANTS[name]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="hybrid-rag-bench-") as directory:
        root = Path(directory)
        env = {
            **os.environ,
            "API_KEY": BENCHMARK_KEY, "GROQ_API_KEY": "offline-benchmark",
            "RETRIEVAL_MODE": "dense", "RERANK_DEPTH": "20", "RATE_LIMIT_PER_MIN": "0",
            "MAX_CONCURRENT_QUERIES": "2", "TRUSTED_HOSTS": "127.0.0.1",
            "MANIFEST_PATH": "artifacts/v3/manifest.json", "TRUSTED_MANIFEST_PATH": "manifests/v3.json",
            "PASSAGES_PATH": "artifacts/v3/passages.parquet", "DENSE_INDEX_PATH": "artifacts/v3/dense_index.faiss",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
            "BENCHMARK_READY_DIR": directory, "BENCHMARK_EMBEDDING_CACHE": str(cache),
            "RERANK_BATCH_SIZE": str(batch or 32),
        }
        command = [
            sys.executable, "-m", "uvicorn", "scripts.manual.serve_no_llm:benchmark_app",
            "--factory", "--host", "127.0.0.1", "--port", str(port), "--workers", str(workers),
            "--no-proxy-headers", "--no-access-log",
        ]
        log_path = root / "server.log"
        with log_path.open("wb") as log:
            start = time.perf_counter()
            process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
            tree = psutil.Process(process.pid)
            memory = MemorySampler(tree)
            memory.thread.start()
            try:
                with httpx.Client(**client_options(base_url)) as client:
                    while True:
                        if process.poll() is not None or time.perf_counter() - start > 180:
                            raise RuntimeError(f"{name} failed to start")
                        if len(list(root.glob("*.ready"))) == workers:
                            try:
                                if client.get("/ready", timeout=1).status_code == 200:
                                    break
                            except httpx.RequestError:
                                pass
                        time.sleep(0.1)
                    cold_start = time.perf_counter() - start
                    meta = client.get("/metadata").raise_for_status().json()
                    if meta["retrieval_mode"] != "dense" or meta["rerank_depth"] != 20:
                        raise RuntimeError("Benchmark requires dense retrieval and rerank depth 20")
                    startup_peak = max(memory.samples)
                    hot_queries = workloads.get(1, [])[::2][:10]
                    for text in hot_queries:
                        if query(client, text)[0] != "200":
                            raise RuntimeError(f"{name} warmup request failed")
                    loads = []
                    for count, workload in workloads.items():
                        offset = max(0, len(memory.samples) - 1)
                        load = measure(base_url, workload, count)
                        load["peak_tree_rss_mib"] = round(max(memory.samples[offset:]) / 2**20, 2)
                        loads.append(load)
                        print(f"{name} clients={count} {load}", flush=True)
                    parity = {}
                    for row in rows:
                        status, _, ids = query(client, row["query"])
                        if status != "200":
                            raise RuntimeError(f"{name} parity request failed: {status}")
                        parity[str(row["qid"])] = ids
                        if len(parity) % 25 == 0:
                            print(f"{name} checked {len(parity)}/{len(rows)} rankings", flush=True)
                return {
                    "workers": workers, "embedding_cache_size": cache,
                    "reranker_batch_size": batch or None, "metadata": meta,
                    "cold_start_seconds": round(cold_start, 3),
                    "startup_peak_tree_rss_mib": round(startup_peak / 2**20, 2),
                    "loads": loads, "top10_by_qid": parity,
                }
            except Exception:
                print(log_path.read_text(encoding="utf-8", errors="replace")[-4000:], file=sys.stderr)
                raise
            finally:
                memory.stop.set()
                memory.thread.join()
                children = tree.children(recursive=True) if process.poll() is None else []
                process.terminate()
                for child in children:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                process.wait(timeout=20)
                _, alive = psutil.wait_procs(children, timeout=10)
                for child in alive:
                    child.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--parity-queries", type=int, default=200)
    parser.add_argument("--variant", choices=VARIANTS, action="append")
    parser.add_argument("--parity-only", action="store_true")
    args = parser.parse_args()
    if args.requests < 20 or args.threads < 1 or args.parity_queries < 0:
        parser.error("Use at least 20 requests, one thread, and a nonnegative parity count")
    if args.parity_only and not args.parity_queries:
        parser.error("--parity-only requires at least one parity query")
    try:
        import psutil
    except ImportError:
        parser.error("Install benchmark-only psutil first: python -m pip install psutil==7.0.0")
    with REFERENCE.open(encoding="utf-8") as source:
        all_rows = [json.loads(line) for line in source]
    rows = all_rows[:args.parity_queries]
    if len(rows) != args.parity_queries:
        parser.error("The parity set exceeds the available reference rows")
    rng = random.Random(42)
    workloads = {} if args.parity_only else {
        count: [
            all_rows[205 + run * 200 + (i // 2) % 200]["query"]
            if i % 2 else all_rows[200 + rng.randrange(5)]["query"]
            for i in range(args.requests)
        ] for run, count in enumerate(CLIENT_COUNTS)
    }
    output = {
        "system": platform.platform(), "cpu": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False), "logical_cores": os.cpu_count(),
        "memory_bytes": psutil.virtual_memory().total, "python": platform.python_version(),
        "psutil": psutil.__version__, "threads_per_worker": args.threads,
        "packages": {
            name: version(name) for name in ("torch", "sentence-transformers", "faiss-cpu", "uvicorn", "httpx")
        },
        "reference": REFERENCE.as_posix(), "reference_sha256": sha256_file(REFERENCE),
        "trusted_manifest_sha256": sha256_file(Path("manifests/v3.json")),
        "requests_per_load": args.requests if workloads else 0,
        "warmup_requests": 10 if workloads else 0, "top_k": 10,
        "backoff_after_429_seconds": 1,
        "parity_queries": args.parity_queries, "workload_seed": 42,
        "query_capacity_per_worker": 2, "rate_limit_per_min": 0, "variants": {},
    }
    expected = {
        str(row["qid"]): row["systems"]["dense+rerank(depth=20)"]["top10"] for row in rows
    }
    baseline = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    variants = args.variant or VARIANTS
    for name in variants:
        result = run_variant(name, workloads, rows, args.threads)
        rankings = result.pop("top10_by_qid")
        if name == "baseline":
            baseline = rankings
        result["top10_mismatches_vs_baseline"] = (
            [qid for qid in rankings if rankings[qid] != baseline[qid]] if baseline and rows else None
        )
        result["top10_mismatches_vs_saved_eval"] = (
            [qid for qid in rankings if rankings[qid] != expected[qid]] if rows else None
        )
        if rows:
            result["top10_by_qid"] = rankings
        output["variants"][name] = result
        args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
