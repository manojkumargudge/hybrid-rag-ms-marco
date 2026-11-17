"""Small dependency-free HTTP load test for the health endpoint."""

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen


def request_latency(url: str) -> float:
    started_at = time.perf_counter()
    with urlopen(url, timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"Unexpected status: {response.status}")
        response.read()
    return (time.perf_counter() - started_at) * 1000


def percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        round((percentage / 100) * (len(ordered) - 1)),
    )
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8010/health")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()

    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        latencies = list(
            executor.map(
                request_latency,
                [args.url] * args.requests,
            )
        )

    elapsed_seconds = time.perf_counter() - started_at
    print(f"requests={len(latencies)} concurrency={args.concurrency}")
    print(f"throughput_rps={len(latencies) / elapsed_seconds:.2f}")
    print(f"p50_ms={statistics.median(latencies):.2f}")
    print(f"p95_ms={percentile(latencies, 95):.2f}")


if __name__ == "__main__":
    main()