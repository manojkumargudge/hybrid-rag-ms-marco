# Monitoring and incident response

## Monitor the service

Check public `/health` and `/ready` from a trusted monitoring path through the
same gateway used by clients. A running and healthy container can still return
503 on readiness. Scrape `/metrics` over a private path. Configure the scraper
to send `X-API-Key` from its secret store and keep it out of dashboards and
scrape logs. Scrapes count against the per-key rate limit, along with query and
metadata traffic. Each worker has its own metrics registry, rate limiter,
query slots, and circuit breaker. Scrape each worker separately if the
deployment is changed from the single-worker configuration.

| Signal | What to watch |
|---|---|
| `http_requests_total{status}` | Request rate and changes in 429 or 5xx. This metric includes probes and other HTTP routes. |
| `http_request_duration_seconds_bucket` | End-to-end latency. Compare p95 with the deployment's SLO at similar traffic. |
| `rag_stage_latency_seconds_bucket{stage}` | Separate retrieval, rerank, and generation latency before changing limits or rolling back. |
| `rag_errors_total{type}` | Break down rate limiting, query saturation, unready service, Groq timeouts, generation failures, and open-circuit fallbacks. Some labels appear only after the first matching event. |
| Docker and host metrics | Inspect CPU saturation or throttling, RSS against the 4 GiB limit, OOM kills, restart count, and available disk. The application exporter does not publish these container measurements. |

Use the following PromQL to distinguish slower inference from rejected work.

```promql
histogram_quantile(0.95, sum by (le, stage) (rate(rag_stage_latency_seconds_bucket[5m])))
sum by (type) (increase(rag_errors_total{type=~"rate_limited|query_capacity_exceeded|service_unavailable|groq_timeout|generation_failed|generation_circuit_open"}[5m]))
```

Start with alerts for three consecutive failed readiness probes, repeated
restarts or OOM kills, sustained 5xx above the error budget, and p95 latency
above the SLO for ten minutes with enough requests to make the window useful.
Alert on sustained capacity rejections and open-circuit fallbacks separately.
Tune durations and volume floors against observed traffic. A short burst of
rate-limit responses alone is usually expected when a client exceeds quota.
These are alert recommendations. The repository supplies no installed
monitoring stack or tested external alert routing.

Application logs include request ID, status, stage timings, and a hashed key
identifier. Correlate client `X-Request-ID` with `request_complete` and stage
events. Use `docker compose logs --tail 150 --no-color hybrid-rag` with the
release selection and project name. Logs can also include server lifecycle
messages. Avoid dumping request bodies, environment variables, or provider
credentials when collecting evidence.

## Triage and recover

Confirm the Compose project and intended image and artifact bundle. Check
the container state, then `/health`, `/ready`, and the relevant logs and metrics.
For container inspection, select fields such as `.Image`, `.State.Status`,
`.State.OOMKilled`, and `.RestartCount` using `--format`.

| Symptom | Response |
|---|---|
| Missing or corrupt artifacts | A start failure can leave no HTTP listener. Check mount source, permissions for the image's non-root user, retrieval mode, and the immutable image ID. Run the release preflight. Fetch the verified files into a new directory and stage a fresh bundle. Use the recorded previous bundle when recovery is time-sensitive. Never edit a trusted digest to accept a mismatched file. |
| Model initialization failure | Inspect startup logs and memory and disk headroom. Confirm the image contains the expected pinned snapshots and that runtime model settings match them. The runtime loads models offline. Restage the correct image or roll back instead of enabling runtime model downloads. |
| Groq outage or persistent timeouts | Watch `groq_timeout`, `generation_failed`, and `generation_circuit_open`. The response can be HTTP 200 with ranked passages, a null answer, and an error reason. Check provider status and host connectivity independently. Inspect credential provisioning privately and rotate only if needed. The per-worker breaker tries one recovery probe after cooldown. Keep retrieval available while it recovers. |
| High latency | Compare request and stage p95, CPU throttling, and memory. Check whether 429s coincide with a full query limit. During rollback compare the candidate to the previous known-good release at similar load. Preserve the 2-query per-worker limit unless capacity has been measured. A second worker nearly doubled peak RSS in the P4 run. |
| 429 | Follow `Retry-After`. `60` means the per-key rate limit and `1` means the concurrent query cap. Reduce client fan-out and use bounded retries. Verify which limit fired through `rag_errors_total` before changing it. |
| 503 or failed readiness | Check if the worker is starting, draining, missing required files or configuration, or unavailable behind the gateway. During a planned stop allow the configured 45-second grace period. If the candidate still fails direct loopback probes, pause routing and restore the prior release. A working `/health` alone is insufficient for recovery. |

Use the [rollback procedure](rollback.md) for a failed release, then confirm
readiness and authenticated metadata before resuming traffic. A successful
live generation probe can confirm provider recovery, while retrieval-only
fallbacks still indicate degraded generation.
