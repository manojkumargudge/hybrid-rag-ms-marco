import httpx

from scripts import load_test


def test_benchmark_model_pins_ignore_environment_overrides(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "other-model")
    monkeypatch.setenv("RERANKER_MODEL_REVISION", "other-revision")
    models = load_test.model_environment()
    assert models["EMBEDDING_MODEL"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert models["RERANKER_MODEL_REVISION"] == "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    assert len(models) == 4


def test_load_report_distinguishes_successful_throughput_from_rejections(monkeypatch):
    backoffs = []

    def handle(request):
        assert request.url.path == "/api/v1/query"
        if b"busy" in request.content:
            return httpx.Response(429)
        return httpx.Response(
            200, json={"citations": [{"passage_id": i} for i in range(10)], "error": None}
        )

    monkeypatch.setattr(
        load_test, "client_options",
        lambda base_url: {"base_url": base_url, "transport": httpx.MockTransport(handle)},
    )
    monkeypatch.setattr(load_test.time, "sleep", backoffs.append)
    report = load_test.measure("http://localhost", ["ok", "busy", "ok", "busy"], clients=2)
    assert report["statuses"] == {"200": 2, "429": 2}
    assert report["attempts"] == 4
    assert report["attempts_per_second"] > report["successful_requests_per_second"] > 0
    assert 0 <= report["p50_ms"] <= report["p95_ms"] <= report["p99_ms"]
    assert backoffs == [1]


def test_nearest_rank_percentiles_include_the_tail_and_handle_no_successes():
    samples = list(range(1, 101))
    assert load_test.percentile(samples, 0.99) == 99
    assert load_test.percentile([23, 11], 0.50) == 11
    assert load_test.percentile([], 0.99) is None


def test_malformed_success_body_is_counted_as_an_error():
    def handle(_request):
        return httpx.Response(200, content=b"not json")

    with httpx.Client(base_url="http://localhost", transport=httpx.MockTransport(handle)) as client:
        status, _, ids = load_test.query(client, "question")
    assert status == "unexpected_response"
    assert ids == []
