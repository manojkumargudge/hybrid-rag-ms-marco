from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest
from groq import APIConnectionError, APIStatusError, APITimeoutError

from app.config import Settings
from app.generation.groq_generator import (
    GroqCircuitBreaker,
    GroqCircuitOpen,
    GroqGenerator,
)


def _response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _generator(client: Mock, max_retries: int = 1, timeout: float = 10.0,
               deadline: float = 20.0) -> GroqGenerator:
    generator = GroqGenerator.__new__(GroqGenerator)
    generator.model_name = "test-model"
    generator.client = client
    generator.max_retries = max_retries
    generator.timeout_seconds = timeout
    generator.deadline_seconds = deadline
    generator.breaker = GroqCircuitBreaker(3, 30)
    return generator


def test_breaker_ignores_stale_results_and_allows_one_recovery_probe(monkeypatch):
    from app.generation import groq_generator

    now = 0.0
    monkeypatch.setattr(groq_generator, "time", SimpleNamespace(monotonic=lambda: now))
    breaker = GroqCircuitBreaker(failure_threshold=1, cooldown_seconds=10)
    stale = breaker.admit()
    breaker.record(breaker.admit(), success=False)
    breaker.record(stale, success=True)
    with pytest.raises(GroqCircuitOpen):
        breaker.admit()

    now = 10.0
    probe = breaker.admit()
    with pytest.raises(GroqCircuitOpen):
        breaker.admit()
    breaker.record(probe, success=False)
    with pytest.raises(GroqCircuitOpen):
        breaker.admit()

    now = 20.0
    breaker.record(breaker.admit(), success=True)
    assert breaker.admit() == breaker.admit()


def test_groq_retries_transient_timeout_and_succeeds():
    client = Mock()
    client.chat.completions.create.side_effect = [
        TimeoutError(),
        _response("grounded answer"),
    ]
    generator = _generator(client)

    with patch("app.generation.groq_generator.time.sleep"):
        answer = generator.generate("question", [{"passage_text": "context"}])

    assert answer == "grounded answer"
    assert client.chat.completions.create.call_count == 2


def test_groq_exhausts_bounded_retries_as_runtime_error():
    client = Mock()
    client.chat.completions.create.side_effect = TimeoutError()
    generator = _generator(client, max_retries=1)

    with patch("app.generation.groq_generator.time.sleep"), pytest.raises(
        RuntimeError,
        match="LLM generation failed",
    ):
        generator.generate("question", [{"passage_text": "context"}])

    assert client.chat.completions.create.call_count == 2


# --- retry policy: only timeouts, 429 and 5xx -------------------------------

_REQUEST = httpx.Request("POST", "https://api.groq.test")


def _status(code: int) -> APIStatusError:
    return APIStatusError("error", response=httpx.Response(code, request=_REQUEST), body=None)


@pytest.mark.parametrize(
    "error",
    [APITimeoutError(request=_REQUEST), TimeoutError(), _status(429), _status(500), _status(503)],
    ids=["api-timeout", "timeout", "429", "500", "503"],
)
def test_timeouts_429_and_5xx_are_retried(error):
    client = Mock()
    client.chat.completions.create.side_effect = [error, _response("ok")]
    with patch("app.generation.groq_generator.time.sleep"):
        assert _generator(client, max_retries=2).generate("q", [{"passage_text": "c"}]) == "ok"
    assert client.chat.completions.create.call_count == 2


@pytest.mark.parametrize(
    "error",
    [_status(400), _status(401), _status(403), _status(404), _status(408), _status(409),
     _status(422), APIConnectionError(request=_REQUEST), ConnectionError(), ValueError()],
    ids=["400", "401", "403", "404", "408", "409", "422", "api-connection", "connection",
         "value"],
)
def test_other_errors_fail_immediately_without_retry(error):
    client = Mock()
    client.chat.completions.create.side_effect = error
    with patch("app.generation.groq_generator.time.sleep") as sleep, pytest.raises(
        RuntimeError, match="LLM generation failed"
    ):
        _generator(client, max_retries=2).generate("q", [{"passage_text": "c"}])
    assert client.chat.completions.create.call_count == 1
    sleep.assert_not_called()


# --- total deadline ------------------------------------------------------------

class FakeClock:
    """time.monotonic/time.sleep stand-in; each failed call costs `cost` s."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _slow_timeouts(clock, cost):
    def create(**kwargs):
        clock.now += min(cost, kwargs["timeout"])
        raise APITimeoutError(request=_REQUEST)
    return create


def test_per_attempt_timeout_is_capped_by_remaining_deadline():
    clock, client = FakeClock(), Mock()
    client.chat.completions.create.side_effect = _slow_timeouts(clock, cost=9.0)
    with patch("app.generation.groq_generator.time", clock), pytest.raises(RuntimeError):
        _generator(client, max_retries=2, timeout=10.0, deadline=20.0).generate(
            "q", [{"passage_text": "c"}])

    timeouts = [c.kwargs["timeout"] for c in client.chat.completions.create.call_args_list]
    # t=0: 10 s; t=9.2: 10 s; t=18.6: only 1.4 s left before the 20 s deadline.
    assert timeouts == pytest.approx([10.0, 10.0, 1.4])
    assert clock.now <= 20.0 + 1e-9


def test_no_attempt_or_backoff_starts_after_the_deadline():
    clock, client = FakeClock(), Mock()
    client.chat.completions.create.side_effect = _slow_timeouts(clock, cost=10.0)
    with patch("app.generation.groq_generator.time", clock), pytest.raises(
        RuntimeError, match="deadline exceeded"
    ):
        _generator(client, max_retries=10, timeout=10.0, deadline=20.0).generate(
            "q", [{"passage_text": "c"}])

    # t=0 -> 10, sleep 0.2, t=10.2 -> 20 (capped at 9.8 left); the next backoff
    # would cross 20 s, so it stops after 2 attempts despite max_retries=10.
    assert client.chat.completions.create.call_count == 2
    assert clock.now <= 20.0 + 1e-9


def test_generation_defaults():
    fields = Settings.model_fields
    assert fields["groq_timeout_seconds"].default == 10.0
    assert fields["generation_deadline_seconds"].default == 20.0
    assert fields["groq_max_retries"].default == 2
