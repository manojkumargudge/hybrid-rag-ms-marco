from copy import deepcopy

import pytest

from scripts import docker_smoke_provider
from scripts.docker_smoke import validate_response


@pytest.fixture
def response():
    return {
        "answer": "Deterministic smoke answer. [pid:123]",
        "citations": [
            {"passage_id": 123, "passage_text": "First passage.", "score": 1.0},
            {"passage_id": 456, "passage_text": "Second passage.", "score": 0.0},
        ],
        "latency_ms": 1.2,
        "error": None,
    }


def test_smoke_contract_checks_exact_citation_order(response):
    validate_response(response, [123, 456])
    shuffled = deepcopy(response)
    shuffled["citations"].reverse()
    with pytest.raises(RuntimeError, match="ranking differ"):
        validate_response(shuffled, [123, 456])


@pytest.mark.parametrize(("key", "invalid"), [
    ("answer", "Deterministic smoke answer. [pid:789]"),
    ("citations", [{"passage_id": "123", "passage_text": "First passage.", "score": 1.0}]),
    ("latency_ms", float("nan")),
    ("error", "generation_failed"),
])
def test_smoke_contract_rejects_bad_responses(response, key, invalid):
    response[key] = invalid
    with pytest.raises(RuntimeError):
        validate_response(response, [123, 456])


def test_smoke_provider_returns_actual_prompt_citation(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(docker_smoke_provider, "CONTROL", tmp_path)
    (tmp_path / "drain-query").write_text("drain me", encoding="utf-8")
    (tmp_path / "release").touch()
    groq = docker_smoke_provider.FakeGroq()
    response = groq.chat.completions.create(messages=[{
        "content": "<retrieved_context>\n[pid:418]\ntext\n</retrieved_context>\nQuestion:\ndrain me\n"
    }])
    groq.close()
    assert response.choices[0].message.content == "Deterministic smoke answer. [pid:418]"
    assert capsys.readouterr().out.splitlines() == ["smoke_provider_waiting", "smoke_provider_closed"]
