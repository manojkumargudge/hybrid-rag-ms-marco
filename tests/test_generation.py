from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from app.generation.groq_generator import GroqGenerator


def _response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _generator(client: Mock, max_retries: int = 1) -> GroqGenerator:
    generator = GroqGenerator.__new__(GroqGenerator)
    generator.model_name = "test-model"
    generator.client = client
    generator.max_retries = max_retries
    return generator


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