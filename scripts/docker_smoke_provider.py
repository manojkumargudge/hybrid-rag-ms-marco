"""Test-only Groq replacement injected into the offline Docker smoke container."""

import re
import time
from pathlib import Path
from types import SimpleNamespace

from app.generation import groq_generator

CONTROL = Path("/smoke")
ANSWER = "Deterministic smoke answer."


class FakeGroq:
    def __init__(self, **_):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.complete))

    def complete(self, *, messages, **_):
        prompt = messages[-1]["content"]
        context, query = prompt.split("</retrieved_context>", 1)
        reference = re.search(r"^\[pid:[0-9]+\]$", context, re.MULTILINE)
        if reference is None:
            raise ValueError("The generation prompt has no returned passage ID")
        if "Question:\n" + (CONTROL / "drain-query").read_text(encoding="utf-8") + "\n" in query:
            print("smoke_provider_waiting", flush=True)
            deadline = time.monotonic() + 40
            while not (CONTROL / "release").exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Smoke driver failed to release the request")
                time.sleep(0.05)
        return SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=f"{ANSWER} {reference[0]}"))
        ])

    def close(self):
        print("smoke_provider_closed", flush=True)


def create_app():
    groq_generator.Groq = FakeGroq
    from app.main import app

    return app
