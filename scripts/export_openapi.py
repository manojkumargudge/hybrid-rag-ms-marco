"""Export the served application's OpenAPI schema without starting the service."""

import json
from pathlib import Path

from app.main import app

SNAPSHOT = Path(__file__).resolve().parents[1] / "docs/openapi.json"


if __name__ == "__main__":
    SNAPSHOT.write_text(json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")
