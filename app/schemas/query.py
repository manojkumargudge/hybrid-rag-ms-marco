from pydantic import BaseModel, Field, field_validator

from app.config import settings


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        strict=True,
        min_length=1,
        description="User question (at most MAX_QUERY_CHARS characters; default 2,000).",
    )
    top_k: int = Field(default=5, strict=True, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def _cap_length(cls, value: str) -> str:
        if len(value) > settings.max_query_chars:
            raise ValueError(
                f"query must be at most {settings.max_query_chars} characters"
            )
        return value


class SourceResponse(BaseModel):
    passage_id: int
    passage_text: str
    score: float


class QueryResponse(BaseModel):
    answer: str | None = Field(
        description="Grounded answer, or null if generation failed or the circuit is open."
    )
    citations: list[SourceResponse]
    latency_ms: float = Field(ge=0)
    error: str | None = Field(
        default=None,
        description="Set to 'generation_failed' or 'generation_circuit_open' on fallback.",
    )
