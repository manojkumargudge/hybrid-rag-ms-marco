from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        strict=True,
        min_length=1,
        max_length=2000,
        description="User question.",
    )
    top_k: int = Field(default=5, strict=True, ge=1, le=20)


class SourceResponse(BaseModel):
    passage_id: int
    passage_text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[SourceResponse]
    latency_ms: float = Field(ge=0)