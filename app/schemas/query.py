from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="User question.",
    )


class SourceResponse(BaseModel):
    passage_id: int
    passage_text: str
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceResponse]