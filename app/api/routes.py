from fastapi import APIRouter, HTTPException, Request

from app.schemas.query import QueryRequest, QueryResponse


router = APIRouter(
    prefix="/api/v1",
    tags=["RAG"],
)


@router.post(
    "/query",
    response_model=QueryResponse,
)
def query_rag(
    request: Request,
    query_request: QueryRequest,
) -> QueryResponse:
    """
    Execute the complete Hybrid RAG pipeline.
    """

    try:
        rag_service = request.app.state.rag_service

        result = rag_service.answer(
            query_request.query
        )

        return QueryResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="RAG pipeline failed.",
        ) from exc