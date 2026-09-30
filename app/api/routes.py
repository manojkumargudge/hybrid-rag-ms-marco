from logging import getLogger

from fastapi import APIRouter, HTTPException, Request

from app.schemas.query import QueryRequest, QueryResponse

logger = getLogger(__name__)

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
        rag_service = request.app.state.get_rag_service()
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "rag_service_unavailable",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=503,
            detail="RAG service is not ready.",
        ) from None

    try:
        result = rag_service.answer(
            query=query_request.query,
            top_k=query_request.top_k,
        )

        return QueryResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        logger.warning(
            "answer_generation_failed",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=502,
            detail="Answer generation failed.",
        ) from None

    except Exception as exc:  # noqa: BLE001 - keep provider details out of responses
        logger.error(
            "rag_pipeline_failed",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=500,
            detail="RAG pipeline failed.",
        ) from None