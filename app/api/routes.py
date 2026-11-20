from logging import getLogger

from fastapi import APIRouter, HTTPException, Request

from app.metrics import ERRORS
from app.schemas.query import QueryRequest, QueryResponse

logger = getLogger(__name__)

# Authentication is enforced for every non-allowlisted path by
# app.middleware.ProductionMiddleware, so routes need no per-route auth.
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
    Run the RAG pipeline (retrieve -> rerank -> generate). Requires X-API-Key.

    If LLM generation fails after its retries, the response still returns the
    reranked passages, with answer=null and error set.
    """

    try:
        rag_service = request.app.state.get_rag_service()
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        ERRORS.labels(type="service_unavailable").inc()
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
        ERRORS.labels(type="invalid_request").inc()
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        ERRORS.labels(type="pipeline_error").inc()
        logger.warning(
            "answer_generation_failed",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=502,
            detail="Answer generation failed.",
        ) from None

    except Exception as exc:  # noqa: BLE001 - keep provider details out of responses
        ERRORS.labels(type="pipeline_error").inc()
        logger.error(
            "rag_pipeline_failed",
            extra={"error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=500,
            detail="RAG pipeline failed.",
        ) from None
