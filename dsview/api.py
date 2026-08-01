import asyncio
import logging
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import HttpUrl
from sqlalchemy.exc import NoResultFound
from sqlmodel import Session
from starlette.responses import JSONResponse
from tenacity import RetryError

from dsview.config import setup_logger
from dsview.db import engine, check_db_connection
from dsview.db.ingest import ContentAlreadyExists, update_content
from dsview.db.query import get_content, get_content_extraction, get_failed_ingestion
from dsview.db.schemas import InputContent
from dsview.ingest_source import IngestPipeline
from dsview.obsidian.sync_vault import (
    CommandFailed,
    api_sync_vault,
    async_pull_changes,
    async_upload_changes,
    config as vault_config,
    init_vault,
)

# TODO merge setup and sync_vault

load_dotenv()
setup_logger()
init_vault()

logger = logging.getLogger(__name__)

ingest_pipeline = IngestPipeline()

app = FastAPI()


# TODO Remove support for path content
# TODO Add relevant images ???


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]

# nginx-ingress on this route has no proxy-read-timeout override (falls back to
# nginx's 60s default). Stay comfortably under that so a response always beats the
# proxy kill, leaving margin for check_db_connection/git pull/response serialization.
INGEST_SYNC_TIMEOUT_SECONDS = 50.0

# Strong references to ingest tasks still running past the sync timeout, so they
# aren't garbage-collected while detached from the request/response cycle.
_background_ingest_tasks: set[asyncio.Task] = set()


def _track_background_ingest(task: asyncio.Task) -> None:
    _background_ingest_tasks.add(task)
    task.add_done_callback(_background_ingest_tasks.discard)
    task.add_done_callback(_log_background_ingest_result)


def _log_background_ingest_result(task: asyncio.Task) -> None:
    if task.cancelled():
        return

    exc = task.exception()
    if exc is None:
        return

    if isinstance(exc, ContentAlreadyExists):
        logger.info("Background ingest found link already existed: %s", exc)
        return

    logger.error("Background ingest task failed: %s", exc, exc_info=exc)


async def _run_ingest_and_sync(content: InputContent) -> Exception | None:
    with Session(engine) as session:
        error = await ingest_pipeline.async_ingest_content(content, session)
        # ContentAlreadyExists, if raised, propagates out of this block and out of
        # this function to whoever is awaiting/tracking the task.

    if error is None and vault_config.github_vault.repository is not None:
        try:
            await async_upload_changes("Adding content")
        except CommandFailed:
            logger.exception(
                "Failed to push vault changes after ingesting %s", content.link
            )

    return error


@app.post("/ingest")
async def ingest(content: InputContent, session: SessionDep):
    # validate content
    check_db_connection(session)

    content = InputContent.model_validate(content)

    if content.source == "None":
        content.source = None

    if vault_config.github_vault.repository is not None:
        await async_pull_changes()

    task = asyncio.create_task(_run_ingest_and_sync(content))
    _track_background_ingest(task)

    _done, pending = await asyncio.wait({task}, timeout=INGEST_SYNC_TIMEOUT_SECONDS)

    if task in pending:
        return JSONResponse(
            status_code=202,
            content={
                "status": "processing",
                "detail": (
                    "Ingestion is running in the background; poll "
                    "GET /ingest/status?link=<link> for progress."
                ),
            },
        )

    exc = task.exception()
    if exc is not None:
        if isinstance(exc, ContentAlreadyExists):
            raise HTTPException(status_code=409, detail=str(exc))
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
            },
        )

    error = task.result()
    if error is not None:
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": error.__class__.__name__,
                "error_message": str(error),
            },
        )


@app.get("/ingest/status")
async def ingest_status(link: HttpUrl, session: SessionDep):
    clean_link = ingest_pipeline._clean_content_url(link)
    content = get_content(clean_link, session)
    if content is None:
        return {"status": "pending"}

    extraction_results, _, _ = get_content_extraction(content.id, session)
    if extraction_results:
        return {"status": "success"}

    failed = get_failed_ingestion(content.id, session)
    if failed is not None:
        return {
            "status": "failed",
            "error_type": failed.error_type,
            "error_message": failed.error_message,
        }

    return {"status": "pending"}


@app.patch("/relevance")
@api_sync_vault
async def relevance(link: HttpUrl, relevance: int, session: SessionDep):
    check_db_connection(session)

    try:
        update_content(
            session,
            content_link=link,
            already_read=True,
            read_priority=0,
            relevance=relevance,
        )
    except NoResultFound:
        raise HTTPException(
            status_code=404,
            detail="Provided link not found, cannot change relevance.",
        )

    logger.info("Updated relevance from link %s to %s", link, relevance)


# TODO Test this endpoint
@app.get("/health")
async def health_check(session: SessionDep):
    """Health check endpoint that verifies database connectivity."""

    try:
        check_db_connection(session)
        # Simple query to test database connection
        return {"status": "healthy", "database": "connected"}

    except RetryError as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}",
        )
