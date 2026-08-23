import asyncio
import logging
from typing import Annotated

import logfire
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import HttpUrl
from sqlalchemy.exc import NoResultFound
from sqlmodel import Session
from starlette.responses import JSONResponse
from tenacity import RetryError

from dsview.config import setup_logger
from dsview.db import check_db_connection, engine
from dsview.db.ingest import update_content
from dsview.db.query import (
    get_content_by_id,
    get_content_extraction,
    get_failed_ingestion,
)
from dsview.db.schemas import InputContent
from dsview.ingest_source import IngestPipeline
from dsview.obsidian.sync_vault import (
    CommandFailed,
    api_sync_vault,
    async_pull_changes,
    async_upload_changes,
    init_vault,
)
from dsview.obsidian.sync_vault import (
    config as vault_config,
)

# TODO merge setup and sync_vault

load_dotenv()
setup_logger(enable_logfire=True, service_name="dsview-api")
init_vault()

logger = logging.getLogger(__name__)

# Built after logfire.configure(): constructing a Mistral provider imports
# providers/mistral.py, which runs MistralAIInstrumentor().instrument() at import
# and binds whatever tracer provider is current.
ingest_pipeline = IngestPipeline()

app = FastAPI()

logfire.instrument_fastapi(app)
# logfire.instrument_sqlalchemy(engine=engine)
# Instrumenting sqlalchemy won't be really usefull because the
# database is tiny

session_content_request = logfire.metric_counter(
    name="session_content_request",
    unit="1",
    description="Number of api ingestion request since last deployment",
)
session_failed_ingestion = logfire.metric_counter(
    name="session_failed_ingestion",
    unit="1",
    description="Number of failed ingestion since last deployment",
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "detail": exc.detail},
    )


# TODO Remove support for path content
# TODO Add relevant images ???


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]

# Strong references to background ingest tasks, so they aren't garbage-collected
# while detached from the request/response cycle.
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

    logger.exception("Background ingest task failed", exc_info=exc)


async def _run_ingest_and_sync(content_id: int) -> None:
    # The row was committed by the request handler, so it is re-read by id here
    # rather than carried over: the request session is closed by now and the
    # instance detached from it.
    with Session(engine) as session:
        content = get_content_by_id(content_id, session)
        link = str(content.link)

        error = await ingest_pipeline.async_ingest_content(
            content, session, already_saved=True
        )

    if error is None and vault_config.github_vault.repository is not None:
        try:
            await async_upload_changes("Adding content")
        except CommandFailed:
            logger.exception("Failed to push vault changes after ingesting %s", link)

    # Re-raised so the task records it and it surfaces as a failure rather than
    # being swallowed by the background task.
    if error is not None:
        session_failed_ingestion.add(1)
        raise error


@app.post("/ingest")
async def ingest(content: InputContent, session: SessionDep):
    # validate content
    check_db_connection(session)

    session_content_request.add(1)

    content = InputContent.model_validate(content)

    if content.source == "None":
        content.source = None

    if ingest_pipeline.get_existing_content(content.link, session) is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Content with link {content.link} already exists",
        )

    if vault_config.github_vault.repository is not None:
        await async_pull_changes()

    # Committed before returning 202 so that polling /ingest/status immediately
    # reports "pending" rather than 404-ing on a row the task hasn't written yet.
    content = ingest_pipeline.register_content(content, session)

    task = asyncio.create_task(_run_ingest_and_sync(content.id))
    _track_background_ingest(task)

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


@app.get("/ingest/status")
async def ingest_status(link: HttpUrl, session: SessionDep):
    content = ingest_pipeline.get_existing_content(link, session)

    # /ingest commits the content row before returning, so a missing row means
    # the link was never submitted. Returned rather than raised: the
    # HTTPException handler would rewrite the body to status "error".
    if content is None:
        return JSONResponse(
            status_code=404,
            content={
                "status": "unknown",
                "detail": f"No ingestion found for link {link}",
            },
        )

    extraction_results, _, _ = get_content_extraction(content.id, session)
    if extraction_results:
        return {"status": "success"}

    # The ingestion failed, but reporting that status is a success: the state
    # belongs in the body, not in the status code.
    failed = get_failed_ingestion(content.id, session)
    if failed is not None:
        return {
            "status": "failed",
            "detail": failed.error_message,
        }

    return JSONResponse(status_code=202, content={"status": "pending"})


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
        logger.exception("Health check failed")
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {e!s}",
        )
