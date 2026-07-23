import logging
from typing import Annotated

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
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
from dsview.extraction.content_loader import PdfUrlLoader, get_content_loader
from dsview.ingest_source import IngestPipeline
from dsview.obsidian.sync_vault import (
    api_sync_vault,
    config as vault_config,
    init_vault,
    pull_changes,
    upload_changes,
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


async def _ingest_and_sync(content: InputContent, session: Session):
    await ingest_pipeline.async_ingest_content(content, session)

    if vault_config.github_vault.repository is not None:
        upload_changes("Adding content")


@app.post("/ingest")
async def ingest(
    content: InputContent, session: SessionDep, background_tasks: BackgroundTasks
):
    # validate content
    check_db_connection(session)

    content = InputContent.model_validate(content)

    if content.source == "None":
        content.source = None

    if vault_config.github_vault.repository is not None:
        pull_changes()

    if isinstance(get_content_loader(content.link), PdfUrlLoader):
        background_tasks.add_task(_ingest_and_sync, content, session)
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

    try:
        error = await ingest_pipeline.async_ingest_content(content, session)
    except ContentAlreadyExists as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    if error is not None:
        raise HTTPException(
            status_code=500,
            detail={
                "error_type": error.__class__.__name__,
                "error_message": str(error),
            },
        )

    if vault_config.github_vault.repository is not None:
        upload_changes("Adding content")


@app.get("/ingest/status")
async def ingest_status(link: HttpUrl, session: SessionDep):
    clean_link = ingest_pipeline._clean_content_url(link)
    content = get_content(clean_link, session)
    if content is None:
        return {"status": "pending"}

    failed = get_failed_ingestion(content.id, session)
    if failed is not None:
        return {
            "status": "failed",
            "error_type": failed.error_type,
            "error_message": failed.error_message,
        }

    extraction_results, _, _ = get_content_extraction(content.id, session)
    if extraction_results:
        return {"status": "success"}

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
