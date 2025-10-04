import logging
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import HttpUrl
from sqlalchemy.exc import NoResultFound
from sqlmodel import Session
from tenacity import RetryError

from dsview.config import load_model_config, setup_logger
from dsview.db import engine, check_db_connection
from dsview.db.ingest import update_content
from dsview.db.schemas import InputContent
from dsview.ingest_source import IngestPipeline
from dsview.obsidian.sync_vault import api_sync_vault, init_vault

# TODO merge setup and sync_vault

load_dotenv()
setup_logger()
init_vault()

model_config = load_model_config()

logger = logging.getLogger(__name__)

ingest_pipeline = IngestPipeline()

app = FastAPI()


# TODO Remove support for path content
# TODO Add relevant images ???


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


@app.post("/ingest")
@api_sync_vault
async def ingest(content: InputContent, session: SessionDep):
    # validate content
    check_db_connection(session)

    content = InputContent.model_validate(content)

    if content.source == "None":
        content.source = None

    await ingest_pipeline.async_ingest_content(content, session)


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
