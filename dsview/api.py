import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import HttpUrl
from sqlalchemy.exc import NoResultFound
from sqlmodel import Session

from dsview.config import load_model_config, setup_logger
from dsview.db.ingest import update_content
from dsview.db.schemas import InputContent, engine
from dsview.ingest_source import IngestPipeline
from dsview.obsidian.sync_vault import api_sync_vault

load_dotenv()
setup_logger()

model_config = load_model_config()

logger = logging.getLogger(__name__)

ingest_pipeline = IngestPipeline()

app = FastAPI()


# TODO Remove support for path content
# TODO Add relevant images ???


@app.post("/ingest")
@api_sync_vault
async def ingest(content: InputContent):
    # validate content

    content = InputContent.model_validate(content)

    if content.source == "None":
        content.source = None

    # ? Maybe it is slower than session dependency
    with Session(engine) as session:
        await ingest_pipeline.async_ingest_content(content, session)


@app.patch("/relevance")
@api_sync_vault
async def relevance(link: HttpUrl, relevance: int):
    try:
        with Session(engine) as session:
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
