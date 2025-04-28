import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import HttpUrl
from sqlalchemy.exc import NoResultFound
from sqlmodel import Session, SQLModel, create_engine

from dsview.config import get_sqlite_url, load_model_config, setup_logger
from dsview.content.content_db_schema import InputContent, update_content
from dsview.ingest_source import IngestPipeline
from dsview.obsidian.sync_vault import api_sync_vault

load_dotenv()
setup_logger()

model_config = load_model_config()

logger = logging.getLogger(__name__)
engine = create_engine(get_sqlite_url())
SQLModel.metadata.create_all(engine)

ingest_pipeline = IngestPipeline()


app = FastAPI()


# TODO Remove support for path content
# TODO Maybe use prompt caching to avoid feeding content multiples times https://platform.openai.com/docs/guides/prompt-caching
# TODO Add relevant images ???
# TODO Add semantic search for topics


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


# TODO Update sqlite db and test locally


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
