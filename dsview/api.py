from dotenv import load_dotenv
import logging

from fastapi import FastAPI, HTTPException
from pydantic import HttpUrl
from sqlalchemy.exc import NoResultFound
from sqlmodel import SQLModel, Session, create_engine, select

from dsview.config import load_model_config, get_sqlite_url, setup_logger
from dsview.content.content_db_schema import InputContent
from dsview.obsidian.sync_vault import api_sync_vault
from dsview.ingest_source import IngestPipeline

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
        ingest_pipeline.ingest_content(content, session)


# TODO Update sqlite db and test locally


@app.patch("/relevance")
@api_sync_vault
async def relevance(link: HttpUrl, relevance: int):
    with Session(engine) as session:
        statement = select(InputContent).where(InputContent.link == link)
        result = session.exec(statement)

        try:
            input_content = result.one()
        except NoResultFound:
            raise HTTPException(
                status_code=404,
                detail="Provided link not found, cannot change relevance.",
            )

        input_content.relevance = relevance
        input_content.read_priority = 0
        input_content.already_read = True

        session.add(input_content)
        session.commit()
        session.refresh(input_content)

    logger.info("Updated relevance from link %s to %s", link, relevance)
