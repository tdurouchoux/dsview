from dotenv import load_dotenv
import logging

from fastapi import FastAPI
from sqlmodel import SQLModel, Session, create_engine

from dsview.config import load_model_config, load_db_config, setup_logger
from dsview.content.content_db_schema import InputContent
from dsview.obsidian.sync_vault import api_sync_vault
from dsview.ingest_source import IngestPipeline

load_dotenv()
setup_logger()

db_config = load_db_config()
model_config = load_model_config()

logger = logging.getLogger(__name__)
engine = create_engine(db_config.sqlite_url)
SQLModel.metadata.create_all(engine)


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

    ingest_pipeline = IngestPipeline()

    # ? Maybe it is slower than session dependency
    with Session(engine) as session:
        ingest_pipeline.ingest_content(content, session)
