from dotenv import load_dotenv
import logging

from langchain_openai import ChatOpenAI
from fastapi import FastAPI

from dsview.config import load_model_config, setup_logger
from dsview.content.input_content import InputContent
from dsview.extraction.content_extraction import ContentExtractor
from dsview.extraction.entity_resolution import ERSolver
from dsview.ingest_source import ingest_single_content
from dsview.obsidian.sync_vault import api_sync_vault

load_dotenv()
setup_logger()

logger = logging.getLogger(__name__)
app = FastAPI()

model_config = load_model_config()


@app.post("/ingest")
@api_sync_vault
async def ingest(content: InputContent):
    llm = ChatOpenAI(temperature=0, model_name=model_config.name)
    content_extractor = ContentExtractor(llm)
    er_solver = ERSolver(llm)

    logger.info("Received new content %s to ingest in knowledge base", content.link)

    ingest_single_content(content, content_extractor, er_solver)
