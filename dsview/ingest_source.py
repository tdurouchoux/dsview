from datetime import datetime
import logging

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import typer

from dsview.config import setup_logger
from dsview.content_extraction import ContentExtractor
from dsview.content_loader import get_content_loader
from dsview.notes_generator import NotesGenerator

load_dotenv()
setup_logger()
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def ingest_content(
    content_path: str,
    already_read: bool = False,
    upload_date: datetime = datetime.now(),
):
    logger.info("Received new content %s to ingest in knowledge base", content_path)

    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    content_extractor = ContentExtractor(llm)
    content_loader = get_content_loader(content_path)

    summary, topics = content_extractor.extract_content(content_loader)

    notes_generator = NotesGenerator(summary, topics, already_read, upload_date)
    notes_generator.generate_subjects_md()
    notes_generator.generate_content_md()

    logger.info("New content added")


if __name__ == "__main__":
    app()
