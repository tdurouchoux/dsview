import logging
import os

from langchain_openai import ChatOpenAI
from rich.progress import track
from sqlmodel import Session, SQLModel, select, Field

from dsview.content.content_loader import (
    get_content_loader,
)
from dsview.content.content_db_schema import InputContent
from dsview.extraction.content_extraction import ContentExtractor
from dsview.extraction.entity_resolution import ERSolver
from dsview.config import load_model_config
from dsview.obsidian.notes_generator import NotesGenerator

logger = logging.getLogger(__name__)
model_config = load_model_config()


class FailedIngestion(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    content_id: int = Field(unique=True, foreign_key="inputcontent.id")
    error_type: str
    error_message: str


class ContentAlreadyExists(Exception):
    def __init__(self, link: str):
        super().__init__(f"Content with link {link} already exists")


class IngestPipeline:
    def __init__(self, rebuild_mode: bool = False):
        self.llm = ChatOpenAI(temperature=0, model_name=model_config.name)
        self.content_extractor = ContentExtractor(self.llm)
        self.er_solver = ERSolver(self.llm)
        self.rebuild_mode = rebuild_mode

    def _add_in_db(self, content: InputContent, session: Session):
        existing_content = session.exec(
            select(InputContent).where(InputContent.link == content.link)
        ).first()

        if existing_content is not None:
            if self.rebuild_mode:
                return existing_content
            raise ContentAlreadyExists(content.link)

        session.add(content)
        session.commit()

        return content

    def _extract(self, content: InputContent, session: Session) -> NotesGenerator:
        content_loader = get_content_loader(content.link, model_config.token_limit)

        summary, content_description, topics, content_links = (
            self.content_extractor.extract_content(
                content_loader,
                session,
                content.id,
            )
        )

        topics = self.er_solver.run_topics_er(topics, session)

        notes_generator = NotesGenerator(
            content,
            content_loader.get_hyperlink(),
            summary,
            content_description,
            topics,
            content_links,
        )

        return notes_generator

    def _ingest(self, notes_generator: NotesGenerator) -> NotesGenerator:
        notes_generator.generate_topics_md()
        notes_generator.generate_content_md()
        notes_generator.insert_in_index()

    def _save_failed_ingestion(
        self, content: InputContent, error: Exception, session: Session
    ):
        logger.info("Saving in failed ingestion db.")

        failed_ingestion = FailedIngestion(
            content_id=content.id,
            error_type=error.__class__.__name__,
            error_message=str(error),
        )
        session.add(failed_ingestion)
        session.commit()

    def ingest_content(self, content: InputContent, session: Session):
        content = self._add_in_db(content, session)

        logger.info("Ingesting content : %s", content.link)

        try:
            notes_generator = self._extract(content, session)
            self._ingest(notes_generator)
            logger.info("Content ingested.")

        except Exception as error:
            logger.exception("Failed to ingest content : %s", content.link)
            self._save_failed_ingestion(content, error, session)

    def ingest_content_list(self, content_list: list[InputContent], session: Session):
        for content in track(content_list):
            self.ingest_content(content, session)
