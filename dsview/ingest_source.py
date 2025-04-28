import asyncio
import logging
from urllib.parse import urlunparse

from pydantic import HttpUrl
from rich.progress import track
from sqlmodel import Field, Session, SQLModel, select

from dsview.config import load_model_config
from dsview.content.content_db_schema import InputContent
from dsview.content.content_loader import (
    get_content_loader,
)
from dsview.extraction.content_extraction import ContentExtractor
from dsview.extraction.entity_resolution import ERSolver
from dsview.obsidian.notes_generator import NotesGenerator

logger = logging.getLogger(__name__)
model_config = load_model_config()

# TODO Add medium hosts as configuration
MEDIUM_HOSTS = ["medium.com", "towardsdatascience.com", "netflixtechblog.com"]
IGNORE_CLEAN_HOSTS = ["www.youtube.com"]


class FailedIngestion(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    content_id: int = Field(unique=True, foreign_key="inputcontent.id")
    original_link: str
    error_type: str
    error_message: str


class ContentAlreadyExists(Exception):
    def __init__(self, link: str):
        super().__init__(f"Content with link {link} already exists")


class IngestPipeline:
    def __init__(self, rebuild_mode: bool = False):
        self.content_extractor = ContentExtractor()
        self.er_solver = ERSolver()
        self.rebuild_mode = rebuild_mode

    def _clean_content_url(self, link: HttpUrl) -> HttpUrl:
        # remove query and fragment from url

        logger.info("Cleaning content url")

        if link.host not in IGNORE_CLEAN_HOSTS:
            clean_url = HttpUrl(
                urlunparse((link.scheme, link.host, link.path, "", "", ""))
            )
        else:
            clean_url = link

        if clean_url.host in MEDIUM_HOSTS:
            logger.info("Received a medium link, redirecting to readmedium")
            clean_url = HttpUrl("https://readmedium.com/" + str(clean_url))

        return clean_url

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

    async def _extract(self, content: InputContent, session: Session) -> NotesGenerator:
        content_loader = get_content_loader(content.link, model_config.token_limit)

        summary, content_description, topics, content_links = (
            await self.content_extractor.extract_content(
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

    def _save_failed_ingestion(
        self,
        content: InputContent,
        original_link: str,
        error: Exception,
        session: Session,
    ):
        logger.info("Saving in failed ingestion db.")

        failed_ingestion = FailedIngestion(
            content_id=content.id,
            original_link=original_link,
            error_type=error.__class__.__name__,
            error_message=str(error),
        )
        session.add(failed_ingestion)
        session.commit()


    async def async_ingest_content(self, content: InputContent, session: Session):
        original_link = str(content.link)

        if isinstance(content.link, HttpUrl):
            content.link = self._clean_content_url(content.link)

        content = self._add_in_db(content, session)

        logger.info("Ingesting content : %s", content.link)

        try:
            notes_generator = await self._extract(content, session)
            self._ingest(notes_generator)
            logger.info("Content ingested.")

        except Exception as error:
            logger.exception("Failed to ingest content : %s", content.link)
            self._save_failed_ingestion(content, original_link, error, session)

    def ingest_content(self, content: InputContent, session: Session):
        asyncio.run(self.async_ingest_content(content, session))

    def ingest_content_list(self, content_list: list[InputContent], session: Session):
        for content in track(content_list):
            self.ingest_content(content, session)
