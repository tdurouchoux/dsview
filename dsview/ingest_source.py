import asyncio
import logging
from urllib.parse import urlunparse

from pydantic import HttpUrl
from rich.progress import track
from sqlmodel import Session

from dsview.db.ingest import save_content, save_failed_ingestion
from dsview.db.query import get_content
from dsview.db.schemas import InputContent
from dsview.db.schemas.extraction_schema import ExtractionResult, ExtractionTopic
from dsview.extraction.content_extraction import ContentExtractor
from dsview.extraction.content_loader import (
    YOUTUBE_HOSTS,
    ContentLoader,
    get_content_loader,
)
from dsview.extraction.models.topics_extraction import DataScienceTopic
from dsview.obsidian import write_notes

logger = logging.getLogger(__name__)

# TODO Add medium hosts as configuration
MEDIUM_HOSTS = ["medium.com", "towardsdatascience.com", "netflixtechblog.com"]
IGNORE_CLEAN_HOSTS = list(YOUTUBE_HOSTS)


class IngestPipeline:
    def __init__(self, rebuild_mode: bool = False):
        self.content_extractor = ContentExtractor()
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
            clean_url = HttpUrl("https://readmedium.com/" + link.path)

        return clean_url

    def get_existing_content(
        self, link: HttpUrl, session: Session
    ) -> InputContent | None:
        return get_content(self._clean_content_url(link), session)

    # make ingest_source load
    def _load(self, content: InputContent) -> ContentLoader:
        logger.info("Loading input content")

        content_loader = get_content_loader(content.link)
        content_loader.load()

        return content_loader

    async def _extract(
        self, content_loader: ContentLoader, content_id: int, session: Session
    ) -> ExtractionResult:
        logger.info("Launching content extraction")

        extraction_result = await self.content_extractor.extract_content(
            content_loader,
            session,
            content_id,
        )

        return extraction_result

    def _write(
        self,
        content: InputContent,
        extraction_result: ExtractionResult,
    ):
        logger.info("Writing extraction to Obsidian notes")

        # must be done before
        write_notes.write_content_note(content, extraction_result)
        write_notes.write_and_update_topic_list_notes(extraction_result.topics)

    async def async_ingest_content(
        self, content: InputContent, session: Session
    ) -> Exception | None:
        original_link = str(content.link)

        if isinstance(content.link, HttpUrl):
            content.link = self._clean_content_url(content.link)

        if not self.rebuild_mode:
            content = save_content(content, session)
            session.commit()

        logger.info("Ingesting content : %s", content.link)

        error: Exception | None = None
        try:
            content_loader = self._load(content)

            extraction_result = await self._extract(content_loader, content.id, session)

            self._write(
                content,
                extraction_result,
            )

            logger.info("Ingestion successful.")

        except Exception as caught_error:
            logger.exception("Failed to ingest content : %s", content.link)
            session.rollback()

            save_failed_ingestion(content, original_link, caught_error, session)
            error = caught_error

        session.commit()
        return error
        

    def ingest_content(self, content: InputContent, session: Session):
        asyncio.run(self.async_ingest_content(content, session))

    def ingest_content_list(self, content_list: list[InputContent], session: Session):
        logger.info("Total number of content to ingest : %s", len(content_list))
        i = 1

        for content in track(content_list):
            logger.info("Content number : %s", i)
            self.ingest_content(content, session)
            i += 1
