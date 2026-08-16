import asyncio
from functools import lru_cache
import logging
from urllib.parse import urlunparse

import logfire
from pydantic import HttpUrl
from rich.progress import track
from sqlmodel import Session

from dsview.db.ingest import save_content, save_failed_ingestion
from dsview.db.query import get_content
from dsview.db.schemas import InputContent
from dsview.db.schemas.extraction_schema import ExtractionResult
from dsview.extraction.content_extraction import ContentExtractor
from dsview.extraction.content_loader import (
    YOUTUBE_HOSTS,
    ContentLoader,
    get_content_loader,
)
from dsview.extraction.topic_er import TopicMerge
from dsview.obsidian import write_notes

logger = logging.getLogger(__name__)

# TODO Add medium hosts as configuration
MEDIUM_HOSTS = ["medium.com", "towardsdatascience.com", "netflixtechblog.com"]
IGNORE_CLEAN_HOSTS = list(YOUTUBE_HOSTS)


@lru_cache(maxsize=100)
def clean_content_url(link: HttpUrl) -> HttpUrl:
    # remove query and fragment from url

    if link.host not in IGNORE_CLEAN_HOSTS:
        clean_url = HttpUrl(urlunparse((link.scheme, link.host, link.path, "", "", "")))
    else:
        clean_url = link

    if clean_url.host in MEDIUM_HOSTS:
        logger.info("Received a medium link, redirecting to readmedium")
        clean_url = HttpUrl("https://readmedium.com/" + link.path)

    return clean_url


class IngestPipeline:
    def __init__(self, rebuild_mode: bool = False):
        self.content_extractor = ContentExtractor()
        self.rebuild_mode = rebuild_mode

    def get_existing_content(
        self, link: HttpUrl, session: Session
    ) -> InputContent | None:
        return get_content(clean_content_url(link), session)

    # make ingest_source load
    def _load(self, content: InputContent) -> ContentLoader:
        content_loader = get_content_loader(content.link)
        content_loader.load()

        return content_loader

    async def _extract(
        self, content_loader: ContentLoader, content_id: int, session: Session
    ) -> tuple[ExtractionResult, list[TopicMerge]]:

        return await self.content_extractor.extract_content(
            content_loader,
            session,
            content_id,
        )

    @logfire.instrument("Updating vault")
    def _write(
        self,
        content: InputContent,
        extraction_result: ExtractionResult,
        topic_merges: list[TopicMerge],
    ):
        logger.info("Writing extraction to Obsidian notes")

        # must be done before
        write_notes.write_content_note(content, extraction_result)
        write_notes.write_and_update_topic_list_notes(
            extraction_result.topics, topic_merges
        )

    def register_content(self, content: InputContent, session: Session) -> InputContent:
        """Persist the content row before any ingestion work happens.

        Lets a caller that ingests in the background commit the row up front, so
        that a later status lookup can tell an unknown link apart from an
        ingestion that simply hasn't written anything yet.
        """
        if isinstance(content.link, HttpUrl):
            content.link = clean_content_url(content.link)

        content = save_content(content, session)
        session.commit()

        return content

    async def async_ingest_content(
        self, content: InputContent, session: Session, already_saved: bool = False
    ) -> Exception | None:
        original_link = str(content.link)

        if isinstance(content.link, HttpUrl):
            content.link = clean_content_url(content.link)

        if not self.rebuild_mode and not already_saved:
            content = save_content(content, session)
            session.commit()

        logger.info("Ingesting content : %s", content.link)

        error: Exception | None = None
        try:
            content_loader = self._load(content)

            extraction_result, topic_merges = await self._extract(
                content_loader, content.id, session
            )

            self._write(
                content,
                extraction_result,
                topic_merges,
            )

            logger.info("Ingestion successful.")

        except Exception as caught_error:
            logger.exception("Failed to ingest content : %s", content.link)
            session.rollback()

            save_failed_ingestion(content, original_link, caught_error, session)
            error = caught_error

        session.commit()
        return error

    @logfire.instrument("Ingestion pipeline")
    def ingest_content(self, content: InputContent, session: Session):
        asyncio.run(self.async_ingest_content(content, session))

    def ingest_content_list(self, content_list: list[InputContent], session: Session):
        logger.info("Total number of content to ingest : %s", len(content_list))
        i = 1

        for content in track(content_list):
            logger.info("Content number : %s", i)
            self.ingest_content(content, session)
            i += 1
