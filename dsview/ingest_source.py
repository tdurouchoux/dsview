import asyncio
import logging
from urllib.parse import urlunparse

from pydantic import HttpUrl
from rich.progress import track
from sqlmodel import Session

from dsview.config import load_model_config
from dsview.db.ingest import save_content, save_failed_ingestion
from dsview.db.schemas import InputContent
from dsview.extraction.content_extraction import ContentExtractor
from dsview.extraction.content_loader import (
    ContentLoader,
    get_content_loader,
)
from dsview.extraction.models.topics_extraction import DataScienceTopic
from dsview.obsidian import write_notes

logger = logging.getLogger(__name__)
model_config = load_model_config()

# TODO Add medium hosts as configuration
MEDIUM_HOSTS = ["medium.com", "towardsdatascience.com", "netflixtechblog.com"]
IGNORE_CLEAN_HOSTS = ["www.youtube.com"]


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
            clean_url = HttpUrl("https://readmedium.com/" + str(clean_url))

        return clean_url

    # make ingest_source load
    def _load(self, content: InputContent) -> ContentLoader:
        logger.info("Loading input content")

        content_loader = get_content_loader(content.link, model_config.token_limit)
        content_loader.load()

        return content_loader

    async def _extract(
        self, content_loader: ContentLoader, content_id: int, session: Session
    ) -> tuple[DataScienceTopic, int]:
        logging.info("Launching content extraction")

        (updated_topics, new_topic_ids) = await self.content_extractor.extract_content(
            content_loader,
            session,
            content_id,
        )

        return (updated_topics, new_topic_ids)

    def _write(
        self,
        content: InputContent,
        hyperlink: str,
        updated_topics: list[DataScienceTopic],
        new_topic_ids: list[int],
        session: Session,
    ):
        logger.info("Writing extraction to Obsidian notes")

        # must be done before
        write_notes.write_content_note(content, hyperlink, session)
        write_notes.update_topic_list_notes(updated_topics, session)
        write_notes.write_topic_list_notes(new_topic_ids, session)

    async def async_ingest_content(self, content: InputContent, session: Session):
        original_link = str(content.link)

        if isinstance(content.link, HttpUrl):
            content.link = self._clean_content_url(content.link)

        if not self.rebuild_mode:
            content = save_content(content, session)

        logger.info("Ingesting content : %s", content.link)

        try:
            content_loader = self._load(content)

            (updated_topics, new_topic_ids) = await self._extract(
                content_loader, content.id, session
            )

            self._write(
                content,
                content_loader.get_hyperlink(),
                updated_topics,
                new_topic_ids,
                session,
            )

            logger.info("Ingestion successful.")

        except Exception as error:
            logger.exception("Failed to ingest content : %s", content.link)
            save_failed_ingestion(content, original_link, error, session)

    def ingest_content(self, content: InputContent, session: Session):
        asyncio.run(self.async_ingest_content(content, session))

    def ingest_content_list(self, content_list: list[InputContent], session: Session):
        for content in track(content_list):
            self.ingest_content(content, session)
