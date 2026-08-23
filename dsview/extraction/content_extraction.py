import asyncio
import logging

import logfire
from sqlmodel import Session

from dsview.db.ingest import embed_and_format_extraction_results
from dsview.db.schemas import ExtractionResult
from dsview.extraction.content_loader import ContentLoader, UrlLoader

from .models.description_generation import (
    ContentDescription,
    DescriptionGenerator,
)
from .models.links_extraction import LinksExtractor, RelevantLink
from .models.summary_generation import SummaryGenerator
from .models.topics_extraction import DataScienceTopic, TopicsExtractor
from .topic_er import TopicMerge, TopicResolver

logger = logging.getLogger(__name__)

# TODO implement llm response monitoring (mainly tokens)
# TODO fix this script this should not be a class
# TODO implement asynchronous requests

# TODO encode and ingest summary


class ContentExtractor:
    def __init__(self) -> None:
        self.summary_generator = SummaryGenerator()
        self.description_generator = DescriptionGenerator()
        self.topics_extractor = TopicsExtractor()
        self.link_extractor = LinksExtractor()
        self.topic_resolver = TopicResolver()

    @logfire.instrument("Extraction LLM calls")
    async def run_extraction(
        self, content_loader: ContentLoader
    ) -> tuple[str, ContentDescription, list[DataScienceTopic], list[RelevantLink]]:
        logger.info("Launching extraction api requests ...")

        tasks = []

        tasks.append(
            self.summary_generator.async_predict({"content": content_loader.content})
        )
        tasks.append(
            self.description_generator.async_predict(
                {"content": content_loader.content}
            )
        )
        tasks.append(
            self.topics_extractor.async_predict({"content": content_loader.content}),
        )

        if isinstance(content_loader, UrlLoader):
            tasks.append(self.link_extractor.async_predict(content_loader))

            result = await asyncio.gather(*tasks)
            return result[0], result[1], result[2].topics, result[3].links

        else:
            result = await asyncio.gather(*tasks)
            return result[0], result[1], result[2].topics, None

    @logfire.instrument("Content extraction")
    async def extract_content(
        self, content_loader: ContentLoader, session: Session, content_id: int
    ) -> tuple[ExtractionResult, list[TopicMerge]]:
        (
            summary,
            content_description,
            topics,
            content_links,
        ) = await self.run_extraction(content_loader)

        with logfire.span("Saving extraction results"):
            extraction_result = await embed_and_format_extraction_results(
                summary,
                content_description,
                content_links,
                content_id,
                session,
            )

        (
            extraction_result.topics,
            topic_merges,
        ) = await self.topic_resolver.resolve_topics(topics, session)
        session.add(extraction_result)

        return extraction_result, topic_merges
