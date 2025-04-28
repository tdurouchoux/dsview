import asyncio
import logging
import time
from typing import List, Tuple

from sqlmodel import Session

from dsview.config import load_extraction_config
from dsview.content.content_loader import ContentLoader, UrlLoader
from dsview.models.llm_models.description_generation import (
    ContentDescription,
    DataScienceTag,
    DescriptionGenerator,
)
from dsview.models.llm_models.links_extraction import LinksExtractor, RelevantLink
from dsview.models.llm_models.summary_generation import SummaryGenerator
from dsview.models.llm_models.topics_extraction import DataScienceTopic, TopicsExtractor

from .extraction_db_schema import ExtractionResult, InvalidTag, InvalidTopic

config = load_extraction_config()

logger = logging.getLogger(__name__)

# TODO implement llm response monitoring (mainly tokens)
# TODO fix this script this should not be a class
# TODO implement asynchronous requests


class ContentExtractor:
    def __init__(self) -> None:
        self.summary_generator = SummaryGenerator()
        self.description_generator = DescriptionGenerator()
        self.topics_extractor = TopicsExtractor()
        self.link_extractor = LinksExtractor()

    @staticmethod
    def select_valid_properties(
        property_list: List,
        property_attribute_values: List[str],
        property_name: str,
        attribute: str = "name",
    ) -> Tuple[List, List]:
        valid_properties = []
        invalid_properties = []

        for prop in property_list:
            if getattr(prop, attribute) in property_attribute_values:
                valid_properties.append(prop)
            else:
                invalid_properties.append(prop)

        if len(invalid_properties) > 0:
            logger.warning(
                "%s extracted %s were invalid",
                len(invalid_properties),
                property_name,
            )

        return valid_properties, invalid_properties

    def _save_extraction_results(
        self,
        invalid_tags: list[DataScienceTag],
        invalid_topics: list[DataScienceTopic],
        content_description: ContentDescription,
        session: Session,
        content_id: int,
    ):
        if session is None:
            return

        if len(invalid_tags) > 0:
            session.add_all(
                [
                    InvalidTag(
                        content_id=content_id,
                        name=tag.name,
                    )
                    for tag in invalid_tags
                ]
            )
            session.commit()

        if len(invalid_topics) > 0:
            session.add_all(
                [
                    InvalidTopic(
                        content_id=content_id,
                        name=topic.name,
                        type=topic.type,
                        description=topic.description,
                    )
                    for topic in invalid_topics
                ]
            )

            session.commit()

        extraction_result = ExtractionResult(
            content_id=content_id,
            title=content_description.title,
            content_type=content_description.content_type.value,
        )

        session.add(extraction_result)
        session.commit()

    async def _send_api_requests(
        self, content_loader: ContentLoader
    ) -> tuple[str, ContentDescription, list[DataScienceTopic], list[RelevantLink]]:
        logger.info("Launching extraction api requests ...")

        tasks = []

        tasks.append(
            self.summary_generator.async_predict({"content": content_loader.content})
        )
        await asyncio.sleep(0.1)
        tasks.append(
            self.description_generator.async_predict(
                {"content": content_loader.content}
            )
        )
        await asyncio.sleep(0.1)
        tasks.append(
            self.topics_extractor.async_predict({"content": content_loader.content}),
        )

        if isinstance(content_loader, UrlLoader):
            await asyncio.sleep(0.1)
            tasks.append(self.link_extractor.async_predict(content_loader))

            result = await asyncio.gather(*tasks)
            return result[0], result[1], result[2].topics, result[3].links

        else:
            result = await asyncio.gather(*tasks)
            return result[0], result[1], result[2].topics, None

    async def extract_content(
        self, content_loader: ContentLoader, session: Session, content_id: int
    ) -> tuple[str, ContentDescription, list[DataScienceTopic], list[RelevantLink]]:
        starting_time = time.perf_counter()

        content_loader.load()

        summary, content_description, topics, content_links = await self._send_api_requests(content_loader)


        content_description.tags, invalid_tags = self.select_valid_properties(
            content_description.tags, config.tags.values(), "tag"
        )

        topics, invalid_topics = self.select_valid_properties(
            topics,
            config.topic_categories.values(),
            "topic",
            attribute="type",
        )

        logger.info("Saving extraction results")
        self._save_extraction_results(
            invalid_tags, invalid_topics, content_description, session, content_id
        )

        logger.info(
            f"Total time for extraction: {time.perf_counter() - starting_time:.1f} seconds"
        )

        return summary, content_description, topics, content_links
