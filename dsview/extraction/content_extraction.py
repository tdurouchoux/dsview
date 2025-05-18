import asyncio
import logging
import time

from sqlmodel import Session

from dsview.config import load_extraction_config
from dsview.db.ingest import (
    embed_and_save_topic,
    embed_and_update_topic,
    save_er_comparison,
    save_extraction_results,
    save_topic_relations,
)
from dsview.db.query import TopicsIndex, get_topic_by_name
from dsview.extraction.content_loader import ContentLoader, UrlLoader
from dsview.extraction.models.er_classification import ERClassifier

from .models.description_generation import (
    ContentDescription,
    DescriptionGenerator,
)
from .models.links_extraction import LinksExtractor, RelevantLink
from .models.summary_generation import SummaryGenerator
from .models.topics_extraction import DataScienceTopic, TopicsExtractor

config = load_extraction_config()

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
        self.er_classifier = ERClassifier()

    async def _send_api_requests(
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

    async def extract_content(
        self, content_loader: ContentLoader, session: Session, content_id: int
    ) -> tuple[list[int], list[int]]:
        starting_time = time.perf_counter()

        (
            summary,
            content_description,
            topics,
            content_links,
        ) = await self._send_api_requests(content_loader)

        logger.info("Saving extraction results")
        save_extraction_results(
            summary,
            content_description,
            content_links,
            content_id,
            session,
        )

        logger.info(
            f"Total time for extraction: {time.perf_counter() - starting_time:.1f} seconds"
        )

        logger.info("Launching topic ER")

        return await self.topics_er(topics, content_id, session)

    async def _single_topic_er(
        self,
        topic: DataScienceTopic,
        topics_index: TopicsIndex,
        session: Session,
    ) -> tuple[tuple[int, DataScienceTopic] | None, int | None]:
        candidate_topics = topics_index.query_topic(topic)

        for candidate_id, candidate_topic in candidate_topics:
            result = await self.er_classifier.async_predict(topic, candidate_topic)

            save_er_comparison(topic, candidate_topic, result, session)

            if result.merge_topic:
                # This means I don't have to update older links
                logger.warning(
                    "Merging topics %s and %s into %s ",
                    topic.name,
                    candidate_topic.name,
                    result.topic.name,
                )

                await embed_and_update_topic(candidate_id, result.topic, session)

                return (candidate_id, candidate_topic), None

        topic_id = await embed_and_save_topic(topic, session)

        return None, topic_id

    async def topics_er(
        self, topics: list[DataScienceTopic], content_id: int, session: Session
    ) -> tuple[tuple[int, DataScienceTopic], int]:
        # Get topics to update
        # update links
        # Get new topics
        # Update links
        # Returns id list of updated topics and

        # !!! I kinda wait topics_id to surface
        # !! TODO Configure embedding size

        existing_topic_ids = []
        tasks = []

        with TopicsIndex() as topics_index:
            for topic in topics:
                existing_topic = get_topic_by_name(topic.name, session)

                if existing_topic is not None:
                    logger.warning("Topic %s already exists", topic.name)
                    existing_topic_ids.append(existing_topic.id)
                    continue

                tasks.append(self._single_topic_er(topic, topics_index, session))

            # ! I am not sure but I may need thread safe session
            er_results = await asyncio.gather(*tasks)

        # take into account the fact that two topics can update the same one
        # Asynchronous code make it not so well handled
        updated_topics = [result[0] for result in er_results if result[1] is None]
        updated_topic_ids = [e[0] for e in updated_topics]
        new_topic_ids = [result[1] for result in er_results if result[0] is None]

        save_topic_relations(
            content_id,
            set(existing_topic_ids + updated_topic_ids + new_topic_ids),
            session,
        )

        return updated_topics, new_topic_ids
