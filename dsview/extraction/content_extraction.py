import asyncio
import logging
import time

from sqlmodel import Session

from dsview.config import load_extraction_config
from dsview.db.ingest import (
    embed_and_save_topic,
    embed_and_update_topic,
    save_er_comparison,
    embed_and_format_extraction_results
)
from dsview.db.query import TopicsIndex, get_topic_by_name
from dsview.db.schemas import ExtractionTopic, ExtractionResult
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
    ) -> ExtractionResult:
        starting_time = time.perf_counter()

        (
            summary,
            content_description,
            topics,
            content_links,
        ) = await self._send_api_requests(content_loader)

        # Make sure there is no duplicates in topics
        deduped_topics = []
        set_topic_name = set()

        for topic in topics:
            if topic.name not in set_topic_name:
                deduped_topics.append(topic)
                set_topic_name.add(topic.name)

        if len(deduped_topics) != len(topics):
            logger.warning("Dropped duplicates topics from LLM response !")

        logger.info("Saving extraction results")

        extraction_result = await embed_and_format_extraction_results(
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

        extraction_topics = await self.topics_er(deduped_topics, content_id, session)

        extraction_result.topics = extraction_topics
        session.add(extraction_result)

        return extraction_result


    async def _single_topic_er(
        self,
        topic: DataScienceTopic,
        topics_index: TopicsIndex,
        session: Session,
    ) -> ExtractionTopic:
        candidate_topics = topics_index.query_topic(topic)

        for candidate_id, candidate_topic_dict in candidate_topics.items():

            candidate_topic = candidate_topic_dict["topic"]
            fts_score = candidate_topic_dict.get("fts_score")
            vss_distance = candidate_topic_dict.get("vss_distance")

            result = await self.er_classifier.async_predict(topic, candidate_topic)

            save_er_comparison(topic, candidate_topic, fts_score, vss_distance, result, session)

            if result.merge_topic:
                # This means I don't have to update older links
                logger.warning(
                    "Merging topics %s and %s into %s ",
                    topic.name,
                    candidate_topic.name,
                    result.topic.name,
                )

                extraction_topic = await embed_and_update_topic(candidate_id, result.topic, session)

                # Avoid the same candidate topic being merged multiple time
                topics_index.delete_rows(f"id={extraction_topic.id}")

                return extraction_topic
                # return (candidate_id, candidate_topic), None
        extraction_topic = await embed_and_save_topic(topic, session)

        return extraction_topic

    async def topics_er(
        self, topics: list[DataScienceTopic], content_id: int, session: Session
    ) -> list[ExtractionTopic]:
        # Get topics to update
        # update links
        # Get new topics
        # Update links
        # Returns id list of updated topics and

        # !!! I kinda wait topics_id to surface
        # !! TODO Configure embedding size

        existing_topics = []
        tasks = []

        with TopicsIndex() as topics_index:
            for topic in topics:
                existing_topic = get_topic_by_name(topic.name, session)

                if existing_topic is not None:
                    logger.warning("Topic %s already exists", topic.name)
                    existing_topics.append(existing_topic)
                    continue

                tasks.append(self._single_topic_er(topic, topics_index, session))

            # ! I am not sure but I may need thread safe session
            extraction_topics = list(await asyncio.gather(*tasks))

        return extraction_topics + existing_topics
