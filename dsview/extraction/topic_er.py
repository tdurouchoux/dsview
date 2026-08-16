import asyncio
import logging
from dataclasses import dataclass

import logfire
from sqlmodel import Session

from dsview.db.ingest import (
    build_er_comparison,
    embed_and_save_topic,
    embed_and_update_topic,
)
from dsview.db.query import TopicsIndex, get_topic_by_name
from dsview.db.schemas import ERComparison, ExtractionTopic

from .models.er_classification import ERClassifier
from .models.topics_extraction import DataScienceTopic

logger = logging.getLogger(__name__)


@dataclass
class TopicMerge:
    """A topic merged in place by ER: same row id, new name/type, so the vault
    notes written under the old name are stale."""

    topic_id: int
    old_name: str
    old_type: str


@dataclass
class TopicERDecision:
    topic: DataScienceTopic
    comparisons: list[ERComparison]
    merge: TopicMerge | None


def _dedup_resolved(topics: list[ExtractionTopic]) -> list[ExtractionTopic]:
    """Failsafe: two extracted topics may resolve to the same existing row."""
    deduped = []
    seen_ids = set()

    for topic in topics:
        if topic.id is not None:
            if topic.id in seen_ids:
                logger.warning("Found duplicate topic after ER")
                continue
            seen_ids.add(topic.id)
        deduped.append(topic)

    # logger.info("Number of topics after ER dedup : %s", len(deduped))

    return deduped


class TopicResolver:
    def __init__(self) -> None:
        self.er_classifier = ERClassifier()

    async def _resolve_topic(
        self, topic: DataScienceTopic, topics_index: TopicsIndex
    ) -> TopicERDecision:
        """Decide whether a topic merges into an existing one. Touches no session."""
        comparisons = []

        for candidate_id, candidate in topics_index.query_topic(topic).items():
            candidate_topic = candidate["topic"]

            result = await self.er_classifier.async_predict(topic, candidate_topic)

            comparisons.append(
                build_er_comparison(
                    topic,
                    candidate_topic,
                    candidate["fts_score"],
                    candidate["vss_distance"],
                    result,
                )
            )

            if result.merge_topic:
                logger.info(
                    "Merging topics %s and %s into %s ",
                    topic.name,
                    candidate_topic.name,
                    result.topic.name,
                )

                # Avoid the same candidate topic being merged multiple times
                topics_index.delete_rows(f"id={candidate_id}")

                # candidate_topic is the current DB row, i.e. the pre-merge
                # identity the vault notes were written under
                return TopicERDecision(
                    result.topic,
                    comparisons,
                    TopicMerge(
                        candidate_id, candidate_topic.name, candidate_topic.type.value
                    ),
                )

        return TopicERDecision(topic, comparisons, None)

    async def _apply_decision(
        self, decision: TopicERDecision, session: Session
    ) -> ExtractionTopic:
        for comparison in decision.comparisons:
            session.add(comparison)

        if decision.merge is not None:
            return await embed_and_update_topic(
                decision.merge.topic_id, decision.topic, session
            )

        return await embed_and_save_topic(decision.topic, session)

    @logfire.instrument("Resolving topics")
    async def resolve_topics(
        self, topics: list[DataScienceTopic], session: Session
    ) -> tuple[list[ExtractionTopic], list[TopicMerge]]:
        deduped_topics = {topic.name: topic for topic in topics}
        if len(deduped_topics) != len(topics):
            logger.warning("Dropped duplicates topics from LLM response !")

        existing_topics = []
        tasks = []

        with TopicsIndex() as topics_index:
            for topic in deduped_topics.values():
                existing_topic = get_topic_by_name(topic.name, session)

                if existing_topic is not None:
                    logger.info("Topic %s already exists", topic.name)
                    existing_topics.append(existing_topic)
                    continue

                tasks.append(self._resolve_topic(topic, topics_index))

            decisions = await asyncio.gather(*tasks)

        # Sequential: SQLAlchemy sessions are not safe for interleaved concurrent use.
        extraction_topics = [
            await self._apply_decision(decision, session) for decision in decisions
        ]

        merges = [
            decision.merge for decision in decisions if decision.merge is not None
        ]

        resolved_topics = _dedup_resolved(extraction_topics + existing_topics)

        n_resolved_topics = len(resolved_topics)
        logfire.info(f"Number of topics after ER : {n_resolved_topics}")

        return resolved_topics, merges
