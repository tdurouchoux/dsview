from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import func
from sqlmodel import Session, select

from ..schemas import ContentTopicRelation, ExtractionTopic, InputContent
from .query_utils import DuckDBIndex

if TYPE_CHECKING:
    from dsview.extraction.models.topics_extraction import DataScienceTopic

TOPICS_FTS_FIELDS = ["name"]  # "description"
N_TOPICS_FTS = 3
N_TOPICS_VSS = 1

# TODO make embedding_size a configuration param


class TopicsIndex(DuckDBIndex):
    def __init__(self, embedding_size: int = 1024):
        # Imported here: the extraction models build enums from config at
        # import time, which must not be required to import this module
        from dsview.extraction.models.topics_extraction import DataScienceTopic

        self.topic_fields = list(DataScienceTopic.model_fields.keys())

        super().__init__(
            ExtractionTopic.__table__,
            self.topic_fields,
            TOPICS_FTS_FIELDS,
            embedding_size=embedding_size,
        )

    def query_topic(
        self,
        topic: DataScienceTopic,
    ) -> dict[int, dict[str, DataScienceTopic | float]]:
        from dsview.extraction.models.topics_extraction import DataScienceTopic

        result_topics = {}

        fts_query_result = self._query_fts_index(topic.name, N_TOPICS_FTS)

        for topic_id, row in fts_query_result.iterrows():
            result_topics[topic_id] = {
                "topic": DataScienceTopic(**row[self.topic_fields]),
                "fts_score": row["fts_score"],
                "vss_distance": self.compute_vss_distance(topic.name, topic_id),
            }

        vss_query_result = self._query_vss_index(topic.description, N_TOPICS_VSS)

        for topic_id, row in vss_query_result.iterrows():
            if topic_id in result_topics:
                result_topics[topic_id]["vss_distance"] = row["vss_distance"]
            else:
                result_topics[topic_id] = {
                    "topic": DataScienceTopic(**row[self.topic_fields]),
                    "fts_score": self.compute_fts_score(topic.description, topic_id),
                    "vss_distance": row["vss_distance"],
                }

        # Also store distance / score results
        # query_result = pd.concat(
        #     (fts_query_result, vss_query_result), ignore_index=False
        # ).drop_duplicates()

        return result_topics


def get_topic_by_name(name: str, session: Session) -> DataScienceTopic:
    statement = select(ExtractionTopic).where(
        func.lower(ExtractionTopic.name) == name.lower()
    )

    return session.exec(statement).first()


def get_topic_list(session: Session) -> list[ExtractionTopic]:
    return session.exec(select(ExtractionTopic)).all()


def get_topic_min_upload_dates(
    topic_ids: Sequence[int], session: Session
) -> dict[int, date]:
    """Earliest upload date among the contents connected to each of `topic_ids`.

    A topic's "creation date" is defined as this minimum - it's brand new once
    that date falls within the current digest week.
    """
    if not topic_ids:
        return {}

    statement = (
        select(
            ContentTopicRelation.topic_id,
            func.min(InputContent.upload_date),
        )
        .join(InputContent, InputContent.id == ContentTopicRelation.content_id)
        .where(ContentTopicRelation.topic_id.in_(topic_ids))
        .group_by(ContentTopicRelation.topic_id)
    )

    return dict(session.exec(statement).all())
