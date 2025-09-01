import pandas as pd
from sqlmodel import Session, select

from dsview.extraction.models.topics_extraction import DataScienceTopic

from ..schemas import ExtractionTopic
from .query_utils import DuckDBIndex

TOPICS_FTS_FIELDS = ["name"] # "description"
N_TOPICS_FTS = 3
N_TOPICS_VSS = 1

# TODO make embedding_size a configuration param


class TopicsIndex(DuckDBIndex):
    def __init__(self, embedding_size: int = 1024):
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
    statement = select(ExtractionTopic).where(ExtractionTopic.name == name)

    return session.exec(statement).first()


def get_topic_list(session: Session) -> list[ExtractionTopic]:
    return session.exec(select(ExtractionTopic)).all()
