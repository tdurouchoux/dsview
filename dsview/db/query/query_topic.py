import pandas as pd
from sqlmodel import Session, select

from dsview.extraction.models.topics_extraction import DataScienceTopic

from ..schemas import ExtractionTopic
from .query_utils import DuckDBIndex

TOPICS_FTS_FIELDS = ["name", "description"]
N_TOPICS_FTS = 3
N_TOPICS_VSS = 3


class TopicsIndex(DuckDBIndex):
    def __init__(self, embedding_size: int = 1536):
        self.topic_fields = list(DataScienceTopic.model_fields.keys())

        super().__init__(
            ExtractionTopic.__tablename__,
            self.topic_fields,
            TOPICS_FTS_FIELDS,
            embedding_size=embedding_size,
        )

    def query_topic(
        self,
        topic: DataScienceTopic,
    ) -> list[tuple[int, DataScienceTopic]]:
        fts_query_result = self._query_fts_index(topic.name, N_TOPICS_FTS)[
            self.topic_fields
        ]

        vss_query_result = self._query_vss_index(topic.description, N_TOPICS_VSS)[
            self.topic_fields
        ]

        query_result = pd.concat(
            (fts_query_result, vss_query_result), ignore_index=False
        ).drop_duplicates()

        return [
            (index, DataScienceTopic(**row.to_dict()))
            for index, row in query_result.iterrows()
        ]


def get_topic_by_name(name: str, session: Session) -> DataScienceTopic:
    statement = select(ExtractionTopic).where(ExtractionTopic.name == name)

    return session.exec(statement).first()
