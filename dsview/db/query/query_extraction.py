from sqlmodel import Session, select

from ..schemas import (
    ContentTopicRelation,
    ExtractionLink,
    ExtractionResult,
    ExtractionTag,
    ExtractionTopic,
)
from .query_utils import DuckDBIndex

EXTRACTION_FTS_FIELDS = ["title", "summary"]


class ExtractionIndex(DuckDBIndex):
    def __init__(self, embedding_size: int = 1024):
        self.extraction_fields = list(ExtractionResult.model_fields.keys())

        super().__init__(
            ExtractionResult.__table__,
            self.extraction_fields,
            EXTRACTION_FTS_FIELDS,
            id_column="content_id",
            embedding_size=embedding_size,
        )

def get_content_extraction(
    content_id: int, session: Session
) -> list[list[ExtractionResult], list[ExtractionLink], list[ExtractionTag]]:
    result = []

    for table in [ExtractionResult, ExtractionLink, ExtractionTag]:
        statement = select(table).where(getattr(table, "content_id") == content_id)
        result.append(session.exec(statement).all())

    return result


def get_content_linked_topics(
    content_id: int,
    session: Session,
) -> list[ExtractionTopic]:
    statement = select(ExtractionTopic).where(
        ExtractionTopic.id.in_(
            select(ContentTopicRelation.topic_id).where(
                ContentTopicRelation.content_id == content_id
            )
        )
    )

    return session.exec(statement).all()
