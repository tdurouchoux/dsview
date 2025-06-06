from sqlmodel import Session, select

from ..schemas import (
    ContentTopicRelation,
    ExtractionLink,
    ExtractionResult,
    ExtractionTag,
    ExtractionTopic,
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


def get_topic_linked_contents(
    topic_id: int, session: Session
) -> list[ExtractionResult]:
    statement = select(ExtractionResult).where(
        ExtractionResult.content_id.in_(
            select(ContentTopicRelation.content_id).where(
                ContentTopicRelation.topic_id == topic_id
            )
        )
    )

    return session.exec(statement).all()
