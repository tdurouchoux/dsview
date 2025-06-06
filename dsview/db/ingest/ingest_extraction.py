from datetime import date

from sqlmodel import Session

from dsview.config import ModelType, load_model_config
from dsview.extraction.models.description_generation import ContentDescription
from dsview.extraction.models.er_classification import ERResult
from dsview.extraction.models.links_extraction import RelevantLink
from dsview.extraction.models.topics_extraction import DataScienceTopic
from dsview.model_utils import get_model_provider

from ..schemas import (
    ContentTopicRelation,
    ERComparison,
    ERDecision,
    ExtractionLink,
    ExtractionResult,
    ExtractionTag,
    ExtractionTopic,
)
from .update import update_instance

model_config = load_model_config(ModelType.INDEX)
model_provider = get_model_provider(model_config)


def save_extraction_results(
    summary: str,
    content_description: ContentDescription,
    content_links: list[RelevantLink],
    content_id: int,
    session: Session,
):
    extraction_result = ExtractionResult(
        content_id=content_id,
        title=content_description.title,
        content_type=content_description.content_type.value,
        summary=summary,
    )

    extraction_tags = (
        ExtractionTag(content_id=content_id, name=tag.name.value)
        for tag in content_description.tags
    )
    if content_links is not None:
        extraction_links = (
            ExtractionLink(
                content_id=content_id,
                name=link.name,
                url=link.url,
                description=link.description,
            )
            for link in content_links
        )

        session.add_all((extraction_result, *extraction_tags, *extraction_links))
    else:
        session.add_all((extraction_result, *extraction_tags))
    session.commit()


# Need to add async
async def embed_and_format(input: str) -> str:
    embedding = await model_provider.async_embed(input)

    return ",".join([str(e) for e in embedding])


# This is kinda dangerous will see what happens
# Async function that write to db has unknown behavior
async def embed_and_save_topic(topic: DataScienceTopic, session: Session) -> int:
    embedding = await embed_and_format(topic.description)

    extraction_topic = ExtractionTopic(
        type=topic.type.value,
        name=topic.name,
        description=topic.description,
        embedding=embedding,
    )

    session.add(extraction_topic)
    session.commit()

    return extraction_topic.id


async def embed_and_update_topic(
    topic_id: int,
    updated_topic: DataScienceTopic,
    session,
):
    embedding = await embed_and_format(updated_topic.description)

    update_instance(
        session,
        ExtractionTopic,
        row_id=topic_id,
        **dict(updated_topic, embedding=embedding),
    )


def save_er_comparison(
    topic_1: DataScienceTopic,
    topic_2: DataScienceTopic,
    result: ERResult,
    session: Session,
):
    # Check if the comparison already exists in the database

    er_comparison = ERComparison(
        name_1=topic_1.name,
        type_1=topic_1.type,
        description_1=topic_1.description,
        name_2=topic_2.name,
        type_2=topic_2.type,
        description_2=topic_2.description,
    )

    session.add(er_comparison)
    session.commit()

    er_decision = ERDecision(
        comparison_id=er_comparison.id,
        decision_date=date.today().isoformat(),
        merge_topic=result.merge_topic,
        merge_name=result.topic.name,
        merge_type=result.topic.type,
        merge_description=result.topic.description,
    )

    session.add(er_decision)
    session.commit()


def save_topic_relations(
    content_id: int,
    topic_ids: list[int],
    session: Session,
):
    session.add_all(
        (
            ContentTopicRelation(content_id=content_id, topic_id=topic_id)
            for topic_id in topic_ids
        )
    )

    session.commit()
