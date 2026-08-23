from datetime import UTC, datetime

from sqlmodel import Session

from dsview.config import ModelType, lazy, load_model_config
from dsview.extraction.models.description_generation import ContentDescription
from dsview.extraction.models.er_classification import ERResult
from dsview.extraction.models.links_extraction import RelevantLink
from dsview.extraction.models.topics_extraction import DataScienceTopic
from dsview.model_utils import get_model_provider

from ..schemas import (
    ERComparison,
    ExtractionLink,
    ExtractionResult,
    ExtractionTag,
    ExtractionTopic,
)
from .update import update_instance

# Lazy: building a provider checks API keys, which must not happen at import
model_provider = lazy(lambda: get_model_provider(load_model_config(ModelType.INDEX)))


async def embed_and_format_extraction_results(
    summary: str,
    content_description: ContentDescription,
    content_links: list[RelevantLink],
    content_id: int,
    session: Session,
):
    # TODO This may be truncated
    embedding = await model_provider.async_embed(summary)

    extraction_result = ExtractionResult(
        content_id=content_id,
        title=content_description.title,
        content_type=content_description.content_type.value,
        summary=summary,
        embedding=embedding,
    )

    extraction_result.tags = [
        ExtractionTag(content_id=content_id, name=tag.name.value)
        for tag in content_description.tags
    ]

    if content_links is not None:
        extraction_result.links = [
            ExtractionLink(
                content_id=content_id,
                name=link.name,
                url=link.url,
                description=link.description,
            )
            for link in content_links
        ]

    return extraction_result


# Need to add async


# This is kinda dangerous will see what happens
# Async function that write to db has unknown behavior
async def embed_and_save_topic(
    topic: DataScienceTopic, session: Session
) -> ExtractionTopic:
    embedding = await model_provider.async_embed(topic.description)

    extraction_topic = ExtractionTopic(
        type=topic.type.value,
        name=topic.name,
        description=topic.description,
        embedding=embedding,
    )

    session.add(extraction_topic)

    return extraction_topic


async def embed_and_update_topic(
    topic_id: int,
    updated_topic: DataScienceTopic,
    session,
) -> ExtractionTopic:
    embedding = await model_provider.async_embed(updated_topic.description)

    update_dict = dict(updated_topic, embedding=embedding)
    update_dict["type"] = updated_topic.type.value

    update_instance(
        session,
        ExtractionTopic,
        row_id=topic_id,
        **update_dict,
    )

    return session.get(ExtractionTopic, topic_id)


def build_er_comparison(
    topic_1: DataScienceTopic,
    topic_2: DataScienceTopic,
    fts_score: float,
    vss_distance: float,
    result: ERResult,
) -> ERComparison:
    return ERComparison(
        name_1=topic_1.name,
        type_1=topic_1.type,
        description_1=topic_1.description,
        name_2=topic_2.name,
        type_2=topic_2.type,
        description_2=topic_2.description,
        fts_score=fts_score,
        vss_distance=vss_distance,
        decision_date=datetime.now(UTC).date().isoformat(),
        merge_topic=result.merge_topic,
        merge_name=(result.topic.name if result.topic else None),
        merge_type=(result.topic.type if result.topic else None),
        merge_description=(result.topic.description if result.topic else None),
    )
