import logging
from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate
from pydantic.v1 import BaseModel, Field
from sqlmodel import Session

from dsview.content.content_loader import ContentLoader, UrlLoader
from dsview.config import load_extraction_config
from .extraction_db_schema import InvalidTag, InvalidTopic
from .prompt_loader import get_prompt


logger = logging.getLogger(__name__)
config = load_extraction_config()


def get_summary_generator(llm):
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", get_prompt("system_generate_summary.txt")),
            ("user", get_prompt("user_generate_summary.txt")),
        ]
    )
    return summarization_prompt | llm


class DataScienceTag(BaseModel):
    name: str = Field(
        description="Data science topic discussed in the source, it should be as precise as possible",
        enum=config.tags,
    )


class ContentDescription(BaseModel):
    title: str = Field(
        description=(
            "Title of the source, should be the actual title when it exists. "
            "If it does not exists, generate one, it can not exceed 50 characters."
        )
    )
    content_type: str = Field(
        description=(
            "Type of the content that most accurately describe the source. "
            "For example, a github link will most of the time be a 'repository'."
            "Or another example, a medium article is a 'blog post'. "
        ),
        enum=config.content_types,
    )
    tags: List[DataScienceTag]


def get_description_generator(llm):
    content_description_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", get_prompt("system_content_description.txt")),
            ("user", get_prompt("user_content_description.txt")),
        ]
    )

    return content_description_prompt | llm.with_structured_output(
        schema=ContentDescription
    )


class DataScienceTopic(BaseModel):
    type: str = Field(
        default=None,
        description="Type of the described topic in the source.",
        enum=config.topic_categories,
    )
    name: str = Field(
        default=None, description="Name of the described topic in the source."
    )
    description: str = Field(
        default=None,
        description=(
            "General and detailed description of the topic, it should not "
            "be a description of the source or how the source "
            "tackle this topic. But the description must be "
            "created using the provided information or any prior knowledge."
        ),
    )


class TopicList(BaseModel):
    topics: List[DataScienceTopic]


def get_topics_extractor(llm):
    topics_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                get_prompt("system_topics_extraction.txt").format(
                    ", ".join(config.tags)
                ),
            ),
            ("user", get_prompt("user_topics_extraction.txt")),
        ]
    )

    return topics_prompt | llm.with_structured_output(schema=TopicList)


def get_link_extractor(llm):
    link_extraction_prompt = ChatPromptTemplate.from_template(
        get_prompt("links_extraction.txt")
    )

    return link_extraction_prompt | llm


# TODO implement llm response monitoring (mainly tokens)


class ContentExtractor:
    def __init__(self, llm) -> None:
        self.summary_generator = get_summary_generator(llm)
        self.description_generator = get_description_generator(llm)
        self.topics_extractor = get_topics_extractor(llm)
        self.link_extractor = get_link_extractor(llm)

    @staticmethod
    def select_valid_properties(
        property_list: List,
        property_attribute_values: List[str],
        property_name: str,
        attribute: str = "name",
    ) -> Tuple[List, List]:
        valid_properties = []
        invalid_properties = []

        for prop in property_list:
            if getattr(prop, attribute) in property_attribute_values:
                valid_properties.append(prop)
            else:
                invalid_properties.append(prop)

        if len(invalid_properties) > 0:
            logger.warning(
                "%s extracted %s were invalid",
                len(invalid_properties),
                property_name,
            )

        return valid_properties, invalid_properties

    def extract_content(
        self, content_loader: ContentLoader, session: Session, content_id: int
    ) -> Tuple[str, ContentDescription, List[DataScienceTopic], str]:
        content_loader.load()

        logger.info("Launching summary generation")
        summary = self.summary_generator.invoke(
            {"content": content_loader.content}
        ).content

        logger.info("Launching description generation")
        content_description = self.description_generator.invoke(
            {"content": content_loader.content}
        )
        content_description.tags, invalid_tags = self.select_valid_properties(
            content_description.tags, config.tags, "tag"
        )

        if session is not None and len(invalid_tags) > 0:
            session.add_all(
                [
                    InvalidTag(
                        content_id=content_id,
                        name=tag.name,
                    )
                    for tag in invalid_tags
                ]
            )
            session.commit()

        logger.info("Launching topics extraction")
        topic_list = self.topics_extractor.invoke({"content": content_loader.content})
        topics, invalid_topics = self.select_valid_properties(
            topic_list.topics, config.topic_categories, "topic", attribute="type"
        )

        if session is not None and len(invalid_topics) > 0:
            session.add_all(
                [
                    InvalidTopic(
                        content_id=content_id,
                        name=topic.name,
                        type=topic.type,
                        description=topic.description,
                    )
                    for topic in invalid_topics
                ]
            )
            session.commit()

        content_links = None
        if isinstance(content_loader, UrlLoader):
            logger.info("Launching links extraction")
            content_links = self.link_extractor.invoke(
                {
                    "url": content_loader.link,
                    "content": content_loader.content,
                    "content_links": "\n".join(content_loader.content_links),
                }
            ).content

        return summary, content_description, topics, content_links
