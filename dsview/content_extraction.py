import logging
from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from dsview.config import load_topic_extraction_config
from dsview.content_loader import ContentLoader
from dsview.prompt_loader import get_prompt

logger = logging.getLogger(__name__)
config = load_topic_extraction_config()


class DataScienceTag(BaseModel):
    name: str = Field(
        default=None,
        description="Data science topic discussed in the source, it should be as precise as possible",
        enum=config.tags,
    )


class DataScienceSubject(BaseModel):
    type: str = Field(
        default=None,
        description="Type of the described subject in the source.",
        enum=config.subjects,
    )
    name: str = Field(
        default=None, description="Name of the described subject in the source."
    )
    description: str = Field(
        default=None,
        description=(
            "General description of the subject, it should not "
            "be a description of the source or how the source "
            "tackle this subject. But the description must be "
            "created using the provided context or any prior knowledge."
        ),
    )


class TopicExtraction(BaseModel):
    title: str = Field(
        description=(
            "Title of the source, should be the actual title when it exists. "
            "If it does not exists, generate one, it can not exceed 50 characters."
        )
    )
    tags: List[DataScienceTag]
    subjects: List[DataScienceSubject]


def get_topic_extractor(llm):
    topic_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", get_prompt("system_topic_extraction.txt")),
            ("human", get_prompt("human_topic_extraction.txt")),
        ]
    )

    return topic_prompt | llm.with_structured_output(schema=TopicExtraction)


def get_summary_generator(llm):
    summarization_prompt = ChatPromptTemplate.from_template(
        get_prompt("generate_summary.txt")
    )
    return summarization_prompt | llm


class ContentExtractor:
    def __init__(self, llm) -> None:
        self.summary_generator = get_summary_generator(llm)
        self.topic_extractor = get_topic_extractor(llm)

    def extract_content(
        self, content_loader: ContentLoader
    ) -> Tuple[str, TopicExtraction]:
        content_loader.load()

        logger.info("Launching summary generation")
        summary = self.summary_generator.invoke({"content": content_loader.content})

        logger.info("Launching topic extraction")
        topics = self.topic_extractor.invoke({"content": content_loader.content})

        return summary.content, topics
