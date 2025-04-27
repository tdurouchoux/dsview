import enum
from typing import List

from pydantic import BaseModel

from dsview.config import ModelType, load_extraction_config, load_model_config

from ..model_utils import LLMModel
from .description_generation import TagsType

extraction_config = load_extraction_config()
topics_extraction_model_config = load_model_config(ModelType.TOPICS_EXTRACTION)

TopicType = enum.StrEnum("TopicType", extraction_config.topic_categories)


class DataScienceTopic(BaseModel):
    type: TopicType
    name: str
    description: str


class TopicList(BaseModel):
    topics: List[DataScienceTopic]


class TopicsExtractor(LLMModel):
    DEFAULT_MODEL_CONFIG = topics_extraction_model_config
    DEFAULT_SYSTEM_PROMPT_FILE = "system_topics_extraction.txt"
    DEFAULT_USER_PROMPT_FILE = "user_topics_extraction.txt"
    # DEFAULT_SYSTEM_PROMPT_FORMAT = [", ".join(TagsType)]
    DEFAULT_STRUCTURED_OUTPUT_CLASS = TopicList

    # TODO clean this up
    def predict(self, input: dict[str, str]) -> TopicList:
        return super().predict(dict(input, tags=", ".join(TagsType)))

    async def async_predict(self, input: dict[str, str]) -> TopicList:
        return await super().async_predict(dict(input, tags=", ".join(TagsType)))
