import enum
from typing import List

from pydantic import BaseModel

from dsview.config import ModelType, load_extraction_config, load_model_config
from dsview.model_utils import LLMModel

extraction_config = load_extraction_config()
description_generation_model_config = load_model_config(
    ModelType.DESCRIPTION_GENERATION
)

TagsType = enum.StrEnum("TagsType", extraction_config.tags)
ContentType = enum.StrEnum("ContentType", extraction_config.content_types)


class DataScienceTag(BaseModel):
    name: TagsType


class ContentDescription(BaseModel):
    title: str
    content_type: ContentType
    tags: List[DataScienceTag]


class DescriptionGenerator(LLMModel):
    DEFAULT_MODEL_CONFIG = description_generation_model_config
    DEFAULT_SYSTEM_PROMPT_FILE = "system_content_description.txt"
    DEFAULT_USER_PROMPT_FILE = "user_content_description.txt"
    DEFAULT_STRUCTURED_OUTPUT_CLASS = ContentDescription
