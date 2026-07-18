import enum
from typing import List

from pydantic import BaseModel

from dsview.config import ModelType, lazy_model_config, load_extraction_config
from dsview.model_utils import LLMModel

# Eager on purpose: pydantic needs the enum types at class-definition time,
# so importing this module requires a valid CONF_DIR (the one exception to
# the lazy-config rule).
_extraction_config = load_extraction_config()
TagsType = enum.StrEnum("TagsType", _extraction_config.tags)
ContentType = enum.StrEnum("ContentType", _extraction_config.content_types)


class DataScienceTag(BaseModel):
    name: TagsType


class ContentDescription(BaseModel):
    analysis: str
    title: str
    content_type: ContentType
    tags: List[DataScienceTag]


class DescriptionGenerator(LLMModel):
    DEFAULT_MODEL_CONFIG = lazy_model_config(ModelType.DESCRIPTION_GENERATION)
    DEFAULT_SYSTEM_PROMPT_FILE = "system_content_description.txt"
    DEFAULT_USER_PROMPT_FILE = "user_content_description.txt"
    DEFAULT_STRUCTURED_OUTPUT_CLASS = ContentDescription
