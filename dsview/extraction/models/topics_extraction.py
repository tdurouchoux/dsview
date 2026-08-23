import enum

from pydantic import BaseModel

from dsview.config import ModelType, lazy_model_config, load_extraction_config
from dsview.model_utils import LLMModel

from .description_generation import TagsType

# Eager on purpose: pydantic needs the enum type at class-definition time,
# so importing this module requires a valid CONF_DIR (the one exception to
# the lazy-config rule).
TopicType = enum.StrEnum("TopicType", load_extraction_config().topic_categories)


class DataScienceTopic(BaseModel):
    type: TopicType
    name: str
    description: str


class TopicList(BaseModel):
    # analysis must come first: structured-output decoding fills fields in
    # schema order, so this is what lets the model work through the source
    # (what it is about, which mentions are genuine topics vs. themes to
    # exclude) before committing to the topic list. Mirrors ERResult.analysis.
    analysis: str
    topics: list[DataScienceTopic]

tags_format_dict = {"tags": ", ".join(TagsType)}

class TopicsExtractor(LLMModel):
    DEFAULT_MODEL_CONFIG = lazy_model_config(ModelType.TOPICS_EXTRACTION)
    DEFAULT_SYSTEM_PROMPT_FILE = "system_topics_extraction.txt"
    DEFAULT_USER_PROMPT_FILE = "user_topics_extraction.txt"
    # DEFAULT_SYSTEM_PROMPT_FORMAT = [", ".join(TagsType)]
    DEFAULT_USER_PROMPT_ADD_FORMAT = tags_format_dict
    DEFAULT_STRUCTURED_OUTPUT_CLASS = TopicList
