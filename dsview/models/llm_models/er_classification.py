from pydantic import BaseModel

from dsview.config import ModelType, load_model_config

from ..model_utils import LLMModel
from .topics_extraction import DataScienceTopic, TopicType

er_classification_model_config = load_model_config(ModelType.ER_CLASSIFICATION)


class ERResult(BaseModel):
    merge_topic: bool
    topic: DataScienceTopic


class ERClassifier(LLMModel):
    DEFAULT_MODEL_CONFIG = er_classification_model_config
    DEFAULT_SYSTEM_PROMPT_FILE = "system_entity_resolution.txt"
    DEFAULT_USER_PROMPT_FILE = "user_entity_resolution.txt"
    DEFAULT_SYSTEM_PROMPT_FORMAT = [",".join(TopicType)]
    DEFAULT_STRUCTURED_OUTPUT_CLASS = ERResult
