from typing import Optional

from pydantic import BaseModel

from dsview.config import ModelType, load_model_config
from dsview.model_utils import LLMModel

from .topics_extraction import DataScienceTopic, TopicType

er_classification_model_config = load_model_config(ModelType.ER_CLASSIFICATION)


class ERResult(BaseModel):
    merge_topic: bool
    topic: Optional[DataScienceTopic]


class ERClassifier(LLMModel):
    DEFAULT_MODEL_CONFIG = er_classification_model_config
    DEFAULT_SYSTEM_PROMPT_FILE = "system_entity_resolution.txt"
    DEFAULT_USER_PROMPT_FILE = "user_entity_resolution.txt"
    DEFAULT_SYSTEM_PROMPT_FORMAT = [",".join(TopicType)]
    DEFAULT_STRUCTURED_OUTPUT_CLASS = ERResult

    def _format_topics(
        self, topic1: DataScienceTopic, topic2: DataScienceTopic
    ) -> dict[str, str]:
        return {
            "name_1": topic1.name,
            "type_1": topic1.type.value,
            "description_1": topic1.description,
            "name_2": topic2.name,
            "type_2": topic2.type.value,
            "description_2": topic2.description,
        }

    # This also have effect on evaluation (also simple er ? )
    def predict(self, topic1: DataScienceTopic, topic2: DataScienceTopic) -> ERResult:
        input = self._format_topics(topic1, topic2)

        return super().predict(input)

    async def async_predict(
        self, topic1: DataScienceTopic, topic2: DataScienceTopic
    ) -> ERResult:
        input = self._format_topics(topic1, topic2)

        return await super().async_predict(input)
