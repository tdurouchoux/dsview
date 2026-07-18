from typing import Optional

from pydantic import BaseModel

from dsview.config import ModelType, lazy_model_config
from dsview.model_utils import LLMModel

from .topics_extraction import DataScienceTopic, TopicType


class ERResult(BaseModel):
    # analysis must come first: structured-output decoding fills fields in
    # schema order, so this is what actually lets the model work through the
    # <topic_comparison> reasoning steps in the prompt before committing to
    # merge_topic. Without it, the prompt's reasoning instructions are inert.
    analysis: str
    merge_topic: bool
    topic: Optional[DataScienceTopic]


class ERClassifier(LLMModel):
    DEFAULT_MODEL_CONFIG = lazy_model_config(ModelType.ER_CLASSIFICATION)
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

    def predict_batch(
        self, topic_pairs: list[tuple[DataScienceTopic, DataScienceTopic]]
    ) -> list[ERResult]:
        inputs = [self._format_topics(topic1, topic2) for topic1, topic2 in topic_pairs]

        return super().predict_batch(inputs)
