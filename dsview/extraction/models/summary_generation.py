from dsview.config import ModelType, load_model_config

from dsview.model_utils import LLMModel

summary_generation_model_config = load_model_config(ModelType.SUMMARY_GENERATION)


class SummaryGenerator(LLMModel):
    DEFAULT_MODEL_CONFIG = summary_generation_model_config
    DEFAULT_SYSTEM_PROMPT_FILE = "system_generate_summary.txt"
    DEFAULT_USER_PROMPT_FILE = "user_generate_summary.txt"
