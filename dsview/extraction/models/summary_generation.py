from dsview.config import ModelType, lazy_model_config
from dsview.model_utils import LLMModel


class SummaryGenerator(LLMModel):
    DEFAULT_MODEL_CONFIG = lazy_model_config(ModelType.SUMMARY_GENERATION)
    DEFAULT_SYSTEM_PROMPT_FILE = "system_generate_summary.txt"
    DEFAULT_USER_PROMPT_FILE = "user_generate_summary.txt"
