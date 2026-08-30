from pydantic import BaseModel, Field

from dsview.config import ModelType, lazy_model_config
from dsview.model_utils import LLMModel


class DigestItemSummary(BaseModel):
    one_liner: str
    bullets: list[str] = Field(min_length=2, max_length=3)


class DigestSummaryGenerator(LLMModel):
    """Shortens an existing content summary into a digest card (gist + bullets).

    Not evaluated: unlike the other extraction tasks, this has no `dsview/evaluation/`
    module or labels table, per an explicit exception to the usual eval requirement.
    """

    DEFAULT_MODEL_CONFIG = lazy_model_config(ModelType.DIGEST_SUMMARY)
    DEFAULT_SYSTEM_PROMPT_FILE = "system_digest_summary.txt"
    DEFAULT_USER_PROMPT_FILE = "user_digest_summary.txt"
    DEFAULT_STRUCTURED_OUTPUT_CLASS = DigestItemSummary
