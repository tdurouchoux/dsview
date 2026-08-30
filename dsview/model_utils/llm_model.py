import os
from functools import reduce
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from dsview.config import ModelConfig, lazy_model_config

from .get_model_provider import get_model_provider

load_dotenv()

# mlflow.mistral.autolog()


class MissingPromptFile(Exception):
    def __init__(self, prompt_filename: str):
        super().__init__(
            self,
            (
                f"Prompt file '{prompt_filename}' is missing "
                f"from prompt directory : {os.getenv('PROMPT_DIR')}."
            ),
        )


def coalesce(*args):
    return reduce(lambda x, y: x if x is not None else y, args)


def get_prompt(prompt_filename: str) -> str:
    prompt_file = Path(os.getenv("PROMPT_DIR")) / prompt_filename

    if not prompt_file.exists():
        raise MissingPromptFile(prompt_filename)

    with open(prompt_file, "r") as txt_file:
        prompt = txt_file.read()

    return prompt


# TODO add thinking step support


class LLMModel:
    DEFAULT_MODEL_CONFIG = lazy_model_config()
    DEFAULT_SYSTEM_PROMPT_FILE: str | None = None
    DEFAULT_USER_PROMPT_FILE: str | None = None
    DEFAULT_SYSTEM_PROMPT_FORMAT: list[str] | dict[str, str] | None = None
    DEFAULT_USER_PROMPT_ADD_FORMAT: dict[str, str] | None = None
    DEFAULT_STRUCTURED_OUTPUT_CLASS: type[BaseModel] | None = None

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        system_prompt_format: list[str] | dict[str, str] | None = None,
        user_prompt_format: dict[str, str] | None = None,
        structured_output_class: type[BaseModel] | None = None,
    ):
        model_config = coalesce(model_config, self.DEFAULT_MODEL_CONFIG)
        self.model_provider = get_model_provider(model_config)

        if system_prompt is None:
            if self.DEFAULT_SYSTEM_PROMPT_FILE is None:
                raise ValueError("No system prompt provided.")

            self.system_prompt = get_prompt(self.DEFAULT_SYSTEM_PROMPT_FILE)
        else:
            self.system_prompt = system_prompt

        system_prompt_format = coalesce(
            system_prompt_format, self.DEFAULT_SYSTEM_PROMPT_FORMAT
        )

        if system_prompt_format is not None:
            if isinstance(system_prompt_format, list):
                self.system_prompt = self.system_prompt.format(*system_prompt_format)
            else:
                self.system_prompt = self.system_prompt.format(**system_prompt_format)

        self.user_prompt_format = coalesce(
            user_prompt_format, self.DEFAULT_USER_PROMPT_ADD_FORMAT, {}
        )

        if user_prompt is None:
            if self.DEFAULT_USER_PROMPT_FILE is None:
                raise ValueError("No user prompt provided.")
            self.user_prompt_template = get_prompt(self.DEFAULT_USER_PROMPT_FILE)
        else:
            self.user_prompt_template = user_prompt

        self.structured_output_class = coalesce(
            structured_output_class, self.DEFAULT_STRUCTURED_OUTPUT_CLASS
        )

    def log_params(self):
        import mlflow

        self.model_provider.log_params()
        mlflow.log_param("system_prompt", self.system_prompt)
        mlflow.log_param("structured_output_class", self.structured_output_class)
        mlflow.log_param("user_prompt_template", self.user_prompt_template)

    def _prepare_messages(self, input: dict[str, str]) -> list[dict[str, str]]:
        user_prompt = self.user_prompt_template.format(
            **input, **self.user_prompt_format
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return messages

    def predict(self, input: dict[str, str]) -> str | BaseModel:
        messages = self._prepare_messages(input)

        return self.model_provider.send_messages(messages, self.structured_output_class)

    async def async_predict(self, input: dict[str, str]) -> str | BaseModel:
        messages = self._prepare_messages(input)

        result = await self.model_provider.async_send_messages(
            messages, self.structured_output_class
        )

        return result

    def predict_batch(self, inputs: list[dict[str, str]]) -> list[str | BaseModel]:
        batch_messages = [self._prepare_messages(input) for input in inputs]

        return self.model_provider.batch_send_messages(
            batch_messages, self.structured_output_class
        )
