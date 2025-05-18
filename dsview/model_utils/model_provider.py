import os
from abc import ABC, abstractmethod
from typing import Union

import mlflow
import numpy as np
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from dsview.config import LLMProvider, ModelConfig, ModelType, load_model_config

default_model_config = load_model_config(ModelType.DEFAULT)


class MissingAPIKey(Exception):
    def __init__(self, provider: LLMProvider, api_key_name: str) -> None:
        super().__init__(
            f"Missing api key for provider {provider},"
            f" please set {api_key_name} environment variable."
        )


class ModelConfigurationError(Exception):
    def __init__(self, provider: LLMProvider, model_name: str, error: str) -> None:
        super().__init__(
            f"Model {model_name} was not found for provider {provider}, "
            f"encountered error : {error}"
        )


class ModelProvider(ABC):
    PROVIDER_API_KEY_NAME = None

    def __init__(
        self,
        model_config: ModelConfig,
        force_provider: bool = False,
    ):
        self.model_config = model_config
        self.force_provider = force_provider

        # ! Kind of annoying that check is made for every single model

        self._check_api_key()
        # self._check_model_config()

    def _check_api_key(self):
        if self.PROVIDER_API_KEY_NAME is None:
            return
        if self.PROVIDER_API_KEY_NAME not in os.environ:
            raise MissingAPIKey(self.model_config.provider, self.PROVIDER_API_KEY_NAME)

    @abstractmethod
    def _retrieve_model(self, model_name: str):
        pass

    def _assert_model_exists(self, model_name: str):
        try:
            self._retrieve_model(model_name)
        except Exception as error:
            raise ModelConfigurationError(self.model_config.provider, model_name, error)

    def _check_model_config(self):
        self._assert_model_exists(self.model_config.chat_model)

        if (
            self.model_config.embedding_model is not None
            and self.model_config.embedding_model != "None"
        ):
            self._assert_model_exists(self.model_config.embedding_model)

    def log_params(self):
        mlflow.log_param("model_config", self.model_config)

    @abstractmethod
    def _embed(self, input: str) -> np.ndarray:
        pass

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(5))
    def embed(self, input: str) -> np.ndarray:
        return self._embed(input)

    @abstractmethod
    async def _async_embed(self, input: str) -> np.ndarray:
        pass

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(5))
    async def async_embed(self, input: str) -> np.ndarray:
        return await self._async_embed(input)

    @abstractmethod
    def _complete(
        self,
        messages: list[dict[str, str]],
        structured_output_class: type[BaseModel] = None,
    ) -> Union[str, BaseModel]:
        pass

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(5))
    def send_messages(
        self,
        messages: list[dict[str, str]],
        structured_output_class: type[BaseModel] = None,
    ) -> Union[str, BaseModel]:
        return self._complete(messages, structured_output_class=structured_output_class)

    @abstractmethod
    async def _async_complete(
        self,
        messages: list[dict[str, str]],
        structured_output_class: type[BaseModel] = None,
    ) -> Union[str, BaseModel]:
        pass

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(5))
    async def async_send_messages(
        self,
        messages: list[dict[str, str]],
        structured_output_class: type[BaseModel] = None,
    ) -> Union[str, BaseModel]:
        return await self._async_complete(
            messages, structured_output_class=structured_output_class
        )
