from typing import Union

import numpy as np
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from dsview.config import ModelConfig

from ..model_provider import ModelProvider


class OpenAIProvider(ModelProvider):
    PROVIDER_API_KEY_NAME = "OPENAI_API_KEY"

    def __init__(self, model_config: ModelConfig):
        self.client = OpenAI()
        self.async_client = AsyncOpenAI()

        super().__init__(model_config)

    def _retrieve_model(self, model_name: str):
        self.client.models.retrieve(model_name)

    def _embed(self, input: str) -> np.ndarray:
        response = self.client.embeddings.create(
            input=input, model=self.model_config.embedding_model
        )

        return np.array(response.data[0].embedding)

    def _complete(
        self,
        messages: list[dict[str, str]],
        structured_output_class: type[BaseModel] = None,
    ) -> Union[str, BaseModel]:
        if structured_output_class is not None:
            response = self.client.beta.chat.completions.parse(
                model=self.model_config.chat_model,
                messages=messages,
                response_format=structured_output_class,
                **self.model_config.model_specific_config,
            )

            return response.choices[0].message.parsed

        else:
            response = self.client.chat.completions.create(
                model=self.model_config.chat_model,
                messages=messages,
                **self.model_config.model_specific_config,
            )

            return response.choices[0].message.content

    async def _async_complete(
        self,
        messages: list[dict[str, str]],
        structured_output_class: type[BaseModel] = None,
    ) -> Union[str, BaseModel]:
        if structured_output_class is not None:
            response = await self.async_client.beta.chat.completions.parse(
                model=self.model_config.chat_model,
                messages=messages,
                response_format=structured_output_class,
                **self.model_config.model_specific_config,
            )

            return response.choices[0].message.parsed

        else:
            response = await self.async_client.chat.completions.create(
                model=self.model_config.chat_model,
                messages=messages,
                **self.model_config.model_specific_config,
            )

            return response.choices[0].message.content
