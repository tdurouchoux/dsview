from typing import Union

import instructor
import numpy as np
from anthropic import Anthropic, AsyncAnthropic
from pydantic import BaseModel

from dsview.config import ModelConfig

from ..model_provider import ModelProvider


class AnthropicProvider(ModelProvider):
    PROVIDER_API_KEY_NAME = "ANTHROPIC_API_KEY"

    def __init__(self, model_config: ModelConfig):
        self.client = Anthropic()
        self.async_client = AsyncAnthropic()

        super().__init__(model_config)

    def _retrieve_model(self, model_name: str):
        self.client.models.retrieve(model_name)

    def _embed(self, input: str) -> np.ndarray:
        raise NotImplementedError("Anthropic do not implement embedding API.")

    async def _async_embed(self, input: str) -> np.ndarray:
        raise NotImplementedError("Anthropic do not implement embedding API.")

    def _complete(
        self,
        messages: list[dict[str, str]],
        structured_output_class: type[BaseModel] = None,
    ) -> Union[str, BaseModel]:
        if structured_output_class is not None:
            struct_client = instructor.from_anthropic(self.client)

            chat_response = struct_client.messages.create(
                model=self.model_config.chat_model,
                messages=messages,
                # max_tokens=1_024,
                response_model=structured_output_class,
                **self.model_config.model_specific_config,
            )

            return chat_response
        else:
            chat_response = self.client.messages.create(
                model=self.model_config.chat_model,
                # max_tokens=1_024,
                messages=messages,
                **self.model_config.model_specific_config,
            )

            return chat_response.content[0].text

    async def _async_complete(
        self,
        messages: list[dict[str, str]],
        structured_output_class: type[BaseModel] = None,
    ) -> Union[str, BaseModel]:
        if structured_output_class is not None:
            struct_client = instructor.AsyncInstructor(
                client=self.async_client,
                create=instructor.patch(
                    create=self.async_client.messages.create,
                    mode=instructor.Mode.ANTHROPIC_TOOLS,
                ),
                mode=instructor.Mode.ANTHROPIC_TOOLS,
            )

            chat_response = await struct_client.chat.completions.create(
                model=self.model_config.chat_model,
                messages=messages,
                # max_tokens=1_024,
                response_model=structured_output_class,
                **self.model_config.model_specific_config,
            )

            return chat_response
        else:
            chat_response = await self.async_client.messages.create(
                model=self.model_config.chat_model,
                # max_tokens=1_024,
                messages=messages,
                **self.model_config.model_specific_config,
            )

            return chat_response.content[0].text
