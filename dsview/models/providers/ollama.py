from typing import Union

import instructor
import numpy as np
import ollama
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

from dsview.config import ModelConfig

from ..model_provider import ModelConfigurationError, ModelProvider

# TODO Configure this
OLLAMA_HOST = "http://localhost:11434"

# TODO add support for instructor
# ? Make instructor optionnal ?


class OllamaProvider(ModelProvider):
    PROVIDER_API_KEY_NAME = None

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)

        self.client = ollama.Client(host=OLLAMA_HOST)
        self.async_client = ollama.AsyncClient(host=OLLAMA_HOST)

        self.struct_client = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            ),
            mode=instructor.Mode.JSON,
        )
        self.async_struct_client = instructor.apatch(
            AsyncOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            ),
            mode=instructor.Mode.JSON,
        )

        model_list = [model.model for model in self.client.list().models]

        if self.model_config.chat_model not in model_list:
            self._pull_model(self.model_config.chat_model)

        if self.model_config.embedding_model not in model_list:
            self._pull_model(self.model_config.embedding_model)

    def _pull_model(self, model_name: str):
        try:
            self.client.pull(model_name)

        except ollama.ResponseError as error:
            raise ModelConfigurationError(
                self.model_config.provider,
                model_name,
                error,
            )

    def _retrieve_model(self, model_name: str):
        return

    def _embed(self, input: str) -> np.ndarray:
        response = self.client.embed(
            model=self.model_config.embedding_model, input=input
        )

        return np.array(response.embeddings[0])

    def _complete(
        self,
        messages: list[dict[str, str]],
        structured_output_class: type[BaseModel] = None,
    ) -> Union[str, BaseModel]:
        if structured_output_class is not None:
            response = self.struct_client.chat.completions.create(
                messages=messages,
                model=self.model_config.chat_model,
                response_model=structured_output_class,
                # **self.model_config.model_specific_config,
            )

            # chat_response = structured_output_class.model_validate_json(
            #     response.message.content
            # )
            return response

        else:
            response = self.client.chat(
                model=self.model_config.chat_model,
                messages=messages,
                **self.model_config.model_specific_config,
            )

            return response.message.content

    async def _async_complete(
        self,
        messages: list[dict[str, str]],
        structured_output_class: type[BaseModel] = None,
    ) -> Union[str, BaseModel]:
        if structured_output_class is not None:
            response = await self.async_struct_client.chat.completions.create(
                messages=messages,
                model=self.model_config.chat_model,
                response_model=structured_output_class,
            )

            # response = await self.async_client.chat(
            #     messages=messages,
            #     model=self.model_config.chat_model,
            #     format=structured_output_class.model_json_schema(),
            #     **self.model_config.model_specific_config,
            # )

            # chat_response = structured_output_class.model_validate_json(
            #     response.message.content
            # )
            return response

        else:
            response = await self.async_client.chat(
                model=self.model_config.chat_model,
                messages=messages,
                **self.model_config.model_specific_config,
            )

            return response.message.content
