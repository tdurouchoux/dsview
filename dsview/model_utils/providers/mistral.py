import os
from typing import Union
import logging

import numpy as np
from mistralai import Mistral
from pydantic import BaseModel
from instructor import from_mistral, Mode

from dsview.config import ModelConfig

from ..model_provider import ModelProvider


# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s - %(message)s'
# )
# debug_logger = logging.getLogger(__name__)


class MistralProvider(ModelProvider):
    PROVIDER_API_KEY_NAME = "MISTRAL_API_KEY"

    def __init__(self, model_config: ModelConfig):
        self.client = Mistral(
            api_key=os.environ[self.PROVIDER_API_KEY_NAME],
            retry_config=None,
            # debug_logger=debug_logger,
        )

        super().__init__(model_config)

    def _retrieve_model(self, model_name: str):
        self.client.models.retrieve(model_id=model_name)

    def _embed(self, input: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model_config.embedding_model,
            inputs=[input],
        )

        return np.array(response.data[0].embedding)

    async def _async_embed(self, input: str) -> np.ndarray:
        response = await self.client.embeddings.create_async(
            model=self.model_config.embedding_model,
            inputs=[input],
        )

        return np.array(response.data[0].embedding)

    def _complete(
        self,
        messages: list[dict[str, str]],
        structured_output_class: type[BaseModel] = None,
    ) -> Union[str, BaseModel]:
        if structured_output_class is not None:
            instructor_client = from_mistral(
                client=self.client,
                model=self.model_config.chat_model,
                mode=Mode.MISTRAL_TOOLS,
                max_tokens=2000,
            )

            resp = instructor_client.messages.create(
                response_model=structured_output_class,
                messages=messages,
            )

            # chat_response = self.client.chat.parse(
            #     model=self.model_config.chat_model,
            #     messages=messages,
            #     response_format=structured_output_class,
            # )

            # return chat_response.choices[0].message.parsed

            return resp
        else:
            chat_response = self.client.chat.complete(
                model=self.model_config.chat_model,
                messages=messages,
            )

            return chat_response.choices[0].message.content

    async def _async_complete(
        self,
        messages: list[dict[str, str]],
        structured_output_class: type[BaseModel] = None,
    ) -> Union[str, BaseModel]:
        if structured_output_class is not None:
            # chat_response = await self.client.chat.parse_async(
            #     model=self.model_config.chat_model,
            #     messages=messages,
            #     response_format=structured_output_class,
            # )

            # return chat_response.choices[0].message.parsed
            #
            instructor_client = from_mistral(
                client=self.client,
                model=self.model_config.chat_model,
                mode=Mode.MISTRAL_TOOLS,
                max_tokens=2000,
                use_async=True,
            )

            resp = await instructor_client.messages.create(
                response_model=structured_output_class,
                messages=messages,
            )

            return resp
        else:
            chat_response = await self.client.chat.complete_async(
                model=self.model_config.chat_model,
                messages=messages,
            )

            return chat_response.choices[0].message.content
