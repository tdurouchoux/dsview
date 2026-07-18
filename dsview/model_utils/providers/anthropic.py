import logging
from typing import Optional, Union

import instructor
import numpy as np
from anthropic import Anthropic, AsyncAnthropic
from pydantic import BaseModel, ValidationError

from dsview.config import ModelConfig

from ..model_provider import ModelProvider, native_batch_disabled

logger = logging.getLogger(__name__)

# Fixed name is fine: each batch request carries its own independent tool scope.
BATCH_TOOL_NAME = "extract_data"


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

    def _split_system_message(
        self, messages: list[dict[str, str]]
    ) -> tuple[Optional[str], list[dict[str, str]]]:
        # The Messages API takes the system prompt as a top-level `system`
        # param, not as a "system"-role entry in `messages`.
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        other_messages = [m for m in messages if m["role"] != "system"]

        system = "\n\n".join(system_parts) if system_parts else None
        return system, other_messages

    def _build_batch_params(
        self,
        messages: list[dict[str, str]],
        structured_output_class: type[BaseModel] = None,
    ) -> dict:
        system, other_messages = self._split_system_message(messages)

        params = {
            "model": self.model_config.chat_model,
            "messages": other_messages,
            **self.model_config.model_specific_config,
        }
        if system is not None:
            params["system"] = system

        if structured_output_class is not None:
            params["tools"] = [
                {
                    "name": BATCH_TOOL_NAME,
                    "description": (
                        structured_output_class.__doc__
                        or f"Extract data matching the {structured_output_class.__name__} schema."
                    ),
                    "input_schema": structured_output_class.model_json_schema(),
                }
            ]
            params["tool_choice"] = {"type": "tool", "name": BATCH_TOOL_NAME}

        return params

    def _parse_batch_result(
        self,
        result,
        structured_output_class: type[BaseModel] = None,
    ) -> Union[str, BaseModel, None]:
        if result.type != "succeeded":
            return None

        content = result.message.content

        if structured_output_class is None:
            text_blocks = [block.text for block in content if block.type == "text"]
            return "".join(text_blocks) if text_blocks else None

        for block in content:
            if block.type == "tool_use" and block.name == BATCH_TOOL_NAME:
                try:
                    return structured_output_class.model_validate(block.input)
                except ValidationError:
                    logger.warning(
                        f"Batch request returned invalid "
                        f"{structured_output_class.__name__} output."
                    )
                    return None

        return None

    def batch_send_messages(
        self,
        batch_messages: list[list[dict[str, str]]],
        structured_output_class: type[BaseModel] = None,
    ) -> list[Union[str, BaseModel]]:
        if native_batch_disabled():
            return super().batch_send_messages(batch_messages, structured_output_class)

        if len(batch_messages) == 0:
            return []

        requests = [
            {
                "custom_id": str(i),
                "params": self._build_batch_params(messages, structured_output_class),
            }
            for i, messages in enumerate(batch_messages)
        ]

        batch = self.client.messages.batches.create(requests=requests)
        batch_id = batch.id

        batch = self._poll_batch_job(
            retrieve_fn=lambda: self.client.messages.batches.retrieve(batch_id),
            is_done=lambda batch: batch.processing_status == "ended",
            completed_count=lambda batch: (
                len(batch_messages) - batch.request_counts.processing
            ),
            total=len(batch_messages),
            desc=f"Batch job {batch_id}",
        )

        results = [None] * len(batch_messages)

        for entry in self.client.messages.batches.results(batch.id):
            custom_id = int(entry.custom_id)
            results[custom_id] = self._parse_batch_result(
                entry.result, structured_output_class
            )

        # Errored/canceled/expired requests fall back to the synchronous path,
        # which retries with tenacity.
        return self._fill_batch_fallback(
            results, batch_messages, structured_output_class
        )
