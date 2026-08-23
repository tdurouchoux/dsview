import json
import logging
import os

import numpy as np
from mistralai.client import Mistral
from mistralai.client.models import File
from mistralai.extra import response_format_from_pydantic_model
from openinference.instrumentation.mistralai import MistralAIInstrumentor
from pydantic import BaseModel, ValidationError

from dsview.config import ModelConfig

from ..model_provider import ModelProvider, native_batch_disabled
from ..observability import embedding_span

logger = logging.getLogger(__name__)

MistralAIInstrumentor().instrument()

# Parity with the max_tokens previously configured on the instructor client.
STRUCTURED_OUTPUT_MAX_TOKENS = 10_000

BATCH_RUNNING_STATUSES = ("QUEUED", "RUNNING", "CANCELLATION_REQUESTED")


class BatchJobFailed(Exception):
    def __init__(self, job_id: str, status: str) -> None:
        super().__init__(f"Batch job {job_id} ended with status {status}.")


class MistralProvider(ModelProvider):
    PROVIDER_API_KEY_NAME = "MISTRAL_API_KEY"

    def __init__(self, model_config: ModelConfig):
        self.client = Mistral(
            api_key=os.environ[self.PROVIDER_API_KEY_NAME],
            retry_config=None,
        )

        super().__init__(model_config)

    def _retrieve_model(self, model_name: str):
        self.client.models.retrieve(model_id=model_name)

    def _embed(self, input: str) -> np.ndarray:
        with embedding_span(self.model_config.embedding_model, input) as record:
            response = self.client.embeddings.create(
                model=self.model_config.embedding_model,
                inputs=[input],
            )
            record(response)

        return np.array(response.data[0].embedding)

    async def _async_embed(self, input: str) -> np.ndarray:
        with embedding_span(self.model_config.embedding_model, input) as record:
            response = await self.client.embeddings.create_async(
                model=self.model_config.embedding_model,
                inputs=[input],
            )
            record(response)

        return np.array(response.data[0].embedding)

    def _complete(
        self,
        messages: list[dict[str, str]],
        structured_output_class: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        if structured_output_class is not None:
            chat_response = self.client.chat.parse(
                model=self.model_config.chat_model,
                messages=messages,
                response_format=structured_output_class,
                max_tokens=STRUCTURED_OUTPUT_MAX_TOKENS,
                **self.model_config.model_specific_config,
            )

            return chat_response.choices[0].message.parsed
        else:
            chat_response = self.client.chat.complete(
                model=self.model_config.chat_model,
                messages=messages,
                **self.model_config.model_specific_config,
            )

            return chat_response.choices[0].message.content

    async def _async_complete(
        self,
        messages: list[dict[str, str]],
        structured_output_class: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        if structured_output_class is not None:
            chat_response = await self.client.chat.parse_async(
                model=self.model_config.chat_model,
                messages=messages,
                response_format=structured_output_class,
                max_tokens=STRUCTURED_OUTPUT_MAX_TOKENS,
                **self.model_config.model_specific_config,
            )

            return chat_response.choices[0].message.parsed
        else:
            chat_response = await self.client.chat.complete_async(
                model=self.model_config.chat_model,
                messages=messages,
                **self.model_config.model_specific_config,
            )

            return chat_response.choices[0].message.content

    def _build_batch_file(
        self,
        batch_messages: list[list[dict[str, str]]],
        structured_output_class: type[BaseModel] | None = None,
    ) -> bytes:
        response_format = (
            response_format_from_pydantic_model(structured_output_class)
            if structured_output_class is not None
            else None
        )

        lines = []
        for custom_id, messages in enumerate(batch_messages):
            body = {"messages": messages, **self.model_config.model_specific_config}
            if response_format is not None:
                body["response_format"] = response_format
                body["max_tokens"] = STRUCTURED_OUTPUT_MAX_TOKENS

            lines.append(json.dumps({"custom_id": str(custom_id), "body": body}))

        return "\n".join(lines).encode()

    def _parse_batch_output(
        self,
        output: bytes,
        results: list,
        structured_output_class: type[BaseModel] | None = None,
    ):
        for line in output.decode().splitlines():
            entry = json.loads(line)
            response = entry.get("response")

            if response is None or response.get("status_code") != 200:
                continue

            content = response["body"]["choices"][0]["message"]["content"]
            custom_id = int(entry["custom_id"])

            if structured_output_class is None:
                results[custom_id] = content
            else:
                try:
                    results[custom_id] = structured_output_class.model_validate_json(
                        content
                    )
                except ValidationError:
                    logger.warning(
                        f"Batch request {custom_id} returned invalid "
                        f"{structured_output_class.__name__} output."
                    )

    def batch_send_messages(
        self,
        batch_messages: list[list[dict[str, str]]],
        structured_output_class: type[BaseModel] | None = None,
    ) -> list[str | BaseModel]:
        if native_batch_disabled():
            return super().batch_send_messages(batch_messages, structured_output_class)

        if len(batch_messages) == 0:
            return []

        batch_file = self._build_batch_file(batch_messages, structured_output_class)

        uploaded_file = self.client.files.upload(
            file=File(file_name="batch_requests.jsonl", content=batch_file),
            purpose="batch",
        )

        job = self.client.batch.jobs.create(
            endpoint="/v1/chat/completions",
            input_files=[uploaded_file.id],
            model=self.model_config.chat_model,
        )
        job_id = job.id

        job = self._poll_batch_job(
            retrieve_fn=lambda: self.client.batch.jobs.get(job_id=job_id),
            is_done=lambda job: job.status not in BATCH_RUNNING_STATUSES,
            completed_count=lambda job: job.completed_requests,
            total=job.total_requests,
            desc=f"Batch job {job_id}",
        )

        if job.status not in ("SUCCESS", "TIMEOUT_EXCEEDED"):
            raise BatchJobFailed(job.id, job.status)

        results = [None] * len(batch_messages)

        if job.output_file is not None:
            output = self.client.files.download(file_id=job.output_file).read()
            self._parse_batch_output(output, results, structured_output_class)

        # Failed or unparseable requests fall back to the synchronous path,
        # which retries with tenacity.
        return self._fill_batch_fallback(
            results, batch_messages, structured_output_class
        )
