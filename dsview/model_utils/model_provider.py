import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm

from dsview.config import LLMProvider, ModelConfig

logger = logging.getLogger(__name__)

# Concurrency cap for the async fallback of batch_send_messages, to stay clear
# of provider rate limits.
BATCH_FALLBACK_MAX_CONCURRENCY = 10

# Poll interval for providers with a native batch job (Mistral, Anthropic).
BATCH_POLL_INTERVAL_SECONDS = 10

# When set (to any non-empty value), providers with a native batch API fall
# back to the synchronous fanout in ModelProvider.batch_send_messages.
DISABLE_NATIVE_BATCH_ENV_VAR = "DSVIEW_DISABLE_NATIVE_BATCH"


def native_batch_disabled() -> bool:
    return bool(os.environ.get(DISABLE_NATIVE_BATCH_ENV_VAR))


def retry_callback(retry_state):
    # Only log for actual retries (attempt_number > 1)
    if retry_state.attempt_number > 1:
        logger.warning(
            f"Retry {retry_state.attempt_number} for {retry_state.fn.__name__} "
            f"due to {retry_state.outcome.exception() if retry_state.outcome else 'unknown error'}"
        )


class MissingAPIKey(Exception):
    def __init__(self, provider: LLMProvider, api_key_name: str) -> None:
        super().__init__(
            f"Missing api key for provider {provider},"
            f" please set {api_key_name} environment variable."
        )


class ProviderConfigurationError(Exception):
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
            raise ProviderConfigurationError(
                self.model_config.provider, model_name, error
            )

    def _check_model_config(self):
        self._assert_model_exists(self.model_config.chat_model)

        if (
            self.model_config.embedding_model is not None
            and self.model_config.embedding_model != "None"
        ):
            self._assert_model_exists(self.model_config.embedding_model)

    def log_params(self):
        import mlflow

        mlflow.log_param("model_config", self.model_config)

    @abstractmethod
    def _embed(self, input: str) -> np.ndarray:
        pass

    @retry(
        wait=wait_random_exponential(multiplier=1, max=20),
        stop=stop_after_attempt(8),
        retry=retry_callback,
    )
    def embed(self, input: str) -> np.ndarray:
        return self._embed(input)

    @abstractmethod
    async def _async_embed(self, input: str) -> np.ndarray:
        pass

    @retry(
        wait=wait_random_exponential(multiplier=1, max=20),
        stop=stop_after_attempt(8),
        retry=retry_callback,
    )
    async def async_embed(self, input: str) -> np.ndarray:
        return await self._async_embed(input)

    @abstractmethod
    def _complete(
        self,
        messages: list[dict[str, str]],
        structured_output_class: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        pass

    @retry(
        wait=wait_random_exponential(multiplier=1, max=20),
        stop=stop_after_attempt(8),
        retry=retry_callback,
    )
    def send_messages(
        self,
        messages: list[dict[str, str]],
        structured_output_class: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        return self._complete(messages, structured_output_class=structured_output_class)

    @abstractmethod
    async def _async_complete(
        self,
        messages: list[dict[str, str]],
        structured_output_class: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        pass

    @retry(
        wait=wait_random_exponential(multiplier=1, max=20),
        stop=stop_after_attempt(8),
        retry=retry_callback,
    )
    async def async_send_messages(
        self,
        messages: list[dict[str, str]],
        structured_output_class: type[BaseModel] | None = None,
    ) -> str | BaseModel:
        return await self._async_complete(
            messages, structured_output_class=structured_output_class
        )

    def batch_send_messages(
        self,
        batch_messages: list[list[dict[str, str]]],
        structured_output_class: type[BaseModel] | None = None,
    ) -> list[str | BaseModel]:
        """Send many independent requests, returning results in input order.

        Providers with a native batch API should override this; the default
        fans out over async_send_messages with bounded concurrency.
        """

        async def _gather():
            semaphore = asyncio.Semaphore(BATCH_FALLBACK_MAX_CONCURRENCY)

            async def _send(messages: list[dict[str, str]]):
                async with semaphore:
                    return await self.async_send_messages(
                        messages, structured_output_class=structured_output_class
                    )

            return await asyncio.gather(
                *[_send(messages) for messages in batch_messages]
            )

        return asyncio.run(_gather())

    def _poll_batch_job(
        self,
        retrieve_fn: Callable[[], Any],
        is_done: Callable[[Any], bool],
        completed_count: Callable[[Any], int],
        total: int,
        desc: str,
    ) -> Any:
        """Poll a provider-native batch job to completion, driving a tqdm bar.

        Shared skeleton for providers with a native batch API (Mistral,
        Anthropic): the sleep/refresh loop is identical, only the job
        object's status/count fields differ, so callers supply small hooks
        instead of duplicating the loop.
        """
        job = retrieve_fn()

        with tqdm(total=total, desc=desc) as progress:
            while not is_done(job):
                time.sleep(BATCH_POLL_INTERVAL_SECONDS)
                job = retrieve_fn()

                progress.n = completed_count(job)
                progress.refresh()

        return job

    def _fill_batch_fallback(
        self,
        results: list,
        batch_messages: list[list[dict[str, str]]],
        structured_output_class: type[BaseModel] | None = None,
    ) -> list:
        """Resolve any still-missing (None) batch results via sync calls.

        Shared by providers with a native batch API: whatever the batch job
        couldn't produce a parsed result for (failed, errored, expired,
        unparseable) falls back to the retried synchronous path.
        """
        missing = [i for i, result in enumerate(results) if result is None]

        if missing:
            logger.warning(
                f"{len(missing)} of {len(batch_messages)} batch requests failed, "
                "falling back to synchronous calls."
            )

        for i in tqdm(missing, desc="Batch fallback"):
            results[i] = self.send_messages(
                batch_messages[i], structured_output_class=structured_output_class
            )

        return results
