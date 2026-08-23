import asyncio

import numpy as np
import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from dsview.config import LLMProvider, ModelConfig
from dsview.model_utils import get_model_provider
from dsview.model_utils.model_provider import MissingAPIKey
from dsview.model_utils.providers.mistral import MistralProvider
from dsview.model_utils.providers.ollama import OllamaProvider
from dsview.model_utils.providers.openai import OpenAIProvider

# TODO async endpoint testing

load_dotenv()

OPENAI_MODEL_CONFIG = ModelConfig(
    chat_model="gpt-4o-mini-2024-07-18",
    embedding_model="text-embedding-3-small",
    provider=LLMProvider.OPENAI,
    token_limit=128_000,
)

MISTRAL_MODEL_CONFIG = ModelConfig(
    chat_model="mistral-medium-latest",
    provider=LLMProvider.MISTRAL,
    token_limit=32_000,
)

OLLAMA_MODEL_CONFIG = ModelConfig(
    chat_model="smollm2:1.7b",
    embedding_model="all-minilm:latest",
    provider=LLMProvider.OLLAMA,
    token_limit=8_192,
)

ANTHROPIC_MODEL_CONFIG = ModelConfig(
    chat_model="claude-3-5-haiku-20241022",
    provider=LLMProvider.ANTHROPIC,
    token_limit=200_000,
    model_specific_config={"max_tokens": 1_024},
)


@pytest.mark.parametrize(
    "model_config, provider_type",
    [
        (OPENAI_MODEL_CONFIG, OpenAIProvider),
        (MISTRAL_MODEL_CONFIG, MistralProvider),
        (OLLAMA_MODEL_CONFIG, OllamaProvider),
        # (ANTHROPIC_MODEL_CONFIG, AnthropicProvider),
    ],
)
def test_get_model_provider(model_config: ModelConfig, provider_type: LLMProvider):
    model_provider = get_model_provider(model_config)
    assert isinstance(model_provider, provider_type)


def test_missing_api_key():
    OpenAIProvider.PROVIDER_API_KEY_NAME = "SOME_OTHER_KEY"

    with pytest.raises(MissingAPIKey):
        _ = OpenAIProvider(OPENAI_MODEL_CONFIG)

    OpenAIProvider.PROVIDER_API_KEY_NAME = "OPENAI_API_KEY"


@pytest.mark.parametrize(
    "model_config",
    [
        OPENAI_MODEL_CONFIG,
        MISTRAL_MODEL_CONFIG,
        OLLAMA_MODEL_CONFIG,
        # ANTHROPIC_MODEL_CONFIG,
    ],
)
def test_send_messages(model_config: ModelConfig):
    messages = [
        {"role": "user", "content": "Hello !"},
    ]

    model_provider = get_model_provider(model_config)

    response = model_provider.send_messages(messages)

    assert isinstance(response, str)


@pytest.mark.parametrize(
    "model_config",
    [
        OPENAI_MODEL_CONFIG,
        MISTRAL_MODEL_CONFIG,
        OLLAMA_MODEL_CONFIG,
        # ANTHROPIC_MODEL_CONFIG,
    ],
)
def test_async_messages(model_config: ModelConfig):
    messages = [
        {"role": "user", "content": "Hello !"},
    ]
    model_provider = get_model_provider(model_config)

    response = asyncio.run(model_provider.async_send_messages(messages))

    assert isinstance(response, str)


class Response(BaseModel):
    is_fine: bool


@pytest.mark.parametrize(
    "model_config",
    [
        OPENAI_MODEL_CONFIG,
        MISTRAL_MODEL_CONFIG,
        OLLAMA_MODEL_CONFIG,
        # ANTHROPIC_MODEL_CONFIG,
    ],
)
def test_send_messaged_structured(model_config: ModelConfig):
    messages = [
        {"role": "user", "content": "Are you fine ?"},
    ]

    model_provider = get_model_provider(model_config)

    response = model_provider.send_messages(messages, structured_output_class=Response)

    assert isinstance(response, Response)


@pytest.mark.parametrize(
    "model_config",
    [
        OPENAI_MODEL_CONFIG,
        MISTRAL_MODEL_CONFIG,
        OLLAMA_MODEL_CONFIG,
        # ANTHROPIC_MODEL_CONFIG,
    ],
)
def test_async_send_messaged_structured(model_config: ModelConfig):
    messages = [
        {"role": "user", "content": "How are you ?"},
    ]

    model_provider = get_model_provider(model_config)

    response = asyncio.run(
        model_provider.async_send_messages(messages, structured_output_class=Response)
    )

    assert isinstance(response, Response)


@pytest.mark.parametrize(
    "model_config",
    [
        OPENAI_MODEL_CONFIG,
        OLLAMA_MODEL_CONFIG,
    ],
)
def test_embed(model_config: ModelConfig):
    input = "What is the meaning of life ?"

    model_provider = get_model_provider(model_config)

    embedding = model_provider.embed(input)

    assert isinstance(embedding, np.ndarray)
