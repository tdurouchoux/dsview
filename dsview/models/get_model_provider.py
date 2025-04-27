from dsview.config import LLMProvider, ModelConfig

from .model_provider import ModelProvider
from .providers import (
    AnthropicProvider,
    MistralProvider,
    OllamaProvider,
    OpenAIProvider,
)


def get_model_provider(model_config: ModelConfig) -> ModelProvider:
    if model_config.provider == LLMProvider.OPENAI:
        return OpenAIProvider(model_config)
    elif model_config.provider == LLMProvider.MISTRAL:
        return MistralProvider(model_config)
    elif model_config.provider == LLMProvider.OLLAMA:
        return OllamaProvider(model_config)
    return AnthropicProvider(model_config)
