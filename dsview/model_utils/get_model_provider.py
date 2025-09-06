from dsview.config import LLMProvider, ModelConfig

from .model_provider import ModelProvider
from .providers import (
    AnthropicProvider,
    MistralProvider,
    OllamaProvider,
    OpenAIProvider,
)


def get_model_provider(model_config: ModelConfig) -> ModelProvider:
    match model_config.provider:
        case LLMProvider.OPENAI:
            return OpenAIProvider(model_config)
        case LLMProvider.MISTRAL:
            return MistralProvider(model_config)
        case LLMProvider.OLLAMA:
            return OllamaProvider(model_config)
        case LLMProvider.ANTHROPIC:
            return AnthropicProvider(model_config)
