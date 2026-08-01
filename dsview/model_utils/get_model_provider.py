from dsview.config import LLMProvider, ModelConfig

from .model_provider import ModelProvider


def get_model_provider(model_config: ModelConfig) -> ModelProvider:
    # Imported lazily, one provider at a time: each provider module imports its own SDK
    # (anthropic/mistralai/openai/ollama) at module scope, so importing all four eagerly
    # here would require every SDK to be installed just to construct one provider.
    match model_config.provider:
        case LLMProvider.OPENAI:
            from .providers.openai import OpenAIProvider

            return OpenAIProvider(model_config)
        case LLMProvider.MISTRAL:
            from .providers.mistral import MistralProvider

            return MistralProvider(model_config)
        case LLMProvider.OLLAMA:
            from .providers.ollama import OllamaProvider

            return OllamaProvider(model_config)
        case LLMProvider.ANTHROPIC:
            from .providers.anthropic import AnthropicProvider

            return AnthropicProvider(model_config)
