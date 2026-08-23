import logging.config
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import cache, partial
from pathlib import Path
from typing import TypeVar, cast

import yaml
from dotenv import load_dotenv
from omegaconf import MISSING, OmegaConf

load_dotenv()


@cache
def load_config(config_class, conf_file: Path):
    default_config = OmegaConf.structured(config_class)
    file_config = OmegaConf.load(Path(os.getenv("CONF_DIR")) / conf_file)
    merged_config = OmegaConf.merge(default_config, file_config)

    return OmegaConf.to_object(merged_config)


class LLMProvider(str, Enum):
    OPENAI = "openai"
    MISTRAL = "mistral"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class ModelType(Enum):
    DEFAULT = 0
    DESCRIPTION_GENERATION = 1
    ER_CLASSIFICATION = 2
    LINKS_EXTRACTION = 3
    SUMMARY_GENERATION = 4
    TOPICS_EXTRACTION = 5
    SEMANTIC_SCORE = 6
    INDEX = 7


@dataclass
class ModelConfig:
    host: str | None = None
    chat_model: str = MISSING
    embedding_model: str | None = None
    provider: LLMProvider = LLMProvider.OPENAI
    token_limit: int = MISSING
    model_specific_config: dict[str, int | str] | None = field(
        default_factory=dict
    )


@dataclass
class ModelSpecificConfig:
    model_type: ModelType = ModelType.DEFAULT
    model_config: ModelConfig = MISSING


@dataclass
class GlobalModelConfig:
    configs: list[ModelSpecificConfig]


class ModelConfigurationError(Exception):
    def __init__(self, model_type: ModelType):
        super().__init__(
            f"No model configuration was found for model type {model_type}, "
            "and no default model config was found."
            f"Could not determine model configuration."
        )


T = TypeVar("T")

_UNSET = object()


class LazyProxy:
    """Stand-in for a module-level object that defers its construction.

    Wraps a zero-argument loader; the target is built on first attribute
    access and memoized, so importing a module that declares one never
    touches config files or environment variables.
    """

    __slots__ = ("_loader", "_target")

    def __init__(self, loader: Callable[[], object]) -> None:
        object.__setattr__(self, "_loader", loader)
        object.__setattr__(self, "_target", _UNSET)

    def _resolve(self):
        target = object.__getattribute__(self, "_target")
        if target is _UNSET:
            target = object.__getattribute__(self, "_loader")()
            object.__setattr__(self, "_target", target)
        return target

    def __getattr__(self, name: str):
        return getattr(self._resolve(), name)

    def __setattr__(self, name: str, value) -> None:
        setattr(self._resolve(), name, value)

    def __repr__(self) -> str:
        return repr(self._resolve())


def lazy(loader: Callable[[], T]) -> T:
    """Typed lazy module-level global.

    The cast lets call sites keep the loaded type for IDE completion and
    type checking while the value is actually a LazyProxy.
    """
    return cast(T, LazyProxy(loader))


def load_model_config(model_type: ModelType = None) -> ModelConfig:
    if model_type is None:
        model_type = ModelType.DEFAULT

    config_list = load_config(GlobalModelConfig, "model.yaml").configs

    model_config = None

    for model_specific_config in config_list:
        if (
            model_type != ModelType.DEFAULT
            and model_specific_config.model_type == ModelType.DEFAULT
        ):
            model_config = model_specific_config.model_config

        if model_specific_config.model_type.value == model_type.value:
            return model_specific_config.model_config

    if model_config is None:
        raise ModelConfigurationError(model_type)
    return model_config


def lazy_model_config(model_type: ModelType = ModelType.DEFAULT) -> ModelConfig:
    """Lazy counterpart of load_model_config for module-level declarations."""
    return lazy(partial(load_model_config, model_type))


@dataclass
class EvaluationConfig:
    model_config: ModelConfig | None = None
    system_prompt_file: Path | None = None
    user_prompt_file: Path | None = None


def load_evaluation_config(config_file: Path) -> EvaluationConfig:
    default_config = OmegaConf.structured(EvaluationConfig)
    file_config = OmegaConf.load(config_file)

    return OmegaConf.to_object(OmegaConf.merge(default_config, file_config))


@dataclass
class ExtractionConfig:
    tags: dict[str, str]
    content_types: dict[str, str]
    topic_categories: dict[str, str]
    er_jaro_threshold: float


load_extraction_config: Callable[[], ExtractionConfig] = partial(
    load_config, ExtractionConfig, "extraction.yaml"
)


@dataclass
class GithubVault:
    repository: str | None = "${oc.env:VAULT_REPOSITORY,null}"
    username: str | None = "${oc.env:GITHUB_USERNAME,null}"
    email: str | None = "${oc.env:GITHUB_USER_EMAIL,null}"
    token: str | None = "${oc.env:GITHUB_TOKEN,null}"

    @property
    def url(self) -> str:
        return (
            f"https://{self.username}:{self.token}"
            f"@{self.repository.replace('https://', '')}"
        )


@dataclass
class ObsidianConfig:
    vault_path: Path = MISSING
    content_directory: str = "contents"
    topic_directory: str = "topics"
    github_vault: GithubVault = field(default_factory=GithubVault)


@dataclass
class PostgresConfig:
    host: str = "${oc.env:POSTGRES_HOST}"
    port: int = 5432
    database: str = "dsview_db"
    user: str = "${oc.env:POSTGRES_USER}"
    password: str = "${oc.env:POSTGRES_PASSWORD}"

    def db_uri(self, sqlalchemy: bool = True) -> str:
        return (
            "postgresql"
            + ("+psycopg" if sqlalchemy else "")
            + f"://{self.user}:{self.password}"
            + f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass
class StorageConfig:
    obsidian: ObsidianConfig = field(default_factory=ObsidianConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)


load_storage_config: Callable[[], StorageConfig] = partial(
    load_config, StorageConfig, "storage.yaml"
)


def load_obsidian_config() -> ObsidianConfig:
    return load_storage_config().obsidian


def load_postgres_config() -> PostgresConfig:
    return load_storage_config().postgres


def setup_logger(enable_logfire: bool = False, service_name: str | None = None):
    with open(Path(os.getenv("CONF_DIR")) / "logging.yaml") as config_file:
        logging_config = yaml.safe_load(config_file)
    logging.config.dictConfig(logging_config)

    if enable_logfire:
        if service_name is None:
            raise ValueError("service_name must be provided if enable_logfire is True")

        # Imported here rather than at module scope: anything under `model_utils`
        # pulls in its `__init__`, which imports this module back. By call time
        # `dsview.config` is fully initialized, so the cycle never forms.
        import logfire

        from dsview.model_utils.observability import MistralUsageSpanProcessor

        send_to_logfire = False if "PYTEST_VERSION" in os.environ else "if-token-present"

        logfire.configure(
            console=False,
            send_to_logfire=send_to_logfire,
            service_name=service_name,
            additional_span_processors=[MistralUsageSpanProcessor()],
        )

        logging.getLogger().addHandler(logfire.LogfireLoggingHandler())
