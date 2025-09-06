import logging.config
import os
from dataclasses import dataclass, field
from enum import Enum
from functools import cache, partial
from pathlib import Path
from typing import Callable, Optional, Union

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
    host: Optional[str] = None
    chat_model: str = MISSING
    embedding_model: Optional[str] = None
    provider: LLMProvider = LLMProvider.OPENAI
    token_limit: int = MISSING
    model_specific_config: Optional[dict[str, Union[int, str]]] = field(
        default_factory=lambda: {}
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
    user: str = "postgres"
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


def setup_logger():
    with open(Path(os.getenv("CONF_DIR")) / "logging.yaml") as config_file:
        logging_config = yaml.safe_load(config_file)
    logging.config.dictConfig(logging_config)
