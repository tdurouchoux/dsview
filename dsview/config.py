from dataclasses import dataclass, field
from functools import partial
import os
from typing import Callable, List, Optional

import logging.config
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf, MISSING
import yaml

load_dotenv()
os.getenv("CONF_DIR")


def load_config(config_class, conf_file: Path):
    default_config = OmegaConf.structured(config_class)
    file_config = OmegaConf.load(Path(os.getenv("CONF_DIR")) / conf_file)
    merged_config = OmegaConf.merge(default_config, file_config)

    return OmegaConf.to_object(merged_config)


@dataclass
class ModelConfig:
    name: str = MISSING
    token_limit: int = MISSING


load_model_config: Callable[[], ModelConfig] = partial(
    load_config, ModelConfig, "model.yaml"
)


@dataclass
class ExtractionConfig:
    tags: List[str]
    content_types: List[str]
    topic_categories: List[str]
    er_jaro_threshold: float


load_extraction_config: Callable[[], ExtractionConfig] = partial(
    load_config, ExtractionConfig, "extraction.yaml"
)


@dataclass
class GithubVault:
    enabled: bool = False
    repository: str | None = "${oc.env:VAULT_REPOSITORY,null}"
    username: str | None = "${oc.env:GITHUB_USERNAME,null}"
    token: str | None = "${oc.env:GITHUB_TOKEN,null}"


@dataclass
class ObsidianConfig:
    vault_path: Path = MISSING
    db_file: str = "vault.db"
    content_directory: str = "contents"
    topic_directory: str = "topics"
    artefact_directory: str = "artefacts"
    index_file: str = "index.md"
    github_vault: GithubVault = field(default_factory=GithubVault)


load_obsidian_config: Callable[[], ObsidianConfig] = partial(
    load_config, ObsidianConfig, "obsidian.yaml"
)

def get_sqlite_url() -> str:

    obsidian_config = load_obsidian_config()
    return f"sqlite:///{obsidian_config.vault_path}/{obsidian_config.db_file}"


def setup_logger():
    with open(Path(os.getenv("CONF_DIR")) / "logging.yaml") as config_file:
        logging_config = yaml.safe_load(config_file)
    logging.config.dictConfig(logging_config)
