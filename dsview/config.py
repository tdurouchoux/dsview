from dataclasses import dataclass
from functools import partial
import logging.config
import os
from pathlib import Path
from typing import List, Callable

from dotenv import load_dotenv
from omegaconf import OmegaConf, MISSING
import yaml

load_dotenv()
os.getenv("CONF_DIR")


@dataclass
class ObsidianConfig:
    vault_path: Path = MISSING


@dataclass
class TopicExtractionConfig:
    tags: List[str]
    subjects: List[str]


def load_config(config_class, conf_file: Path):
    default_config = OmegaConf.structured(config_class)
    file_config = OmegaConf.load(Path(os.getenv("CONF_DIR")) / conf_file)
    merged_config = OmegaConf.merge(default_config, file_config)

    return OmegaConf.to_object(merged_config)


load_obsidian_config: Callable[[], ObsidianConfig] = partial(
    load_config, ObsidianConfig, "obsidian.yaml"
)


load_topic_extraction_config: Callable[[], TopicExtractionConfig] = partial(
    load_config, TopicExtractionConfig, "topic_extraction.yaml"
)


def setup_logger():
    with open(Path(os.getenv("CONF_DIR")) / "logging.yaml") as config_file:
        logging_config = yaml.safe_load(config_file)
    logging.config.dictConfig(logging_config)
