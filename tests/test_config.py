import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from dsview.config import (
    ExtractionConfig,
    GithubVault,
    GlobalModelConfig,
    LLMProvider,
    ModelConfig,
    ModelConfigurationError,
    ModelType,
    ObsidianConfig,
    get_sqlite_url,
    load_config,
    load_extraction_config,
    load_model_config,
    load_obsidian_config,
)

MODEL_CONFIG_TEST_1 = """
configs:
  - model_type: DEFAULT
    model_config:
      chat_model: gpt-4o-mini-2024-07-18
      embedding_model: text-embedding-3-small
      provider: OPENAI
      token_limit: 128000
  - model_type: LINKS_EXTRACTION
    model_config:
      chat_model: mistral
      provider: MISTRAL
      token_limit: 60000

"""

MODEL_CONFIG_TEST_2 = """
configs:
  - model_type: LINKS_EXTRACTION
    model_config:
      chat_model: mistral
      provider: MISTRAL
      token_limit: 60000

"""
EXTRACTION_CONFIG_TEST = """
    tags:
        MACHINE_LEARNING: "Machine Learning"
    content_types:
        COURSE: Course
    topic_categories:
        LIBRARY: Library
    er_jaro_threshold: 0.5
"""

OBSIDIAN_CONFIG_TEST = """
    vault_path: dsview_vault
    db_file: vault.db
    content_directory: "contents"
    topic_directory: "topics"
"""

CONF_DIR = Path("test_config")

# Env variables for obsidian configuration test
os.environ["VAULT_REPOSITORY"] = "repo_url"
os.environ["GITHUB_USERNAME"] = "me"
os.environ["GITHUB_USER_EMAIL"] = "me@mail.com"
os.environ["GITHUB_TOKEN"] = "some_token"


@contextmanager
def generate_config(filename: str, conf_content: str):

    curr_conf_dir = os.getenv("CONF_DIR")

    os.environ["CONF_DIR"] = str(CONF_DIR)
    CONF_DIR.mkdir(exist_ok=True)

    tmp_filepath = CONF_DIR / filename

    with open(tmp_filepath, "w") as tmp_file:
        tmp_file.write(conf_content)

    try:
        yield tmp_filepath
    finally:
        tmp_filepath.unlink()
        CONF_DIR.rmdir()

        os.environ["CONF_DIR"] = curr_conf_dir


def test_load_config():
    with generate_config("model.yaml", MODEL_CONFIG_TEST_1):
        config = load_config(GlobalModelConfig, "model.yaml")

        assert isinstance(config, GlobalModelConfig)


def test_model_config_1():
    with generate_config("model.yaml", MODEL_CONFIG_TEST_1):
        expected_model_config = ModelConfig(
            chat_model="gpt-4o-mini-2024-07-18",
            embedding_model="text-embedding-3-small",
            provider=LLMProvider.OPENAI,
            token_limit=128000,
        )

        model_config = load_model_config()
        assert expected_model_config == model_config


def test_model_config_2():
    with generate_config("model.yaml", MODEL_CONFIG_TEST_1):
        expected_model_config = ModelConfig(
            chat_model="mistral",
            embedding_model=None,
            provider=LLMProvider.MISTRAL,
            token_limit=60000,
        )

        model_config = load_model_config(ModelType.LINKS_EXTRACTION)
        assert expected_model_config == model_config


def test_model_config_3():
    with generate_config("model.yaml", MODEL_CONFIG_TEST_2):
        with pytest.raises(ModelConfigurationError):
            load_model_config()


def test_extraction_config():
    with generate_config("extraction.yaml", EXTRACTION_CONFIG_TEST):
        expected_config = ExtractionConfig(
            tags={"MACHINE_LEARNING": "Machine Learning"},
            content_types={"COURSE": "Course"},
            topic_categories={"LIBRARY": "Library"},
            er_jaro_threshold=0.5,
        )

        extraction_config = load_extraction_config()

        assert isinstance(extraction_config, ExtractionConfig)
        assert extraction_config == expected_config


def test_obsidian_config():
    with generate_config("obsidian.yaml", OBSIDIAN_CONFIG_TEST):
        expected_config = ObsidianConfig(
            vault_path=Path("dsview_vault"),
            db_file="vault.db",
            content_directory="contents",
            topic_directory="topics",
            artefact_directory="artefacts",
            github_vault=GithubVault(
                repository="repo_url",
                username="me",
                email="me@mail.com",
                token="some_token",
            ),
        )

        obsidian_config = load_obsidian_config()

        assert isinstance(obsidian_config, ObsidianConfig)
        assert obsidian_config == expected_config


def test_sqlite_url():
    with generate_config("obsidian.yaml", OBSIDIAN_CONFIG_TEST):
        sqlite_url = get_sqlite_url()

        assert sqlite_url == "sqlite:///dsview_vault/vault.db"
