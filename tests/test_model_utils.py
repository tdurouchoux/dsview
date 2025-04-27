import os
from pathlib import Path

import pytest

from dsview.config import LLMProvider, ModelConfig
from dsview.models.model_utils import LLMModel, MissingPromptFile, get_prompt
from dsview.models.providers import MistralProvider

TEST_SYSTEM_PROMPT = "You are a helpfull history assistant specialized in {}"
TEST_USER_PROMPT = "Who is {input} ?"


@pytest.fixture
def prompt_dir(tmp_path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    os.environ["PROMPT_DIR"] = str(prompt_dir.resolve())
    return prompt_dir


@pytest.fixture
def user_prompt_file(prompt_dir: Path):
    user_prompt_path = prompt_dir / "user_prompt.txt"
    user_prompt_path.write_text(TEST_USER_PROMPT)
    return user_prompt_path.name


@pytest.fixture
def system_prompt_file(prompt_dir: Path):
    system_prompt_path = prompt_dir / "system_prompt.txt"
    system_prompt_path.write_text(TEST_SYSTEM_PROMPT)
    return system_prompt_path.name


def test_get_prompt(user_prompt_file):
    user_prompt = get_prompt(user_prompt_file)
    assert user_prompt == TEST_USER_PROMPT


def test_error_get_prompt():
    with pytest.raises(MissingPromptFile):
        get_prompt("random_prompt.txt")


@pytest.fixture
def llm_model_test_class(system_prompt_file, user_prompt_file):
    class LLMModelTest(LLMModel):
        DEFAULT_SYSTEM_PROMPT_FILE = system_prompt_file
        DEFAULT_USER_PROMPT_FILE = user_prompt_file
        DEFAULT_SYSTEM_PROMPT_FORMAT = ["France history"]

    return LLMModelTest


@pytest.fixture
def llm_model_test(llm_model_test_class):
    llm_model_test = llm_model_test_class()
    return llm_model_test


def test_llm_model_system_prompt(llm_model_test):
    assert llm_model_test.system_prompt == TEST_SYSTEM_PROMPT.format("France history")


def test_llm_model_user_prompt(llm_model_test):
    assert llm_model_test.user_prompt_template == TEST_USER_PROMPT


def test_llm_model_predict(llm_model_test):
    response = llm_model_test.predict({"input": "Napoleon"})

    assert isinstance(response, str)


def test_override_model_config(llm_model_test_class):
    test_model_config = ModelConfig(
        chat_model="mistral-small-latest", provider=LLMProvider.MISTRAL, token_limit=10
    )

    llm_model_test = llm_model_test_class(model_config=test_model_config)

    assert isinstance(llm_model_test.model_provider, MistralProvider)


def test_override_system_prompt(llm_model_test_class):
    test_override_system_prompt = (
        "You are a useless assistant that ignore everything about {}"
    )

    llm_model_test = llm_model_test_class(system_prompt=test_override_system_prompt)

    assert llm_model_test.system_prompt == test_override_system_prompt.format(
        "France history"
    )


def test_override_user_prompt(llm_model_test_class):
    test_override_user_prompt = "Where is {input}?"
    llm_model_test = llm_model_test_class(user_prompt=test_override_user_prompt)

    assert llm_model_test.user_prompt_template == test_override_user_prompt


def test_override_system_prompt_format(llm_model_test_class):
    llm_model_test = llm_model_test_class(system_prompt_format=["Germany"])
    assert llm_model_test.system_prompt == TEST_SYSTEM_PROMPT.format("Germany")
