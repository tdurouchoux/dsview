from pathlib import Path

import pytest
from dotenv import load_dotenv

from dsview.config import ObsidianConfig
from dsview.obsidian.obsidian_utils import clean_note_title, get_topic_link

load_dotenv()
config = ObsidianConfig(vault_path=Path("test_vault"))


@pytest.mark.parametrize(
    "note_title,cleaned_note_title",
    [
        (" This   is .,'a title ", "This is a title"),
        ("   Another,, -** title", "Another title"),
    ],
)
def test_clean_note_title(note_title: str, cleaned_note_title: str):
    assert clean_note_title(note_title) == cleaned_note_title


def test_get_topic_link():
    topic_link = get_topic_link("LLM", "Concept")

    assert topic_link == "![](topics/Concept/LLM)"
