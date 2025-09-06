import shutil
import urllib.parse
from functools import partial
from pathlib import Path
from typing import List

import regex as re

from dsview.config import load_obsidian_config

config = load_obsidian_config()


def clean_note_title(note_name: str) -> str:
    cleaned_name = re.sub(r"\W+", " ", note_name)
    return re.sub(r"\s+", " ", cleaned_name).strip()


def get_topic_link(topic_name: str, topic_type: str) -> str:
    topic_name_url = urllib.parse.quote(clean_note_title(topic_name))
    return f"![]({config.topic_directory}/{topic_type}/{topic_name_url})"


def get_topic_path(topic_name: str, topic_type: str) -> Path:
    return (
        config.vault_path
        / config.topic_directory
        / topic_type
        / f"{clean_note_title(topic_name)}.md"
    )


def get_content_path(content_title: str, content_type: str) -> Path:
    return (
        config.vault_path
        / config.content_directory
        / content_type
        / f"{clean_note_title(content_title)}.md"
    )


def get_content_url_link(note_title: str) -> str:
    cleaned_title = clean_note_title(note_title)
    title_query = urllib.parse.quote(cleaned_title)
    content_url = f"obsidian://open?vault={config.vault_path.name}&file={title_query}"

    return content_url


class InvalidNoteDirectory(Exception):
    def __init__(self, note_dir: str):
        super().__init__(
            self,
            f"note_dir must take the value {config.topic_directory} or "
            f"{config.content_directory}. '{note_dir}' was provided",
        )


def retrieve_notes_path(note_dir: str) -> List[Path]:
    notes_directory = config.vault_path / note_dir

    notes_path = []

    for sub_directory in notes_directory.iterdir():
        notes_path += [note for note in sub_directory.glob("*.md")]

    return notes_path


retrieve_topics_path = partial(retrieve_notes_path, config.topic_directory)
retrieve_contents_path = partial(retrieve_notes_path, config.content_directory)


def clear_vault():
    topic_dir = config.vault_path / config.topic_directory
    content_dir = config.vault_path / config.content_directory

    if topic_dir.exists():
        shutil.rmtree(topic_dir)

    if content_dir.exists():
        shutil.rmtree(content_dir)

    topic_dir.mkdir()
    content_dir.mkdir()
