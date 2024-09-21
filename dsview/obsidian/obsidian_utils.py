from functools import partial
import shutil
from typing import List

from pathlib import Path

from dsview.config import load_obsidian_config

config = load_obsidian_config()


def get_topic_link(topic_name: str, topic_type: str) -> str:
    return f"[[{config.topic_directory}/{topic_type}/{topic_name.replace('/', ' ')}]]"


def get_topic_path(topic_name: str, topic_type: str) -> Path:
    return (
        config.vault_path
        / config.topic_directory
        / topic_type
        / f"{topic_name.replace('/', ' ')}.md"
    )


def get_content_path(content_title: str, content_type: str) -> Path:
    return (
        config.vault_path
        / config.content_directory
        / content_type
        / f"{content_title.replace('/', ' ')}.md"
    )


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


def get_pdf_filepath(pdf_filename: str) -> Path:
    return config.vault_path / config.artefact_directory / pdf_filename


def clear_vault():
    topic_dir = config.vault_path / config.topic_directory
    content_dir = config.vault_path / config.content_directory
    artefact_dir = config.vault_path / config.artefact_directory

    shutil.rmtree(topic_dir)
    shutil.rmtree(content_dir)
    shutil.rmtree(artefact_dir)

    topic_dir.mkdir()
    content_dir.mkdir()
    artefact_dir.mkdir()
