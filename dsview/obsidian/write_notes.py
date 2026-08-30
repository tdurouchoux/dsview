from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import frontmatter

from dsview.db.schemas import (
    ExtractionTopic,
    InputContent,
)
from dsview.db.schemas.extraction_schema import ExtractionResult

from .obsidian_utils import get_content_path, get_topic_link, get_topic_path

if TYPE_CHECKING:
    # Imported here: topic_er pulls in the extraction models, whose enums are
    # built from config at import time, and cli.py imports this module
    from dsview.extraction.topic_er import TopicMerge

logger = logging.getLogger(__name__)


class ObsidianNoteShouldNotExists(Exception):
    def __init__(self, note_path: Path) -> None:
        super().__init__(
            f"Attempt to write note {note_path} "
            "and found that note already exists. "
            "It may indicates a poor synchronisation "
            "between Obsidian and DB."
        )


def write_note(note: frontmatter.Post, note_path: Path):
    note_path.parent.mkdir(parents=True, exist_ok=True)

    with open(note_path, "w") as note_file:
        frontmatter.dump(note, note_file)


def write_topic_note(topic: ExtractionTopic):
    topic_path = get_topic_path(topic.name, topic.type)

    # if topic_path.exists():
    #     raise ObsidianNoteShouldNotExists(topic_path)

    note = frontmatter.Post(topic.description)
    note["type"] = topic.type

    write_note(note, topic_path)


def update_topic_note(topic: ExtractionTopic, old_name: str, old_type: str):
    """Rewrite the vault after a merge renamed `topic` from (old_name, old_type).

    The pre-merge identity is passed in rather than read back from the SQLAlchemy
    attribute history, which any flush resets (see #55).
    """
    logger.info("Updating topic note %s to %s", old_name, topic.name)

    # Deleting old note. A merge often keeps the name and type, in which case
    # this is the note write_topic_note is about to write again
    old_topic_path = get_topic_path(old_name, old_type)

    if old_topic_path.exists():
        old_topic_path.unlink()
    else:
        logger.error("Could not find topic note : %s. Continuing anyway", old_name)

    write_topic_note(topic)

    old_topic_link = get_topic_link(old_name, old_type)
    new_topic_link = get_topic_link(topic.name, topic.type)

    for extraction in topic.extractions:
        content_path = get_content_path(extraction.title, extraction.content_type)

        if not content_path.exists():
            logger.error(
                "Could not find content note : %s. Continuing anyway", extraction.title
            )
            continue

        content_note = frontmatter.load(content_path)

        # Matching the bare link: frontmatter strips trailing newlines, so the
        # last embed of a note has no "\n\n" after it
        content_note.content = content_note.content.replace(
            old_topic_link,
            new_topic_link,
        )

        write_note(content_note, content_path)


def write_and_update_topic_list_notes(
    topics: list[ExtractionTopic], merges: list[TopicMerge]
):
    merged_topics = {merge.topic_id: merge for merge in merges}

    for topic in topics:
        merge = merged_topics.get(topic.id)

        if merge is None:
            write_topic_note(topic)
        else:
            update_topic_note(topic, merge.old_name, merge.old_type)


def update_content_note_metadata(
    content: InputContent, extraction_result: ExtractionResult
) -> None:
    """Refresh a content note's frontmatter after a DB-only field update.

    Used by the /relevance endpoint: already_read/read_priority/relevance are
    updated on the InputContent row without re-running extraction, so the note
    body is left untouched and only the frontmatter is rewritten from `content`.
    """
    content_path = get_content_path(
        extraction_result.title, extraction_result.content_type
    )

    if not content_path.exists():
        logger.error(
            "Could not find content note : %s. Continuing anyway",
            extraction_result.title,
        )
        return

    note = frontmatter.load(content_path)
    note.metadata.update(content.get_str_dict())

    write_note(note, content_path)


# ? What about jinja template for this
def write_content_note(content: InputContent, extraction_result: ExtractionResult):
    # Need more than extracted content > query
    note_content = str(content.link) + "\n\n"
    note_content += "## Summary\n\n" + extraction_result.summary

    note_content += "\n\n## Links\n\n"
    for link in extraction_result.links:
        note_content += f"- [{link.name}]({link.url}) : {link.description}\n"

    note_content += "\n\n## Topics\n\n"
    for topic in extraction_result.topics:
        note_content += f"{get_topic_link(topic.name, topic.type)}\n\n"

    note = frontmatter.Post(note_content, **content.get_str_dict())
    note["type"] = "Content"

    note["tags"] = [tag.name.replace(" ", "_") for tag in extraction_result.tags]

    content_path = get_content_path(
        extraction_result.title, extraction_result.content_type
    )

    write_note(note, content_path)

    logger.info("Main content note generation completed")
