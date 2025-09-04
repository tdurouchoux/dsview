import logging
from pathlib import Path

import frontmatter
from sqlmodel import Session, inspect

from dsview.db.schemas import (
    ExtractionTopic,
    InputContent,
)
from dsview.db.schemas.extraction_schema import ExtractionResult

from .obsidian_utils import get_content_path, get_topic_link, get_topic_path

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

    with open(note_path, "wb") as note_file:
        frontmatter.dump(note, note_file)


def write_topic_note(topic: ExtractionTopic):
    topic_path = get_topic_path(topic.name, topic.type)

    # if topic_path.exists():
    #     raise ObsidianNoteShouldNotExists(topic_path)

    note = frontmatter.Post(topic.description)
    note["type"] = topic.type

    write_note(note, topic_path)


def update_topic_note(topic: ExtractionTopic):
    topic_inspection = inspect(topic)

    old_topic_name = topic_inspection.attrs.name.history.non_added()[0]
    old_topic_type = topic_inspection.attrs.type.history.non_added()[0]

    logger.warning("Updating topic note %s to %s", old_topic_name, topic.name)

    # Deleting old note
    old_topic_path = get_topic_path(
        old_topic_name,
        old_topic_type,
    )
    if old_topic_path.exists():
        old_topic_path.unlink()
    else:
        logger.error(
            "Could not find topic note : %s. Continuing anyway", old_topic_name
        )

    write_topic_note(topic)

    # Writing updated note
    old_topic_link = get_topic_link(old_topic_name, old_topic_type)
    new_topic_link = get_topic_link(topic.name, topic.type)

    for extraction in topic.extractions:
        content_path = get_content_path(extraction.title, extraction.content_type)
        content_note = frontmatter.load(content_path)

        content_note.content = content_note.content.replace(
            f"{old_topic_link}\n\n",
            f"{new_topic_link}\n\n",
        )

        write_note(content_note, content_path)


def write_and_update_topic_list_notes(topics: list[ExtractionTopic]):
    for topic in topics:
        topic_inspection = inspect(topic)

        if topic_inspection.modified and not topic_inspection.pending:
            update_topic_note(topic)
        else:
            write_topic_note(topic)


# TODO finish updating write_notes to be cleaner


def update_topic_list_notes(updated_topics: list[ExtractionTopic], session: Session):
    for topic_id, old_topic in updated_topics:
        new_topic = session.get(ExtractionTopic, topic_id)
        update_topic_note(old_topic, new_topic, session)


# ? What about jinja template for this
def write_content_note(content: InputContent, extraction_result: ExtractionResult):
    # Need more than extracted content > query
    note_content = str(content.link) + "\n"
    note_content += "## Summary\n\n" + extraction_result.summary

    note_content += "\n## Links\n\n"
    for link in extraction_result.links:
        note_content += f"- [{link.name}]({link.url}) : {link.description}\n"

    note_content += "\n## Topics\n\n"
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
