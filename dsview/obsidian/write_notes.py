import logging
from pathlib import Path

import frontmatter
from sqlmodel import Session

from dsview.db.query import (
    get_content_extraction,
    get_content_linked_topics,
    get_topic_linked_contents,
)
from dsview.db.schemas import (
    ExtractionTopic,
    InputContent,
)
from dsview.extraction.models.topics_extraction import DataScienceTopic

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


def write_topic_note(topic: ExtractionTopic, topic_path: Path, session: Session):
    if topic_path.exists():
        raise ObsidianNoteShouldNotExists(topic_path)

    note = frontmatter.Post(topic.description)
    note["type"] = topic.type

    write_note(note, topic_path)


def write_topic_list_notes(topics_id: list[int], session: Session):
    for topic_id in topics_id:
        topic = session.get(ExtractionTopic, topic_id)
        topic_path = get_topic_path(topic.name, topic.type)
        write_topic_note(topic, topic_path, session)


def update_topic_note(
    old_topic: DataScienceTopic, new_topic: ExtractionTopic, session: Session
):
    # Deleting old note
    old_topic_path = get_topic_path(
        old_topic.name,
        old_topic.type.value,
    )
    old_topic_link = get_topic_link(old_topic.name, old_topic.type.value)
    old_topic_path.unlink()

    # Writing updated note
    new_topic_path = get_topic_path(new_topic.name, new_topic.type)
    new_topic_link = get_topic_link(new_topic.name, new_topic.type)
    write_topic_note(new_topic, new_topic_path, session)

    # Replacing stale topic links
    extracted_content_list = get_topic_linked_contents(new_topic.id, session)

    for content in extracted_content_list:
        content_path = get_content_path(content.title, content.content_type)
        content_note = frontmatter.load(content_path)

        content_note.content = content_note.content.replace(
            f"{old_topic_link}\n\n",
            f"{new_topic_link}\n\n",
        )

        write_note(content_note, content_path)


def update_topic_list_notes(
    updated_topics: list[tuple[int, DataScienceTopic]], session: Session
):
    for topic_id, old_topic in updated_topics:
        new_topic = session.get(ExtractionTopic, topic_id)
        update_topic_note(old_topic, new_topic, session)


# ? What about jinja template for this
def write_content_note(content: InputContent, hyperlink: str, session: Session):
    # Need more than extracted content > query
    extracted_content, extracted_links, extracted_tags = get_content_extraction(
        content.id, session
    )
    linked_topics = get_content_linked_topics(content.id, session)

    note_content = hyperlink + "\n"
    note_content += "## Summary\n\n" + extracted_content[0].summary

    note_content += "\n## Links\n\n"
    for link in extracted_links:
        note_content += f"- [{link.name}]({link.url}) : {link.description}\n"

    note_content += "\n## Topics\n\n"
    for topic in linked_topics:
        note_content += f"{get_topic_link(topic.name, topic.type)}\n\n"

    note = frontmatter.Post(note_content, **content.get_str_dict())
    note["type"] = "Content"

    note["tags"] = [tag.name.replace(" ", "_") for tag in extracted_tags]

    content_path = get_content_path(
        extracted_content[0].title, extracted_content[0].content_type
    )

    write_note(note, content_path)

    logger.info("Main content note generation completed")
