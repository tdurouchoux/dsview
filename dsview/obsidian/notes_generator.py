from dataclasses import dataclass
from datetime import date
import logging
from typing import List

import frontmatter

from dsview.content.input_content import InputContent
from dsview.extraction.content_extraction import ContentDescription, DataScienceTopic
from dsview.obsidian.obsidian_utils import (
    get_topic_link,
    get_topic_path,
    get_content_path,
)

logger = logging.getLogger(__name__)


@dataclass
class NotesGenerator:
    content: InputContent
    hyperlink: str
    summary: str
    content_description: ContentDescription
    topics: List[DataScienceTopic]
    content_links: str = None

    def generate_topics_md(self):
        logger.info("Starting topics markdown notes generation")

        for topic in self.topics:
            logger.info("Writing note for topic %s", topic.name)

            topic_path = get_topic_path(topic.name, topic.type)
            topic_path.parent.mkdir(parents=True, exist_ok=True)

            if topic_path.exists():
                logger.warning("Skipped subject %s, it already exists.", topic.name)
                continue

            note = frontmatter.Post(topic.description)
            note["date"] = self.content.upload_date.isoformat()
            note["type"] = topic.type

            with open(topic_path, "wb") as note_file:
                frontmatter.dump(note, note_file)

        logger.info("Subjects notes generation completed")

    def generate_content_md(self):
        logger.info("Starting main content note generation")

        note_content = self.hyperlink + "\n"
        note_content += "## Summary\n\n" + self.summary

        if self.content_links is not None:
            note_content += "\n## Links\n\n" + self.content_links

        note_content += "\n## Topics\n\n"
        for topic in self.topics:
            note_content += f"- {get_topic_link(topic.name, topic.type)}\n"

        note = frontmatter.Post(note_content, **self.content.get_str_dict())
        note["type"] = "Content"

        if len(self.content_description.tags) > 0:
            note["tags"] = [
                tag.name.replace(" ", "_")
                for tag in self.content_description.tags
                if tag is not None
            ]

        content_path = get_content_path(
            self.content_description.title, self.content_description.content_type
        )
        content_path.parent.mkdir(parents=True, exist_ok=True)

        with open(content_path, "wb") as note_file:
            frontmatter.dump(note, note_file)

        logger.info("Main content note generation completed")
