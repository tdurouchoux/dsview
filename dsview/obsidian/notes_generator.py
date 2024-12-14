from dataclasses import dataclass
from datetime import datetime
import logging
import regex as re
from typing import List

import frontmatter

from dsview.content.content_db_schema import InputContent
from dsview.extraction.content_extraction import ContentDescription, DataScienceTopic
from dsview.obsidian.obsidian_utils import (
    get_topic_link,
    get_topic_path,
    get_content_path,
    get_index_path,
    get_content_url_link,
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
                logger.warning("Skipped topic %s, it already exists.", topic.name)
                continue

            note = frontmatter.Post(topic.description)
            note["date"] = self.content.upload_date.isoformat()
            note["type"] = topic.type

            with open(topic_path, "wb") as note_file:
                frontmatter.dump(note, note_file)

        logger.info("Subjects notes generation completed")

    def generate_content_md(self):
        logger.info("Starting main content note generation")

        # Adding summary
        note_content = self.hyperlink + "\n"
        note_content += "## Summary\n\n" + self.summary

        # Adding links
        if self.content_links is not None:
            note_content += "\n## Links\n\n" + self.content_links

        # Adding topics
        note_content += "\n## Topics\n\n"
        for topic in self.topics:
            note_content += f"{get_topic_link(topic.name, topic.type)}\n\n"

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

    def insert_in_index(self) -> None:
        index_path = get_index_path()

        update_time_line = "> Last updated on : " + datetime.now().isoformat() + "\n\n"

        table_header = "| Upload Date | Read priority | Source | Note |\n"
        table_header += "|-------------|---------------|--------|------|\n"

        table_line = (
            f"| {self.content.upload_date.isoformat()} | {self.content.read_priority} |"
        )
        table_line += f" {self.content.source} | {get_content_url_link(self.content_description.title)} |\n"

        if not index_path.exists():
            index_content = "# Index of Input Contents\n\n"
            index_content += update_time_line

            index_content += table_header
            index_content += table_line
        else:
            with open(index_path, "r") as index_file:
                index_content = index_file.read()

            index_content = re.sub(
                "> Last updated on : .*\n\n", update_time_line, index_content
            )

            index_content = index_content.replace(
                table_header, table_header + table_line
            )

        with open(index_path, "wb") as index_file:
            index_file.write(index_content.encode("utf-8"))
