import logging
from dataclasses import dataclass

import frontmatter

from dsview.content.content_db_schema import InputContent
from dsview.extraction.content_extraction import (
    ContentDescription,
    DataScienceTopic,
    RelevantLink,
)
from dsview.obsidian.obsidian_utils import (
    get_content_path,
    get_topic_link,
    get_topic_path,
)

logger = logging.getLogger(__name__)


@dataclass
class NotesGenerator:
    content: InputContent
    hyperlink: str
    summary: str
    content_description: ContentDescription
    topics: list[DataScienceTopic]
    content_links: list[RelevantLink] = None

    def generate_topics_md(self):
        logger.info("Starting topics markdown notes generation")

        for topic in self.topics:
            logger.info("Writing note for topic %s", topic.name)

            topic_path = get_topic_path(topic.name, topic.type.value)
            topic_path.parent.mkdir(parents=True, exist_ok=True)

            if topic_path.exists():
                logger.warning("Skipped topic %s, it already exists.", topic.name)
                continue

            note = frontmatter.Post(topic.description)
            note["date"] = self.content.upload_date.isoformat()
            note["type"] = topic.type.value

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
            note_content += "\n## Links\n\n"
            for link in self.content_links:
                note_content += f"- [{link.name}]({link.url}) : {link.description}\n"

        # Adding topics
        note_content += "\n## Topics\n\n"
        for topic in self.topics:
            note_content += f"{get_topic_link(topic.name, topic.type.value)}\n\n"

        note = frontmatter.Post(note_content, **self.content.get_str_dict())
        note["type"] = "Content"

        if len(self.content_description.tags) > 0:
            note["tags"] = [
                tag.name.value.replace(" ", "_")
                for tag in self.content_description.tags
                if tag is not None
            ]

        content_path = get_content_path(
            self.content_description.title, self.content_description.content_type.value
        )
        content_path.parent.mkdir(parents=True, exist_ok=True)

        with open(content_path, "wb") as note_file:
            frontmatter.dump(note, note_file)

        logger.info("Main content note generation completed")
