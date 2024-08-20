from dataclasses import dataclass
from datetime import date
import logging

import frontmatter

from dsview.config import load_obsidian_config
from dsview.content_extraction import TopicExtraction

logger = logging.getLogger(__name__)
config = load_obsidian_config()


@dataclass
class NotesGenerator:
    summary: str
    topics: TopicExtraction
    already_read: bool
    upload_date: date
    source: str = None

    def generate_content_md(self):
        logger.info("Starting main content note generation")

        note_content = "## Summary\n\n" + self.summary
        note_content += "\n\n## Subjects\n"

        for subject in self.topics.subjects:
            note_content += f"- [[subjects/{subject.name}]]\n"

        note = frontmatter.Post(note_content)
        note["date"] = self.upload_date.date().isoformat()
        note["type"] = "content"
        note["tags"] = [tag.name.replace(" ", "_") for tag in self.topics.tags]

        if self.source is not None:
            note["source"] = self.source

        note_filepath = config.vault_path / "content" / f"{self.topics.title}.md"
        with open(note_filepath, "wb") as note_file:
            frontmatter.dump(note, note_file)

        logger.info("Main content note generation completed")

    def generate_subjects_md(self):
        logger.info("Starting subjects markdown notes generation")

        for subject in self.topics.subjects:
            logger.info("Writing notes for subject %s", subject.name)

            note_filepath = config.vault_path / "subjects" / f"{subject.name}.md"

            if note_filepath.exists():
                logger.info("Skipped subject %s, it already exists.", subject.name)
                continue

            note = frontmatter.Post(subject.description)
            note["date"] = self.upload_date.date().isoformat()
            note["type"] = subject.type

            with open(note_filepath, "wb") as note_file:
                frontmatter.dump(note, note_file)

        logger.info("Subjects notes generation completed")
