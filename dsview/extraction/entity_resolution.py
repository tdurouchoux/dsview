import logging
from pathlib import Path
from typing import List

import frontmatter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from Levenshtein import jaro
import obsidiantools.api as otools

from dsview.obsidian.obsidian_utils import get_topic_link, retrieve_topics_path
from dsview.config import load_extraction_config, load_obsidian_config
from .content_extraction import DataScienceTopic
from .prompt_loader import get_prompt

logger = logging.getLogger(__name__)

obsidian_config = load_obsidian_config()
content_extraction_config = load_extraction_config()

ER_THRESHOLD = 0.75


class ERResult(BaseModel):
    merge_topic: bool = Field(
        description="Wether or not the two provided topics should be merged."
    )
    topic: DataScienceTopic = Field(
        default=None,
        description="Result of the merge between the two topics, only provided if topics should be merged",
    )


def get_er_classifier(llm):
    er_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                get_prompt("system_entity_resolution.txt").format(
                    ", ".join(content_extraction_config.topic_categories)
                ),
            ),
            ("user", get_prompt("user_entity_resolution.txt")),
        ]
    )

    return er_prompt | llm.with_structured_output(schema=ERResult)


class ERSolver:
    def __init__(self, llm) -> None:
        self.er_classifier = get_er_classifier(llm)
        self.vault = otools.Vault(obsidian_config.vault_path).connect()
        self.topic_note_list: List[Path] = None

    def _find_close_note(self, topic_name: str) -> List[Path]:
        close_notes_dict = {}
        for topic_note in self.topic_note_list:
            name_similarity = jaro(topic_name, topic_note.stem)
            if name_similarity > ER_THRESHOLD:
                close_notes_dict[topic_note] = name_similarity

        sorted_close_notes_dict = dict(
            sorted(close_notes_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return list(sorted_close_notes_dict.keys())

    def _prepare_merge(self, old_note: Path, new_topic: DataScienceTopic):
        new_topic_link = get_topic_link(new_topic.name, new_topic.type)

        old_note_name = str(old_note.relative_to(obsidian_config.vault_path)).replace(
            ".md", ""
        )
        old_backlinks = self.vault.get_backlinks(old_note_name)

        for backlink in old_backlinks:
            content_path = (
                obsidian_config.vault_path / self.vault.md_file_index[backlink]
            )
            content_note = frontmatter.load(content_path)

            content_note.content = content_note.content.replace(
                f"[[{old_note_name}]]",
                new_topic_link,
            )

            with open(content_path, "wb") as content_file:
                frontmatter.dump(content_note, content_file)

        old_note.unlink()
        self.topic_note_list.remove(old_note)

    def _single_topic_er(self, topic: DataScienceTopic) -> DataScienceTopic:
        close_notes = self._find_close_note(topic.name)

        if len(close_notes) == 0:
            return topic

        for note in close_notes:
            note_content = frontmatter.load(note)

            result: ERResult = self.er_classifier.invoke(
                {
                    "name_1": topic.name,
                    "type_1": topic.type,
                    "description_1": topic.description,
                    "name_2": note.stem,
                    "type_2": note_content["type"],
                    "description_2": note_content.content,
                }
            )

            if result.merge_topic:
                logger.warning(
                    "Merging topics %s and %s into %s ",
                    topic.name,
                    note.stem,
                    result.topic.name,
                )
                self._prepare_merge(note, result.topic)
                return result.topic

        return topic

    def run_topics_er(self, topics: List[DataScienceTopic]) -> DataScienceTopic:
        logger.info("Launching topics ER ")

        self.vault = otools.Vault(obsidian_config.vault_path).connect()
        self.topic_note_list = retrieve_topics_path()

        return [self._single_topic_er(topic) for topic in topics]
