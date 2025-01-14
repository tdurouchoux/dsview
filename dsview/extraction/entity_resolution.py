from datetime import date
import logging
from pathlib import Path
from typing import List

import frontmatter
from langchain_core.prompts import ChatPromptTemplate
from Levenshtein import jaro
import obsidiantools.api as otools
from openai import OpenAI
from pydantic.v1 import BaseModel, Field
from sqlmodel import Session

from dsview.obsidian.obsidian_utils import get_topic_link, retrieve_topics_path
from dsview.config import load_extraction_config, load_obsidian_config
from .content_extraction import DataScienceTopic
from .extraction_db_schema import ERDecision, find_or_add_comparison
from .prompt_loader import get_prompt

ER_SYSTEM_PROMPT_FILE = "system_entity_resolution.txt"
ER_USER_PROMPT_FILE = "user_entity_resolution.txt"


logger = logging.getLogger(__name__)

obsidian_config = load_obsidian_config()
content_extraction_config = load_extraction_config()

# TODO : Give examples in input prompt
# TODO : Bettter search for close topics
# TODO : only one query for 1 topic


class SimpleERResult(BaseModel):
    merge_topic: bool = Field(
        description="Wether or not the two provided topics should be merged."
    )


class ERResult(BaseModel):
    merge_topic: bool = Field(
        description="Wether or not the two provided topics should be merged."
    )
    topic: DataScienceTopic = Field(
        default=None,
        description="Result of the merge between the two topics, only provided if topics should be merged",
    )


def get_er_predict_func(
    model_name: str, system_prompt: str, user_prompt: str
) -> callable:
    client = OpenAI()

    if system_prompt is None:
        system_prompt = get_prompt(ER_SYSTEM_PROMPT_FILE)

    system_prompt = system_prompt.format(
        ", ".join(content_extraction_config.topic_categories)
    )

    if user_prompt is None:
        user_prompt = get_prompt(ER_USER_PROMPT_FILE)

    def er_predict(topic_comparison: dict[str, str]) -> ERResult:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.format(**topic_comparison)},
            ],
            response_format=ERResult,
        )
        return completion.choices[0].message.parsed

    return er_predict


def get_er_classifier(llm, system_prompt: str, user_prompt: str):
    if system_prompt is None:
        system_prompt = get_prompt(ER_SYSTEM_PROMPT_FILE)

    system_prompt = system_prompt.format(
        ", ".join(content_extraction_config.topic_categories)
    )

    if user_prompt is None:
        user_prompt = get_prompt(ER_USER_PROMPT_FILE)

    er_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", user_prompt),
        ]
    )

    return er_prompt | llm.with_structured_output(schema=ERResult)


# def get_er_classifier()
# def predict(topic_comparison)


class ERSolver:
    def __init__(self, llm, system_prompt: str = None, user_prompt: str = None) -> None:
        self.er_classifier = get_er_classifier(llm, system_prompt, user_prompt)
        self.vault = otools.Vault(obsidian_config.vault_path).connect()
        self.topic_note_list: List[Path] = None

    # TODO Could be cleaned with dedicated topic_list storing name and types

    def _find_close_note(self, topic_name: str) -> List[Path]:
        close_notes_dict = {}
        for topic_note in self.topic_note_list:
            name_similarity = jaro(topic_name, topic_note.stem)
            if name_similarity > content_extraction_config.er_jaro_threshold:
                close_notes_dict[topic_note] = name_similarity

        sorted_close_notes_dict = dict(
            sorted(close_notes_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return list(sorted_close_notes_dict.keys())

    def _prepare_merge(self, old_note: Path, new_topic: DataScienceTopic):
        new_topic_link = get_topic_link(new_topic.name, new_topic.type)

        old_backlinks = self.vault.get_backlinks(old_note.stem)

        for backlink in old_backlinks:
            content_path = (
                obsidian_config.vault_path / self.vault.md_file_index[backlink]
            )
            content_note = frontmatter.load(content_path)

            content_note.content = content_note.content.replace(
                f"{get_topic_link(old_note.stem, old_note.parent.name)}\n\n",
                f"{new_topic_link}\n\n",
            )

            with open(content_path, "wb") as content_file:
                frontmatter.dump(content_note, content_file)

        old_note.unlink()
        self.topic_note_list.remove(old_note)

    def _single_topic_er(
        self, topic: DataScienceTopic, session: Session
    ) -> DataScienceTopic:
        close_notes = self._find_close_note(topic.name)

        if len(close_notes) == 0:
            return topic

        for note in close_notes:
            note_content = frontmatter.load(note)

            if topic.name == note.stem and topic.type == note.parent.name:
                #! not really clean
                logger.warning("Topic %s already exists, skipping ER.", topic.name)
                return topic

            topic_comparison = {
                "name_1": topic.name,
                "type_1": topic.type,
                "description_1": topic.description,
                "name_2": note.stem,
                "type_2": note.parent.name,
                "description_2": note_content.content,
            }

            er_comparison = find_or_add_comparison(topic_comparison, session)

            result: ERResult = self.er_classifier.invoke(topic_comparison)

            if result.merge_topic:
                logger.warning(
                    "Merging topics %s and %s into %s ",
                    topic.name,
                    note.stem,
                    result.topic.name,
                )
                self._prepare_merge(note, result.topic)

                er_decision = ERDecision(
                    comparison_id=er_comparison.id,
                    decision_date=date.today().isoformat(),
                    merge_topic=result.merge_topic,
                    merge_name=result.topic.name,
                    merge_type=result.topic.type,
                    merge_description=result.topic.description,
                )

                session.add(er_decision)
                session.commit()

                return result.topic

            er_decision = ERDecision(
                comparison_id=er_comparison.id,
                decision_date=date.today().isoformat(),
                merge_topic=result.merge_topic,
            )

            session.add(er_decision)
            session.commit()

        return topic

    def run_topics_er(
        self, topics: List[DataScienceTopic], session: Session
    ) -> DataScienceTopic:
        logger.info("Launching topics ER ")

        self.vault = otools.Vault(obsidian_config.vault_path).connect()
        self.topic_note_list = retrieve_topics_path()

        return [self._single_topic_er(topic, session) for topic in topics]
