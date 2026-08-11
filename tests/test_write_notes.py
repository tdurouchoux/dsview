from datetime import date

import frontmatter
import pytest
from sqlmodel import Session

from dsview.db.schemas import InputContent
from dsview.db.schemas.extraction_schema import (
    ContentTopicRelation,
    ExtractionResult,
    ExtractionTopic,
)
from dsview.extraction.topic_er import TopicMerge
from dsview.obsidian.obsidian_utils import (
    get_content_path,
    get_topic_link,
    get_topic_path,
)
from dsview.obsidian.write_notes import (
    update_topic_note,
    write_and_update_topic_list_notes,
    write_topic_note,
)


@pytest.fixture
def topic(session: Session) -> ExtractionTopic:
    """A topic with its note already on disk, as an earlier ingestion left it."""
    extraction_topic = ExtractionTopic(
        name="BigQuery", type="Platform", description="a warehouse"
    )
    session.add(extraction_topic)
    session.commit()

    write_topic_note(extraction_topic)

    return extraction_topic


def link_content(
    session: Session, topic: ExtractionTopic, title: str, topics: list[str]
) -> InputContent:
    """Link a content to `topic` and write its note embedding `topics`, in order."""
    content = InputContent(
        link=f"https://example.com/{title}", upload_date=date(2026, 1, 1)
    )
    session.add(content)
    session.commit()

    session.add(
        ExtractionResult(
            content_id=content.id,
            title=title,
            content_type="Blog post",
            summary="summary",
            embedding=None,
        )
    )
    session.add(ContentTopicRelation(content_id=content.id, topic_id=topic.id))
    session.commit()

    body = "## Topics\n\n" + "\n\n".join(
        get_topic_link(name, "Platform") for name in topics
    )
    note_path = get_content_path(title, "Blog post")
    note_path.parent.mkdir(parents=True, exist_ok=True)

    with open(note_path, "w") as note_file:
        frontmatter.dump(frontmatter.Post(body), note_file)

    return content


def rename(session: Session, topic: ExtractionTopic, name: str, type_: str):
    topic.name = name
    topic.type = type_
    session.commit()


# --- update_topic_note --------------------------------------------------------


def test_renamed_topic_note_moves_and_content_embeds_follow(session, vault, topic):
    # the merged topic is the *last* embed: frontmatter strips the trailing
    # newlines, so a match on "link\n\n" would silently miss it
    link_content(session, topic, "Title A", ["Pandas", "BigQuery"])
    rename(session, topic, "Google BigQuery", "Tool")

    update_topic_note(topic, "BigQuery", "Platform")

    assert not get_topic_path("BigQuery", "Platform").exists()
    assert get_topic_path("Google BigQuery", "Tool").exists()

    note = frontmatter.load(get_content_path("Title A", "Blog post"))
    assert get_topic_link("BigQuery", "Platform") not in note.content
    assert get_topic_link("Google BigQuery", "Tool") in note.content
    assert get_topic_link("Pandas", "Platform") in note.content


def test_merge_keeping_name_and_type_rewrites_the_note_in_place(session, vault, topic):
    # 40% of real merges keep the candidate's name, so the note deleted as "old"
    # is the one being written again
    topic.description = "a merged warehouse"
    session.commit()

    update_topic_note(topic, "BigQuery", "Platform")

    note_path = get_topic_path("BigQuery", "Platform")
    assert note_path.exists()
    assert frontmatter.load(note_path).content == "a merged warehouse"


def test_missing_content_note_is_skipped(session, vault, topic):
    link_content(session, topic, "Title A", ["BigQuery"])
    get_content_path("Title A", "Blog post").unlink()
    rename(session, topic, "Google BigQuery", "Tool")

    update_topic_note(topic, "BigQuery", "Platform")

    # the ingestion is not aborted, and the renamed topic note is still written
    assert get_topic_path("Google BigQuery", "Tool").exists()


# --- write_and_update_topic_list_notes ----------------------------------------


def test_merged_topic_is_updated_even_once_the_session_is_flushed(
    session, vault, topic
):
    link_content(session, topic, "Title A", ["BigQuery"])
    rename(session, topic, "Google BigQuery", "Tool")

    # the commit above flushed, so inspect(topic).modified is False: routing must
    # come from the merge list, not from ORM state
    write_and_update_topic_list_notes(
        [topic], [TopicMerge(topic.id, "BigQuery", "Platform")]
    )

    assert not get_topic_path("BigQuery", "Platform").exists()
    note = frontmatter.load(get_content_path("Title A", "Blog post"))
    assert get_topic_link("Google BigQuery", "Tool") in note.content


def test_unmerged_topic_is_only_written(session, vault, topic):
    content = link_content(session, topic, "Title A", ["BigQuery"])
    note_before = get_content_path("Title A", "Blog post").read_text()

    write_and_update_topic_list_notes([topic], [])

    assert get_topic_path("BigQuery", "Platform").exists()
    assert get_content_path("Title A", "Blog post").read_text() == note_before
    assert content.id is not None
