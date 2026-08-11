from datetime import date, datetime

from sqlalchemy import text
from sqlmodel import Session

from dsview.db.schemas import InputContent
from dsview.db.schemas.extraction_schema import ContentTopicRelation
from dsview.obsidian.check_vault_sync import (
    check_content_notes,
    check_topic_content_relations,
    check_topic_notes,
    check_vault_sync,
    format_report,
    render_category_table,
    render_summary_table,
)
from dsview.obsidian.obsidian_utils import (
    get_content_path,
    get_topic_link,
    get_topic_path,
)


def _add_content(session: Session, link: str) -> InputContent:
    content = InputContent(link=link, upload_date=date.today())
    session.add(content)
    session.commit()
    session.refresh(content)
    return content


def _add_extraction(session: Session, content_id: int, title: str, content_type: str):
    session.exec(
        text(
            "INSERT INTO extraction.extractionresult "
            "(content_id, title, content_type, summary, embedding, extraction_time) "
            "VALUES (:content_id, :title, :content_type, 'summary', NULL, :extraction_time)"
        ),
        params={
            "content_id": content_id,
            "title": title,
            "content_type": content_type,
            "extraction_time": datetime.now().isoformat(),
        },
    )
    session.commit()


def _add_topic(
    session: Session, name: str, type_: str, description: str = "desc"
) -> int:
    session.exec(
        text(
            "INSERT INTO extraction.extractiontopic (type, name, description, embedding) "
            "VALUES (:type, :name, :description, NULL)"
        ),
        params={"type": type_, "name": name, "description": description},
    )
    session.commit()
    return session.exec(
        text("SELECT id FROM extraction.extractiontopic WHERE name = :name"),
        params={"name": name},
    ).one()[0]


def _add_relation(session: Session, content_id: int, topic_id: int):
    session.add(ContentTopicRelation(content_id=content_id, topic_id=topic_id))
    session.commit()


def _write_note(path, body: str = "note body"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


# --- check_content_notes -----------------------------------------------------


def test_content_note_present_produces_no_discrepancy(session, vault):
    content = _add_content(session, "https://example.com/a")
    _add_extraction(session, content.id, "Title A", "Blog post")
    _write_note(get_content_path("Title A", "Blog post"))

    discrepancies, missing_extraction_count = check_content_notes(session)

    assert discrepancies == []
    assert missing_extraction_count == 0


def test_content_without_note_is_missing(session, vault):
    content = _add_content(session, "https://example.com/a")
    _add_extraction(session, content.id, "Title A", "Blog post")

    discrepancies, _ = check_content_notes(session)

    assert len(discrepancies) == 1
    d = discrepancies[0]
    assert d.kind == "content"
    assert d.category == "missing_note"
    assert d.content_id == content.id
    assert d.entity == "Title A"
    assert d.path == get_content_path("Title A", "Blog post")


def test_content_without_extraction_is_skipped_not_flagged(session, vault):
    _add_content(session, "https://example.com/a")

    discrepancies, missing_extraction_count = check_content_notes(session)

    assert discrepancies == []
    assert missing_extraction_count == 1


# --- check_topic_notes --------------------------------------------------------


def test_topic_note_present_produces_no_discrepancy(session, vault):
    _add_topic(session, "LLM", "Concept")
    _write_note(get_topic_path("LLM", "Concept"))

    discrepancies = check_topic_notes(session)

    assert discrepancies == []


def test_topic_without_note_is_missing(session, vault):
    topic_id = _add_topic(session, "LLM", "Concept")

    discrepancies = check_topic_notes(session)

    assert len(discrepancies) == 1
    assert discrepancies[0].kind == "topic"
    assert discrepancies[0].category == "missing_note"
    assert discrepancies[0].topic_id == topic_id
    assert discrepancies[0].entity == "LLM (Concept)"
    assert discrepancies[0].path == get_topic_path("LLM", "Concept")


def test_orphan_topic_file_is_flagged(session, vault):
    # Simulates a topic renamed/merged in the DB whose old note was never removed.
    _write_note(get_topic_path("Old Name", "Concept"))

    discrepancies = check_topic_notes(session)

    assert len(discrepancies) == 1
    assert discrepancies[0].kind == "topic"
    assert discrepancies[0].category == "orphan_file"
    assert discrepancies[0].content_id is None
    assert discrepancies[0].topic_id is None
    assert discrepancies[0].entity == "Concept/Old Name"
    assert discrepancies[0].path == get_topic_path("Old Name", "Concept")


# --- check_topic_content_relations -------------------------------------------


def test_matching_relation_and_embed_produce_no_discrepancy(session, vault):
    content = _add_content(session, "https://example.com/a")
    _add_extraction(session, content.id, "Title A", "Blog post")
    topic_id = _add_topic(session, "LLM", "Concept")
    _add_relation(session, content.id, topic_id)
    _write_note(
        get_content_path("Title A", "Blog post"),
        body=f"body\n\n## Topics\n\n{get_topic_link('LLM', 'Concept')}\n\n",
    )

    discrepancies = check_topic_content_relations(session)

    assert discrepancies == []


def test_relation_without_embed_is_missing_edge(session, vault):
    content = _add_content(session, "https://example.com/a")
    _add_extraction(session, content.id, "Title A", "Blog post")
    topic_id = _add_topic(session, "LLM", "Concept")
    _add_relation(session, content.id, topic_id)
    _write_note(
        get_content_path("Title A", "Blog post"), body="body with no topic embeds"
    )

    discrepancies = check_topic_content_relations(session)

    assert len(discrepancies) == 1
    assert discrepancies[0].kind == "relation"
    assert discrepancies[0].category == "missing_edge"
    assert discrepancies[0].entity == "'LLM' (Concept)"
    assert discrepancies[0].content_id == content.id
    assert discrepancies[0].topic_id == topic_id
    assert discrepancies[0].path == get_content_path("Title A", "Blog post")


def test_embed_without_relation_is_extra_edge(session, vault):
    content = _add_content(session, "https://example.com/a")
    _add_extraction(session, content.id, "Title A", "Blog post")
    _add_topic(session, "LLM", "Concept")
    # Topic exists in the DB, but no relation row: the note still embeds it anyway.
    _write_note(
        get_content_path("Title A", "Blog post"),
        body=f"body\n\n## Topics\n\n{get_topic_link('LLM', 'Concept')}\n\n",
    )

    discrepancies = check_topic_content_relations(session)

    assert len(discrepancies) == 1
    assert discrepancies[0].category == "extra_edge"
    assert discrepancies[0].entity == "'LLM' (Concept)"
    assert discrepancies[0].content_id == content.id
    # extra_edge deliberately doesn't resolve against the topics table (see prior review).
    assert discrepancies[0].topic_id is None


def test_content_with_zero_relations_reports_extra_edge_for_stray_embed(session, vault):
    # Regression guard for the outer join: a content with no contenttopicrelation
    # rows at all must still be checked, not silently dropped from the query.
    # The embedded topic doesn't even exist in the DB (e.g. merged away) - extra_edge
    # reporting must not depend on resolving it against the topics table.
    content = _add_content(session, "https://example.com/a")
    _add_extraction(session, content.id, "Title A", "Blog post")
    _write_note(
        get_content_path("Title A", "Blog post"),
        body="body\n\n## Topics\n\n![[topics/Concept/Gone Topic]]\n\n",
    )

    discrepancies = check_topic_content_relations(session)

    assert len(discrepancies) == 1
    assert discrepancies[0].category == "extra_edge"
    assert discrepancies[0].entity == "'Gone Topic' (Concept)"
    assert discrepancies[0].content_id == content.id
    assert discrepancies[0].topic_id is None


def test_relation_skipped_when_content_note_missing(session, vault):
    content = _add_content(session, "https://example.com/a")
    _add_extraction(session, content.id, "Title A", "Blog post")
    topic_id = _add_topic(session, "LLM", "Concept")
    _add_relation(session, content.id, topic_id)
    # No note written for the content at all.

    discrepancies = check_topic_content_relations(session)

    # Already reported as content/missing_note; must not also surface as a relation issue.
    assert discrepancies == []


# --- check_vault_sync (integration) ------------------------------------------


def test_check_vault_sync_aggregates_all_discrepancy_kinds(session, vault):
    content = _add_content(session, "https://example.com/a")
    _add_extraction(session, content.id, "Title A", "Blog post")
    # Content note left unwritten -> content/missing_note.

    _add_content(session, "https://example.com/b")
    # No extraction -> counted in stats, not a discrepancy.

    _write_note(get_topic_path("Orphan", "Concept"))
    # No matching DB topic -> topic/orphan_file.

    discrepancies, stats = check_vault_sync(session)

    assert stats == {"content_missing_extraction": 1}
    categories = {(d.kind, d.category) for d in discrepancies}
    assert ("content", "missing_note") in categories
    assert ("topic", "orphan_file") in categories


def test_check_vault_sync_clean_state_has_no_discrepancies(session, vault):
    content = _add_content(session, "https://example.com/a")
    _add_extraction(session, content.id, "Title A", "Blog post")
    topic_id = _add_topic(session, "LLM", "Concept")
    _add_relation(session, content.id, topic_id)
    _write_note(get_topic_path("LLM", "Concept"))
    _write_note(
        get_content_path("Title A", "Blog post"),
        body=f"body\n\n## Topics\n\n{get_topic_link('LLM', 'Concept')}\n\n",
    )

    discrepancies, stats = check_vault_sync(session)

    assert discrepancies == []
    assert stats == {"content_missing_extraction": 0}


# --- rendering / formatting ---------------------------------------------------


def test_render_summary_table_empty():
    table = render_summary_table([])

    assert table.row_count == 1  # the "no discrepancies found" placeholder row


def test_render_category_table_filters_by_kind_and_category(session, vault):
    content = _add_content(session, "https://example.com/a")
    _add_extraction(session, content.id, "Title A", "Blog post")
    discrepancies, _ = check_content_notes(session)

    table = render_category_table(discrepancies, "content", "missing_note")

    assert table.row_count == 1


def test_format_report_no_discrepancies():
    report = format_report([], {"content_missing_extraction": 0})

    assert "fully" in report.lower() or "no discrepancies" in report.lower()


def test_format_report_lists_each_group(session, vault):
    _write_note(get_topic_path("Orphan", "Concept"))
    discrepancies, stats = check_vault_sync(session)

    report = format_report(discrepancies, stats)

    assert "topic / orphan_file (1)" in report
    assert "Orphan" in report
