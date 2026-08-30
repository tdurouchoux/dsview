from dataclasses import dataclass
from datetime import date

import pytest

from dsview.notification import digest as digest_module
from dsview.notification.digest import (
    ContentRef,
    Item,
    Stats,
    Topic,
    digest_subject,
    render_digest,
    vault_content_url,
    vault_topic_url,
)


@dataclass
class FakeObsidianConfig:
    vault_base_url: str = "https://example-vault.test"
    content_directory: str = "contents"
    topic_directory: str = "topics"


@dataclass
class FakeNotificationConfig:
    first_issue_week_start: str = "2026-08-24"


@pytest.fixture(autouse=True)
def fake_vault_config(monkeypatch):
    monkeypatch.setattr(digest_module, "obsidian_config", FakeObsidianConfig())
    monkeypatch.setattr(digest_module, "notification_config", FakeNotificationConfig())


def test_render_digest_includes_new_content_and_must_read_items():
    new_content = [
        ContentRef(
            title="A new paper",
            note_url=vault_content_url("A new paper", "Blog post"),
        )
    ]
    must_read = [
        Item(
            content_type="Scientific article",
            title="Attention Is All You Need",
            source_url="https://example.com/attention",
            upload_date=date(2026, 8, 25),
            note_url=vault_content_url(
                "Attention Is All You Need", "Scientific article"
            ),
            meta="Transformer",
            gist="Introduces the Transformer architecture.",
            bullets=[
                "Replaces recurrence with self-attention",
                "State of the art on WMT14",
            ],
        )
    ]
    stats = Stats(contents=1, contents_delta=3, new_topics=0, sources=1, failed=0)

    html = render_digest(
        new_content,
        must_read,
        [],
        date(2026, 8, 24),
        date(2026, 8, 30),
        stats=stats,
    )

    assert "A new paper" in html
    # New contents link to the vault note, not an external source.
    assert f'href="{vault_content_url("A new paper", "Blog post")}"' in html
    assert "Attention Is All You Need" in html
    assert "Introduces the Transformer architecture." in html
    assert "Replaces recurrence with self-attention" in html
    assert "Transformer" in html  # Item.meta (main tag) shown on the card
    assert "Aug 25, 2026" in html  # Item.upload_date shown on the card
    assert "+3" in html  # Stats.contents_delta
    assert "No.&nbsp;1" in html  # week_start == first_issue_week_start


def test_render_digest_includes_revisit_section_with_why():
    revisit = [
        Item(
            content_type="Blog post",
            title="An older post",
            source_url="https://example.com/older",
            why="Ties back to Transformer",
        )
    ]
    stats = Stats(contents=0, new_topics=0, sources=0, failed=0)

    html = render_digest(
        [], [], revisit, date(2026, 8, 24), date(2026, 8, 30), stats=stats
    )

    assert "An older post" in html
    assert "Ties back to Transformer" in html
    assert "Worth revisiting" in html


def test_render_digest_omits_bullets_for_revisit_items():
    revisit = [
        Item(
            content_type="Blog post",
            title="An older post",
            source_url="https://example.com/older",
            why="Ties back to Transformer",
            bullets=["This bullet should not be rendered"],
        )
    ]
    stats = Stats(contents=0, new_topics=0, sources=0, failed=0)

    html = render_digest(
        [], [], revisit, date(2026, 8, 24), date(2026, 8, 30), stats=stats
    )

    assert "This bullet should not be rendered" not in html


def test_render_digest_omits_sections_with_no_items():
    stats = Stats(contents=0, new_topics=0, sources=0, failed=0)

    html = render_digest([], [], [], date(2026, 8, 24), date(2026, 8, 30), stats=stats)

    assert "Must read" not in html
    assert "Worth revisiting" not in html


def test_render_digest_includes_new_topics_section():
    new_topics = [
        Topic(
            name="polars",
            type="Library",
            url=vault_topic_url("polars", "Library"),
        )
    ]
    stats = Stats(contents=1, new_topics=1, sources=1, failed=0)

    html = render_digest(
        [],
        [],
        [],
        date(2026, 8, 24),
        date(2026, 8, 30),
        stats=stats,
        new_topics=new_topics,
    )

    assert "New topics" in html
    assert "polars" in html
    assert f'href="{vault_topic_url("polars", "Library")}"' in html


def test_render_digest_omits_new_topics_section_when_empty():
    stats = Stats(contents=0, new_topics=0, sources=0, failed=0)

    html = render_digest([], [], [], date(2026, 8, 24), date(2026, 8, 30), stats=stats)

    assert "New topics" not in html


def test_render_digest_shows_default_footer_note():
    stats = Stats(contents=0, new_topics=0, sources=0, failed=0)

    html = render_digest([], [], [], date(2026, 8, 24), date(2026, 8, 30), stats=stats)

    assert (
        "DSView weekly digest — generated from the knowledge base, Sunday 06:00."
        in html
    )
    assert "<title>DSView Weekly — Issue 1</title>" in html
    assert ">None<" not in html
    assert "None" not in html


def test_issue_number_counts_sequentially_from_first_issue_week():
    stats = Stats(contents=0, new_topics=0, sources=0, failed=0)

    html = render_digest([], [], [], date(2026, 9, 7), date(2026, 9, 13), stats=stats)

    assert "No.&nbsp;3" in html  # two full weeks after 2026-08-24 -> issue 3


def test_digest_subject_uses_sequential_issue_number():
    assert digest_subject(date(2026, 8, 24)) == "DSView Weekly — Issue 1"
    assert digest_subject(date(2026, 8, 31)) == "DSView Weekly — Issue 2"


def test_vault_url_helpers_slugify_names():
    # Matches Quartz's own slug transform (lowercase, hyphenated) - verified
    # against the actually published vault, e.g. .../contents/scientific-article/
    # and .../topics/library/pytorch.
    assert vault_content_url(
        "Attention Is All You Need", "Scientific article"
    ).endswith("/contents/scientific-article/attention-is-all-you-need")
    assert vault_topic_url("PyTorch", "Library").endswith("/topics/library/pytorch")


def test_vault_content_url_strips_punctuation_like_quartz():
    assert vault_content_url("A: Weird/Title!!", "Blog post").endswith(
        "/contents/blog-post/a-weird-title"
    )
