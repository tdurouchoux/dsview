"""Render the DSView weekly digest email from `weekly_digest.html.j2`.

Rendering only: this module has no DB dependency. Callers fetch the week's
content/topics from `dsview.db.query` and hand already-built `ContentRef` /
`Item` sequences to `render_digest`.

Field notes
-----------
* `logo_url` must be an absolute https URL (email clients cannot load local files).
* `Item.source_url` is the primary CTA, `Item.note_url` the Quartz vault page.
* `Topic.type` must be one of the extraction config's `topic_categories`
  (Library / Tool / Model / Platform / Concept / Dataset) — it selects the
  colour used in the vault graph. Unknown values fall back to the blue accent.
* Keep the rendered HTML under ~100KB: Gmail clips longer messages. Roughly,
  cap `must_read_topics` at 5, `resurfaced_topics` at 10, and `new_content` at 8-10.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, fields
from datetime import UTC, date, datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from dsview.config import notification_config
from dsview.obsidian.obsidian_utils import config as obsidian_config

TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_NAME = "weekly_digest.html.j2"
LOGO_PATH = "static/icon.png"


@dataclass(slots=True)
class Issue:
    number: int
    period: str  # e.g. "24-30 Aug"


@dataclass(slots=True)
class Stats:
    contents: int
    new_topics: int
    sources: int
    failed: int
    contents_delta: int | None = None
    new_topics_delta: int | None = None
    sources_delta: int | None = None


@dataclass(slots=True)
class Topic:
    name: str
    type: str  # topic_categories value
    url: str


@dataclass(slots=True)
class ContentRef:
    title: str
    note_url: str


@dataclass(slots=True)
class Item:
    """A card in `must_read_topics` or `resurfaced_topics`."""

    content_type: str  # content_types value, e.g. "Scientific article"
    title: str
    source_url: str
    upload_date: date | None = None
    note_url: str | None = None
    meta: str | None = None  # main tag, or "read Feb 2026" for revisits
    gist: str | None = None  # digest one-liner
    bullets: Sequence[str] = ()
    why: str | None = None  # revisit rationale; omit for must-reads


@dataclass(slots=True)
class FooterLink:
    label: str
    url: str


@dataclass(slots=True)
class DigestContext:
    issue: Issue
    stats: Stats
    logo_url: str
    vault_url: str
    vault_contents_url: str
    footer_links: Sequence[FooterLink]
    preheader: str = ""
    recap_line: str | None = None
    new_topics: Sequence[Topic] = ()
    new_contents: Sequence[ContentRef] = ()
    new_contents_remaining: int = 0
    must_read: Sequence[Item] = ()
    revisit: Sequence[Item] = ()
    revisit_subtitle: str | None = None
    subject: str | None = None
    footer_note: str | None = None
    lang: str = "en"


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=False,
    )


def _issue_period(week_start: date, week_end: date) -> str:
    if week_start.month == week_end.month:
        return f"{week_start.day}-{week_end.day} {week_end:%b}"
    return f"{week_start:%d %b}-{week_end:%d %b}"


def _issue_number(week_start: date, first_issue_week_start: date) -> int:
    """Sequential issue number, counting `first_issue_week_start` as Issue #1."""
    return (week_start - first_issue_week_start).days // 7 + 1


def digest_subject(week_start: date) -> str:
    first_issue_week_start = date.fromisoformat(
        notification_config.first_issue_week_start
    )
    return f"DSView Weekly — Issue {_issue_number(week_start, first_issue_week_start)}"


def render_digest(
    new_content: Sequence[ContentRef],
    must_read_topics: Sequence[Item],
    resurfaced_topics: Sequence[Item],
    week_start: date,
    week_end: date,
    *,
    stats: Stats,
    new_topics: Sequence[Topic] = (),
) -> str:
    """Return the full email HTML for the week `[week_start, week_end]`."""
    vault_base_url = obsidian_config.vault_base_url
    first_issue_week_start = date.fromisoformat(
        notification_config.first_issue_week_start
    )

    context = DigestContext(
        issue=Issue(
            number=_issue_number(week_start, first_issue_week_start),
            period=_issue_period(week_start, week_end),
        ),
        stats=stats,
        logo_url=f"{vault_base_url}/{LOGO_PATH}",
        vault_url=f"{vault_base_url}/",
        vault_contents_url=f"{vault_base_url}/{obsidian_config.content_directory}/",
        footer_links=[
            FooterLink("Vault", f"{vault_base_url}/"),
            FooterLink("Repo", "https://github.com/tdurouchoux/dsview"),
        ],
        preheader=(
            f"{len(new_content)} new contents this week, "
            f"{len(must_read_topics)} must-reads."
        ),
        new_contents=new_content,
        new_topics=new_topics,
        must_read=must_read_topics,
        revisit=resurfaced_topics,
        footer_note=(
            "DSView weekly digest — generated from the knowledge base, "
            f"{datetime.now(UTC):%b %d, %Y %H:%M} UTC."
        ),
    )

    template = _environment().get_template(TEMPLATE_NAME)
    # `context` is a slots dataclass, so it has no __dict__ for vars() to read.
    render_kwargs = {f.name: getattr(context, f.name) for f in fields(context)}
    return template.render(**render_kwargs)


# --- URL helpers -------------------------------------------------------------
# Quartz slugifies every path segment for the published site (lowercase, runs of
# non-word characters collapsed to a single hyphen) - this is Quartz's own slug
# transform, distinct from dsview.obsidian.obsidian_utils.clean_note_title (which
# only sanitizes the *local* note filename and keeps case/spaces).


def _quartz_slug(value: str) -> str:
    return re.sub(r"[^\w]+", "-", value.lower()).strip("-")


def vault_content_url(
    title: str, content_type: str, base_url: str | None = None
) -> str:
    base_url = base_url or obsidian_config.vault_base_url
    return (
        f"{base_url}/{obsidian_config.content_directory}"
        f"/{_quartz_slug(content_type)}/{_quartz_slug(title)}"
    )


def vault_topic_url(name: str, topic_type: str, base_url: str | None = None) -> str:
    base_url = base_url or obsidian_config.vault_base_url
    return (
        f"{base_url}/{obsidian_config.topic_directory}"
        f"/{_quartz_slug(topic_type)}/{_quartz_slug(name)}"
    )
