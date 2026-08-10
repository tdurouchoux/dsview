"""
Audit topological coherence between the database (source of truth) and the
Obsidian vault: for every content/topic row, does the note it implies exist on
disk, does every note on disk trace back to a DB row, and does every
content<->topic relation (extraction.contenttopicrelation) show up as the
matching `![[topics/...]]` embed in the content note?

This intentionally only tracks the shape of the note graph (nodes and the
edges between them) - note text/frontmatter content is out of scope. Path
collisions (two DB rows resolving to the same note path) aren't tracked here
either - that's meant to be prevented by a DB constraint, not caught after the fact.
"""

import dataclasses
import logging
from collections import defaultdict
from pathlib import Path

import regex as re
from rich.table import Table
from sqlmodel import Session, select

from dsview.db.query import get_content_list
from dsview.db.schemas.extraction_schema import (
    ContentTopicRelation,
    ExtractionResult,
    ExtractionTopic,
)

from . import obsidian_utils
from .obsidian_utils import (
    clean_note_title,
    get_content_path,
    get_topic_path,
    retrieve_contents_path,
    retrieve_topics_path,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Discrepancy:
    kind: str  # "content" | "topic" | "relation"
    category: str
    entity: str
    content_id: int | None = None
    topic_id: int | None = None
    path: Path | None = None


def _orphan_discrepancies(
    kind: str, expected_paths: set[Path], actual_paths: set[Path]
) -> list[Discrepancy]:
    return [
        Discrepancy(kind, "orphan_file", f"{path.parent.name}/{path.stem}", path=path)
        for path in sorted(actual_paths - expected_paths)
    ]


def check_content_notes(session: Session) -> tuple[list[Discrepancy], int]:
    discrepancies = []
    expected_paths: set[Path] = set()
    missing_extraction_count = 0

    for content in get_content_list(session):
        extraction = session.get(ExtractionResult, content.id)

        if extraction is None:
            # No note is expected until extraction has run, mirrors regen_vault.
            missing_extraction_count += 1
            continue

        path = get_content_path(extraction.title, extraction.content_type)
        expected_paths.add(path)

        if not path.exists():
            discrepancies.append(
                Discrepancy(
                    "content",
                    "missing_note",
                    extraction.title,
                    content_id=content.id,
                    path=path,
                )
            )

    discrepancies += _orphan_discrepancies(
        "content", expected_paths, set(retrieve_contents_path())
    )

    return discrepancies, missing_extraction_count


def check_topic_notes(session: Session) -> list[Discrepancy]:
    discrepancies = []
    expected_paths: set[Path] = set()

    topics = session.exec(
        select(ExtractionTopic.id, ExtractionTopic.name, ExtractionTopic.type)
    ).all()

    for topic_id, name, type_ in topics:
        path = get_topic_path(name, type_)
        expected_paths.add(path)

        if not path.exists():
            discrepancies.append(
                Discrepancy(
                    "topic",
                    "missing_note",
                    f"{name} ({type_})",
                    topic_id=topic_id,
                    path=path,
                )
            )

    discrepancies += _orphan_discrepancies(
        "topic", expected_paths, set(retrieve_topics_path())
    )

    return discrepancies


# Matches the exact embed format written by get_topic_link(): "![[<topic_dir>/<type>/<name>]]"
def _topic_embed_pattern() -> "re.Pattern":
    topic_dir = re.escape(obsidian_utils.config.topic_directory)
    return re.compile(rf"!\[\[{topic_dir}/([^/\]]+)/([^\]]+)\]\]")


def _read_topic_embeds(path: Path, pattern: "re.Pattern") -> set[tuple[str, str]]:
    return {match.groups() for match in pattern.finditer(path.read_text())}


def _diff_content_relations(
    content_id: int,
    path: Path,
    expected: dict[tuple[str, str], int],
    embed_pattern: "re.Pattern",
) -> list[Discrepancy]:
    if not path.exists():
        return []  # already reported as content/missing_note

    try:
        actual_embeds = _read_topic_embeds(path, embed_pattern)
    except OSError:
        return [
            Discrepancy(
                "relation",
                "unreadable_note",
                path.stem,
                content_id=content_id,
                path=path,
            )
        ]

    discrepancies = []

    for embed_key in expected.keys() - actual_embeds:
        topic_type, topic_name = embed_key
        discrepancies.append(
            Discrepancy(
                "relation",
                "missing_edge",
                f"'{topic_name}' ({topic_type})",
                content_id=content_id,
                topic_id=expected[embed_key],
                path=path,
            )
        )

    for topic_type, topic_name in actual_embeds - expected.keys():
        discrepancies.append(
            Discrepancy(
                "relation",
                "extra_edge",
                f"'{topic_name}' ({topic_type})",
                content_id=content_id,
                path=path,
            )
        )

    return discrepancies


def check_topic_content_relations(session: Session) -> list[Discrepancy]:
    # Outer join so a content with zero DB relations still appears (topic=None) -
    # its note still needs checking for embeds with no relation behind them at all.
    # Selecting individual columns (not the full entities) skips the embedding
    # ARRAY columns, which dominate transfer time for no benefit here.
    rows = session.exec(
        select(
            ExtractionResult.content_id,
            ExtractionResult.title,
            ExtractionResult.content_type,
            ExtractionTopic.id,
            ExtractionTopic.type,
            ExtractionTopic.name,
        )
        .join(
            ContentTopicRelation,
            ContentTopicRelation.content_id == ExtractionResult.content_id,
            isouter=True,
        )
        .join(
            ExtractionTopic,
            ExtractionTopic.id == ContentTopicRelation.topic_id,
            isouter=True,
        )
    ).all()

    # One entry per content: its note path, and its expected embeds keyed to their topic id.
    info_by_content: dict[int, tuple[Path, dict[tuple[str, str], int]]] = {}
    for content_id, title, content_type, topic_id, topic_type, topic_name in rows:
        path = get_content_path(title, content_type)
        _, expected = info_by_content.setdefault(content_id, (path, {}))
        if topic_id is not None:
            expected[(topic_type, clean_note_title(topic_name))] = topic_id

    embed_pattern = _topic_embed_pattern()

    return [
        d
        for content_id, (path, expected) in info_by_content.items()
        for d in _diff_content_relations(content_id, path, expected, embed_pattern)
    ]


def check_vault_sync(session: Session) -> tuple[list[Discrepancy], dict]:
    content_discrepancies, missing_extraction_count = check_content_notes(session)
    topic_discrepancies = check_topic_notes(session)
    relation_discrepancies = check_topic_content_relations(session)

    stats = {"content_missing_extraction": missing_extraction_count}

    return content_discrepancies + topic_discrepancies + relation_discrepancies, stats


CATEGORY_DESCRIPTIONS = {
    "missing_note": "DB row has no matching file in the vault",
    "orphan_file": "file in the vault has no matching DB row (shouldn't exist)",
    "missing_edge": "DB content<->topic relation not reflected as a note embed",
    "extra_edge": "note embeds a topic with no matching DB relation",
    "unreadable_note": "file exists but could not be read",
}

KIND_ORDER = {"content": 0, "topic": 1, "relation": 2}


def render_summary_table(discrepancies: list[Discrepancy]) -> Table:
    table = Table(title="Vault / database topology check")
    table.add_column("Kind", style="bold")
    table.add_column("Category")
    table.add_column("Count", justify="right")
    table.add_column("Description")

    by_group = defaultdict(list)
    for d in discrepancies:
        by_group[(d.kind, d.category)].append(d)

    if not by_group:
        table.add_row("-", "-", "0", "no discrepancies found")
        return table

    for kind, category in sorted(
        by_group, key=lambda kc: (KIND_ORDER.get(kc[0], 9), kc[1])
    ):
        items = by_group[(kind, category)]
        table.add_row(
            kind,
            category,
            str(len(items)),
            CATEGORY_DESCRIPTIONS.get(category, ""),
        )

    return table


def render_category_table(
    discrepancies: list[Discrepancy], kind: str, category: str
) -> Table:
    title = f"{kind} / {category}"
    if description := CATEGORY_DESCRIPTIONS.get(category):
        title += f" - {description}"

    table = Table(title=title)
    table.add_column("Entity")
    table.add_column("Content ID", justify="right")
    table.add_column("Topic ID", justify="right")
    table.add_column("Path")

    for d in discrepancies:
        if d.category == category and (kind is None or d.kind == kind):
            table.add_row(
                d.entity,
                str(d.content_id) if d.content_id is not None else "-",
                str(d.topic_id) if d.topic_id is not None else "-",
                str(d.path) if d.path is not None else "-",
            )

    return table


def format_report(discrepancies: list[Discrepancy], stats: dict) -> str:
    lines = []

    if stats.get("content_missing_extraction"):
        lines.append(
            f"(info) {stats['content_missing_extraction']} content row(s) have no "
            "extraction yet - no note is expected for them, not counted below.\n"
        )

    if not discrepancies:
        lines.append(
            "Vault and database are topologically in sync. No discrepancies found."
        )
        return "\n".join(lines)

    by_group = defaultdict(list)
    for d in discrepancies:
        by_group[(d.kind, d.category)].append(d)

    lines.append(
        f"Found {len(discrepancies)} discrepancies across {len(by_group)} (kind, category) groups:\n"
    )

    for (kind, category), items in sorted(
        by_group.items(), key=lambda kc: (KIND_ORDER.get(kc[0][0], 9), kc[0][1])
    ):
        description = CATEGORY_DESCRIPTIONS.get(category, "")
        lines.append(f"## {kind} / {category} ({len(items)}) - {description}")
        for d in items:
            ids = [
                f"{name}={value}"
                for name, value in (
                    ("content_id", d.content_id),
                    ("topic_id", d.topic_id),
                    ("path", d.path),
                )
                if value is not None
            ]
            id_str = f" [{', '.join(ids)}]" if ids else ""
            lines.append(f"- {d.entity}{id_str}")
        lines.append("")

    return "\n".join(lines)
