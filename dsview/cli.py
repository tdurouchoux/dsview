import logging
import os
import shutil
from datetime import datetime, date, timezone
from pathlib import Path

import logfire
import typer
from dotenv import load_dotenv
from sqlmodel import Session

from dsview import db
from dsview.config import load_postgres_config, setup_logger
from dsview.db import schemas
from dsview.db.query import get_content_list, get_failed_ingestions, get_topic_list
from dsview.db.schemas.extraction_schema import ExtractionResult
from dsview.evaluation.cli import evaluate_app
from dsview.obsidian.sync_vault import run_cmd
from dsview.obsidian.write_notes import (
    write_content_note,
    write_topic_note,
)

load_dotenv()

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="DSView CLI - A tool for content ingestion, processing, and knowledge base management"
)


@app.callback()
def cli_setup():
    # Runs before any command (but not for --help), so a bare `dsview --help`
    # works without CONF_DIR or database environment variables.
    setup_logger(enable_logfire=True, service_name="dsview_cli")


app.add_typer(evaluate_app, name="evaluate")


class MissingPostgresTool(Exception):
    """Exception raised when pg_dump/pg_restore is not installed."""

    def __init__(self, tool: str) -> None:
        super().__init__(
            f"'{tool}' not found on PATH. Install the postgresql-client package."
        )


def _require_pg_tool(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise MissingPostgresTool(tool)
    return path


def _pg_connection_args() -> tuple[list[str], dict]:
    pg_config = load_postgres_config()
    args = [
        "-h",
        pg_config.host,
        "-p",
        str(pg_config.port),
        "-U",
        pg_config.user,
        "-d",
        pg_config.database,
    ]
    env = {**os.environ, "PGPASSWORD": pg_config.password}
    return args, env


@app.command(help="Backup the whole database to a pg_dump file in the backup directory")
def backup():
    """
    Dump the entire database (all schemas/tables) with pg_dump so the backup
    is exhaustive by construction and restore preserves ids/foreign keys.
    """
    pg_dump = _require_pg_tool("pg_dump")

    backup_path = Path("backup")
    backup_path.mkdir(exist_ok=True)

    output_file = backup_path / f"dsview_{datetime.now():%Y%m%d_%H%M%S}.dump"
    connection_args, env = _pg_connection_args()

    run_cmd(
        [pg_dump, "-Fc", "--no-owner", "--no-privileges"]
        + connection_args
        + ["-f", str(output_file)],
        "pg_dump backup",
        env=env,
    )

    logger.info("Backup written to %s", output_file)


@app.command(help="Ingest a single piece of content from a URL or link")
@logfire.instrument("Manual ingestion")
def ingest(
    link: str = typer.Argument(..., help="URL or link to the content to ingest"),
    already_read: bool = typer.Option(False, help="Mark content as already read"),
    upload_date: datetime = typer.Option(
        datetime.now(), help="Date when content was uploaded"
    ),
    read_priority: int = typer.Option(
        0, help="Reading priority (higher numbers = higher priority)"
    ),
    relevance: int = typer.Option(0, help="Content relevance score"),
    source: str = typer.Option(None, help="Source identifier for the content"),
):
    """
    Ingest a single piece of content into the system.
    This will download, process, and extract information from the provided link.
    """
    from .ingest_source import IngestPipeline

    ingest_pipeline = IngestPipeline()

    content = schemas.InputContent(
        link=link,
        upload_date=upload_date.date(),
        already_read=already_read,
        read_priority=read_priority,
        relevance=relevance,
        source=source,
    )
    with Session(db.engine) as session:
        ingest_pipeline.ingest_content(content, session)


@app.command(help="Retry processing of previously failed ingestions")
@logfire.instrument("Retrying failed ingestions")
def retry_failed(
    ignore: list[str] | None = None,
):
    """
    Retry ingestion of content that previously failed to process.
    This clears the failed ingestion records and attempts to process them again.
    """
    from .ingest_source import IngestPipeline

    logger.info("Retrying failed ingestions excluding : %s", ",".join(ignore))

    if ignore is None:
        ignore = ["WebRequestFailure"]

    with Session(db.engine) as session:
        ingest_pipeline = IngestPipeline(rebuild_mode=True)

        content_list = get_failed_ingestions(session, ignore_errors=ignore)

        n_failed_content = len(content_list)
        logfire.info(f"Found {n_failed_content} failed ingestions")

    schemas.drop_tables([schemas.FailedIngestion], db.engine, reset=True)

    with Session(db.engine) as session:
        ingest_pipeline.ingest_content_list(content_list, session)


@app.command(help="Rebuild content processing for a range of content IDs")
@logfire.instrument("Full knowledge base rebuild")
def rebuild(
    start: int = typer.Option(0, help="Starting content ID (inclusive)"),
    end: int = typer.Option(
        None, help="Ending content ID (inclusive, None for all remaining)"
    ),
):
    """
    Rebuild content processing for a specified range of content.
    This will reprocess existing content using the current extraction pipeline.
    """
    from .ingest_source import IngestPipeline

    ingest_pipeline = IngestPipeline(rebuild_mode=True)

    with Session(db.engine) as session:
        content_list = get_content_list(
            session,
            start_id=start,
            end_id=end,
        )

        n_content = len(content_list)
        logfire.info(f"Rebuilding extraction for {n_content} content")

        ingest_pipeline.ingest_content_list(content_list, session)


def regen_content_notes(content_list: list[schemas.InputContent], session: Session):
    """
    Regenerate Obsidian notes for a list of content items.

    Args:
        content_list: List of InputContent items to process
        session: Database session

    Returns:
        int: Number of content items with missing extractions
    """
    missing_extraction_count = 0

    for content in content_list:
        extraction = session.get(ExtractionResult, content.id)

        if extraction is None:
            missing_extraction_count += 1
            continue
        write_content_note(content, extraction)

    return missing_extraction_count


@app.command(help="Regenerate all Obsidian vault content notes and topic pages")
def regen_vault():
    """
    Regenerate the entire Obsidian vault from the database.
    This creates fresh notes for all content and topic pages.
    """
    with Session(db.engine) as session:
        content_list = get_content_list(session)
        missing_extraction_count = regen_content_notes(content_list, session)

        logger.info(
            "Missing %s extraction out of %s input content",
            missing_extraction_count,
            len(content_list),
        )

        topic_list = get_topic_list(session)

        for topic in topic_list:
            write_topic_note(topic)


@app.command(
    help="Check coherence between the database and the Obsidian vault (read-only)"
)
def check_vault_sync(
    details: str = typer.Option(
        None,
        help="Print every discrepancy for a single category "
        "(missing_note, orphan_file, missing_edge, extra_edge, unreadable_note)",
    ),
    kind: str = typer.Option(
        None,
        help="Restrict --details to one kind of node (content, topic, relation)",
    ),
    output: Path = typer.Option(
        None,
        help="Optional path to write the full itemized report to (not just the summary)",
    ),
):
    """
    Compare every content/topic note the database implies with what is actually on
    disk in the vault. The database is always assumed correct; this only reports
    discrepancies, it never modifies the database or the vault.
    """
    from rich.console import Console

    from dsview.obsidian.check_vault_sync import check_vault_sync as run_check
    from dsview.obsidian.check_vault_sync import (
        format_report,
        render_category_table,
        render_summary_table,
    )

    with Session(db.engine) as session:
        discrepancies, stats = run_check(session)

    console = Console()

    if stats.get("content_missing_extraction"):
        console.print(
            f"[dim]{stats['content_missing_extraction']} content row(s) have no "
            "extraction yet - no note is expected for them, not counted below.[/dim]"
        )

    console.print(render_summary_table(discrepancies))

    if details:
        matching = [
            d
            for d in discrepancies
            if d.category == details and (not kind or d.kind == kind)
        ]
        if not matching:
            console.print(f"No discrepancies found in category '{details}'.")
        else:
            console.print(render_category_table(discrepancies, kind, details))
    elif discrepancies:
        console.print(
            "[dim]Use --details <category> (optionally with --kind) to list "
            "individual discrepancies.[/dim]"
        )

    if output:
        output.write_text(format_report(discrepancies, stats))
        console.print(f"Full itemized report written to {output}")

    if discrepancies:
        raise typer.Exit(code=1)


@app.command(help="Reset database tables (with confirmation prompts)")
def reset_db():
    """
    Reset database tables by dropping extraction results and related data.
    User will be prompted to confirm deletion of extraction results.
    """

    tables_to_drop = [
        schemas.ExtractionResult,
        schemas.FailedIngestion,
        schemas.ExtractionTopic,
        schemas.ContentTopicRelation,
        schemas.ERComparison,
        schemas.ExtractionLink,
        schemas.ExtractionTag,
    ]

    logger.info(
        "This command will delete the following tables : %s",
        ", ".join([t.__tablename__ for t in tables_to_drop]),
    )

    delete_extraction = typer.confirm("Clear extraction results ? ")

    if delete_extraction:
        logger.info("Deleting extraction results")
        schemas.drop_tables(tables_to_drop, db.engine, reset=True)


@app.command(help="Clear the entire Obsidian vault (DESTRUCTIVE)")
def reset_vault():
    """
    Clear the entire Obsidian vault directory.
    WARNING: This is a destructive operation that cannot be undone.
    """
    from dsview.obsidian.obsidian_utils import clear_vault

    delete = typer.confirm(
        "This action will clear the entire obsidian vault. "
        "Are you sure you want to reset the vault ?"
    )
    if not delete:
        logger.info("Aborting reset")
        raise typer.Abort()

    logger.info("Launching reset")
    clear_vault()
    logger.info("Reset completed")


@app.command(help="Reset labelling data (content and extraction result labels)")
def reset_labelling():
    """
    Reset labelling data with separate confirmations for content and ER labelling.
    This allows selective clearing of different types of label data.
    """

    delete_content_labelling = typer.confirm("Clear content labelling ?")
    if delete_content_labelling:
        logger.info("Clearing content labels")
        schemas.drop_tables(
            [
                schemas.LabelledContent,
                schemas.TitleLabels,
                schemas.ContentTypeLabels,
                schemas.TagLabels,
                schemas.TopicsLabels,
                schemas.LinksLabels,
            ],
            db.engine,
        )

    delete_er_labelling = typer.confirm("Clear ER labelling ?")
    if delete_er_labelling:
        logger.info("Clearing ER labels")
        schemas.drop_tables([schemas.ERLabels], db.engine)


class MissingBackupDirectory(Exception):
    """Exception raised when backup directory is not found."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"No backup directory found at {path}")


class NoBackupFileFound(Exception):
    """Exception raised when no dump file is found in the backup directory."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"No '*.dump' file found in {path}")


def _latest_backup_file(backup_path: Path) -> Path:
    dump_files = sorted(backup_path.glob("*.dump"), key=lambda f: f.stat().st_mtime)
    if not dump_files:
        raise NoBackupFileFound(backup_path)
    return dump_files[-1]


@app.command(help="Restore the database from a pg_dump backup file (DESTRUCTIVE)")
def restore_db(
    backup_file: str = typer.Option(
        None,
        help="Dump file to restore. Defaults to the most recent '*.dump' in --backup-dir",
    ),
    backup_dir: str = typer.Option("backup", help="Directory containing backup files"),
):
    """
    Restore the database from a pg_dump backup file.
    WARNING: this drops existing objects (pg_restore --clean) before recreating them.
    """
    backup_path = Path(backup_dir)
    if not backup_path.exists():
        raise MissingBackupDirectory(backup_path)

    dump_file = Path(backup_file) if backup_file else _latest_backup_file(backup_path)

    delete = typer.confirm(
        f"This will drop and recreate existing database objects from {dump_file}. "
        "Are you sure you want to restore ?"
    )
    if not delete:
        logger.info("Aborting restore")
        raise typer.Abort()

    pg_restore = _require_pg_tool("pg_restore")
    connection_args, env = _pg_connection_args()

    run_cmd(
        [pg_restore, "--clean", "--if-exists", "--no-owner", "--no-privileges"]
        + connection_args
        + [str(dump_file)],
        "pg_restore restore",
        env=env,
    )

    logger.info("Database restored from %s", dump_file)


@app.command(help="Export extraction results as a graph")
def export_graph(
    output_file: str = typer.Option("dsview_graph.graphml", help="Output file path"),
):
    from dsview.graph.build_graph import build_graph

    with Session(db.engine) as session:
        graph = build_graph(session)

    graph.write_graphml(output_file)

    logger.info(f"Graph exported to {output_file}")


def main():
    app()


if __name__ == "__main__":
    main()
