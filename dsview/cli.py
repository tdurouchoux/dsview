import logging
from datetime import datetime
from pathlib import Path
from typing import Type

import pandas as pd
import typer
from dotenv import load_dotenv
from sqlmodel import Session, SQLModel

from dsview import db
from dsview.config import setup_logger
from dsview.db import schemas
from dsview.db.query import get_content_list, get_failed_ingestions, get_topic_list
from dsview.db.schemas.extraction_schema import ExtractionResult
from dsview.evaluation.cli import evaluate_app
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
    setup_logger()


app.add_typer(evaluate_app, name="evaluate")

BACKUP_TABLES = [
    schemas.InputContent,
    schemas.LabelledContent,
    schemas.TitleLabels,
    schemas.ContentTypeLabels,
    schemas.TagLabels,
    schemas.TopicsLabels,
    schemas.LinksLabels,
    schemas.ERLabels,
]


@app.command(help="Backup database tables to parquet files in the backup directory")
def backup():
    """
    Create backup files for all important database tables.
    Saves data as parquet files in a 'backup' directory.
    """
    backup_path = Path("backup")

    if not backup_path.exists():
        backup_path.mkdir()

    for table in BACKUP_TABLES:
        df = pd.read_sql(
            f"SELECT * FROM {table.__table__}",
            con=db.engine,
        ).drop(columns=["id"])

        df.to_parquet(backup_path / f"{table.__tablename__}.parquet")


@app.command(help="Ingest a single piece of content from a URL or link")
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

    # mlflow.set_tracking_uri("http://localhost:5001")
    # mlflow.set_experiment("Dsview ingest")

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
def retry_failed(
    ignore: list[str] = ["WebRequestFailure"],
):
    """
    Retry ingestion of content that previously failed to process.
    This clears the failed ingestion records and attempts to process them again.
    """
    from .ingest_source import IngestPipeline

    logger.info("Retrying failed ingestions ...")

    with Session(db.engine) as session:
        ingest_pipeline = IngestPipeline(rebuild_mode=True)

        content_list = get_failed_ingestions(session, ignore_errors=ignore)

        logger.info("Found %s failed ingestions", len(content_list))

    schemas.drop_tables([schemas.FailedIngestion], db.engine, reset=True)

    with Session(db.engine) as session:
        logger.info("Starting ingestions ...")
        ingest_pipeline.ingest_content_list(content_list, session)


@app.command(help="Rebuild content processing for a range of content IDs")
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

    # mlflow.set_experiment(experiment_name="Rebuild tasks")

    ingest_pipeline = IngestPipeline(rebuild_mode=True)

    with Session(db.engine) as session:
        content_list = get_content_list(
            session,
            start_id=start,
            end_id=end,
        )

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
        ', '.join([t.__tablename__ for t in tables_to_drop])
    )

    delete_extraction = typer.confirm("Clear extraction results ? ")

    if delete_extraction:
        logger.info("Deleting extraction results")
        schemas.drop_tables(
            tables_to_drop,
            db.engine,
            reset=True
        )

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


def restore_one_table(
    backup_path: Path,
    table: Type[SQLModel],
    session,
):
    """
    Restore a single table from a parquet backup file.

    Args:
        backup_path: Path to the backup directory
        table: SQLModel table class to restore
        session: Database session
    """
    df = pd.read_parquet(backup_path / f"{table.__tablename__}.parquet")

    instances = []
    for _, row in df.iterrows():
        instances.append(table(**row))
    session.add_all(instances)
    session.commit()


@app.command(help="Restore database tables from parquet backup files")
def restore_db(
    backup_dir: str = typer.Option("backup", help="Directory containing backup files"),
):
    """
    Restore database tables from parquet backup files.
    This will restore all tables listed in BACKUP_TABLES from the specified directory.
    """
    backup_path = Path(backup_dir)

    if not backup_path.exists():
        raise MissingBackupDirectory(backup_path)

    with Session(db.engine) as session:
        for table in BACKUP_TABLES:
            restore_one_table(backup_path, table, session)


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
