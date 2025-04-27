import logging
from datetime import date, datetime

import frontmatter
import mlflow
import typer
from dotenv import load_dotenv
from rich.progress import track
from sqlmodel import Session, SQLModel, create_engine, select

from dsview.config import get_sqlite_url, setup_logger
from dsview.content.content_db_schema import InputContent
from dsview.extraction.extraction_db_schema import (
    ERComparison,
    ERDecision,
    ExtractionResult,
    InvalidTag,
    InvalidTopic,
)

from .ingest_source import FailedIngestion

load_dotenv()
setup_logger()

logger = logging.getLogger(__name__)

engine = create_engine(get_sqlite_url())
SQLModel.metadata.create_all(engine)

app = typer.Typer()

# TODO Implement evaluation cli
# app.add_typer(evaluate_app, name="evaluate")


@app.command()
def ingest(
    link: str,
    already_read: bool = False,
    upload_date: datetime = datetime.now(),
    read_priority: int = 0,
    relevance: int = 0,
    source: str = None,
):
    from .ingest_source import IngestPipeline

    ingest_pipeline = IngestPipeline()

    content = InputContent(
        link=link,
        upload_date=upload_date.date(),
        already_read=already_read,
        read_priority=read_priority,
        relevance=relevance,
        source=source,
    )
    with Session(engine) as session:
        ingest_pipeline.ingest_content(content, session)


@app.command()
def save():
    from dsview.obsidian.obsidian_utils import retrieve_contents_path

    with Session(engine) as session:
        logger.info("Starting dsview snapshot creation.")
        content_notes_path = retrieve_contents_path()

        for note_path in track(content_notes_path):
            note = frontmatter.load(note_path)
            note_dict = dict(note)

            upload_date = (
                date.fromisoformat(note_dict["upload_date"])
                if isinstance(note_dict["upload_date"], str)
                else note_dict["upload_date"]
            )

            relevance = note_dict["relevance"] if "relevance" in note_dict else 0

            source = (
                note_dict["source"]
                if "source" in note_dict
                and note_dict["source"] not in ["None", "Aucune"]
                else None
            )

            input_content = InputContent(
                link=note_dict["link"],
                upload_date=upload_date,
                already_read=note_dict["already_read"],
                read_priority=note_dict["read_priority"],
                relevance=relevance,
                source=source,
            )

            session.add(input_content)
            session.commit()


@app.command()
def retry_failed():
    from .ingest_source import FailedIngestion, IngestPipeline

    with Session(engine) as session:
        ingest_pipeline = IngestPipeline(rebuild_mode=True)

        failed_ingestions = session.exec(select(FailedIngestion)).all()
        SQLModel.metadata.drop_all(engine, tables=[FailedIngestion.__table__])

        for failed_ingestion in failed_ingestions:
            statement = select(InputContent).where(
                InputContent.id == failed_ingestion.content_id
            )
            content = session.exec(statement).first()

            ingest_pipeline.ingest_content(content, session)


@app.command()
def rebuild(start: int = 0, end: int = None):
    from .ingest_source import IngestPipeline

    mlflow.set_experiment(experiment_name="rebuild")

    ingest_pipeline = IngestPipeline(rebuild_mode=True)

    with Session(engine) as session:
        statement = select(InputContent).order_by(InputContent.upload_date.asc())
        content_list = session.exec(statement).all()
        if end is not None:
            content_list = content_list[start:end]
        else:
            content_list = content_list[start:]

        ingest_pipeline.ingest_content_list(content_list, session)


@app.command()
def reset_db():
    delete_input_content = typer.confirm("Clear input content?")

    if delete_input_content:
        logger.info("Deleting input content")
        SQLModel.metadata.drop_all(engine, tables=[InputContent.__table__])

    delete_extraction = typer.confirm("Clear extraction results ? ")

    if delete_extraction:
        logger.info("Deleting extraction results")
        SQLModel.metadata.drop_all(
            engine,
            tables=[
                FailedIngestion.__table__,
                ERComparison.__table__,
                ERDecision.__table__,
                InvalidTag.__table__,
                InvalidTopic.__table__,
                ExtractionResult.__table__,
            ],
        )


@app.command()
def reset_vault():
    from dsview.obsidian.obsidian_utils import clear_vault

    delete = typer.confirm(
        "This action will clear the entire obsidian vault. "
        "Are you sur you want to reset the vault ?"
    )
    if not delete:
        logger.info("Aborting reset")
        raise typer.Abort()

    logger.info("Launching reset")
    clear_vault()
    logger.info("Reset completed")


@app.command()
def reset_labelling():
    from dsview.evaluation.labels_schema import (
        clear_content_labelling,
        clear_er_labelling,
    )

    delete_content_labelling = typer.confirm("Clear content labelling ?")
    if delete_content_labelling:
        logger.info("Clearing content labels")
        clear_content_labelling(engine)

    delete_er_labelling = typer.confirm("Clear ER labelling ?")
    if delete_er_labelling:
        logger.info("Clearing ER labels")
        clear_er_labelling(engine)


def main():
    app()


if __name__ == "__main__":
    main()
