import logging
from datetime import datetime

import mlflow
import typer
from dotenv import load_dotenv
from sqlmodel import Session, SQLModel

from dsview.config import setup_logger
from dsview.db.query import get_content_list, get_failed_ingestions
from dsview.db.schemas import (
    ContentTopicRelation,
    ERComparison,
    ERDecision,
    ExtractionLink,
    ExtractionResult,
    ExtractionTag,
    ExtractionTopic,
    FailedIngestion,
    InputContent,
    drop_tables,
    engine,
)

load_dotenv()
setup_logger()

logger = logging.getLogger(__name__)

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
def retry_failed():
    from .ingest_source import IngestPipeline

    with Session(engine) as session:
        ingest_pipeline = IngestPipeline(rebuild_mode=True)

        content_list = get_failed_ingestions(session)
        drop_tables([FailedIngestion], engine)
        SQLModel.metadata.create_all(engine)

        ingest_pipeline.ingest_content_list(content_list, session)


@app.command()
def rebuild(start: int = 0, end: int = None):
    from .ingest_source import IngestPipeline

    mlflow.set_experiment(experiment_name="rebuild")

    ingest_pipeline = IngestPipeline(rebuild_mode=True)

    with Session(engine) as session:
        content_list = get_content_list(
            session,
            start_id=start,
            end_id=end,
        )

        ingest_pipeline.ingest_content_list(content_list, session)


@app.command()
def reset_db():
    delete_input_content = typer.confirm("Clear input content?")

    if delete_input_content:
        logger.info("Deleting input content")
        drop_tables([InputContent], engine)

    delete_extraction = typer.confirm("Clear extraction results ? ")

    if delete_extraction:
        logger.info("Deleting extraction results")
        drop_tables(
            [
                FailedIngestion,
                ContentTopicRelation,
                # ERComparison,
                ERDecision,
                ExtractionLink,
                ExtractionResult,
                ExtractionTag,
                ExtractionTopic,
            ],
            engine,
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
    from dsview.db.schemas import (
        ContentTypeLabels,
        ERLabels,
        LabelledContent,
        LinksLabels,
        TagLabels,
        TitleLabels,
        TopicsLabels,
    )

    delete_content_labelling = typer.confirm("Clear content labelling ?")
    if delete_content_labelling:
        logger.info("Clearing content labels")
        drop_tables(
            [
                LabelledContent,
                TitleLabels,
                ContentTypeLabels,
                TagLabels,
                TopicsLabels,
                LinksLabels,
            ],
            engine,
        )

    delete_er_labelling = typer.confirm("Clear ER labelling ?")
    if delete_er_labelling:
        logger.info("Clearing ER labels")
        drop_tables([ERLabels], engine)


def main():
    app()


if __name__ == "__main__":
    main()
