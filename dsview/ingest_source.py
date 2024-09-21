from datetime import datetime
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
import frontmatter
from langchain_openai import ChatOpenAI
from rich.progress import track
import typer

from dsview.content.content_loader import (
    get_content_loader,
    WebRequestFailure,
)
from dsview.content.input_content import InputContent
from dsview.extraction.content_extraction import ContentExtractor
from dsview.extraction.entity_resolution import ERSolver
from dsview.config import load_model_config, setup_logger
from dsview.obsidian.obsidian_utils import clear_vault, retrieve_contents_path
from dsview.obsidian.notes_generator import NotesGenerator

load_dotenv()
setup_logger()
logger = logging.getLogger(__name__)
model_config = load_model_config()

app = typer.Typer()


OUTPUT_DIRECTORY = Path("data/output")

# TODO Follow failed request url

def ingest_single_content(
    content: InputContent, content_extractor: ContentExtractor, er_solver: ERSolver
):
    content_loader = get_content_loader(content.link, model_config.token_limit)

    summary, content_description, topics, content_links = (
        content_extractor.extract_content(content_loader)
    )

    topics = er_solver.run_topics_er(topics)

    notes_generator = NotesGenerator(
        content,
        content_loader.get_hyperlink(),
        summary,
        content_description,
        topics,
        content_links,
    )

    notes_generator.generate_topics_md()
    notes_generator.generate_content_md()
    logger.info("New content added")


@app.command()
def ingest(
    link: str,
    already_read: bool = False,
    upload_date: datetime = datetime.now(),
):
    llm = ChatOpenAI(temperature=0, model_name=model_config.name)
    content_extractor = ContentExtractor(llm)
    er_solver = ERSolver(llm)

    content = InputContent(
        link=link, upload_date=upload_date.date(), already_read=already_read
    )

    logger.info("Received new content %s to ingest in knowledge base", content.link)

    ingest_single_content(content, content_extractor, er_solver)


@app.command()
def batch_ingest(content_list: Path, start: int = 0):
    if content_list.suffix != ".json":
        raise ValueError("Batch ingest only accepts json files.")

    logger.info("Reading content json file")

    with open(content_list, "r") as json_file:
        content_dict_list = json.load(json_file)

    logger.info("Received %s content_links to ingest", len(content_dict_list) - start)

    llm = ChatOpenAI(temperature=0, model_name=model_config.name)
    content_extractor = ContentExtractor(llm)
    er_solver = ERSolver(llm)

    failed_content = []

    i = 0
    for content_dict in track(content_dict_list[start:]):
        logger.info("Starting ingestion of file at index %s", i)

        content = InputContent(**content_dict)

        try:
            ingest_single_content(content, content_extractor, er_solver)
        except WebRequestFailure as web_error:
            logger.exception(
                "Failed to load content %s with error %s", content.link, web_error
            )

            failed_content.append(content.get_str_dict())
            continue
        i += 1

    if len(failed_content) > 0:
        logger.info(
            "%s content link failed ingestion, saving to failed_content.json",
            len(failed_content),
        )

        with open(OUTPUT_DIRECTORY / "failed_content.json", "w") as json_file:
            json_file.write(json.dumps(failed_content, indent=4))


# TODO Generate csv output with same format as input
# TODO keep this csv list somewhere safe  s


@app.command()
def save():
    logger.info("Starting dsview snapshot creation.")
    content_notes_path = retrieve_contents_path()

    content_dict_list = []

    for note_path in content_notes_path:
        note = frontmatter.load(note_path)
        note_dict = dict(note)
        content_dict_list.append(
            {
                key: value
                for key, value in note_dict.items()
                if key not in ["type", "tags"] and value is not None
            }
        )

    logger.info("Writing input content to %s", "snapshot_content.json")
    with open(OUTPUT_DIRECTORY / "snapshot_content.json", "w") as json_file:
        json_file.write(json.dumps(content_dict_list, indent=4))

    # use pydantic csv model for frontmatter attribute selection

    # retrieve_all_content_notes
    pass


@app.command()
def reset():
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


def main():
    app()


if __name__ == "__main__":
    main()
