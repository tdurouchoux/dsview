"""Delete all rows for a given content_id across the content and extraction schemas."""

import argparse
import logging

from sqlmodel import Session, text

from dsview.db import engine

logger = logging.getLogger(__name__)

# Order matters: children of extractionresult first, then extractionresult,
# then failedingestion, then inputcontent last (everything FKs to it).
# extractiontopic is untouched: topics are shared across contents.
DELETE_STATEMENTS = [
    "DELETE FROM extraction.extractiontag WHERE content_id = :content_id",
    "DELETE FROM extraction.extractionlink WHERE content_id = :content_id",
    "DELETE FROM extraction.contenttopicrelation WHERE content_id = :content_id",
    "DELETE FROM extraction.extractionresult WHERE content_id = :content_id",
    "DELETE FROM content.failedingestion WHERE content_id = :content_id",
    "DELETE FROM content.inputcontent WHERE id = :content_id",
]


def delete_content(content_id: int) -> None:
    with Session(engine) as session:
        for statement in DELETE_STATEMENTS:
            result = session.exec(text(statement), params={"content_id": content_id})
            logger.info("%s -> %d row(s)", statement, result.rowcount)
        session.commit()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("content_id", type=int)
    args = parser.parse_args()

    delete_content(args.content_id)
