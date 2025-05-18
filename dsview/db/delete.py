import logging
from typing import Type

from sqlmodel import Session, SQLModel

logger = logging.getLogger(__name__)


def delete_rows(table: Type[SQLModel], engine, row_ids: list[int]):
    # Implementation of delete_row function

    with Session(engine) as session:
        for row_id in row_ids:
            row = session.get(table, row_id)
            if row:
                session.delete(row)
            else:
                raise ValueError(f"Row with ID {row_id} not found")
        session.commit()
