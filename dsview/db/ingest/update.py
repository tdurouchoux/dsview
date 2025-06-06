from typing import Any, Type

from sqlmodel import Session, SQLModel, select


class UpdateQueryError(Exception):
    def __init__(self, table: Type[SQLModel], query: str, n_results: int):
        super().__init__(
            f"Update query found {n_results} for "
            f"table {table} and query {query}. "
            "Excepted one and only one result."
        )


def update_instance(
    session: Session,
    table: Type[SQLModel],
    row_id: Any = None,
    filter_attributes: dict[str, Any] = None,
    **update_attributes,
):
    if row_id is not None:
        instance = session.get(table, row_id)
        if instance is None:
            raise UpdateQueryError(table, f"id = {row_id}", 0)

    elif filter_attributes is not None:
        stmt = select(table)
        for attr, value in filter_attributes.items():
            stmt = stmt.where(getattr(table, attr) == value)

        results = session.exec(stmt).all()

        if n_results := len(results) != 1:
            raise UpdateQueryError(table, str(filter_attributes), n_results)

        instance = results[0]

    else:
        raise ValueError(
            "At least one of row_id or filter_attributes "
            "should be provided to perform an update."
        )

    for attr, value in update_attributes.items():
        setattr(instance, attr, value)

    session.add(instance)
    session.commit()
    session.refresh(instance)
