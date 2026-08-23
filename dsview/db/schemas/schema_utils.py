
from sqlmodel import SQLModel


def drop_tables(
    table_models: list[type[SQLModel]],
    engine,
    reset: bool = False,
):
    table_list = [model.__table__ for model in table_models]
    SQLModel.metadata.drop_all(engine, tables=table_list)

    if reset:
        SQLModel.metadata.create_all(engine, tables=table_list)
