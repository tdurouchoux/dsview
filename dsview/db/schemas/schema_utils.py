from typing import Type

from sqlmodel import SQLModel

def drop_tables(
    table_models: list[Type[SQLModel]],
    engine,
):
    SQLModel.metadata.drop_all(engine, tables=[getattr(model, "__table__") for model in table_models])
