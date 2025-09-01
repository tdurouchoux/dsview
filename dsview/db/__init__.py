
from sqlmodel import SQLModel, create_engine

from dsview.config import load_postgres_config
from . import schemas

postgres_config = load_postgres_config()
engine = create_engine(postgres_config.db_uri())
SQLModel.metadata.create_all(engine)
