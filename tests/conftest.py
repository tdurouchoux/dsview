import pytest
from sqlalchemy import ARRAY, event
from sqlalchemy.ext.compiler import compiles
from sqlmodel import Session, SQLModel, create_engine

import dsview.db.schemas  # noqa: F401  registers every table on SQLModel.metadata
from dsview.config import ObsidianConfig
from dsview.obsidian import obsidian_utils

POSTGRES_SCHEMAS = ("content", "extraction", "labels")


# Embedding columns are Postgres ARRAY(Float) and SQLite has no array binding,
# so the DDL just needs a renderable type; tests insert a NULL embedding.
@compiles(ARRAY, "sqlite")
def _compile_array_sqlite(element, compiler, **kw):
    return "JSON"


@pytest.fixture
def session():
    """Throwaway in-memory database with the full schema, for hermetic DB tests."""
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def attach_schemas(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        for schema in POSTGRES_SCHEMAS:
            cursor.execute(f"ATTACH DATABASE ':memory:' AS {schema}")
        cursor.close()

    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        yield session


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """Throwaway Obsidian vault, for hermetic note-writing tests."""
    vault_path = tmp_path / "vault"
    (vault_path / "contents").mkdir(parents=True)
    (vault_path / "topics").mkdir(parents=True)

    monkeypatch.setattr(
        obsidian_utils,
        "config",
        ObsidianConfig(
            vault_path=vault_path,
            content_directory="contents",
            topic_directory="topics",
        ),
    )
    return vault_path
