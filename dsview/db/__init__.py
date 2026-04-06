from sqlmodel import Session, SQLModel, create_engine, text
from tenacity import (
    retry,
    stop_after_attempt,
)

from dsview.config import load_postgres_config
from . import schemas

postgres_config = load_postgres_config()

# Configure engine with proper connection pooling to handle stale connections
engine = create_engine(
    postgres_config.db_uri(),
    # Connection pool settings
    pool_size=5,  # Number of connections to maintain in the pool
    max_overflow=10,  # Additional connections that can be created on demand
    pool_timeout=30,  # Timeout in seconds to get connection from pool
    pool_recycle=3600,  # Recycle connections after 1 hour (3600 seconds)
    pool_pre_ping=True,  # Test connections before use to detect stale connections
    # Additional connection arguments
    connect_args={
        "connect_timeout": 10,  # Connection timeout in seconds
        "application_name": "dsview_api",  # Identify your app in PostgreSQL logs
    },
)

# SQLModel.metadata.create_all(engine)


@retry(reraise=True, stop=stop_after_attempt(5))
def check_db_connection(session: Session):
    session.exec(text("SELECT 1")).first()
