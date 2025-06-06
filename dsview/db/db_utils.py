from functools import cache

from sqlmodel import create_engine

from dsview.config import get_sqlite_url


@cache
def get_engine():
    return create_engine(get_sqlite_url())
