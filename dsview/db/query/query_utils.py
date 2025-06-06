import logging

import duckdb
import pandas as pd

from dsview.config import ModelType, get_db_path, load_model_config
from dsview.model_utils import get_model_provider

logger = logging.getLogger(__name__)
model_config = load_model_config(ModelType.INDEX)

EXTENSION_LIST = ["vss", "fts", "sqlite"]

# ! Not safe sql ingestion is possible
# TODO implement a way to check table names and columns > by checking for table in db (engine.execute(something))


class DuckDBIndex:
    def __init__(
        self,
        table_name: str,
        select_columns: list[str],
        fts_columns: list[str],
        id_column: str = "id",
        embedding_column: str = "embedding",
        embedding_size: int = 1536,
    ):
        self.table_name = table_name
        self.select_columns = select_columns
        self.fts_columns = fts_columns
        self.id_column = id_column
        self.embedding_column = embedding_column
        self.embedding_size = embedding_size

        if self.id_column not in self.select_columns:
            self.select_columns = [self.id_column] + self.select_columns

        self.conn = None

        self.model_provider = get_model_provider(model_config)

    @property
    def index_table(self):
        return f"{self.table_name}_index"

    def _init_duckdb(self):
        logger.info("Initializing Duckdb for %s table indexing", self.table_name)

        conn = duckdb.connect(":memory:")

        for ext in EXTENSION_LIST:
            conn.install_extension(ext)
            conn.load_extension(ext)

        connect_sqlite_query = f"""
            ATTACH '{get_db_path()}' AS sqlite_db (TYPE sqlite);
        """

        conn.execute(connect_sqlite_query)

        return conn

    def _create_index_table(self):
        logger.info("Creating index table")

        create_index_table_query = f"""
            CREATE TABLE {self.index_table} AS
            SELECT
                {",".join(self.select_columns)},
                list_transform(
                    string_split({self.embedding_column}, ','),
                    x -> CAST(trim(x) AS FLOAT)
                )::FLOAT[{self.embedding_size}] AS embedding
            FROM
                sqlite_db.{self.table_name};
        """

        self.conn.execute(
            create_index_table_query,
        )

    def _build_fts_index(self):
        diff_columns = set(self.fts_columns).difference(set(self.select_columns))

        if len(diff_columns) > 0:
            raise ValueError(
                "Every selected columns for FTS must be"
                f"in index table. Missing {diff_columns} "
                "from select_columns"
            )

        create_index_query = f"""
            PRAGMA create_fts_index(
                '{self.index_table}',
                '{self.id_column}',
                '{"', '".join(self.fts_columns)}'
            );
        """

        self.conn.execute(create_index_query)

    def _build_vss_index(self):
        create_index_query = f"""
        CREATE INDEX idx ON {self.index_table} USING HNSW (embedding);
        """

        self.conn.execute(create_index_query)

    def build(self):
        self.conn = self._init_duckdb()
        self._create_index_table()

        logger.info("Creating indexes")
        self._build_fts_index()
        self._build_vss_index()

    def close(self):
        self.conn.close()
        self.conn = None

    def __enter__(self):
        self.build()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _query_fts_index(
        self, input: str, n_query: int, score_threshold: float = 1
    ) -> pd.DataFrame:
        fts_query = f"""
            SELECT *, fts_main_{self.index_table}.match_bm25(
                {self.id_column},
                '{input}',
                fields := '{",".join(self.fts_columns)}'
            ) AS score
            FROM {self.index_table}
            WHERE score IS NOT NULL AND score > {score_threshold}
            ORDER BY score DESC
            LIMIT {n_query};
        """

        return self.conn.execute(fts_query).df().set_index(self.id_column)

    def _query_vss_index(
        self, input: str, n_query: int, distance_threshold: float = 0.95
    ) -> pd.DataFrame:
        input_embedding = self.model_provider.embed(input)

        vss_query = f"""
            SELECT
                *,
                array_distance(
                    {self.embedding_column},
                    [{",".join([str(e) for e in input_embedding])}]::FLOAT[{self.embedding_size}]
                ) as distance
            FROM {self.index_table}
            WHERE distance < {distance_threshold}
            ORDER BY distance
            LIMIT {n_query};
        """

        return self.conn.execute(vss_query).df().set_index(self.id_column)

    def query(
        self, input: str, n_fts: int = 5, n_vss: int = 5
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Received %s input for duckdb indexing", input)

        return (
            self._query_fts_index(input, n_fts),
            self._query_vss_index(input, n_vss),
        )
