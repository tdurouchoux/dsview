import logging

import logfire
import duckdb
import pandas as pd

from dsview.config import ModelType, lazy, lazy_model_config, load_postgres_config
from dsview.model_utils import get_model_provider

logger = logging.getLogger(__name__)
model_config = lazy_model_config(ModelType.INDEX)
postgres_config = lazy(load_postgres_config)

EXTENSION_LIST = ["vss", "fts", "postgres"]

# ! Not safe sql ingestion is possible
# TODO implement a way to check table names and columns > by checking for table in db (engine.execute(something))


class DuckDBIndex:
    def __init__(
        self,
        source_table: str,
        select_columns: list[str],
        fts_columns: list[str],
        id_column: str = "id",
        embedding_column: str = "embedding",
        embedding_size: int = 1024,
    ):
        self.source_table = source_table
        self.select_columns = select_columns
        self.fts_columns = fts_columns
        self.id_column = id_column
        self.embedding_column = embedding_column
        self.embedding_size = embedding_size

        if self.id_column not in self.select_columns:
            self.select_columns = [self.id_column] + self.select_columns

        if self.embedding_column in self.select_columns:
            self.select_columns.remove(self.embedding_column)

        self.conn = None

        self.model_provider = get_model_provider(model_config)

    @property
    def index_table(self):
        return f"{str(self.source_table).split('.')[-1]}_index"

    def _init_duckdb(self):
        logger.info("Initializing Duckdb for %s table indexing", self.source_table)

        conn = duckdb.connect(":memory:")

        for ext in EXTENSION_LIST:
            conn.install_extension(ext)
            conn.load_extension(ext)

        connect_sqlite_query = f"""
            ATTACH '{postgres_config.db_uri(sqlalchemy=False)}' AS dsview_db (TYPE postgres);
        """

        conn.execute(connect_sqlite_query)

        return conn

    def _create_index_table(self):
        logger.info("Creating index table")

        create_index_table_query = f"""
            CREATE TABLE {self.index_table} AS
            SELECT
                {",".join(self.select_columns)},
                {self.embedding_column}::FLOAT[{self.embedding_size}] AS embedding
            FROM
                dsview_db.{self.source_table};
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
        with logfire.span("Building indexes"):
            self.conn = self._init_duckdb()
            self._create_index_table()

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

    @staticmethod
    def _clean_input(input: str) -> str:
        return input.replace("'", "")

    def _query_fts_index(
        self,
        input: str,
        n_query: int,
        score_threshold: float = 1,
        where_clause: str = "1",
    ) -> pd.DataFrame:

        fts_query = f"""
            SELECT *, fts_main_{self.index_table}.match_bm25(
                {self.id_column},
                '{self._clean_input(input)}',
                fields := '{",".join(self.fts_columns)}'
            ) AS fts_score
            FROM {self.index_table}
            WHERE {where_clause} AND fts_score IS NOT NULL AND fts_score > {score_threshold}
            ORDER BY fts_score DESC
            LIMIT {n_query};
        """

        return (
            self.conn.execute(fts_query)
            .df()
            .drop(columns=self.embedding_column)
            .set_index(self.id_column)
        )

    def compute_fts_score(self, input: str, row_id: str) -> float:
        """
        Compute FTS score between two content IDs using the existing FTS index.

        Args:
            input (str) :
            id : Second content ID (used as query)

        Returns:
            FTS BM25 score between the two contents
        """

        # Use the existing FTS index to compute score between the two documents
        fts_query = f"""
            SELECT fts_main_{self.index_table}.match_bm25(
                {self.id_column},
                '{self._clean_input(input)}',
                fields := '{",".join(self.fts_columns)}'
            ) AS fts_score
            FROM {self.index_table}
            WHERE {self.id_column} = '{row_id}';
        """

        result = self.conn.execute(fts_query).df()
        if len(result) == 0 or pd.isna(result["fts_score"].iloc[0]):
            return 0.0

        return float(result["fts_score"].iloc[0])

    def _query_vss_index(
        self,
        input: str,
        n_query: int,
        distance_threshold: float = 0.95,
        where_clause: str = "1",
    ) -> pd.DataFrame:
        input_embedding = self.model_provider.embed(input)

        vss_query = f"""
            SELECT
                *,
                array_distance(
                    {self.embedding_column},
                    [{",".join([str(e) for e in input_embedding])}]::FLOAT[{self.embedding_size}]
                ) as vss_distance
            FROM {self.index_table}
            WHERE {where_clause} AND vss_distance < {distance_threshold}
            ORDER BY vss_distance
            LIMIT {n_query};
        """

        return (
            self.conn.execute(vss_query)
            .df()
            .drop(columns=self.embedding_column)
            .set_index(self.id_column)
        )

    def compute_vss_distance(self, input: str, row_id: str) -> float:
        """
        Compute cosine distance between embeddings of two content IDs.

        Args:
            id1: First content ID
            id2: Second content ID

        Returns:
            Cosine distance between the two embeddings
        """

        input_embedding = self.model_provider.embed(input)

        # Compute distance directly using indexed embeddings
        distance_query = f"""
            SELECT
                array_distance(
                    {self.embedding_column},
                    [{",".join([str(e) for e in input_embedding])}]::FLOAT[{self.embedding_size}]
                ) as vss_distance
            FROM {self.index_table}
            WHERE {self.id_column} = '{row_id}';
        """

        result = self.conn.execute(distance_query).df()
        if len(result) == 0:
            raise ValueError(f"Could not find ID {row_id} in the index")

        return float(result["vss_distance"].iloc[0])

    def query(
        self,
        input: str,
        n_fts: int = 5,
        n_vss: int = 5,
        filters: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Received %s input for duckdb indexing", input)

        where_clause = " AND ".join(filters) if filters else "1"

        return (
            self._query_fts_index(input, n_fts, where_clause=where_clause),
            self._query_vss_index(input, n_vss, where_clause=where_clause),
        )

    def _merge_results_with_rff(
        self,
        fts_results: pd.DataFrame,
        vss_results: pd.DataFrame,
        limit: int,
        k: int,
    ) -> pd.DataFrame:
        # Add rank columns using pandas operations
        fts_with_rank = fts_results.reset_index(names=self.id_column).drop(
            columns=["fts_score"]
        )
        fts_with_rank["rank"] = range(1, len(fts_with_rank) + 1)
        fts_with_rank["score"] = 1 / (k + fts_with_rank["rank"])

        vss_with_rank = vss_results.reset_index(names=self.id_column).drop(
            columns=["vss_distance"]
        )
        vss_with_rank["rank"] = range(1, len(vss_with_rank) + 1)
        vss_with_rank["score"] = 1 / (k + vss_with_rank["rank"])

        merged_results = pd.concat((fts_with_rank, vss_with_rank))

        rff_score = (
            merged_results.groupby(self.id_column)["score"].sum().rename("rff_score")
        )

        return (
            merged_results.drop_duplicates(subset=[self.id_column])
            .set_index(self.id_column)
            .drop(columns=["rank", "score"])
            .join(rff_score)
            .sort_values("rff_score", ascending=False)
            .head(limit)
        )

    def query_with_rff(
        self,
        input: str,
        limit: int,
        n_fts: int | None = None,
        n_vss: int | None = None,
        k: int = 60,
        filters: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Query and return merged results using Reciprocal Rank Fusion.

        Args:
            input: Query string
            limit: Maximum number of expected results
            n_fts: Number of FTS results to retrieve
            n_vss: Number of VSS results to retrieve
            k: RRF parameter (typically 60)

        Returns:
            Merged DataFrame sorted by RRF score (descending)
        """
        logger.info("Received %s input for duckdb indexing", input)

        if n_fts is None:
            n_fts = 2 * limit
        if n_vss is None:
            n_vss = 2 * limit

        # Get FTS and VSS results
        fts_results, vss_results = self.query(
            input, n_fts=n_fts, n_vss=n_vss, filters=filters
        )

        return self._merge_results_with_rff(fts_results, vss_results, limit, k)

    def delete_rows(self, where_clause: str) -> int:
        """
        Delete rows from the index table based on a WHERE clause.

        Args:
            where_clause: SQL WHERE clause (without the "WHERE" keyword)
                         Example: "status = 'inactive' AND category = 'old'"

        Returns:
            Number of rows deleted

        Raises:
            ValueError: If where_clause is empty
        """
        if not where_clause or not where_clause.strip():
            raise ValueError("WHERE clause cannot be empty")

        # Count rows before deletion for return value
        count_query = (
            f"SELECT COUNT(*) as count FROM {self.index_table} WHERE {where_clause};"
        )
        count_result = self.conn.execute(count_query).df()
        rows_to_delete = int(count_result["count"].iloc[0])

        if rows_to_delete == 0:
            logger.info("No rows found matching the deletion criteria")
            return 0

        # Execute deletion
        delete_query = f"DELETE FROM {self.index_table} WHERE {where_clause};"
        self.conn.execute(delete_query)

        logger.info("Deleted %d rows from %s", rows_to_delete, self.index_table)
        return rows_to_delete
