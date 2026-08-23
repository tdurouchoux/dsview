# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair==5.5.0",
#     "duckdb==1.3.2",
#     "marimo",
#     "numpy==2.2.6",
#     "openai==1.99.9",
#     "pandas==2.3.1",
#     "python-frontmatter==1.1.0",
#     "sqlglot==27.7.0",
#     "sqlite-vec==0.1.6",
#     "sqlmodel==0.0.24",
#     "tqdm==4.67.1",
#     "umap-learn==0.5.9.post2",
# ]
# ///

import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""## Retrieve topics""")


@app.cell
def _():
    import frontmatter

    from dsview.extraction.models.topics_extraction import (
        DataScienceTopic,
        TopicType,
    )
    from dsview.obsidian.obsidian_utils import retrieve_topics_path

    return DataScienceTopic, TopicType, frontmatter, retrieve_topics_path


@app.cell
def _(DataScienceTopic, TopicType, frontmatter, retrieve_topics_path):
    def load_topics():
        topics_path = retrieve_topics_path()

        topics = []

        for path in topics_path:
            note = frontmatter.load(path)
            topics.append(
                DataScienceTopic(
                    name=path.stem,
                    type=TopicType(path.parent.name),
                    description=note.content,
                )
            )
        return topics

    return (load_topics,)


@app.cell
def _(load_topics):
    topics = load_topics()
    return (topics,)


@app.cell
def _():
    ## Embed topic
    return


@app.cell
def _():
    from openai import OpenAI
    from tqdm import tqdm

    return OpenAI, tqdm


@app.cell
def _(DataScienceTopic, OpenAI, tqdm):
    def embed_topics(topics: list[DataScienceTopic]) -> list[list[float]]:
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required, but unused
        )

        embeddings = []

        for topic in tqdm(topics):
            # input = f"""
            #     name: {topic.name}
            #     description: {topic.description}
            # """

            input = topic.description

            embeddings.append(
                client.embeddings.create(
                    model="bge-m3",
                    input=input,
                )
                .data[0]
                .embedding
            )

        return embeddings

    return (embed_topics,)


@app.cell
def _(mo):
    mo.md(r"""## export to parquet""")


@app.cell
def _():
    from pathlib import Path

    import pandas as pd

    return Path, pd


@app.cell
def _(Path, embed_topics, pd, topics):
    topics_embedding_path = Path("notebooks/topics_embedding.parquet")

    if not topics_embedding_path.exists():
        topics_embeddings = embed_topics(topics)

        df_topics = pd.DataFrame(
            [
                dict(topic, embedding=embedding)
                for topic, embedding in zip(topics, topics_embeddings)
            ]
        )
        df_topics.to_parquet(topics_embedding_path)
    else:
        df_topics = pd.read_parquet(topics_embedding_path)
    return (df_topics,)


@app.cell
def _(df_topics):
    df_topics


@app.cell
def _(mo):
    mo.md(r"""## Ingest into db""")


@app.cell
def _():
    from sqlmodel import Session

    from dsview.config import get_sqlite_engine
    from dsview.db.ingest import embed_and_save_topic

    return Session, embed_and_save_topic, get_sqlite_engine


@app.cell
def _(get_sqlite_engine):
    ingest_engine = get_sqlite_engine()

    return (ingest_engine,)


@app.cell
def _(topics):
    len(topics)


@app.cell
def _(Session, embed_and_save_topic, ingest_engine, topics, tqdm):
    with Session(ingest_engine) as session:
        for topic in tqdm(topics):
            embed_and_save_topic(session, topic)


@app.cell
def _(df_topics):
    df_topics.type.loc[0]


@app.cell
def _(mo):
    mo.md(r"""## Plot""")


@app.cell
def _():
    import altair as alt
    import numpy as np
    import umap

    return alt, np, umap


@app.cell
def _(n_neighbors, umap):
    reducer = umap.UMAP(n_neighbors=n_neighbors.value)
    return (reducer,)


@app.cell
def _(df_topics, np, reducer):
    df_topics_red = df_topics.copy()

    df_topics_red[["x", "y"]] = reducer.fit_transform(
        np.array(df_topics_red["embedding"]).tolist()
    )
    return (df_topics_red,)


@app.cell
def _(df_topics):
    df_topics


@app.cell
def _(mo):
    n_neighbors = mo.ui.slider(start=2, stop=100)
    n_neighbors
    return (n_neighbors,)


@app.cell
def _(alt, df_topics_red):
    alt.Chart(df_topics_red[["name", "type", "x", "y"]]).mark_point().encode(
        x="x:Q", y="y:Q", color="type", tooltip=["name"]
    )


@app.cell
def _(mo):
    mo.md(r"""## duckdb search""")


@app.cell
def _():
    import duckdb

    return (duckdb,)


@app.cell
def _(create_engine):
    test_engine = create_engine("sqlite:///test_ingest.db")
    return (test_engine,)


@app.cell
def _(df_topics, test_engine):
    df_topics_ingest = df_topics.copy()
    df_topics_ingest["embedding"] = df_topics_ingest["embedding"].apply(
        lambda arr: ",".join([str(e) for e in arr])
    )
    df_topics_ingest.to_sql("topics_ingest", test_engine, if_exists="fail", index=False)


@app.cell
def _(mo, test_engine, topics_ingest):
    _df = mo.sql(
        """
        SELECT * FROM topics_ingest
        """,
        engine=test_engine,
    )


@app.cell
def _(duckdb):
    con = duckdb.connect()
    con.install_extension("spatial")
    con.load_extension("spatial")


@app.cell
def _(mo):
    _ = mo.sql(
        """
        INSTALL vss; LOAD vss;
        INSTALL fts; LOAD fts;
        INSTALL sqlite; LOAD sqlite;
        """
    )


@app.cell
def _(mo):
    _df = mo.sql(
        """
        ATTACH 'test_ingest.db' AS w (TYPE sqlite);
        """
    )


@app.cell
def _(mo, sqlite_db, topics_ingest):
    _df = mo.sql(
        """
        CREATE TABLE topics_index AS
        SELECT
            type,
            name,
            description,
            list_transform(
                string_split(embedding, ','),
                x -> CAST(trim(x) AS FLOAT)
            )::FLOAT[1024] AS embedding
        FROM
            sqlite_db.topics_ingest;
        CREATE INDEX idx ON topics_index USING HNSW (embedding);
        """
    )


@app.cell
def _(mo):
    _df = mo.sql(
        """
        PRAGMA create_fts_index('topics_index', 'name', 'description');
        """
    )


@app.cell
def _(mo):
    _df = mo.sql(
        """
        DROP TABLE topics_index
        """
    )


@app.cell
def _(mo, topics_index):
    _df = mo.sql(
        """
        SELECT * FROM topics_index
        """
    )


@app.cell
def _(OpenAI):
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    )

    input = "node embeddings"
    input_embedding = (
        client.embeddings.create(
            model="bge-m3",
            input=input,
        )
        .data[0]
        .embedding
    )
    return input, input_embedding


@app.cell
def _():
    return


@app.cell
def _(input_embedding):
    len(",".join([str(e) for e in input_embedding]))


@app.cell
def _(mo):
    mo.md(r"""x3 12ko très ok""")


@app.cell
def _(input_embedding, mo, topics_index):
    _df = mo.sql(
        f"""
        SELECT * FROM topics_index ORDER BY array_distance(embedding, {input_embedding}::FLOAT[1024]) LIMIT 5
        """
    )


@app.cell
def _(input, mo, topics_index):
    _df = mo.sql(
        f"""
        SELECT name, description, score
        FROM (
            SELECT *, fts_main_topics_index.match_bm25(
                name,
                '{input}',
                fields := 'description'
            ) AS score
            FROM topics_index
        ) sq
        WHERE score IS NOT NULL
        ORDER BY score DESC
        LIMIT 5;
        """
    )


@app.cell
def _(mo):
    mo.md(r"""## sqlite3 search""")


@app.cell
def _(df_topics):
    df_topics.shape


@app.cell
def _(df_topics):
    df_topics[df_topics.duplicated("name")]


@app.cell
def _(df_topics):
    df_topics["name"].nunique()


@app.cell
def _():
    import sqlite3

    import sqlite_vec
    from sqlmodel import create_engine

    return create_engine, sqlite3, sqlite_vec


@app.cell
def _(sqlite3, sqlite_vec):
    db = sqlite3.connect(":memory:")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return (db,)


@app.cell
def _(create_engine):
    engine = create_engine("sqlite:///:memory:")
    return (engine,)


@app.cell
def _(engine, mo):
    _df = mo.sql(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS topics_fts USING fts5 (name, description);
        """,
        engine=engine,
    )


@app.cell
def _(df_topics, engine):
    df_topics[["name", "description"]].to_sql(
        "topics_fts", engine, if_exists="append", index=False
    )


@app.cell
def _():
    search_topic = "reinforcment learning"
    return (search_topic,)


@app.cell
def _(engine, mo, search_topic):
    _df = mo.sql(
        f"""
        SELECT
            *
        FROM
            topics_fts
        WHERE
            topics_fts MATCH 'NEAR({search_topic}, 20)'
        ORDER BY rank LIMIT 5
        """,
        engine=engine,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    -- If the database schema is:
    CREATE TABLE t1 (a, b, c, d INTEGER PRIMARY KEY);
    CREATE VIRTUAL TABLE ft USING fts5(a, c, content=t1, content_rowid=d);

    -- Fts5 may issue queries such as:
    SELECT d, a, c FROM t1 WHERE d = ?;
    """
    )


@app.cell
def _(db):
    db.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS topics_vss USING vec0(embedding float[1024])"
    )


@app.cell
def _(df_topics):
    df_topics.drop_duplicates()


@app.cell
def _(df_topics):
    df_topics.index


@app.cell
def _():
    return


@app.cell
def _(df_topics):
    df_topics.loc[1, "embedding"]


@app.cell
def _(db, df_topics, struct):
    for row in df_topics.itertuples(index=True):
        db.execute(
            "INSERT INTO topics_vss(rowid, embedding) VALUES (?, ?)",
            (row.Index, struct.pack("%sf" % 1024, *row.embedding)),
        )


app._unparsable_cell(
    r"""
    \" search_topic_embedding = (
        client.embeddings.create(
            model=\"bge-m3\",
            input=search_topic,
        )
        .data[0]
        .embedding
    )

    pack_embedding = struct.pack(\"%sf\" % 1024, *search_topic_embedding)
    """,
    name="_",
)


@app.cell
def _():
    import struct

    return (struct,)


@app.cell
def _(db, df_topics, pack_embedding):
    close_rows = db.execute(
        """SELECT
                rowid,
                distance
            FROM topics_vss
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT 5
        """,
        [pack_embedding],
    ).fetchall()

    print(close_rows)

    df_topics.loc[[row[0] for row in close_rows]]


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
