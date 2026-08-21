import json
import logging
import time
from datetime import date
from functools import lru_cache
from typing import Annotated, Literal, Optional

import logfire
import igraph as ig
from mcp.server.fastmcp import FastMCP, Icon
from pydantic import BaseModel, Field
from sqlmodel import Session
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from dsview.config import lazy, load_extraction_config, setup_logger
from dsview.db import engine
from dsview.db.query import ExtractionIndex, TopicsIndex, get_filtered_content
from dsview.db.schemas import ExtractionResult, ExtractionTopic, InputContent
from dsview.graph import build_graph, get_node_neighborhood, get_ranked_nodes

INDEX_TTL = 3_600

setup_logger(enable_logfire=True, service_name="dsview_mcp")

# TODO maybe should handle async calls ?
logger = logging.getLogger(__name__)
extraction_config = lazy(load_extraction_config)

logfire.instrument_mcp()


def get_ttl_hash(seconds: int):
    """Return the same value withing `seconds` time period"""
    return round(time.time() / seconds)


@lru_cache(maxsize=1)
def get_extraction_index(ttl_hash: int = None) -> ExtractionIndex:

    extraction_index = ExtractionIndex()
    extraction_index.build()
    return extraction_index


@lru_cache(maxsize=1)
def get_topics_index(ttl_hash: int = None) -> TopicsIndex:

    topics_index = TopicsIndex()
    topics_index.build()
    return topics_index


@lru_cache(maxsize=1)
def get_graph(ttl_hash: int = None) -> ig.Graph:
    with Session(engine) as session:
        return build_graph(session)


icon = Icon(src="mini_icon.png", mimeType="image/png", sizes=["64x64"])

mcp = FastMCP(
    "DSview mcp server",
    host="0.0.0.0",
    port=8000,
    icons=[icon],
    instructions="""
        Provide tools for exploring a dsview vault, a custom
        Data Science knowledge database managed by the user. It contains a curated
        list of **contents** (research papers, online courses, blog
        posts, github repositories, ...) and **topics** that are
        extracted from them (Machine Learning concepts,
        Development tools, libraries, ...). It serves also as a reading
        list, storing information about what has been and should
        be read.

        The database is structured as a graph of contents linked
        to topics. It allows to navigate the knowledge graph and
        explore adjacent information.

        This is intented to be used in order to help the user
        remind of relevant information about a subject or
        help them during the ideation phase of a new project /
        task.

        Most times, this server will first be used to search
        content or topics (based on fuzzy and semantic search), which
        translates to finding a relevant nodes within the graph.
        And then exploring their neighbors.
    """,
    stateless_http=True,
    json_response=True,
)


class FixAcceptHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        accept = request.headers.get("accept", "")
        if not accept or accept.strip() == "*/*":
            new_headers = [
                (name, b"application/json, text/event-stream")
                if name.lower() == b"accept"
                else (name, value)
                for name, value in request.scope["headers"]
            ]
            if not any(
                name.lower() == b"accept" for name, _ in request.scope["headers"]
            ):
                new_headers.append((b"accept", b"application/json, text/event-stream"))
            request.scope["headers"] = new_headers
        return await call_next(request)


# @mcp.resource("stats://graph")
# def get_graph_stats() -> str:

#     return """
#         Number of contents :
#         Number of topics :
#         Total number of nodes :
#         Total number of edges :
#     """


class DsviewContent(BaseModel):
    id: int = Field(
        description="Unique id of the content, also referenced as content_id"
    )
    link: str = Field(description="Content url")
    upload_date: date = Field(description="Date of content upload by the user")
    already_read: bool = Field(
        description="Wether or not the user have already read the content"
    )
    read_priority: int = Field(
        description="If the content has not been read, gives a priority rating"
    )
    relevance: int = Field(
        description="If the content has been read, rating from the user on how relevant the content is"
    )
    source: Optional[str] = Field(
        default=None, description="Source where the user found the content"
    )
    title: Optional[str] = Field(
        default=None,
        description="Title automatically detected by dsview, None if extraction failed",
    )
    type: Optional[str] = Field(
        default=None,
        description="Content type infered by dsview, None if extraction failed",
    )
    tags: Optional[list[str]] = Field(
        default=None,
        description="Content tags infered by dsview, None if extraction failed",
    )
    summary: Optional[str] = Field(
        default=None,
        description="Content summary generated by dsview, None if extraction failed",
    )


def build_dsview_content(
    input_content: InputContent, extraction_result: ExtractionResult | None = None
) -> DsviewContent:
    """Build a DsviewContent from an InputContent and an ExtractionResult"""
    dsview_content = DsviewContent(**input_content.model_dump())

    if extraction_result is not None:
        dsview_content.title = extraction_result.title
        dsview_content.type = extraction_result.content_type
        dsview_content.tags = [tag.name for tag in extraction_result.tags]
        dsview_content.summary = extraction_result.summary
    return dsview_content


@mcp.tool()
def get_content(content_id: int) -> DsviewContent:
    """Retrieve a stored content from its id"""

    with Session(engine) as db_session:
        input_content = db_session.get(InputContent, content_id)

        if input_content is None:
            raise ValueError(f"Content id {content_id} was not found in database")

        extraction_result = db_session.get(ExtractionResult, content_id)

        return build_dsview_content(input_content, extraction_result)


class DsviewContentList(BaseModel):
    contents: list[DsviewContent]


# @mcp.resource
# def get_sources() -> str:
#     return


@mcp.tool()
def query_content(
    already_read: Optional[bool] = None,
    read_priority: Optional[int] = None,
    relevance: Optional[int] = None,
    source: Optional[str] = None,
    date_ordering: Optional[Literal["asc", "desc"]] = None,
    limit: int = 20,
) -> DsviewContentList:
    """
    Simple query method to retrieve contents, from
    the metadata attached when a content is uploaded
    to the knowledge base.

    Usefull for simple search, for example to search
    for the last uploaded contents, most relevant
    contents, ...

    Defaults to listing last ingested contents.
    """

    with Session(engine) as db_session:
        input_content_list = get_filtered_content(
            db_session,
            already_read,
            read_priority,
            relevance,
            source,
            date_ordering,
            limit,
        )

        contents = []

        for input_content in input_content_list:
            extraction_result = db_session.get(ExtractionResult, input_content.id)
            contents.append(build_dsview_content(input_content, extraction_result))

        return DsviewContentList(contents=contents)


@mcp.tool()
def get_connected_contents(topic_id: int) -> DsviewContentList:
    """
    Get in depth information about contents mentionning
    this topic.

    Also usefull to iteratively explore the knowledge graph,
    while still getting in depth information about the nodes.
    """
    with Session(engine) as db_session:
        topic = db_session.get(ExtractionTopic, topic_id)

        if topic is None:
            raise ValueError(f"Topic id {topic_id} was not found in database")

        contents = []
        for extraction_result in topic.extractions:
            input_content = db_session.get(InputContent, extraction_result.content_id)
            contents.append(build_dsview_content(input_content, extraction_result))

        return DsviewContentList(contents=contents)


@mcp.resource("config://content_types")
def get_content_types() -> str:
    """Retrieve all possible content types"""
    types = list(extraction_config.content_types.values())
    return json.dumps(types)


def _build_type_filter(
    column: str, types: Optional[list[str]], valid_types: list[str]
) -> Optional[list[str]]:
    """
    Build a safe `column IN (...)` filter clause for the DuckDB index.

    `types` is client-supplied and gets interpolated directly into a raw SQL
    string by `DuckDBIndex`, so every value must be checked against the
    known-good set of types from config before it is allowed anywhere near
    the query string.
    """
    if types is None:
        return None

    if len(types) == 0:
        return ["1=0"]

    unknown_types = set(types) - set(valid_types)
    if unknown_types:
        raise ValueError(
            f"Unknown type(s) {sorted(unknown_types)}, expected one of {valid_types}"
        )

    quoted_types = ",".join(f"'{type_}'" for type_ in types)
    return [f"{column} IN ({quoted_types})"]


# ! I don't know if type value or keys are stored in db
@mcp.tool()
def search_content(
    search_term: str,
    limit: int = 10,
    types: Annotated[Optional[list[str]], "Filter content by types"] = None,
) -> DsviewContentList:
    """
    Search contents that are most closely related to the search term.
    It is performed using semantic and fuzzy search on the title and
    summary of contents within the database. This is one of the two
    entrypoints in the knowledge graph.

    Carefull the query index are updated only every hour, so it may
    not return the most fresh information, `query_content` should be
    used to get the last ingested contents.
    """

    with Session(engine) as db_session:
        extraction_index = get_extraction_index(get_ttl_hash(INDEX_TTL))

        filters = _build_type_filter(
            "content_type", types, list(extraction_config.content_types.values())
        )

        result = extraction_index.query_with_rff(search_term, limit, filters=filters)

        contents = []

        for content_id, row in result.iterrows():
            input_content = db_session.get(InputContent, content_id)
            dsview_content = DsviewContent(**input_content.model_dump())
            dsview_content.title = row["title"]
            dsview_content.summary = row["summary"]

            contents.append(dsview_content)

        return DsviewContentList(contents=contents)


class DsviewTopic(BaseModel):
    id: int
    type: str
    name: str
    description: str


class DsviewTopicList(BaseModel):
    topics: list[DsviewTopic]


@mcp.resource("config://topic_types")
def get_topic_types() -> str:
    """Retrieve all possible topic types"""
    types = list(extraction_config.topic_categories.values())
    return json.dumps(types)


@mcp.tool()
def get_connected_topics(content_id: int) -> DsviewTopicList:
    """
    Get in depth information about topics mentionned in one
    content.

    Also usefull to iteratively explore the knowledge graph,
    while still getting in depth information about the nodes.
    """
    with Session(engine) as db_session:
        extraction_result = db_session.get(ExtractionResult, content_id)

        if extraction_result is None:
            raise ValueError(
                f"Content id {content_id} extraction result was not found in database"
            )

        topics = []

        for topic in extraction_result.topics:
            topics.append(DsviewTopic(**topic.model_dump(exclude=["embedding"])))

        return DsviewTopicList(topics=topics)


@mcp.tool()
def search_topic(
    search_term: str,
    limit: int = 10,
    types: Annotated[Optional[list[str]], "The type of topic to search for"] = None,
) -> DsviewTopicList:
    """
    Search for topics in the knowledge graph. This is method
    use both semantic and orthographic distance to find the
    most relevant topics.

    Carefull the query index are updated only every hour, so it may
    not return the most fresh information, `query_content` should be
    used to get the last ingested contents.
    """

    with Session(engine) as db_session:
        topics_index = get_topics_index(get_ttl_hash(INDEX_TTL))

        filters = _build_type_filter(
            "type", types, list(extraction_config.topic_categories.values())
        )

        result = topics_index.query_with_rff(search_term, limit, filters=filters)

        topics = []

        for topic_id, _ in result.iterrows():
            topic = db_session.get(ExtractionTopic, topic_id)
            dsview_topic = DsviewTopic(**topic.model_dump(exclude=["embedding"]))

            topics.append(dsview_topic)

        return DsviewTopicList(topics=topics)


# Deep explore tool, only returns ids and title
# Give and understanding about structure bfs
# What about cycles (prohibit them implicitely)


class GraphNode(BaseModel):
    id: int
    kind: Literal["topic", "content"]
    label: str = Field(description="Title for a content node, name for a topic.")
    type: str


class NeighborNode(GraphNode):
    distance: int = Field(description="How far away from query node")


class Neighborhood(BaseModel):
    nodes: list[NeighborNode]


@mcp.tool()
def explore_graph(
    node_id: int,
    node_kind: Literal["topic", "content"],
    radius: int = 3,
) -> Neighborhood:
    """
    Explore the neighborhood of a node in the
    knowledge graph. Handle both topics nodes and
    content nodes. The output will contains a list
    node and their distance to the origin node.

    It is usefull to quickly deeply explore the
    neighborhood of a node and get a sense of the
    information stored in the knowledge graph.

    Because it returns few information on the content
    of the nodes in the neighborhood, it should most of the time
    be used in combinaison with more in depth exploration.
    For example using the `get_content` or `get_topic` tools
    to get detailed information on the most relevant nodes
    in the neighborhood (depending on the user query).
    """
    graph = get_graph(get_ttl_hash(INDEX_TTL))

    results = get_node_neighborhood(
        graph,
        node_id,
        node_kind,
        radius=radius,
    )

    nodes = []

    for node, distance in results:
        node_attributes = node.attributes()
        node_attributes["id"] = node_attributes["name"].split("_")[1]
        node_attributes["distance"] = distance

        nodes.append(NeighborNode(**node_attributes))

    return Neighborhood(nodes=nodes)


class RankedNode(GraphNode):
    score: float


class NodeRanking(BaseModel):
    nodes: list[RankedNode]


@mcp.tool()
def get_nodes_ranking(
    metric: Literal["betweenness", "degree", "pagerank"] = "betweenness",
    node_kind: Optional[Literal["content", "topic"]] = None,
    limit: int = 20,
):
    """
    Get a list of the most important nodes in the knowledge
    graph with regards to a graph metric. The default metric
    is betweenesss centrality. The output will contain a list
    of node ordered based on the score attribute.

    This can be used to identify what is the core knowledge stored
    in the graph. It can be filtered on only contents or topics.

    Betweenness centrality is the most relevant scoring metric, as the
    underlining graph is created iteratively with a focus on having
    a connected graph, this score allows to surface the core topics
    or most crucial contents. It can also be used to prioritize, when
    the author asks what they should read next (when this tool is combined
    with anoter get tool to get the full node information).

    Degree score should only be used with node_kind equal to "topic", in
    this setup it will surface the most mentionned topics, and can also
    be usefull to get an idea of important topics in the graph.
    """
    graph = get_graph(get_ttl_hash(INDEX_TTL))

    results = get_ranked_nodes(
        graph,
        metric,
        node_kind=node_kind,
        limit=limit,
    )

    nodes = []
    for node, score in results:
        node_attributes = node.attributes()
        node_attributes["id"] = node_attributes["name"].split("_")[1]
        node_attributes["score"] = score

        nodes.append(RankedNode(**node_attributes))

    return NodeRanking(nodes=nodes)


if __name__ == "__main__":
    import uvicorn

    app = mcp.streamable_http_app()
    app.add_middleware(FixAcceptHeaderMiddleware)
    uvicorn.run(app, host="0.0.0.0", port=8000)
