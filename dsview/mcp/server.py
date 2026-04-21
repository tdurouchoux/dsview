import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from functools import cache
from typing import Annotated, Literal, Optional

from mcp.server.fastmcp import Context, FastMCP, Icon
from mcp.server.session import ServerSession
from pydantic import BaseModel, Field
from sqlmodel import Session
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from dsview.config import load_extraction_config
from dsview.db import engine
from dsview.db.query import ExtractionIndex, TopicsIndex, get_filtered_content
from dsview.db.schemas import ExtractionResult, ExtractionTopic, InputContent

# TODO maybe should handle async calls ?
logger = logging.getLogger(__name__)
extraction_config = load_extraction_config()

INDEX_TTL = 3_600


@cache
def get_extraction_index(ttl_hash: int = None) -> ExtractionIndex:

    extraction_index = ExtractionIndex()
    extraction_index.build()
    return extraction_index


@cache
def get_topics_index(ttl_hash: int = None) -> TopicsIndex:

    topics_index = TopicsIndex()
    topics_index.build()
    return topics_index


def get_ttl_hash(seconds: int):
    """Return the same value withing `seconds` time period"""
    return round(time.time() / seconds)


@dataclass
class AppContext:
    db_session: Session
    extraction_index: ExtractionIndex
    topics_index: TopicsIndex


# ? Why async
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle with type-safe context."""
    # Initialize on startup
    db_session = Session(engine)
    extraction_index = get_extraction_index(INDEX_TTL)
    topics_index = get_topics_index(INDEX_TTL)

    try:
        yield AppContext(
            db_session=db_session,
            extraction_index=extraction_index,
            topics_index=topics_index,
        )
    finally:
        # Cleanup on shutdown
        db_session.close()
        extraction_index.close()
        topics_index.close()


icon = Icon(src="mini_icon.png", mimeType="image/png", sizes=["64x64"])

mcp = FastMCP(
    "DSview mcp server",
    lifespan=app_lifespan,
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
def get_content(
    content_id: int, ctx: Context[ServerSession, AppContext]
) -> DsviewContent:
    """Retrieve a stored content from its id"""

    db_session = ctx.request_context.lifespan_context.db_session

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
    ctx: Context[ServerSession, AppContext],
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

    db_session = ctx.request_context.lifespan_context.db_session

    input_content_list = get_filtered_content(
        db_session, already_read, read_priority, relevance, source, date_ordering, limit
    )

    contents = []

    for input_content in input_content_list:
        extraction_result = db_session.get(ExtractionResult, input_content.id)
        contents.append(build_dsview_content(input_content, extraction_result))

    return DsviewContentList(contents=contents)


@mcp.tool()
def get_connected_contents(
    topic_id: int,
    ctx: Context[ServerSession, AppContext],
) -> DsviewContentList:
    db_session = ctx.request_context.lifespan_context.db_session

    topic = db_session.get(ExtractionTopic, topic_id)

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


# ! I don't know if type value or keys are stored in db
@mcp.tool()
def search_content(
    search_term: str,
    ctx: Context[ServerSession, AppContext],
    limit: int = 10,
    types: Annotated[Optional[list[str]], "Filter content by types"] = None,
) -> DsviewContentList:
    """
    Search contents that are most closely related to the search term.
    It is performed using semantic and fuzzy search on the title and
    summary of contents within the database. This is one of the two
    entrypoints in the knowledge graph.
    """

    db_session = ctx.request_context.lifespan_context.db_session
    extraction_index = ctx.request_context.lifespan_context.extraction_index

    filters = (
        ["content_type IN ('" + "','".join(types) + "')"] if types is not None else None
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
def get_connected_topics(
    content_id: str,
    ctx: Context[ServerSession, AppContext],
) -> DsviewTopicList:

    db_session = ctx.request_context.lifespan_context.db_session

    extraction_result = db_session.get(ExtractionResult, content_id)

    topics = []

    for topic in extraction_result.topics:
        topics.append(DsviewTopic(**topic.model_dump(exclude=["embedding"])))

    return DsviewTopicList(topics=topics)


@mcp.tool()
def search_topic(
    search_term: str,
    ctx: Context[ServerSession, AppContext],
    limit: int = 10,
    types: Annotated[Optional[list[str]], "The type of topic to search for"] = None,
) -> DsviewTopicList:
    """
    Search for topics in the knowledge graph.
    """

    db_session = ctx.request_context.lifespan_context.db_session
    topics_index = ctx.request_context.lifespan_context.topics_index

    filters = ["type IN ('" + "','".join(types) + "')"] if types is not None else None

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


if __name__ == "__main__":
    import uvicorn

    app = mcp.streamable_http_app()
    app.add_middleware(FixAcceptHeaderMiddleware)
    uvicorn.run(app, host="0.0.0.0", port=8000)
