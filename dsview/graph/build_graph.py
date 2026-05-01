from typing import NamedTuple

import igraph as ig
from sqlmodel import Session, select

from dsview.db.schemas import (
    ContentTopicRelation,
    ExtractionResult,
    ExtractionTopic,
)

TOPICS_ID = "id"
TOPICS_ATTR_MAPPING = {
    "name": "label",
    "type": "type",
}

EXTRACTIONS_ID = "content_id"
EXTRACTIONS_ATTR_MAPPING = {
    "title": "label",
    "content_type": "type",
}


def format_content_id(content_id: int) -> str:
    return f"c_{content_id}"


def format_topic_id(topic_id: int) -> str:
    return f"t_{topic_id}"


def retrieve_edges(session: Session) -> list[NamedTuple]:
    """
    Retrieve relations from ContentTopicRelation table
    """
    return session.exec(select(ContentTopicRelation)).all()


def retrieve_columns_from_table(
    node_class, columns: list[str], session: Session
) -> list[NamedTuple]:
    return session.exec(select(*(getattr(node_class, col) for col in columns))).all()


def add_extraction_nodes(
    graph: ig.Graph,
    include_nodes_attributes: bool,
    session: Session,
) -> ig.Graph:

    add_extraction_columns = []
    if include_nodes_attributes:
        add_extraction_columns += list(EXTRACTIONS_ATTR_MAPPING.keys())

    extractions = retrieve_columns_from_table(
        ExtractionResult, [EXTRACTIONS_ID] + add_extraction_columns, session
    )

    for extraction in extractions:
        node_id = format_content_id(getattr(extraction, EXTRACTIONS_ID))

        extraction_attrs = {
            EXTRACTIONS_ATTR_MAPPING[col]: getattr(extraction, col)
            for col in add_extraction_columns
        }

        graph.add_vertex(
            node_id,
            kind="content",
            **extraction_attrs,
        )

    return graph


def add_topic_nodes(
    graph: ig.Graph,
    include_nodes_attributes: bool,
    session: Session,
) -> ig.Graph:
    """Add topic nodes to graph."""
    add_topic_columns = []
    if include_nodes_attributes:
        add_topic_columns += list(TOPICS_ATTR_MAPPING.keys())

    topics = retrieve_columns_from_table(
        ExtractionTopic, [TOPICS_ID] + add_topic_columns, session
    )

    for topic in topics:
        node_id = format_topic_id(getattr(topic, TOPICS_ID))

        topic_attrs = {
            TOPICS_ATTR_MAPPING[col]: getattr(topic, col) for col in add_topic_columns
        }

        graph.add_vertex(
            node_id,
            kind="topic",
            **topic_attrs,
        )

    return graph


def build_graph(
    session: Session,
    include_nodes_attributes: bool = True,
) -> ig.Graph:
    """Build an igraph Graph from database records."""

    graph = ig.Graph()

    graph = add_extraction_nodes(graph, include_nodes_attributes, session)
    graph = add_topic_nodes(graph, include_nodes_attributes, session)

    edges = retrieve_edges(session)

    # Add edges
    for edge in edges:
        graph.add_edge(
            format_content_id(edge.content_id), format_topic_id(edge.topic_id)
        )

    return graph
