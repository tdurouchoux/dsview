"""
Graph analysis core for content-topic graphs.
"""

from typing import Literal

import igraph as ig

from .build_graph import format_content_id, format_topic_id


def get_node_neighborhood(
    graph: ig.Graph,
    node_id: str,
    node_kind: Literal["content", "topic"],
    radius: int = 1,
) -> list[tuple[ig.Vertex, int]]:
    """
    Get the neighborhood of a node within a given radius.

    Args:
        graph: The igraph Graph
        node_id: The ID of the node to get the neighborhood for
        node_kind: Node kind necessary to identify the node within the graph
        radius: The radius of the neighborhood (default: 1)

    Returns:
        List of tuples containing the node and its distance from the original node
    """
    if node_kind == "topic":
        formatted_id = format_topic_id(node_id)
    elif node_kind == "content":
        formatted_id = format_content_id(node_id)
    else:
        raise ValueError("Invalid node type")

    results = []

    for order in range(1, radius + 1):
        neighborhood = graph.neighborhood(
            vertices=formatted_id, order=order, mindist=order
        )

        results.extend((graph.vs[node], order) for node in neighborhood)

    return results


def _select_node_kind(
    graph: ig.Graph,
    node_kind: str,
) -> list[int]:
    """Select nodes of a given kind."""
    return [v.index for v in graph.vs if v["kind"] == node_kind]


def _compute_metric(
    graph: ig.Graph,
    metric_name: str,
    nodes: list[str] | None = None,
    **kwargs,
) -> dict[int, float]:
    """Helper to compute a metric for all nodes or a subset."""
    scores = getattr(graph, metric_name)(vertices=nodes, **kwargs)

    if nodes is None:
        return {i: score for i, score in enumerate(scores)}

    return {nodes[i]: score for i, score in enumerate(scores)}


def degree(
    graph: ig.Graph,
    nodes: list[str] | None = None,
) -> dict[int, float]:
    """
    Compute degree for nodes.

    Args:
        graph: The igraph Graph
        nodes: Optional list of node IDs. If None, compute for all nodes.

    Returns:
        Dictionary mapping node IDs to their degree scores.
    """
    return _compute_metric(
        graph,
        "degree",
        nodes=nodes,
    )


def betweenness_centrality(
    graph: ig.Graph,
    nodes: list[str] | None = None,
) -> dict[int, float]:
    """
    Compute betweenness centrality for nodes.

    Args:
        graph: The igraph Graph
        nodes: Optional list of node IDs. If None, compute for all nodes.

    Returns:
        Dictionary mapping node IDs to their betweenness centrality scores.
    """
    return _compute_metric(graph, "betweenness", nodes)


def pagerank(
    graph: ig.Graph,
    nodes: list[str] | None = None,
    damping: float = 0.85,
) -> dict[int, float]:
    """
    Compute PageRank for nodes.

    Args:
        graph: The igraph Graph
        nodes: Optional list of node IDs. If None, compute for all nodes.
        damping: Damping factor

    Returns:
        Dictionary mapping node IDs to their PageRank scores.
    """
    return _compute_metric(graph, "pagerank", nodes, damping=damping)


METRIC_REPOSITORY = {
    "betweenness": betweenness_centrality,
    "degree": degree,
    "pagerank": pagerank,
}


def get_ranked_nodes(
    graph: ig.Graph,
    metric: Literal["betweenness", "degree", "pagerank"],
    node_kind: str | None = None,
    limit: int | None = 20,
    **kwargs,
) -> list[tuple[ig.Vertex, float]]:
    if metric not in METRIC_REPOSITORY:
        raise ValueError(f"Metric {metric} not supported")

    if node_kind is not None:
        nodes = _select_node_kind(graph, node_kind)
    else:
        nodes = None

    scores = METRIC_REPOSITORY[metric](graph, nodes, **kwargs)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if limit is not None:
        sorted_scores = sorted_scores[:limit]

    ranked_nodes = [(graph.vs[index], score) for index, score in sorted_scores]

    return ranked_nodes
