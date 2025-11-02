# -*- coding: utf-8 -*-
"""
NOEMA • concept package

"""

# --- از clustering ---
from .clustering import (
    ConceptNode,
    collect_vectors,
    auto_k,
    kmeans_cluster,
    build_concepts,
    save_concepts,
    load_concepts,
    run_end_to_end as cluster_run_end_to_end,  # alias
)

# --- از graph ---
from .graph import (
    build_graph_from_concepts_and_episodes,
    save_graph_json,
    load_graph_json,
    nearest_concepts_to_vector,
    suggest_next_concepts,
    k_hop_neighbors,
    run_end_to_end as graph_run_end_to_end,   # alias
)

__all__ = [
    # clustering
    "ConceptNode",
    "collect_vectors",
    "auto_k",
    "kmeans_cluster",
    "build_concepts",
    "save_concepts",
    "load_concepts",
    "cluster_run_end_to_end",
    # graph
    "build_graph_from_concepts_and_episodes",
    "save_graph_json",
    "load_graph_json",
    "nearest_concepts_to_vector",
    "suggest_next_concepts",
    "k_hop_neighbors",
    "graph_run_end_to_end",
]
