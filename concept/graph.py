# -*- coding: utf-8 -*-
"""
NOEMA • concept/graph.py — Concept graph (V0 with NetworkX, file-based)

Nodes:
  - ConceptNode entries (output of concept/clustering.py)
Edges:
  - Temporal adjacency between concepts (i.e., concept i tends to be followed by j)

Use cases:
  - Build graph from concepts + episodes
  - Assign each episode to its nearest concept
  - Query neighbors / suggest next concepts
  - Save/load the graph as a compact JSON

Dependencies: numpy, networkx, stdlib
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json
import math

import numpy as np
import networkx as nx

from concept.clustering import ConceptNode, load_concepts
from memory.episodic import EpisodeStore, make_key_vector  # type: ignore


# ----------------------------- Vector helpers -----------------------------

def _l2(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cos(a: List[float], b: List[float]) -> float:
    na = _l2(a) or 1.0
    nb = _l2(b) or 1.0
    return float(sum(x * y for x, y in zip(a, b)) / (na * nb))


def _centers_matrix(concepts: List[ConceptNode]) -> np.ndarray:
    if not concepts:
        return np.zeros((0, 0), dtype=np.float32)
    D = max(1, len(concepts[0].center))
    M = np.zeros((len(concepts), D), dtype=np.float32)
    for i, c in enumerate(concepts):
        v = np.asarray(c.center, dtype=np.float32)
        n = float(np.linalg.norm(v)) or 1.0
        M[i] = v / n
    return M


def _nearest_center(vec: List[float], centers_L2N: np.ndarray) -> int:
    if centers_L2N.size == 0:
        return -1
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v)) or 1.0
    v = v / n
    sims = centers_L2N @ v  # [K]
    return int(np.argmax(sims))


# ----------------------------- Build graph -----------------------------

def build_graph_from_concepts_and_episodes(
    concepts: List[ConceptNode],
    store: EpisodeStore,
    *,
    assign_mode: str = "mean",     # "mean" | "z" | "s"
    dim: Optional[int] = None,     # if None, inferred from center length
    limit_tail: int = 10000,       # how many recent episodes
    min_conf: float = 0.0,
    edge_decay: float = 0.995,     # mild forgetfulness on outgoing edges
) -> nx.DiGraph:
    """
    Build a directed graph: edge i→j means "j tends to follow i".
    Edge weight w is decayed over time and reinforced on transitions.
    """
    G = nx.DiGraph()

    # add nodes
    for c in concepts:
        G.add_node(c.id, **{
            "center": list(c.center),
            "count": int(c.count),
            "radius": float(c.radius),
            "intents": list(c.intents),
            "actions": list(c.actions),
            "tags": list(c.tags),
        })

    if not concepts:
        return G

    D = len(concepts[0].center) if dim is None else int(dim)
    centers = _centers_matrix(concepts)  # [K, D]

    # read episodes and map each to the nearest concept
    episodes = store.tail(n=limit_tail)
    episodes = [ep for ep in episodes if float(getattr(ep, "conf", 0.0)) >= float(min_conf)]
    if not episodes:
        return G

    episodes.sort(key=lambda e: float(getattr(e, "ts", 0.0)))

    last_node: Optional[str] = None
    for ep in episodes:
        key = make_key_vector(ep, mode=assign_mode, dim_limit=D)
        if not key:
            continue
        k = _nearest_center(key, centers)
        if k < 0 or k >= len(concepts):
            continue
        node_id = concepts[k].id

        # temporal edge update
        if last_node is not None and last_node != node_id:
            # decay all outgoing edges from last_node
            for _, v, data in G.out_edges(last_node, data=True):
                data["w"] = float(data.get("w", 0.0)) * float(edge_decay)
            # reinforce last_node -> node_id
            if G.has_edge(last_node, node_id):
                G[last_node][node_id]["w"] = float(G[last_node][node_id].get("w", 0.0)) + 1.0
                G[last_node][node_id]["n"] = int(G[last_node][node_id].get("n", 0)) + 1
            else:
                G.add_edge(last_node, node_id, w=1.0, n=1)
        last_node = node_id

    # optional normalization: convert outgoing weights to probabilities
    for u in G.nodes:
        out_edges = list(G.out_edges(u, data=True))
        s = sum(float(d.get("w", 0.0)) for _, _, d in out_edges)
        if s > 0:
            for _, v, d in out_edges:
                d["p"] = float(d.get("w", 0.0)) / s
        else:
            for _, v, d in out_edges:
                d["p"] = 0.0
    return G


# ----------------------------- Save/Load -----------------------------

def save_graph_json(G: nx.DiGraph, path: str | Path = "data/concepts/graph.json") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "directed": True,
        "nodes": [{"id": n, **{k: v for k, v in G.nodes[n].items()}} for n in G.nodes],
        "edges": [{"u": u, "v": v, **{k: d[k] for k in d}} for u, v, d in G.edges(data=True)],
    }
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_graph_json(path: str | Path = "data/concepts/graph.json") -> nx.DiGraph:
    p = Path(path)
    G = nx.DiGraph()
    if not p.exists():
        return G
    raw = json.loads(p.read_text(encoding="utf-8"))
    for nd in raw.get("nodes", []):
        nid = nd.pop("id")
        G.add_node(nid, **nd)
    for ed in raw.get("edges", []):
        u = ed.pop("u")
        v = ed.pop("v")
        G.add_edge(u, v, **ed)
    return G


# ----------------------------- Queries -----------------------------

def nearest_concepts_to_vector(vec: List[float], concepts: List[ConceptNode], k: int = 5) -> List[Tuple[str, float]]:
    """Return k nearest concepts by cosine similarity to the centroid."""
    out: List[Tuple[str, float]] = []
    for c in concepts:
        out.append((c.id, _cos(vec, c.center)))
    out.sort(key=lambda t: t[1], reverse=True)
    return out[:max(1, int(k))]


def suggest_next_concepts(G: nx.DiGraph, current_id: str, topk: int = 5) -> List[Tuple[str, float]]:
    """Suggest likely next concepts based on outgoing probabilities p(u→v)."""
    if (current_id not in G) or (G.out_degree(current_id) == 0):
        return []
    cand: List[Tuple[str, float]] = []
    for _, v, d in G.out_edges(current_id, data=True):
        cand.append((v, float(d.get("p", d.get("w", 0.0)))))
    cand.sort(key=lambda t: t[1], reverse=True)
    return cand[:max(1, int(topk))]


def k_hop_neighbors(G: nx.DiGraph, node_id: str, k: int = 2) -> List[str]:
    """Return nodes within k hops via outgoing edges."""
    if node_id not in G:
        return []
    frontier = {node_id}
    seen = {node_id}
    for _ in range(max(0, int(k))):
        nxt = set()
        for u in frontier:
            for _, v in G.out_edges(u):
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)
        frontier = nxt
        if not frontier:
            break
    seen.discard(node_id)
    return list(seen)


# ----------------------------- End-to-end -----------------------------

def run_end_to_end(
    concepts_path: str | Path = "data/concepts/concepts.json",
    episodes_root: str | Path = "data/episodes",
    out_path: str | Path = "data/concepts/graph.json",
    *,
    assign_mode: str = "mean",
    limit_tail: int = 10000,
    min_conf: float = 0.0,
) -> nx.DiGraph:
    """
    Concepts + Episodes → Graph (saved to JSON)
    """
    concepts = load_concepts(concepts_path)
    store = EpisodeStore(episodes_root)
    G = build_graph_from_concepts_and_episodes(
        concepts, store, assign_mode=assign_mode, limit_tail=limit_tail, min_conf=min_conf
    )
    save_graph_json(G, out_path)
    print(f"✅ concept graph saved: {out_path}  | nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    return G


if __name__ == "__main__":
    G = run_end_to_end()
    nodes = list(G.nodes())
    if nodes:
        cur = nodes[0]
        print("node:", cur, "→", suggest_next_concepts(G, cur, topk=5))
