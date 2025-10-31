# -*- coding: utf-8 -*-
"""
NOEMA • concept/graph.py — گراف مفاهیم (V0 با NetworkX، فایل‌محور)
- گره‌ها: ConceptNodeها (خروجی concept/clustering.py)
- یال‌ها: «هم‌وقوعیِ زمانی» بین مفاهیم متوالی در اپیزودها (temporal adjacency)
- کاربردها:
    * ساخت گراف از روی concepts + اپیزودها
    * انتساب هر اپیزود به نزدیک‌ترین مفهوم
    * جست‌وجوی همسایه‌ها/پیشنهاد گام بعد
    * ذخیره/بارگذاری گراف (JSON ساده)

وابستگی‌ها: numpy, networkx (سبک)، stdlib
"""

from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json, math, numpy as np
import networkx as nx

from concept.clustering import ConceptNode, load_concepts
from memory.episodic import EpisodeStore, make_key_vector

# ----------------------------- ابزار برداری -----------------------------

def _l2(v: List[float]) -> float:
    return math.sqrt(sum(x*x for x in v))

def _cos(a: List[float], b: List[float]) -> float:
    na = _l2(a) or 1.0
    nb = _l2(b) or 1.0
    return float(sum(x*y for x, y in zip(a, b)) / (na * nb))

def _centers_matrix(concepts: List[ConceptNode]) -> np.ndarray:
    if not concepts:
        return np.zeros((0, 0), dtype=np.float32)
    D = len(concepts[0].center)
    M = np.zeros((len(concepts), D), dtype=np.float32)
    for i, c in enumerate(concepts):
        v = np.asarray(c.center, dtype=np.float32)
        n = np.linalg.norm(v) or 1.0
        M[i] = v / n
    return M

def _nearest_center(vec: List[float], centers_L2N: np.ndarray) -> int:
    if centers_L2N.size == 0:
        return -1
    v = np.asarray(vec, dtype=np.float32)
    v = v / (np.linalg.norm(v) or 1.0)
    sims = centers_L2N @ v  # [K]
    return int(np.argmax(sims))

# ----------------------------- ساخت گراف -----------------------------

def build_graph_from_concepts_and_episodes(
    concepts: List[ConceptNode],
    store: EpisodeStore,
    *,
    assign_mode: str = "mean",     # "mean" | "z" | "s"
    dim: Optional[int] = None,     # اگر None باشد از center طول می‌گیرد
    limit_tail: int = 10000,       # چند اپیزود آخر
    min_conf: float = 0.0,
    edge_decay: float = 0.995,     # فراموشی ملایم روی یال‌ها
) -> nx.DiGraph:
    """
    گراف جهت‌دار می‌سازد: یال i→j یعنی «پس از مفهوم i، بیشتر j آمده».
    وزن یال = شمارش با فراموشیِ نمایی (approx).
    """
    G = nx.DiGraph()
    # افزودن گره‌ها
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

    # خواندن اپیزودها و نگاشت هرکدام به نزدیک‌ترین مفهوم
    episodes = store.tail(n=limit_tail)
    # فیلتر اطمینان
    episodes = [ep for ep in episodes if float(getattr(ep, "conf", 0.0)) >= min_conf]
    if not episodes:
        return G

    # ترتیب زمانی
    episodes.sort(key=lambda e: float(e.ts))

    last_node: Optional[str] = None
    decay_state = 1.0
    for ep in episodes:
        key = make_key_vector(ep, mode=assign_mode, dim_limit=D)
        if not key:
            continue
        k = _nearest_center(key, centers)
        if k < 0 or k >= len(concepts):
            continue
        node_id = concepts[k].id

        # بروز رسانی یال زمانی
        if last_node is not None and last_node != node_id:
            # فراموشی ملایم روی همه‌ی یال‌های خارج‌شونده از last_node
            for _, j, data in G.out_edges(last_node, data=True):
                data["w"] = data.get("w", 0.0) * edge_decay
            # تقویت یال last_node -> node_id
            if G.has_edge(last_node, node_id):
                G[last_node][node_id]["w"] = G[last_node][node_id].get("w", 0.0) + 1.0
                G[last_node][node_id]["n"] = G[last_node][node_id].get("n", 0) + 1
            else:
                G.add_edge(last_node, node_id, w=1.0, n=1)
        last_node = node_id

    # نرمال‌سازی انتخابی: وزن‌های خروجی هر گره را به احتمال تبدیل کن
    for u in G.nodes:
        out_edges = list(G.out_edges(u, data=True))
        s = sum(d.get("w", 0.0) for _, _, d in out_edges)
        if s > 0:
            for _, v, d in out_edges:
                d["p"] = d["w"] / s
        else:
            for _, v, d in out_edges:
                d["p"] = 0.0
    return G

# ----------------------------- ذخیره/بارگذاری -----------------------------

def save_graph_json(G: nx.DiGraph, path: str | Path = "data/concepts/graph.json") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "directed": True,
        "nodes": [
            {"id": n, **{k: v for k, v in G.nodes[n].items()}}
            for n in G.nodes
        ],
        "edges": [
            {"u": u, "v": v, **{k: d[k] for k in d}}
            for u, v, d in G.edges(data=True)
        ],
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
        u = ed.pop("u"); v = ed.pop("v")
        G.add_edge(u, v, **ed)
    return G

# ----------------------------- پرس‌وجوهای رایج -----------------------------

def nearest_concepts_to_vector(vec: List[float], concepts: List[ConceptNode], k: int = 5) -> List[Tuple[str, float]]:
    """k مفهوم نزدیک از روی شباهت کسینوسی به center."""
    out: List[Tuple[str, float]] = []
    for c in concepts:
        out.append((c.id, _cos(vec, c.center)))
    out.sort(key=lambda t: t[1], reverse=True)
    return out[:k]

def suggest_next_concepts(G: nx.DiGraph, current_id: str, topk: int = 5) -> List[Tuple[str, float]]:
    """پیشنهاد گام بعدی بر اساس احتمال‌های خروجی p(u→v)."""
    if (current_id not in G) or (G.out_degree(current_id) == 0):
        return []
    cand = []
    for _, v, d in G.out_edges(current_id, data=True):
        cand.append((v, float(d.get("p", d.get("w", 0.0)))))
    cand.sort(key=lambda t: t[1], reverse=True)
    return cand[:topk]

def k_hop_neighbors(G: nx.DiGraph, node_id: str, k: int = 2) -> List[str]:
    """گره‌های در فاصله‌ی حداکثر k (از یال‌های خروجی)."""
    if node_id not in G:
        return []
    frontier = {node_id}
    seen = {node_id}
    for _ in range(k):
        nxt = set()
        for u in frontier:
            for _, v in G.out_edges(u):
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)
        frontier = nxt
        if not frontier:
            break
    seen.remove(node_id)
    return list(seen)

# ----------------------------- اجرای یک‌مرحله‌ای -----------------------------

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
    Concepts + Episodes → Graph (JSON)
    """
    concepts = load_concepts(concepts_path)
    store = EpisodeStore(episodes_root)
    G = build_graph_from_concepts_and_episodes(
        concepts, store, assign_mode=assign_mode, limit_tail=limit_tail, min_conf=min_conf
    )
    save_graph_json(G, out_path)
    print(f"✅ concept graph saved: {out_path}  | nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    return G

# ----------------------------- تست سریع -----------------------------

if __name__ == "__main__":
    # اگر concepts.json خالی بود، فقط گراف خالی می‌سازد.
    G = run_end_to_end()
    # نمایش چند پیشنهاد از اولین گره (اگر وجود داشته باشد)
    nodes = list(G.nodes())
    if nodes:
        cur = nodes[0]
        print("node:", cur, "→", suggest_next_concepts(G, cur, topk=5))
