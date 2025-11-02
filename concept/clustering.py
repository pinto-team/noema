# -*- coding: utf-8 -*-
"""
NOEMA • concept/clustering.py — Cluster episodic representations into concept nodes (V0)

Inputs:
  - Episodic data (memory/episodic.EpisodeStore) and a key vector per episode.
Outputs:
  - Concept nodes with center vector, size, representative examples, and
    distributions over intents/actions.
Persistence:
  - data/concepts/concepts.json (consumed by concept/graph.py)

Idea:
  1) Collect semantic vectors per episode (mean of z/s or either one)
  2) MiniBatchKMeans clustering (k=auto unless specified)
  3) Pick representatives (closest episodes to the centroid) and summarize meta

API:
  from concept.clustering import (
      collect_vectors, auto_k, kmeans_cluster, build_concepts,
      save_concepts, load_concepts, run_end_to_end
  )

Requires: scikit-learn, numpy
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json
import math

import numpy as np

try:
    from sklearn.cluster import MiniBatchKMeans  # type: ignore
    _HAS_SK = True
except Exception:
    _HAS_SK = False


# ----------------------------- Concept data -----------------------------

@dataclass
class ConceptNode:
    id: str
    center: List[float]                         # cluster centroid
    count: int                                  # cluster size
    examples: List[Dict[str, Any]]              # representative episodes
    intents: List[Tuple[str, int]]              # (intent, count), sorted desc
    actions: List[Tuple[str, int]]              # (action, count), sorted desc
    tags: List[str]                             # dominant tags (optional)
    radius: float                               # mean intra-cluster distance (cosine or L2)


# ----------------------------- Vector utilities -----------------------------

def l2_normalize_rows(X: np.ndarray) -> None:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X /= norms


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) or 1.0
    nb = np.linalg.norm(b) or 1.0
    return float(1.0 - float(np.dot(a, b)) / (na * nb))


# ----------------------------- Collection -----------------------------

def collect_vectors(
    store: Any,
    *,
    key_mode: str = "mean",       # "mean" | "z" | "s"
    dim: int = 64,
    normalize: bool = True,
    limit: Optional[int] = 5000,
    min_conf: float = 0.0,        # keep episodes with conf >= min_conf
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Read episodes from EpisodeStore and build a key vector for each.

    Returns:
      - X: np.ndarray of shape [N, D]
      - metas: list of dicts with text/intent/action/ts/...
    """
    from memory.episodic import make_key_vector  # local import to avoid cycles

    assert hasattr(store, "tail"), "store must implement .tail(n=...)"
    episodes = store.tail(n=limit or 5000)

    vecs: List[List[float]] = []
    metas: List[Dict[str, Any]] = []

    for ep in episodes:
        if float(getattr(ep, "conf", 0.0)) < float(min_conf):
            continue
        key = make_key_vector(ep, mode=key_mode, dim_limit=dim)
        if not key:
            continue
        vecs.append(key)
        metas.append({
            "ts": float(getattr(ep, "ts", 0.0)),
            "intent": getattr(ep, "intent", "") or "",
            "action": getattr(ep, "action_name", "") or "",
            "text_in": getattr(ep, "text_in", "") or "",
            "text_out": getattr(ep, "text_out", "") or "",
            "tags": list(getattr(ep, "tags", []) or []),
        })

    if not vecs:
        return np.zeros((0, dim), dtype=np.float32), []

    X = np.asarray(vecs, dtype=np.float32)
    if normalize:
        l2_normalize_rows(X)
    return X, metas


# ----------------------------- Auto k heuristic -----------------------------

def auto_k(n: int, k_min: int = 4, k_max: int = 64) -> int:
    """
    Simple heuristic:
      k ≈ clamp( round(sqrt(n / 2)), k_min, k_max )
    """
    if n <= 0:
        return 0
    k = int(round(math.sqrt(max(1.0, n / 2.0))))
    return max(k_min, min(k, k_max))


# ----------------------------- KMeans clustering -----------------------------

def kmeans_cluster(
    X: np.ndarray,
    *,
    k: Optional[int] = None,
    batch_size: int = 1024,
    max_iter: int = 100,
    n_init: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MiniBatchKMeans clustering.
    Args:
      X: [N, D] float32 array
    Returns:
      labels: [N] int32
      centers: [k, D] float32
    """
    assert _HAS_SK, "scikit-learn is not installed."
    N = int(X.shape[0])
    if N == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, X.shape[1]), dtype=np.float32)
    k = k or auto_k(N)
    k = max(2, min(k, N))
    km = MiniBatchKMeans(
        n_clusters=k,
        batch_size=min(batch_size, max(256, k * 10)),
        max_iter=max_iter,
        n_init=n_init,
        random_state=seed,
        verbose=0,
    )
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    return labels.astype(np.int32), centers.astype(np.float32)


# ----------------------------- Build concepts -----------------------------

def build_concepts(
    X: np.ndarray,
    metas: List[Dict[str, Any]],
    labels: np.ndarray,
    centers: np.ndarray,
    *,
    topk_examples: int = 6,
) -> List[ConceptNode]:
    """
    For each cluster, compute:
      - size, centroid, representatives (closest to centroid),
      - distributions over intents/actions/tags, and
      - mean intra-cluster distance (radius).
    """
    assert X.shape[0] == len(metas) == labels.shape[0]
    K = int(centers.shape[0])

    # members per cluster
    members: List[List[int]] = [[] for _ in range(K)]
    for i, c in enumerate(labels.tolist()):
        if 0 <= c < K:
            members[c].append(i)

    concepts: List[ConceptNode] = []
    for c in range(K):
        idxs = members[c]
        if not idxs:
            continue
        C = centers[c]
        dists = [cosine_distance(X[i], C) for i in idxs]
        order = [i for _, i in sorted(zip(dists, idxs), key=lambda t: t[0])]
        reps = order[:max(1, int(topk_examples))]

        examples: List[Dict[str, Any]] = []
        intents_count: Dict[str, int] = {}
        actions_count: Dict[str, int] = {}
        tags_counter: Dict[str, int] = {}

        for i in reps:
            meta = metas[i]
            examples.append({
                "text_in": meta.get("text_in", ""),
                "text_out": meta.get("text_out", ""),
                "intent": meta.get("intent", ""),
                "action": meta.get("action", ""),
                "ts": meta.get("ts", 0.0),
            })

        for i in idxs:
            m = metas[i]
            intents_count[m.get("intent", "")] = intents_count.get(m.get("intent", ""), 0) + 1
            actions_count[m.get("action", "")] = actions_count.get(m.get("action", ""), 0) + 1
            for t in (m.get("tags") or []):
                tags_counter[t] = tags_counter.get(t, 0) + 1

        radius = float(np.mean([cosine_distance(X[i], C) for i in idxs])) if idxs else 0.0

        node = ConceptNode(
            id=f"C-{c:04d}",
            center=centers[c].astype(np.float32).tolist(),
            count=len(idxs),
            examples=examples,
            intents=sorted(intents_count.items(), key=lambda x: (-x[1], x[0]))[:8],
            actions=sorted(actions_count.items(), key=lambda x: (-x[1], x[0]))[:8],
            tags=[k for k, _ in sorted(tags_counter.items(), key=lambda x: (-x[1], x[0]))[:8]],
            radius=radius,
        )
        concepts.append(node)

    return concepts


# ----------------------------- Save/Load -----------------------------

def save_concepts(concepts: List[ConceptNode], path: str | Path = "data/concepts/concepts.json") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in concepts], f, ensure_ascii=False, indent=2)
    return p


def load_concepts(path: str | Path = "data/concepts/concepts.json") -> List[ConceptNode]:
    p = Path(path)
    if not p.exists():
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: List[ConceptNode] = []
    for obj in raw:
        out.append(ConceptNode(**obj))
    return out


# ----------------------------- End-to-end -----------------------------

def run_end_to_end(
    episode_store: Any,
    *,
    key_mode: str = "mean",
    dim: int = 64,
    normalize: bool = True,
    limit: Optional[int] = 5000,
    min_conf: float = 0.0,
    k: Optional[int] = None,
    out_path: str | Path = "data/concepts/concepts.json",
) -> List[ConceptNode]:
    """
    One-shot pipeline: collect → cluster → build nodes → save
    """
    X, metas = collect_vectors(
        episode_store,
        key_mode=key_mode,
        dim=dim,
        normalize=normalize,
        limit=limit,
        min_conf=min_conf,
    )
    if X.shape[0] == 0:
        print("⚠️ No vectors available for clustering.")
        return []

    if not _HAS_SK:
        raise RuntimeError("scikit-learn is not installed; MiniBatchKMeans unavailable.")

    labels, centers = kmeans_cluster(X, k=k)
    concepts = build_concepts(X, metas, labels, centers)
    path = save_concepts(concepts, out_path)
    print(f"✅ concepts saved to: {path}  (k={len(concepts)}, N={X.shape[0]})")
    return concepts


if __name__ == "__main__":
    try:
        from memory.episodic import EpisodeStore  # type: ignore
        store = EpisodeStore()
    except Exception as e:
        print("⚠️ Please implement memory/episodic.py (EpisodeStore).")
        raise

    concepts = run_end_to_end(store, key_mode="mean", dim=64, normalize=True, limit=1000, k=None)
    for c in concepts[:3]:
        print(f"{c.id} count={c.count} radius={c.radius:.3f} intents={c.intents[:3]}")
