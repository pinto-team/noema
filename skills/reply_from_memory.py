"""NOEMA • skills.reply_from_memory — Retrieve replies from episodic memory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import math
import time

try:
    from perception.encoder import encode as _encode_text  # type: ignore
except Exception:
    _encode_text = None  # type: ignore

try:
    from memory import EpisodeStore, build_from_episode_store, make_key_vector  # type: ignore
except Exception:
    EpisodeStore = None  # type: ignore
    build_from_episode_store = None  # type: ignore
    make_key_vector = None  # type: ignore


_INDEX_CACHE: Optional[Any] = None
_INDEX_DIM: int = 64
_INDEX_STAMP: float = 0.0
_INDEX_TTL: float = 90.0


def _soft_hash(text: str, dim: int = 32) -> List[float]:
    """Simple deterministic encoder used when no model is available."""
    vec = [0.0] * dim
    for idx, ch in enumerate(text or ""):
        vec[idx % dim] += (ord(ch) % 23) / 23.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _pad(vec: List[float], dim: int) -> List[float]:
    if len(vec) > dim:
        return vec[:dim]
    if len(vec) < dim:
        return vec + [0.0] * (dim - len(vec))
    return vec


def _encode_vector(text: str, dim: int) -> List[float]:
    if callable(_encode_text):
        try:
            vec = list(_encode_text(text))
            if vec:
                return _pad(vec, dim)
        except Exception:
            pass
    return _pad(_soft_hash(text, dim=max(dim, 1)), dim)


def _guess_dim(store: Any) -> int:
    if store is None or make_key_vector is None:
        return 64
    try:
        tail = store.tail(1)
    except Exception:
        return 64
    if not tail:
        return 64
    key = make_key_vector(tail[0], mode="mean")
    return len(key) if key else 64


def _ensure_index(refresh: bool = False) -> Optional[Tuple[Any, int, str]]:
    global _INDEX_CACHE, _INDEX_DIM, _INDEX_STAMP
    if EpisodeStore is None or build_from_episode_store is None:
        return None
    now = time.time()
    if (
        _INDEX_CACHE is not None
        and not refresh
        and (now - _INDEX_STAMP) < _INDEX_TTL
    ):
        metric = getattr(getattr(_INDEX_CACHE, "cfg", None), "metric", "ip")
        return _INDEX_CACHE, _INDEX_DIM, str(metric)
    try:
        store = EpisodeStore()
        dim = _guess_dim(store)
        idx = build_from_episode_store(
            store,
            key_mode="mean",
            dim=max(dim, 16),
            normalize=True,
        )
    except Exception:
        _INDEX_CACHE = None
        _INDEX_STAMP = now
        _INDEX_DIM = 64
        return None
    _INDEX_STAMP = now
    if idx.ntotal() == 0:
        _INDEX_CACHE = None
        return None
    _INDEX_CACHE = idx
    _INDEX_DIM = int(getattr(idx.cfg, "dim", max(dim, 16)))
    metric = getattr(idx.cfg, "metric", "ip")
    return idx, _INDEX_DIM, str(metric)


def _score_from_distance(dist: float, metric: str) -> float:
    if metric == "l2":
        return 1.0 / (1.0 + float(dist))
    return max(-1.0, min(1.0, float(dist)))


def _search_memory(text: str, top_k: int, refresh: bool = False) -> List[Dict[str, Any]]:
    info = _ensure_index(refresh=refresh)
    if not info:
        return []
    idx, dim, metric = info
    query_vec = _encode_vector(text, dim)
    try:
        dists, inds, metas = idx.search(query_vec, k=max(1, top_k))
    except Exception:
        return []
    hits: List[Dict[str, Any]] = []
    for dist, ind, meta in zip(list(dists), list(inds), list(metas)):
        if int(ind) < 0:
            continue
        payload = dict(meta or {})
        payload["score"] = _score_from_distance(float(dist), metric)
        payload["score_raw"] = float(dist)
        payload["metric"] = metric
        hits.append(payload)
    hits.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return hits


def _simple_similarity(q: str, t: str, n: int = 3) -> float:
    if not q or not t:
        return 0.0
    q = q.strip()
    t = t.strip()
    if not q or not t:
        return 0.0
    n = max(1, n)
    qgrams = {q[i : i + n] for i in range(max(1, len(q) - n + 1))}
    tgrams = {t[i : i + n] for i in range(max(1, len(t) - n + 1))}
    if not qgrams or not tgrams:
        return 0.0
    return len(qgrams & tgrams) / float(len(qgrams | tgrams) or 1)


def _load_demos(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    demos: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            demos.append(json.loads(line))
        except Exception:
            continue
    return demos


def _reply_from_demos(user_text: str) -> Dict[str, Any]:
    mem_path = Path("data/demo_memory.jsonl")
    demos = _load_demos(mem_path)
    if not demos:
        return {
            "intent": "memory.reply",
            "text_out": "No episodic memory yet.",
            "meta": {"confidence": 0.35, "u": 0.55, "risk": 0.0, "r_total": 0.0},
            "extras": {"matches": []},
            "label_ok": True,
        }
    try:
        import numpy as np  # type: ignore
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from scipy.sparse import csr_matrix  # type: ignore

        idx_path = Path("data/demo_index.npz")
        vocab_path = Path("data/demo_vocab.json")
        if idx_path.exists() and vocab_path.exists():
            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
            vectorizer = TfidfVectorizer(vocabulary=vocab)
            query_vec = vectorizer.fit_transform([user_text or ""])
            data = np.load(idx_path)
            matrix = csr_matrix(
                (data["data"], data["indices"], data["indptr"]), shape=data["shape"]
            )
            scores = (matrix @ query_vec.T).toarray().ravel()
            best_idx = int(scores.argmax())
            score = float(scores[best_idx])
            output = str(demos[best_idx].get("output", "").strip())
            conf = 0.55 + 0.4 * (1.0 if score > 0 else 0.0)
            return {
                "intent": "memory.reply",
                "text_out": output or "No episodic memory yet.",
                "meta": {"confidence": min(0.9, conf), "u": 0.25, "risk": 0.0, "r_total": 0.0},
                "extras": {"matches": [{"index": best_idx, "score": score}]},
                "label_ok": True,
            }
    except Exception:
        pass

    sims = [(_simple_similarity(user_text or "", str(item.get("input", ""))), idx) for idx, item in enumerate(demos)]
    sims.sort(key=lambda item: item[0], reverse=True)
    best_score, best_idx = sims[0]
    output = str(demos[best_idx].get("output", "").strip())
    conf = 0.5 + 0.35 * min(1.0, float(best_score))
    return {
        "intent": "memory.reply",
        "text_out": output or "No episodic memory yet.",
        "meta": {"confidence": min(0.85, conf), "u": 0.3, "risk": 0.0, "r_total": 0.0},
        "extras": {"matches": [{"index": int(best_idx), "score": float(best_score)}]},
        "label_ok": True,
    }


def run(user_text: str = "", *, k: int = 3, refresh_index: bool = False, **kwargs) -> Dict[str, Any]:
    query = str(user_text or "")
    hits = _search_memory(query, top_k=max(1, int(k)), refresh=bool(refresh_index))
    if hits:
        best = hits[0]
        score = float(best.get("score", 0.0))
        text_out = str(best.get("text_out") or best.get("text_in") or "").strip()
        conf = 0.55 + 0.35 * max(0.0, min(1.0, score))
        return {
            "intent": "memory.reply",
            "text_out": text_out or "No episodic memory yet.",
            "meta": {
                "confidence": min(0.92, conf),
                "u": max(0.15, 0.5 - conf * 0.3),
                "risk": 0.0,
                "r_total": 0.0,
            },
            "extras": {"matches": hits[: max(1, int(k))]},
            "label_ok": True,
        }
    return _reply_from_demos(query)

