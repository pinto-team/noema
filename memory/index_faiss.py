# -*- coding: utf-8 -*-
"""
NOEMA • memory/index_faiss.py — Minimal ANN Index with FAISS (V0)

Purpose:
  Vector search over episodic records using a key vector (z/s/mean).

Persistence:
  - <prefix>.index       : FAISS index (if FAISS is available)
  - <prefix>.meta.jsonl  : line-delimited metadata aligned with index order

If FAISS is not installed, a simple NumPy-based brute-force index is used.

API:
  idx = FaissIndex(IndexConfig(dim=64, kind="HNSW32", metric="ip"))
  idx.add([vec1, vec2, ...], metas=[{...}, ...])
  D, I, M = idx.search(query_vec, k=5)
  idx.save(prefix="data/index/faiss")
  idx2 = FaissIndex.load(prefix="data/index/faiss")

Notes:
  - For cosine similarity use L2-normalized vectors with metric="ip".
  - For L2 distance set metric="l2" and avoid normalization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pathlib import Path
import json
import numpy as np

# Try FAISS
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


# ---------------------- helpers ----------------------

def _as_float32(a: Iterable[float]) -> np.ndarray:
    return np.asarray(list(a), dtype=np.float32)

def _l2_normalize_inplace(X: np.ndarray) -> None:
    # X: [N, D]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X /= norms


# ---------------------- brute-force fallback ----------------------

class _BruteANN:
    """Simple O(ND) search used when FAISS is unavailable."""
    def __init__(self, dim: int, metric: str = "ip"):
        assert metric in ("ip", "l2")
        self.dim = int(dim)
        self.metric = metric
        self._X: Optional[np.ndarray] = None  # [N, D]

    def add(self, X: np.ndarray) -> None:
        if self._X is None:
            self._X = X.copy()
        else:
            self._X = np.vstack([self._X, X])

    def ntotal(self) -> int:
        return 0 if self._X is None else int(self._X.shape[0])

    def search(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        assert q.shape == (self.dim,), f"query shape {q.shape} != ({self.dim},)"
        if self._X is None or self.ntotal() == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        X = self._X  # [N, D]
        if self.metric == "ip":
            scores = X @ q  # [N]
            I = np.argsort(-scores)[:k]
            D = scores[I]
        else:  # l2
            dif = X - q[None, :]
            d2 = np.sum(dif * dif, axis=1)
            I = np.argsort(d2)[:k]
            D = d2[I]
        return D.astype(np.float32), I.astype(np.int64)


# ---------------------- FAISS wrapper ----------------------

@dataclass
class IndexConfig:
    dim: int = 64
    kind: str = "HNSW32"       # "FLAT", "HNSW32", "IVF100,PQ16", ...
    metric: str = "ip"         # "ip" (cosine via dot) or "l2"
    l2_normalize: bool = True  # True for cosine/IP workflows


class FaissIndex:
    """Thin wrapper over FAISS + sidecar metadata file."""

    def __init__(self, cfg: IndexConfig, index_obj: Any = None, metas: Optional[List[Dict[str, Any]]] = None):
        self.cfg = cfg
        self.metas: List[Dict[str, Any]] = metas or []
        self._index = index_obj or self._create_index(cfg)
        self._fallback = None
        if not _HAS_FAISS:
            self._fallback = _BruteANN(cfg.dim, cfg.metric)

    # ---------- index creation ----------
    @staticmethod
    def _create_index(cfg: IndexConfig):
        if not _HAS_FAISS:
            return None

        metric_const = faiss.METRIC_INNER_PRODUCT if cfg.metric == "ip" else faiss.METRIC_L2

        def _flat():
            if cfg.metric == "ip":
                return faiss.IndexFlatIP(cfg.dim)
            return faiss.IndexFlatL2(cfg.dim)

        kind = (cfg.kind or "FLAT").upper()
        try:
            if kind == "FLAT":
                return _flat()
            elif kind.startswith("HNSW"):
                # e.g., HNSW32 -> M=32 ; FAISS may or may not accept a metric arg by version.
                try:
                    m = int(kind.replace("HNSW", ""))
                except Exception:
                    m = 32
                try:
                    # newer FAISS: IndexHNSWFlat(dim, M, metric)
                    return faiss.IndexHNSWFlat(cfg.dim, m, metric_const)  # type: ignore[arg-type]
                except Exception:
                    # fallback: default metric (typically L2). With unit vectors, L2 rankings ≈ cosine.
                    return faiss.IndexHNSWFlat(cfg.dim, m)
            else:
                # Try IVF,PQ style (e.g., "IVF100,PQ16")
                try:
                    parts = kind.split(",")
                    nlist = int(parts[0].replace("IVF", ""))
                    pqm = int(parts[1].replace("PQ", "")) if len(parts) > 1 else 16
                    quantizer = _flat()
                    # Some FAISS versions place metric on index or quantizer; try safest form:
                    index = faiss.IndexIVFPQ(quantizer, cfg.dim, nlist, pqm, 8)
                    index.metric_type = metric_const  # may be ignored in older versions
                    index.nprobe = min(8, nlist)
                    return index
                except Exception:
                    return _flat()
        except Exception:
            return _flat()

    # ---------- add ----------
    def add(self, vecs: List[List[float]], metas: Optional[List[Dict[str, Any]]] = None) -> None:
        X = np.asarray(vecs, dtype=np.float32)
        assert X.ndim == 2 and X.shape[1] == self.cfg.dim, f"Expected [N,{self.cfg.dim}] got {X.shape}"
        if self.cfg.l2_normalize and self.cfg.metric == "ip":
            _l2_normalize_inplace(X)

        # FAISS
        if self._index is not None:
            if hasattr(self._index, "is_trained") and not self._index.is_trained:
                self._index.train(X)
            self._index.add(X)

        # Fallback
        if self._fallback is not None:
            self._fallback.add(X)

        # Metadata
        metas = metas or [{} for _ in range(X.shape[0])]
        self.metas.extend(metas)

    # ---------- count ----------
    def ntotal(self) -> int:
        if self._index is not None:
            return int(self._index.ntotal)
        if self._fallback is not None:
            return self._fallback.ntotal()
        return 0

    # ---------- search ----------
    def search(self, q: List[float], k: int = 5) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        if self.ntotal() == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64), []
        qv = _as_float32(q)
        assert qv.shape == (self.cfg.dim,), f"query dim mismatch: {qv.shape}"
        if self.cfg.l2_normalize and self.cfg.metric == "ip":
            qv = qv / (np.linalg.norm(qv) or 1.0)

        if self._index is not None:
            qv2 = qv[None, :]  # [1, D]
            D, I = self._index.search(qv2, min(k, self.ntotal()))
            D, I = D[0], I[0]
        else:
            D, I = self._fallback.search(qv, min(k, self.ntotal()))  # type: ignore

        metas = [self.metas[int(i)] for i in I if 0 <= int(i) < len(self.metas)]
        return D.astype(np.float32), I.astype(np.int64), metas

    # ---------- save ----------
    def save(self, prefix: str | Path = "data/index/faiss") -> None:
        prefix = str(prefix)
        idx_path = Path(prefix + ".index")
        meta_path = Path(prefix + ".meta.jsonl")
        idx_path.parent.mkdir(parents=True, exist_ok=True)

        if _HAS_FAISS and self._index is not None:
            faiss.write_index(self._index, str(idx_path))

        with meta_path.open("w", encoding="utf-8") as f:
            for m in self.metas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # ---------- load ----------
    @classmethod
    def load(cls, prefix: str | Path = "data/index/faiss", cfg: Optional[IndexConfig] = None) -> "FaissIndex":
        prefix = str(prefix)
        idx_path = Path(prefix + ".index")
        meta_path = Path(prefix + ".meta.jsonl")

        metas: List[Dict[str, Any]] = []
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if ln:
                        try:
                            metas.append(json.loads(ln))
                        except Exception:
                            continue

        index_obj = None
        if _HAS_FAISS and idx_path.exists():
            try:
                index_obj = faiss.read_index(str(idx_path))
                dim = int(index_obj.d)
                if cfg is None:
                    cfg = IndexConfig(dim=dim)  # metric/kind cannot be reliably read across FAISS variants
            except Exception:
                index_obj = None

        if cfg is None:
            cfg = IndexConfig()

        return cls(cfg=cfg, index_obj=index_obj, metas=metas)


# ---------------------- EpisodeStore integration ----------------------

def build_from_episode_store(
    store: Any,
    *,
    key_mode: str = "mean",
    dim: int = 64,
    metric: str = "ip",
    kind: str = "HNSW32",
    normalize: bool = True,
    limit_days: Optional[int] = None,
) -> FaissIndex:
    """
    Build an index from stored episodes.

    Args:
      store: EpisodeStore (memory/episodic.EpisodeStore)
      key_mode: "mean" | "z" | "s"
      limit_days: if set, only use episodes from the last `limit_days` days.
    """
    from datetime import datetime, timedelta
    from memory.episodic import make_key_vector, Episode  # type: ignore

    cfg = IndexConfig(dim=dim, kind=kind, metric=metric, l2_normalize=normalize and (metric == "ip"))
    idx = FaissIndex(cfg)

    # Collect episodes
    episodes: List[Episode] = []
    if limit_days is None:
        episodes = store.tail(n=5000)
    else:
        today = datetime.utcnow().date()
        start = today - timedelta(days=max(0, int(limit_days) - 1))
        for ep in store.iter_days(start.isoformat(), today.isoformat()):
            episodes.append(ep)

    if not episodes:
        return idx

    vecs: List[List[float]] = []
    metas: List[Dict[str, Any]] = []
    for ep in episodes:
        key = make_key_vector(ep, mode=key_mode, dim_limit=dim)
        if not key:
            continue
        vecs.append(key)
        metas.append({
            "ts": float(ep.ts),
            "session_id": ep.session_id,
            "intent": ep.intent,
            "action": ep.action_name,
            "text_in": ep.text_in,
            "text_out": ep.text_out,
        })

    if vecs:
        idx.add(vecs, metas=metas)
    return idx


# ---------------------- manual test ----------------------
if __name__ == "__main__":
    try:
        from memory.episodic import EpisodeStore  # type: ignore
    except Exception:
        raise RuntimeError("Please provide memory/episodic.py (EpisodeStore) for testing.")

    store = EpisodeStore()
    if not store.tail(1):
        store.log(
            session_id="S-TEST",
            text_in="hello",
            text_out="hi there!",
            intent="greeting",
            action_kind="skill",
            action_name="reply_greeting",
            r_total=1.0, r_int=0.2, r_ext=0.8,
            u=0.1, conf=0.9,
            s_vec=[0.1]*64, z_vec=[0.12]*64,
        )
        store.log(
            session_id="S-TEST",
            text_in="2+2?",
            text_out="4",
            intent="compute",
            action_kind="tool",
            action_name="invoke_calc",
            r_total=1.0, r_int=0.3, r_ext=0.7,
            u=0.1, conf=0.9,
            s_vec=[0.2]*64, z_vec=[0.21]*64,
        )

    idx = build_from_episode_store(store, key_mode="mean", dim=64, metric="ip", kind="HNSW32", normalize=True)
    print("ntotal:", idx.ntotal())

    q = np.array([0.12]*64, dtype=np.float32)
    q /= (np.linalg.norm(q) or 1.0)
    D, I, M = idx.search(q.tolist(), k=3)
    print("search D:", D)
    print("search I:", I)
    print("search metas:", M[:2])

    idx.save("data/index/faiss")
    idx2 = FaissIndex.load("data/index/faiss")
    print("loaded ntotal:", idx2.ntotal())
